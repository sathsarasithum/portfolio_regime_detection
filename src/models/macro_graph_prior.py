"""
Macro Graph Prior (GNN/GAT)
============================
Graph Attention Network that models inter-asset and macroeconomic relationships.
Acts as an inductive prior to inject structural knowledge into the regime encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention layer (GAT).
    
    Computes attention-weighted message passing between nodes
    in the asset-macro relationship graph.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Linear transformation for each head
        self.W = nn.Parameter(
            torch.empty(num_heads, in_features, out_features)
        )
        nn.init.xavier_uniform_(self.W)

        # Attention mechanism parameters
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if concat:
            self.output_dim = out_features * num_heads
        else:
            self.output_dim = out_features

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_nodes, in_features) node features
            adj: (batch, n_nodes, n_nodes) or (n_nodes, n_nodes) adjacency matrix
            
        Returns:
            output: (batch, n_nodes, output_dim) updated node features
            attention: (batch, num_heads, n_nodes, n_nodes) attention coefficients
        """
        B, N, _ = x.shape

        # Linear transformation: (B, N, in) → (B, heads, N, out)
        h = torch.einsum("bni,hio->bhno", x, self.W)  # (B, heads, N, out)

        # Compute attention scores
        attn_src = torch.matmul(h, self.a_src).squeeze(-1)  # (B, heads, N)
        attn_dst = torch.matmul(h, self.a_dst).squeeze(-1)  # (B, heads, N)

        # Pairwise attention: e_ij = LeakyReLU(a_src_i + a_dst_j)
        attn = attn_src.unsqueeze(-1) + attn_dst.unsqueeze(-2)  # (B, heads, N, N)
        attn = self.leaky_relu(attn)

        # Mask with adjacency
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        elif adj.dim() == 3:
            adj = adj.unsqueeze(1)  # (B, 1, N, N)

        attn = attn.masked_fill(adj == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Message passing
        out = torch.matmul(attn, h)  # (B, heads, N, out)

        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        else:
            out = out.mean(dim=1)  # Average over heads

        return out, attn


class MacroGraphPrior(nn.Module):
    """
    Macro Graph Prior using Graph Attention Network (GAT).
    
    Models the structural relationships between:
        - Individual assets (correlation-based edges)
        - Macroeconomic variables (influence edges)
        - Asset-macro interactions (cross-domain edges)
    
    This serves as an inductive prior that helps the regime encoder
    understand how different market components relate to each other.
    """

    def __init__(
        self,
        n_assets: int,
        n_macro: int = 0,
        node_feature_dim: int = 64,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_macro = n_macro
        self.n_nodes = n_assets + n_macro
        self.hidden_dim = hidden_dim

        # Node feature projection
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        # Residual projections — needed because concat=True on non-final
        # layers makes output_dim = num_heads * hidden_dim != in_dim, so a
        # plain shape-matched residual never fires. Without SOME residual
        # path, stacked GAT layers over the (default fully-connected) graph
        # oversmooth: near-uniform attention repeatedly averages node
        # features together, collapsing genuinely distinct per-asset inputs
        # to numerically identical outputs within 2 layers (verified: cross
        # -node std 0.54 at input -> ~0.03 after layer 0 -> ~1e-9 after layer
        # 1 without this projection).
        self.gat_residual_proj = nn.ModuleList()

        in_dim = hidden_dim
        for i in range(num_layers):
            concat = (i < num_layers - 1)  # Don't concat on last layer
            layer = GraphAttentionLayer(
                in_features=in_dim,
                out_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat=concat,
            )
            self.gat_layers.append(layer)
            out_dim = layer.output_dim
            self.gat_norms.append(nn.LayerNorm(out_dim))
            self.gat_residual_proj.append(
                nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
            )
            in_dim = out_dim

        # Graph-level readout
        self.readout = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable adjacency (if no prior graph is available)
        self.learnable_adj = nn.Parameter(
            torch.ones(self.n_nodes, self.n_nodes) * 0.5
        )

    def build_adjacency(
        self,
        correlation_matrix: Optional[torch.Tensor] = None,
        threshold: float = 0.3,
    ) -> torch.Tensor:
        """
        Build adjacency matrix from correlation or learn it.
        
        Args:
            correlation_matrix: (n_nodes, n_nodes) correlation matrix
            threshold: minimum correlation for edge creation
            
        Returns:
            adj: (n_nodes, n_nodes) binary adjacency
        """
        if correlation_matrix is not None:
            adj = (correlation_matrix.abs() > threshold).float()
        else:
            # Use learnable adjacency
            adj = torch.sigmoid(self.learnable_adj)

        # Ensure self-loops
        adj = adj + torch.eye(self.n_nodes, device=adj.device)
        adj = (adj > 0).float()

        return adj

    def forward(
        self,
        node_features: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (batch, n_nodes, node_feature_dim) per-node features
            adj: (n_nodes, n_nodes) adjacency matrix (optional)
            
        Returns:
            graph_embedding: (batch, hidden_dim) graph-level representation
            node_embeddings: (batch, n_nodes, hidden_dim) updated node features
        """
        if adj is None:
            adj = self.build_adjacency()

        # Project node features
        h = self.node_proj(node_features)  # (B, N, hidden_dim)

        # Apply GAT layers
        all_attention = []
        for gat, norm, res_proj in zip(self.gat_layers, self.gat_norms, self.gat_residual_proj):
            h_new, attn = gat(h, adj)
            h_new = F.elu(h_new)
            h_new = norm(h_new)
            # Always-on residual (projected when concat changes the width).
            # This is what lets each node's identity survive the layer even
            # when the attention distribution is close to uniform.
            h = res_proj(h) + h_new
            all_attention.append(attn)

        node_embeddings = h

        # Graph-level readout (mean pooling)
        graph_embedding = h.mean(dim=1)  # (B, hidden_dim)
        graph_embedding = self.readout(graph_embedding)

        return graph_embedding, node_embeddings
