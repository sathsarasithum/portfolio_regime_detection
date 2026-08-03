"""
Variable Selection Network (VSN) + LSTM
========================================
V-VSN: Variable-wise Variable Selection Network with LSTM memory.
Part of the Differentiable Regime Encoder.

The VSN learns to select and weight the most relevant input features
at each timestep, filtering noise before the LSTM processes temporal dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) — core building block of the VSN.
    Applies gated skip connections with ELU activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Primary layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Context projection (optional)
        self.context_proj = None
        if context_dim > 0:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        # Gating mechanism
        self.gate = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection projection if dimensions differ
        self.skip_proj = None
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        # Skip connection
        residual = x if self.skip_proj is None else self.skip_proj(x)

        # Primary path
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # Gating
        gate = torch.sigmoid(self.gate(h))
        h = gate * h

        # Add & Norm
        output = self.layer_norm(h + residual)
        return output
        


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) — vectorised implementation.

    Instead of one GRN per variable (expensive for 700+ features), this version:
      1. Projects all variables jointly to hidden_dim via a shared linear layer.
      2. Computes per-variable selection weights via a GRN on the flattened input.
      3. Weights and sums the projected per-variable embeddings.

    This gives the same selective attention behaviour at O(1) parameter cost
    relative to n_variables, making it practical for 700+ features.
    """

    def __init__(self, n_variables: int, hidden_dim: int,
                 dropout: float = 0.1, context_dim: int = 0):
        super().__init__()
        self.n_variables = n_variables
        self.hidden_dim = hidden_dim

        # Shared linear projection: each scalar feature → hidden_dim embedding
        # Uses a single weight matrix (n_variables, hidden_dim) applied in parallel
        self.var_proj = nn.Linear(n_variables, hidden_dim * n_variables, bias=False)

        # Variable selection weights via GRN on the flat input
        self.selection_grn = GatedResidualNetwork(
            n_variables, hidden_dim, n_variables, dropout, context_dim
        )

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_variables) input features
            context: optional context vector

        Returns:
            selected: (batch, seq_len, hidden_dim) selected feature representation
            weights: (batch, seq_len, n_variables) variable importance weights
        """
        B, T, N = x.shape

        # ── Variable selection weights ────────────────────────────────
        flat_x = x.reshape(B * T, N)
        flat_context = None
        if context is not None:
            flat_context = context.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        weights = self.selection_grn(flat_x, flat_context)   # (B*T, N)
        weights = F.softmax(weights, dim=-1)                  # (B*T, N)
        weights = weights.reshape(B, T, N)                   # (B, T, N)

        # ── Per-variable projection (vectorised) ──────────────────────
        # Project: (B*T, N) → (B*T, hidden_dim, N) via reshape trick
        # var_proj maps N scalars to N*hidden_dim values; we reshape to (N, hidden_dim)
        projected = self.var_proj(flat_x)                    # (B*T, N*hidden_dim)
        projected = projected.view(B * T, N, self.hidden_dim)  # (B*T, N, hidden_dim)

        # ── Weighted combination ──────────────────────────────────────
        w = weights.reshape(B * T, N).unsqueeze(-1)          # (B*T, N, 1)
        selected = (projected * w).sum(dim=1)                # (B*T, hidden_dim)
        selected = selected.reshape(B, T, self.hidden_dim)   # (B, T, hidden_dim)

        return selected, weights


class VSNLSTM(nn.Module):
    """
    V-VSN + LSTM: Variable Selection followed by LSTM temporal memory.

    Architecture:
        Input x_t → VSN (feature selection) → LSTM (temporal dynamics) → h_t
    """

    def __init__(
        self,
        n_features: int,
        vsn_hidden_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.lstm_hidden_dim = lstm_hidden_dim

        # Vectorised Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            n_variables=n_features,
            hidden_dim=vsn_hidden_dim,
            dropout=dropout,
        )

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=vsn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # Normalizes the LSTM output scale. Without it a stacked LSTM at
        # default init attenuates the signal ~4x per layer, so h_temp ends up
        # almost independent of the observation and the policy emits the same
        # allocation every day.
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)

        self._init_lstm()

    def _init_lstm(self):
        """
        Orthogonal recurrent weights + unit forget-gate bias (Jozefowicz et
        al., 2015). The forget bias keeps the cell state from decaying to zero
        across the window, which is what preserves input sensitivity.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                for i in range(0, param.shape[0], self.lstm_hidden_dim):
                    nn.init.orthogonal_(param[i:i + self.lstm_hidden_dim])
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # PyTorch gate order is [input, forget, cell, output]
                param.data[self.lstm_hidden_dim:2 * self.lstm_hidden_dim] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Args:
            x: (batch, seq_len, n_features) temporal context window
            hidden: optional LSTM hidden state

        Returns:
            output: (batch, seq_len, lstm_hidden_dim) LSTM outputs
            var_weights: (batch, seq_len, n_features) variable importance
            hidden: updated LSTM hidden state
        """
        # Variable selection
        selected, var_weights = self.vsn(x)   # (B, T, vsn_hidden_dim)

        # LSTM temporal processing
        lstm_out, hidden = self.lstm(selected, hidden)   # (B, T, lstm_hidden_dim)

        # Project output
        output = self.output_proj(self.lstm_norm(lstm_out))

        return output, var_weights, hidden
