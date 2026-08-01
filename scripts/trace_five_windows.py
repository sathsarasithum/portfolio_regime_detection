"""
FIVE-WINDOW END-TO-END TRACE
=============================
Walks 5 real sliding windows (built from data/raw/*.csv) through every module of
the PPO agent, printing the input, output, shape and parameters of each block,
then the reward, the backward pass and the resulting parameter change --
before moving on to the next window.

Uses the project's own classes (no re-implementation), plus manual
recomputation of the key tensor algebra to verify each module's math.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import get_config, DATA_RAW_DIR
from src.data.data_loader import CSEDataLoader
from src.data.preprocessor import Preprocessor
from src.environment.market_env import MarketEnvironment
from src.environment.reward import RewardCalculator
from src.models.ppo_agent import PPOAgent
from src.training.advantage import compute_gae, normalize_advantages, RewardNormalizer
from src.training.optimizer import create_optimizer, create_scheduler

np.set_printoptions(precision=6, suppress=True, linewidth=150)
torch.set_printoptions(precision=6, sci_mode=False, linewidth=150)

N_WINDOWS = 5
N_ASSETS = 47          # main.py default
FOCUS_ASSET = 0        # asset index printed in full detail
SEED = 42


def hr(title, ch="="):
    print("\n" + ch * 100)
    print(title)
    print(ch * 100)


def sub(title):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def vec(x, n=8):
    """Compact view of a 1-D tensor/array: first n entries + stats."""
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    flat = a.ravel()
    head = np.array2string(flat[:n], precision=6, suppress_small=True)
    return (f"{head}{' ...' if flat.size > n else ''}  "
            f"[min={flat.min():.6f} max={flat.max():.6f} mean={flat.mean():.6f} std={flat.std():.6f}]")


def pcount(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def param_table(module, name):
    print(f"\n  Parameters of {name}  (total trainable = {pcount(module):,})")
    for pn, p in module.named_parameters():
        print(f"    {pn:<48} {str(tuple(p.shape)):<20} {p.numel():>10,}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 0 -- RAW DATA -> WINDOWS  (real pipeline, exactly as main.py does it)
# ══════════════════════════════════════════════════════════════════════════════
hr("PART 0 :: RAW DATA -> ENGINEERED FEATURES -> NORMALISED -> 5 WINDOWS")

torch.manual_seed(SEED)
np.random.seed(SEED)

config = get_config(seed=SEED, device="cpu")
config.data.n_assets = N_ASSETS
config.data.n_macro_variables = 0
# main.py's CLI defaults (what `python main.py` actually runs with)
config.ppo.total_timesteps = 10_000

loader = CSEDataLoader(raw_data_dir=DATA_RAW_DIR)
prices, volumes = loader.load_prices_and_volumes(n_assets=N_ASSETS)
config.data.asset_names = list(prices.columns)
config.data.n_assets = len(config.data.asset_names)

sub("0.1  RAW price matrix straight out of CSEDataLoader (data/raw/2021..2025_banking_sector.csv)")
print(f"price_matrix  shape = {prices.shape}   (dates x assets)")
print(f"volume_matrix shape = {volumes.shape}")
print(f"date range          = {prices.index[0].date()}  ->  {prices.index[-1].date()}")
print(f"n_assets selected   = {config.data.n_assets}")
print(f"asset order (node order fed to the GAT) = {config.data.asset_names}")
FA = config.data.asset_names[FOCUS_ASSET]
print(f"\nFOCUS ASSET for detailed printing: index {FOCUS_ASSET} = '{FA}'")
print(f"\nFirst 6 raw rows, first 6 assets (Rs. close price):")
print(prices.iloc[:6, :6].to_string())
print(f"\nFirst 6 raw rows, first 6 assets (share volume):")
print(volumes.iloc[:6, :6].to_string())

preprocessor = Preprocessor(window_size=config.data.window_size,
                            normalize_method=config.data.normalize_method)

sub("0.2  FEATURE ENGINEERING  (Preprocessor.engineer_features)")
features_all = preprocessor.engineer_features(prices, volumes, macro_data=None)
print(f"engineer_features output shape = {features_all.shape}  "
      f"(T={features_all.shape[0]} usable dates, {features_all.shape[1]} raw feature columns)")
print(f"rows lost to rolling windows / diffs: {len(prices) - len(features_all)}")
features_all = preprocessor.reorder_per_asset(features_all, config.data.asset_names)
F_PER_ASSET = preprocessor.n_features_per_asset
print(f"after reorder_per_asset       = {features_all.shape}   "
      f"= n_assets({config.data.n_assets}) x n_features_per_asset({F_PER_ASSET})")
print(f"\nThe {F_PER_ASSET} features every asset contributes, in order:")
for i, s in enumerate(preprocessor.PER_ASSET_SUFFIXES):
    print(f"    f{i:<2} {FA}{s}")

prices = prices.loc[features_all.index]
volumes = volumes.loc[features_all.index]

sub("0.3  CHRONOLOGICAL SPLIT + NORMALISATION (scaler fitted on TRAIN only)")
n_total = len(features_all)
n_train = int(n_total * config.data.train_ratio)
n_val = int(n_total * config.data.val_ratio)
features_train = features_all.iloc[:n_train]
prices_train = prices.loc[features_train.index]
print(f"train = {len(features_train)} steps  ({features_train.index[0].date()} -> {features_train.index[-1].date()})")
print(f"val   = {n_val} steps,  test = {n_total - n_train - n_val} steps")

norm_train = preprocessor.fit_normalize(features_train)
scaler = preprocessor.scalers["features"]
windows_train = preprocessor.create_windows(norm_train.values)
prices_train_aligned = prices_train.iloc[config.data.window_size - 1:].values
W = config.data.window_size
print(f"\ncreate_windows -> windows_train shape = {windows_train.shape}  "
      f"= (n_windows, window_size={W}, n_features={windows_train.shape[2]})")
print(f"prices_train_aligned shape = {prices_train_aligned.shape}  (price on each window's END date)")

sub("0.4  THE 5 WINDOWS WE WILL TRACE")
dates = features_train.index
for w in range(N_WINDOWS):
    print(f"  window {w+1}: rows {w}..{w+W-1}   dates {dates[w].date()} -> {dates[w+W-1].date()}   "
          f"decision date = {dates[w+W-1].date()}, reward earned over {dates[w+W-1].date()} -> {dates[w+W].date()}")

sub(f"0.5  RAW (un-normalised) engineered features for '{FA}' on the 5 decision dates")
cols_fa = [f"{FA}{s}" for s in preprocessor.PER_ASSET_SUFFIXES]
raw_focus = features_train[cols_fa].iloc[W - 1: W - 1 + N_WINDOWS]
print(raw_focus.T.to_string())

sub(f"0.6  Z-SCORED values of the same cells  (x - mean)/std, StandardScaler fitted on train")
means = pd.Series(scaler.mean_, index=features_train.columns)
stds = pd.Series(np.sqrt(scaler.var_), index=features_train.columns)
chk = pd.DataFrame({
    "raw": raw_focus.iloc[0],
    "train_mean": means[cols_fa],
    "train_std": stds[cols_fa],
    "z = (raw-mean)/std": (raw_focus.iloc[0] - means[cols_fa]) / stds[cols_fa],
    "norm_train value": norm_train[cols_fa].iloc[W - 1],
})
print(f"verification for decision date {dates[W-1].date()}:")
print(chk.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 -- MODEL / ENV / TRAINER CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
hr("PART 1 :: MODEL CONSTRUCTION -- every block, its shape contract and its parameters")

env = MarketEnvironment(
    prices=prices_train_aligned,
    features=windows_train,
    initial_value=config.environment.initial_portfolio_value,
    transaction_cost_rate=config.reward.transaction_cost_rate,
    slippage=config.environment.slippage,
    allow_short=config.environment.allow_short_selling,
    max_position_size=config.environment.max_position_size,
    vol_target=config.environment.vol_target,
    vol_ewma_span=config.environment.vol_ewma_span,
)

agent = PPOAgent(
    n_features_per_asset=F_PER_ASSET,
    n_assets=config.data.n_assets,
    n_macro=0,
    vsn_hidden_dim=config.regime_encoder.vsn_hidden_dim,
    lstm_hidden_dim=config.regime_encoder.lstm_hidden_dim,
    lstm_num_layers=config.regime_encoder.lstm_num_layers,
    mha_num_heads=config.regime_encoder.mha_num_heads,
    gnn_hidden_dim=config.regime_encoder.gnn_hidden_dim,
    gnn_num_heads=config.regime_encoder.gnn_num_heads,
    gnn_num_layers=config.regime_encoder.gnn_num_layers,
    n_regimes=config.regime_encoder.n_regimes,
    latent_state_dim=config.regime_encoder.latent_state_dim,
    actor_hidden_dims=config.actor.hidden_dims,
    critic_hidden_dims=config.critic.hidden_dims,
    encoder_dropout=config.regime_encoder.lstm_dropout,
    mha_dropout=config.regime_encoder.mha_dropout,
    actor_dropout=config.actor.dropout,
    critic_dropout=config.critic.dropout,
    allow_short=config.environment.allow_short_selling,
)

enc = agent.regime_encoder
print("\nParameter budget:")
for k, v in agent.count_parameters().items():
    print(f"  {k:<16} {v:>12,}")
print(f"\n  breakdown inside the regime encoder:")
for nm, mod in [("vsn_lstm.vsn", enc.vsn_lstm.vsn), ("vsn_lstm.lstm", enc.vsn_lstm.lstm),
                ("vsn_lstm.lstm_norm", enc.vsn_lstm.lstm_norm),
                ("vsn_lstm.output_proj", enc.vsn_lstm.output_proj),
                ("temporal_attention", enc.temporal_attention),
                ("macro_graph", enc.macro_graph), ("fusion", enc.fusion),
                ("regime_classifier", enc.regime_classifier)]:
    print(f"    {nm:<28} {pcount(mod):>12,}")
print(f"    {'regime_temperature':<28} {enc.regime_temperature.numel():>12,}")

param_table(enc.vsn_lstm.vsn, f"VariableSelectionNetwork (shared by all {config.data.n_assets} assets)")
param_table(enc.vsn_lstm.lstm, "LSTM (2 layers)")
param_table(enc.temporal_attention, "TemporalAttentionBlock (MHA + FFN)")
param_table(enc.macro_graph, "MacroGraphPrior (GAT)")
param_table(agent.actor, "ActorNetwork")
param_table(agent.critic, "CriticNetwork")

# real trainer pieces
optimizer = create_optimizer(agent, config.ppo.learning_rate)
total_updates = config.ppo.total_timesteps // config.ppo.rollout_length * config.ppo.n_epochs
scheduler = create_scheduler(optimizer, total_updates)
reward_calc = RewardCalculator()
reward_normalizer = RewardNormalizer(gamma=config.ppo.gamma)

print(f"\nOptimizer = AdamW, 3 param groups")
for g in optimizer.param_groups:
    print(f"    group '{g['name']:<14}' initial_lr={g['initial_lr']:.3e}  current lr={g['lr']:.3e}  "
          f"n_tensors={len(g['params'])}")
print(f"scheduler: LambdaLR, warmup_steps=1000, total_steps={total_updates} "
      f"(total_timesteps={config.ppo.total_timesteps} // rollout_length={config.ppo.rollout_length} "
      f"* n_epochs={config.ppo.n_epochs})")
print(f"    lr_lambda(0) = 0/1000 = 0.0  -> the FIRST optimizer.step() runs at lr = 0 exactly.")
print(f"    warmup_steps(1000) > total_steps({total_updates}), so with these defaults the LR only ever")
print(f"    ramps to {total_updates}/1000 = {total_updates/1000:.0%} of base and never reaches the cosine phase.")

sub("Environment configuration")
print(f"  prices                 {env.prices.shape}   (aligned to window end dates)")
print(f"  features (windows)     {env.features.shape}")
print(f"  n_steps                {env.n_steps}")
print(f"  initial_value          {env.initial_value:,.2f} LKR")
print(f"  transaction_cost_rate  {env.tc_rate}      slippage {env.slippage}")
print(f"  allow_short            {env.allow_short}   -> actor uses SOFTMAX (long-only)")
print(f"  max_position_size      {env.max_position_size}")
print(f"  vol_target             {env.vol_target}  (inactive: only applies when allow_short=True)")

obs = env.reset()
print(f"\nenv.reset() -> observation dict:")
print(f"    features         {obs['features'].shape}")
print(f"    current_weights  {obs['current_weights'].shape}  = {vec(obs['current_weights'], 5)}")
print(f"    portfolio_value  {obs['portfolio_value']}")

history = []
grad_norms = []

# ══════════════════════════════════════════════════════════════════════════════
# PER-WINDOW TRACE
# ══════════════════════════════════════════════════════════════════════════════
for w in range(N_WINDOWS):
    step_idx = env.current_step
    dec_date = dates[step_idx + W - 1]
    nxt_date = dates[step_idx + W]
    hr(f"WINDOW {w+1} / {N_WINDOWS}   |   env.current_step = {step_idx}   |   "
       f"window {dates[step_idx].date()} -> {dec_date.date()}   |   reward over {dec_date.date()} -> {nxt_date.date()}",
       ch="#")

    # ── forward in eval mode: dropout off, exactly as trainer.collect_rollout ──
    agent.eval()

    # ---------------------------------------------------------------- STAGE 1
    sub("STAGE 1 :: OBSERVATION TENSOR  (env -> agent)")
    obs_np = obs["features"]
    obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
    print(f"  obs['features']            {obs_np.shape}      (window_size={W}, n_features={obs_np.shape[1]})")
    print(f"  after .unsqueeze(0)        {tuple(obs_t.shape)}  (batch, T, N*F)")
    B, T, NF = obs_t.shape
    x = obs_t.view(B, T, enc.n_assets, enc.n_features_per_asset).permute(0, 2, 1, 3)
    print(f"  encoder reshape+permute -> {tuple(x.shape)}  (batch, n_assets, T, F_per_asset)")
    print(f"     view(B,T,N,F) then permute(0,2,1,3): column (asset i, feature j) of the flat")
    print(f"     {NF}-vector lands at x[0, i, :, j] -- this is why reorder_per_asset() is required.")
    dfw = pd.DataFrame(x[0, FOCUS_ASSET].numpy(),
                       index=[d.date() for d in dates[step_idx: step_idx + W]],
                       columns=[f"f{i}" for i in range(F_PER_ASSET)])
    if w == 0:
        print(f"\n  x[0, {FOCUS_ASSET}] = the full ({T}, {F_PER_ASSET}) z-scored window for '{FA}':")
        print(dfw.to_string(float_format=lambda v: f"{v:8.4f}"))
    else:
        print(f"\n  the window SLID one trading day: row {dates[step_idx-1].date()} dropped off the front,")
        print(f"  row {dates[step_idx+W-1].date()} entered at the back. Last 3 rows of x[0,{FOCUS_ASSET}] ('{FA}'):")
        print(dfw.tail(3).to_string(float_format=lambda v: f"{v:8.4f}"))
        print(f"  (the other {W-1} rows are identical to window {w} -- 29/30 of the input is shared,")
        print(f"   which is why consecutive decisions are highly correlated)")
    print(f"\n  verify against norm_train: max|x[0,{FOCUS_ASSET}] - norm_train[{FA} cols][{step_idx}:{step_idx+W}]| = "
          f"{np.abs(x[0, FOCUS_ASSET].numpy() - norm_train[cols_fa].values[step_idx:step_idx+W]).max():.3e}")
    print(f"\n  whole-tensor stats: {vec(x, 6)}")

    # ---------------------------------------------------------------- STAGE 2
    sub("STAGE 2 :: VARIABLE SELECTION NETWORK (V-VSN)   [src/models/vsn_lstm.py]")
    vsn = enc.vsn_lstm.vsn
    x_flat = x.reshape(B * enc.n_assets, T, enc.n_features_per_asset)
    print(f"  INPUT   x_flat            {tuple(x_flat.shape)}  = (B*N={B*enc.n_assets}, T={T}, F={F_PER_ASSET})")
    print(f"          all {enc.n_assets} assets are folded into the batch -> ONE shared VSN, channel-independent")
    with torch.no_grad():
        selected, var_weights = vsn(x_flat)
        # manual recomputation
        Bf, Tf, Nf = x_flat.shape
        flat = x_flat.reshape(Bf * Tf, Nf)
        raw_sel = vsn.selection_grn(flat)
        w_soft = F.softmax(raw_sel, dim=-1)
        proj = vsn.var_proj(flat).view(Bf * Tf, Nf, vsn.hidden_dim)
        manual_selected = (proj * w_soft.unsqueeze(-1)).sum(dim=1).reshape(Bf, Tf, vsn.hidden_dim)
    print(f"\n  2a) selection_grn : GatedResidualNetwork(in={Nf}, hidden={vsn.selection_grn.fc1.out_features}, out={Nf})")
    print(f"      flat input                {tuple(flat.shape)}  (B*N*T rows)")
    print(f"      -> raw selection logits   {tuple(raw_sel.shape)}")
    print(f"      -> softmax over F         {tuple(w_soft.shape)}   sums to 1 per (asset,timestep)")
    i_focus = FOCUS_ASSET * T + (T - 1)      # row for focus asset, last timestep
    print(f"\n      logits  for '{FA}' at t={T-1}: {np.array2string(raw_sel[i_focus].numpy(), precision=4)}")
    print(f"      WEIGHTS for '{FA}' at t={T-1}:")
    for j, s in enumerate(preprocessor.PER_ASSET_SUFFIXES):
        print(f"        f{j:<2} {FA + s:<28} input={flat[i_focus, j].item():9.4f}   weight={w_soft[i_focus, j].item():.6f}")
    print(f"        sum of weights = {w_soft[i_focus].sum().item():.6f}")
    top = torch.argsort(w_soft[i_focus], descending=True)[:3]
    print(f"        top-3 selected features: "
          + ", ".join(f"{preprocessor.PER_ASSET_SUFFIXES[k] or '<log-return>'}={w_soft[i_focus,k]:.4f}" for k in top))
    print(f"\n  2b) var_proj : Linear({Nf}, {Nf}*{vsn.hidden_dim}={Nf*vsn.hidden_dim}, bias=False) then view(.,{Nf},{vsn.hidden_dim})")
    print(f"      -> projected              {tuple(proj.shape)}   (one {vsn.hidden_dim}-dim embedding per input variable)")
    print(f"      projected[{FA} t={T-1}, f0, :6] = {np.array2string(proj[i_focus,0,:6].numpy(), precision=5)}")
    print(f"\n  2c) weighted sum over the F variable embeddings")
    print(f"      OUTPUT  selected          {tuple(selected.shape)}  = (B*N, T, vsn_hidden={vsn.hidden_dim})")
    print(f"      var_weights               {tuple(var_weights.shape)}")
    print(f"      selected[{FA}, t={T-1}] = {vec(selected[FOCUS_ASSET, -1], 8)}")
    print(f"      manual recomputation matches module: max abs diff = "
          f"{(manual_selected - selected).abs().max().item():.3e}")
    print(f"      cross-asset spread of selected[:, -1, :] std over assets = "
          f"{selected[:, -1, :].std(dim=0).mean().item():.6f}  (0 would mean assets are indistinguishable)")

    # ---------------------------------------------------------------- STAGE 3
    sub("STAGE 3 :: LSTM  (temporal dynamics)   [src/models/vsn_lstm.py]")
    lstm = enc.vsn_lstm.lstm
    print(f"  INPUT   selected          {tuple(selected.shape)}")
    print(f"  nn.LSTM(input_size={lstm.input_size}, hidden_size={lstm.hidden_size}, "
          f"num_layers={lstm.num_layers}, batch_first=True, dropout={lstm.dropout})")
    print(f"  init: weight_hh orthogonal per gate block, weight_ih xavier, forget-gate bias = 1.0")
    print(f"        check bias_ih_l0 forget block [128:256] mean = "
          f"{lstm.bias_ih_l0.data[128:256].mean().item():.4f}")
    with torch.no_grad():
        lstm_out, (h_n, c_n) = lstm(selected)
        normed = enc.vsn_lstm.lstm_norm(lstm_out)
        proj_out = enc.vsn_lstm.output_proj(normed)
    print(f"\n  OUTPUT  lstm_out          {tuple(lstm_out.shape)}   (B*N, T, lstm_hidden={lstm.hidden_size})")
    print(f"          h_n               {tuple(h_n.shape)}    (num_layers, B*N, hidden)")
    print(f"          c_n               {tuple(c_n.shape)}")
    print(f"    lstm_out[{FA}, t={T-1}]  = {vec(lstm_out[FOCUS_ASSET, -1])}")
    print(f"    lstm_out[{FA}, t=0]      = {vec(lstm_out[FOCUS_ASSET, 0], 6)}")
    print(f"    LayerNorm -> normed[{FA}, t={T-1}] = {vec(normed[FOCUS_ASSET, -1])}")
    print(f"    output_proj Linear(128,128) -> {vec(proj_out[FOCUS_ASSET, -1])}")
    print(f"    (trainer note: no hidden state is carried between windows -- each window is self-contained)")

    # ---------------------------------------------------------------- STAGE 4
    sub("STAGE 4 :: TEMPORAL MULTI-HEAD ATTENTION   [src/models/temporal_attention.py]")
    tab = enc.temporal_attention
    mha = tab.mha
    print(f"  INPUT   output_proj result  {tuple(proj_out.shape)}")
    print(f"  d_model={mha.d_model}, num_heads={mha.num_heads}, d_k={mha.d_k}, ffn_dim={tab.ffn[0].out_features}")
    with torch.no_grad():
        xp = mha.pos_encoding(proj_out)
        Q = mha.W_q(xp).view(Bf, T, mha.num_heads, mha.d_k).transpose(1, 2)
        K = mha.W_k(xp).view(Bf, T, mha.num_heads, mha.d_k).transpose(1, 2)
        V = mha.W_v(xp).view(Bf, T, mha.num_heads, mha.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(mha.d_k)
        attnw_manual = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attnw_manual, V)
        ctx = ctx.transpose(1, 2).contiguous().view(Bf, T, mha.d_model)
        mha_out_manual = mha.layer_norm(mha.W_o(ctx) + xp)
        attn_out, attn_weights = tab(proj_out)
    print(f"\n  4a) positional encoding (sinusoidal, added): pe[0,0,:6] = "
          f"{np.array2string(mha.pos_encoding.pe[0,0,:6].numpy(), precision=4)}")
    print(f"      x + pe -> {tuple(xp.shape)}")
    print(f"  4b) Q = W_q(x): {tuple(Q.shape)}  (B*N, heads, T, d_k) ; same for K, V")
    print(f"      Q[{FA}, head0, t={T-1}, :6] = {np.array2string(Q[FOCUS_ASSET,0,-1,:6].numpy(), precision=4)}")
    print(f"  4c) scores = QK^T/sqrt(d_k) -> {tuple(scores.shape)}")
    print(f"      scores[{FA}, head0, t={T-1}, :8] = {np.array2string(scores[FOCUS_ASSET,0,-1,:8].numpy(), precision=4)}")
    print(f"  4d) attn_weights = softmax(scores) -> {tuple(attn_weights.shape)}")
    aw = attnw_manual[FOCUS_ASSET, 0, -1]
    print(f"      attention paid by the LAST timestep ({dec_date.date()}) across the {T}-day window, head 0:")
    for k in range(0, T, 5):
        print("        " + "  ".join(f"t{kk:02d}={aw[kk]:.4f}" for kk in range(k, min(k + 5, T))))
    print(f"      sum={aw.sum():.6f}   argmax=t{int(aw.argmax())} ({dates[step_idx+int(aw.argmax())].date()}) "
          f"weight={aw.max():.4f}")
    print(f"  4e) context = attn @ V, concat heads, W_o, +residual, LayerNorm -> {tuple(mha_out_manual.shape)}")
    print(f"  4f) FFN Linear(128,256)-GELU-Linear(256,128) + residual + LayerNorm")
    print(f"      OUTPUT attn_out       {tuple(attn_out.shape)}")
    print(f"      attn_out[{FA}, t={T-1}] = {vec(attn_out[FOCUS_ASSET, -1])}")
    print(f"      manual MHA sub-block matches module: max abs diff = "
          f"{(mha_out_manual - tab.mha(proj_out)[0]).abs().max().item():.3e}")
    with torch.no_grad():
        H = attn_out[:, -1, :].reshape(B, enc.n_assets, enc.lstm_hidden_dim)
    print(f"\n  4g) take LAST timestep only -> per-asset temporal embedding")
    print(f"      H = {tuple(H.shape)}  (batch, n_assets, lstm_hidden)")
    print(f"      H[0,{FOCUS_ASSET}] ('{FA}') = {vec(H[0, FOCUS_ASSET])}")
    print(f"      H[0,1] ('{config.data.asset_names[1]}') = {vec(H[0,1], 6)}")
    print(f"      cross-asset std (mean over dims) = {H[0].std(dim=0).mean().item():.6f}")

    # ---------------------------------------------------------------- STAGE 5
    sub("STAGE 5 :: MACRO GRAPH PRIOR (GAT)   [src/models/macro_graph_prior.py]")
    mg = enc.macro_graph
    with torch.no_grad():
        adj = mg.build_adjacency()
        h0 = mg.node_proj(H)
        h_run = h0
        gat_dbg = []
        for li, (gat, norm, res) in enumerate(zip(mg.gat_layers, mg.gat_norms, mg.gat_residual_proj)):
            h_new, attn = gat(h_run, adj)
            h_new = F.elu(h_new)
            h_new = norm(h_new)
            h_run = res(h_run) + h_new
            gat_dbg.append((li, gat, attn, h_run.clone()))
        node_embeddings = h_run
        graph_embedding = mg.readout(node_embeddings.mean(dim=1))
        ge_ref, ne_ref = mg(H, None)
    print(f"  INPUT   H                 {tuple(H.shape)}  -> {enc.n_assets} graph NODES (one per asset), n_macro=0")
    print(f"  adjacency: learnable_adj {tuple(mg.learnable_adj.shape)}, all entries = "
          f"{mg.learnable_adj.data.unique().tolist()} -> sigmoid = "
          f"{torch.sigmoid(mg.learnable_adj.data[0,0]).item():.4f} > 0")
    print(f"             build_adjacency() -> adj {tuple(adj.shape)}, sum = {adj.sum().item():.0f} of "
          f"{adj.numel()} -> FULLY CONNECTED graph (every bank linked to every bank)")
    print(f"  node_proj Linear({mg.node_proj.in_features},{mg.node_proj.out_features}) -> {tuple(h0.shape)}")
    print(f"    h0[0,{FOCUS_ASSET}] = {vec(h0[0, FOCUS_ASSET])}")
    for li, gat, attn, hout in gat_dbg:
        print(f"\n  GAT layer {li}: in={gat.in_features} out={gat.out_features} heads={gat.num_heads} "
              f"concat={gat.concat} -> output_dim={gat.output_dim}")
        print(f"    W       {tuple(gat.W.shape)}   a_src {tuple(gat.a_src.shape)}   a_dst {tuple(gat.a_dst.shape)}")
        print(f"    e_ij = LeakyReLU(0.2)(a_src.Wh_i + a_dst.Wh_j), masked by adj, softmax over j")
        print(f"    attention  {tuple(attn.shape)}  (batch, heads, N, N)")
        a0 = attn[0, 0, FOCUS_ASSET]
        print(f"    node {FOCUS_ASSET} ('{FA}') attention over its {enc.n_assets} neighbours (head 0):")
        print(f"      first 8: {np.array2string(a0[:8].numpy(), precision=5)}")
        print(f"      sum={a0.sum():.6f}  self-weight={a0[FOCUS_ASSET]:.5f}  "
              f"max=node {int(a0.argmax())} ('{config.data.asset_names[int(a0.argmax())]}') {a0.max():.5f}  "
              f"min={a0.min():.5f}  uniform would be {1/enc.n_assets:.5f}")
        print(f"    after ELU + LayerNorm + residual: h {tuple(hout.shape)}   "
              f"cross-node std = {hout[0].std(dim=0).mean().item():.6f}")
    print(f"\n  OUTPUT  node_embeddings   {tuple(node_embeddings.shape)}")
    print(f"          graph_embedding   {tuple(graph_embedding.shape)}  (mean-pool + readout MLP; "
          f"NOT used downstream in this architecture)")
    print(f"    node_embeddings[0,{FOCUS_ASSET}] = {vec(node_embeddings[0, FOCUS_ASSET])}")
    print(f"    manual == module: nodes {abs((ne_ref-node_embeddings)).max().item():.3e}, "
          f"graph {abs((ge_ref-graph_embedding)).max().item():.3e}")

    # ---------------------------------------------------------------- STAGE 6
    sub("STAGE 6 :: PER-ASSET FUSION  ->  h_asset ,  MEAN-POOL  ->  h_market")
    with torch.no_grad():
        fused = torch.cat([H, node_embeddings], dim=-1)
        h_asset = enc.fusion(fused.reshape(B * enc.n_assets, -1)).reshape(B, enc.n_assets, enc.latent_state_dim)
        h_market = h_asset.mean(dim=1)
    print(f"  concat[H({H.shape[-1]}), node_emb({node_embeddings.shape[-1]})] -> fused {tuple(fused.shape)}")
    print(f"  fusion = Linear(192,128) -> GELU -> Dropout({config.regime_encoder.lstm_dropout}) "
          f"-> Linear(128,128) -> LayerNorm")
    print(f"  OUTPUT  h_asset           {tuple(h_asset.shape)}   -> goes to the ACTOR")
    print(f"          h_market          {tuple(h_market.shape)}   -> goes to the CRITIC + regime classifier")
    print(f"    h_asset[0,{FOCUS_ASSET}] ('{FA}')  = {vec(h_asset[0, FOCUS_ASSET])}")
    print(f"    h_asset[0,1] ('{config.data.asset_names[1]}') = {vec(h_asset[0,1], 6)}")
    print(f"    h_market[0]                = {vec(h_market[0])}")
    print(f"    cross-asset std of h_asset = {h_asset[0].std(dim=0).mean().item():.6f}")

    # ---------------------------------------------------------------- STAGE 7
    sub("STAGE 7 :: REGIME CLASSIFIER  (Bull / Bear / Sideways, differentiable)")
    with torch.no_grad():
        regime_logits = enc.regime_classifier(h_market)
        temp = enc.regime_temperature.clamp(min=0.1)
        regime_probs = F.softmax(regime_logits / temp, dim=-1)
    print(f"  INPUT   h_market          {tuple(h_market.shape)}")
    print(f"  Linear(128,64) -> GELU -> Dropout -> Linear(64,3)")
    print(f"  learnable regime_temperature = {enc.regime_temperature.item():.6f} (clamped min 0.1)")
    print(f"  OUTPUT  regime_logits     {tuple(regime_logits.shape)} = {regime_logits[0].numpy()}")
    print(f"          regime_probs      = softmax(logits/T) = {regime_probs[0].numpy()}   sum={regime_probs.sum():.6f}")
    print(f"          -> regime {int(regime_probs.argmax())} "
          f"({['Bull','Bear','Sideways'][int(regime_probs.argmax())]}) with p={regime_probs.max():.4f}")
    print(f"          entropy = {-(regime_probs*torch.log(regime_probs+1e-10)).sum().item():.6f} "
          f"(max possible ln3={math.log(3):.4f})")

    # ---------------------------------------------------------------- STAGE 8
    sub("STAGE 8 :: ACTOR (policy pi)   [src/models/actor.py]")
    act = agent.actor
    with torch.no_grad():
        feat = act.feature_net(h_asset)
        regime_signal = act.regime_embed(regime_probs)
        feat_cond = feat + regime_signal.unsqueeze(1)
        raw_head = act.mean_head(feat_cond).squeeze(-1)
        mean = act.logit_scale * torch.tanh(raw_head / act.logit_scale)
        std = act.log_std.exp().expand_as(mean)
    print(f"  INPUT   h_asset {tuple(h_asset.shape)} + regime_probs {tuple(regime_probs.shape)}")
    print(f"  trunk (SHARED across assets, applied on last dim):")
    print(f"    Linear(128,256)+LayerNorm+ReLU+Dropout -> Linear(256,128)+LayerNorm+ReLU+Dropout")
    print(f"    -> features {tuple(feat.shape)}")
    print(f"  regime_embed Linear(3,128) -> {tuple(regime_signal.shape)}, broadcast-added to every asset")
    print(f"    regime_signal[0,:6] = {np.array2string(regime_signal[0,:6].numpy(), precision=6)}")
    print(f"  mean_head Linear(128,1) -> one logit per asset {tuple(raw_head.shape)}")
    print(f"  mean = logit_scale*tanh(raw/logit_scale), logit_scale={act.logit_scale}")
    print(f"    raw_head[0,:8] = {np.array2string(raw_head[0,:8].numpy(), precision=6)}")
    print(f"    mean[0]      = {vec(mean[0])}")
    print(f"  log_std (learnable, {tuple(act.log_std.shape)}) = {vec(act.log_std, 5)}  -> std = {std[0,0].item():.6f}")
    torch.manual_seed(SEED + w)   # make the sample reproducible per window
    with torch.no_grad():
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        weights_t = F.softmax(raw_action, dim=-1)
    print(f"\n  a) Gaussian policy N(mean, std) per asset, sample via rsample()")
    print(f"     raw_action {tuple(raw_action.shape)} = {vec(raw_action[0])}")
    print(f"     log_prob   = sum_i log N(a_i; mu_i, sigma_i) = {log_prob.item():.6f}")
    print(f"     entropy    = sum_i 0.5*log(2*pi*e*sigma_i^2) = {entropy.item():.6f}")
    print(f"     manual entropy check: {config.data.n_assets*(0.5*math.log(2*math.pi*math.e*std[0,0].item()**2)):.6f}")
    print(f"  b) squash to allocation: allow_short={act.allow_short} -> SOFTMAX over {enc.n_assets} assets")
    print(f"     action (portfolio weights) = {vec(weights_t[0])}")
    print(f"     sum={weights_t.sum().item():.6f}  max={weights_t.max().item():.6f} "
          f"(asset {int(weights_t.argmax())} = {config.data.asset_names[int(weights_t.argmax())]})")
    print(f"     equal weight would be 1/{enc.n_assets} = {1/enc.n_assets:.6f}")

    # ---------------------------------------------------------------- STAGE 9
    sub("STAGE 9 :: CRITIC (value V)   [src/models/critic.py]")
    cr = agent.critic
    with torch.no_grad():
        cfeat = cr.feature_net(h_market)
        crs = cr.regime_embed(regime_probs)
        value_t = cr.value_head(cfeat + crs).squeeze(-1)
    print(f"  INPUT   h_market {tuple(h_market.shape)} + regime_probs {tuple(regime_probs.shape)}")
    print(f"  Linear(128,256)+LN+ReLU+Drop -> Linear(256,128)+LN+ReLU+Drop -> +regime_embed(3->128) -> Linear(128,1)")
    print(f"    critic features[0,:6] = {np.array2string(cfeat[0,:6].numpy(), precision=6)}")
    print(f"  OUTPUT  V(s) = {value_t.item():.6f}")

    # cross-check the whole forward against agent.forward()
    torch.manual_seed(SEED + w)
    with torch.no_grad():
        full = agent(obs_t)
    print(f"\n  cross-check vs agent.forward(): |action|={abs(full['action']-weights_t).max().item():.3e} "
          f"|value|={abs(full['value']-value_t).max().item():.3e} "
          f"|log_prob|={abs(full['log_prob']-log_prob).max().item():.3e} "
          f"|regime|={abs(full['regime_probs']-regime_probs).max().item():.3e}")

    action_np = weights_t.numpy()[0]
    raw_action_np = raw_action.numpy()[0]

    # ---------------------------------------------------------------- STAGE 10
    sub("STAGE 10 :: MARKET ENVIRONMENT STEP  (real prices, real costs)")
    old_weights = env.current_weights.copy()
    pv_before = env.portfolio_value
    p_now = env.prices[env.current_step]
    p_nxt = env.prices[env.current_step + 1]
    clipped = np.clip(action_np, -env.max_position_size, env.max_position_size)
    renorm = clipped / clipped.sum()
    asset_ret = (p_nxt - p_now) / (p_now + 1e-10)
    print(f"  action in                 {action_np.shape}  {vec(action_np, 6)}")
    print(f"  _clip_weights: clip to +/-{env.max_position_size} then renormalise (long-only, sum->1)")
    print(f"    n weights above cap = {(action_np > env.max_position_size).sum()}   "
          f"clipped sum = {clipped.sum():.6f} -> divide by it")
    print(f"    new_weights = {vec(renorm, 6)}")
    print(f"  turnover = sum|w_new - w_old| = {np.abs(renorm - old_weights).sum():.6f}   "
          f"(w_old sum = {old_weights.sum():.4f})")
    tc = env.tc_rate * np.abs(renorm - old_weights).sum() * pv_before
    sl = env.slippage * np.abs(renorm - old_weights).sum() * pv_before
    print(f"  transaction cost = {env.tc_rate} * turnover * PV = {tc:,.2f} LKR")
    print(f"  slippage         = {env.slippage} * turnover * PV = {sl:,.2f} LKR      total = {tc+sl:,.2f} LKR")
    print(f"\n  REAL PRICES used for the return:")
    hdr = pd.DataFrame({
        f"P({dec_date.date()})": p_now[:8],
        f"P({nxt_date.date()})": p_nxt[:8],
        "simple return": asset_ret[:8],
        "weight": renorm[:8],
        "contribution": (renorm * asset_ret)[:8],
    }, index=config.data.asset_names[:8])
    print(hdr.to_string(float_format=lambda v: f"{v:12.6f}"))
    print(f"  ... {len(p_now)} assets total.  asset_returns: {vec(asset_ret, 6)}")
    obs_next, portfolio_return, done, info = env.step(action_np)
    print(f"\n  portfolio_return = w . r = {portfolio_return:.8f}  ({portfolio_return*100:.4f} %)")
    print(f"  portfolio_value  = {pv_before:,.2f} * (1 + {portfolio_return:.6f}) - {info['total_cost']:,.2f} "
          f"= {env.portfolio_value:,.2f} LKR")
    print(f"  done = {done}   (env.n_steps={env.n_steps}, current_step now {env.current_step})")

    # ---------------------------------------------------------------- STAGE 11
    sub("STAGE 11 :: REWARD CALCULATION   [src/environment/reward.py]")
    print(f"  Net reward = rolling Sharpe  +  EVaR penalty  +  cost penalty")
    print(f"  buffer length before this step = {len(reward_calc.return_buffer)}")
    reward_info = reward_calc.compute_reward(portfolio_return, old_weights, action_np)
    buf = np.array(reward_calc.return_buffer)
    print(f"  daily_rf = (1+{reward_calc.risk_free_rate})^(1/252)-1 = {reward_calc.daily_rf:.8f}")
    print(f"  return buffer (n={len(buf)}) = {np.array2string(buf, precision=6)}")
    if len(buf) >= 2:
        exc = buf[-reward_calc.sharpe_window:] - reward_calc.daily_rf
        print(f"    excess = r - daily_rf ; mean={exc.mean():.8f}  std={exc.std():.8f}")
        print(f"    Sharpe = mean/std*sqrt(252) = {exc.mean()/(exc.std()+1e-10)*math.sqrt(252):.6f}")
    else:
        print(f"    only 1 observation -> Sharpe component falls back to the raw return")
    print(f"  EVaR penalty: needs >= sharpe_window({reward_calc.sharpe_window}) observations -> "
          f"{'active' if len(buf) >= reward_calc.sharpe_window else 'INACTIVE (0.0) at this stage'}")
    turn = np.abs(action_np - old_weights).sum()
    print(f"  cost penalty = -(tc_rate*turnover + turnover_penalty*turnover^2)")
    print(f"               = -({reward_calc.tc_rate}*{turn:.6f} + {reward_calc.turnover_penalty}*{turn:.6f}^2) "
          f"= {reward_info['cost_penalty']:.8f}")
    print(f"\n  COMPONENTS: sharpe={reward_info['sharpe_component']:.6f}  "
          f"evar={reward_info['evar_penalty']:.6f}  cost={reward_info['cost_penalty']:.6f}")
    print(f"  NET REWARD r_{w} = {reward_info['net_reward']:.6f}")

    # ---------------------------------------------------------------- STAGE 12
    sub("STAGE 12 :: BACKPROPAGATION + PARAMETER UPDATE (PPO, joint end-to-end)")
    print(f"  NOTE: the real trainer buffers rollout_length={config.ppo.rollout_length} steps before updating.")
    print(f"        Here we update after EVERY window (batch of 1) as requested, using the same")
    print(f"        RewardNormalizer / compute_gae / PPO-loss code paths.")

    reward_normalizer.reset()
    rewards_np = reward_normalizer(np.array([reward_info["net_reward"]]), np.array([float(done)]))
    print(f"\n  12a) RewardNormalizer: discounted acc = {reward_info['net_reward']:.6f}, "
          f"running std = {reward_normalizer.rms.std:.6f} (count={reward_normalizer.rms.count:.4f})")
    print(f"       scaled reward = {reward_info['net_reward']:.6f} / {reward_normalizer.rms.std:.6f} "
          f"= {rewards_np[0]:.6f}")

    # value of the next state, for GAE bootstrapping
    agent.eval()
    with torch.no_grad():
        if obs_next is not None:
            nxt_t = torch.FloatTensor(obs_next["features"]).unsqueeze(0)
            last_value = agent(nxt_t)["value"].item()
        else:
            last_value = 0.0
    adv, ret = compute_gae(torch.FloatTensor(rewards_np), torch.FloatTensor([value_t.item()]),
                           torch.FloatTensor([float(done)]), torch.tensor(last_value),
                           config.ppo.gamma, config.ppo.gae_lambda)
    print(f"\n  12b) GAE(gamma={config.ppo.gamma}, lambda={config.ppo.gae_lambda}) on T=1")
    print(f"       V(s_t)={value_t.item():.6f}   V(s_t+1)={last_value:.6f}   done={float(done)}")
    print(f"       delta = r + gamma*V(s_t+1)*(1-done) - V(s_t) = {rewards_np[0]:.6f} + "
          f"{config.ppo.gamma}*{last_value:.6f} - {value_t.item():.6f} = {adv[0].item():.6f}")
    print(f"       advantage = {adv[0].item():.6f}   return (target for critic) = {ret[0].item():.6f}")
    adv_norm = normalize_advantages(adv)
    print(f"       trainer then calls normalize_advantages() -> {adv_norm[0].item()}")
    print(f"       *** with a single sample, (adv-adv.mean())/(adv.std()+1e-8) is 0/nan because torch's")
    print(f"       std uses ddof=1 -> undefined for n=1. We therefore keep the RAW advantage for this")
    print(f"       1-step demo so the policy term is visible. (With rollout_length=256 as the real")
    print(f"       trainer uses, the normalisation is well defined.)")

    obs_b = obs_t.clone()
    act_b = torch.FloatTensor(raw_action_np).unsqueeze(0)
    olp_b = torch.FloatTensor([log_prob.item()])
    ret_b = ret.clone()
    adv_b = adv.clone()

    watch = {
        "encoder.vsn.selection_grn.fc1.weight": enc.vsn_lstm.vsn.selection_grn.fc1.weight,
        "encoder.vsn.var_proj.weight": enc.vsn_lstm.vsn.var_proj.weight,
        "encoder.lstm.weight_ih_l0": enc.vsn_lstm.lstm.weight_ih_l0,
        "encoder.mha.W_q.weight": enc.temporal_attention.mha.W_q.weight,
        "encoder.macro_graph.learnable_adj": enc.macro_graph.learnable_adj,
        "encoder.macro_graph.gat0.W": enc.macro_graph.gat_layers[0].W,
        "encoder.fusion.0.weight": enc.fusion[0].weight,
        "encoder.regime_temperature": enc.regime_temperature,
        "actor.mean_head.weight": agent.actor.mean_head.weight,
        "actor.log_std": agent.actor.log_std,
        "critic.value_head.weight": agent.critic.value_head.weight,
    }
    before = {k: v.detach().clone() for k, v in watch.items()}

    agent.train()   # dropout ON for the update, matching trainer.update()
    print(f"\n  12c) PPO epochs (n_epochs={config.ppo.n_epochs}), agent.train() -> dropout active")
    for epoch in range(config.ppo.n_epochs):
        eval_out = agent.evaluate_actions(obs_b, act_b)
        new_lp, values_new, ent = eval_out["log_prob"], eval_out["value"], eval_out["entropy"]
        ratio = torch.exp(new_lp - olp_b)
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1 - config.ppo.clip_epsilon, 1 + config.ppo.clip_epsilon) * adv_b
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values_new, ret_b)
        entropy_loss = -ent.mean()
        total_loss = (policy_loss + config.ppo.value_loss_coeff * value_loss
                      + config.ppo.entropy_coeff * entropy_loss)

        optimizer.zero_grad()
        total_loss.backward()

        gn = {}
        for cname, comp in [("regime_encoder", enc), ("actor", agent.actor), ("critic", agent.critic)]:
            s = sum((p.grad.detach() ** 2).sum().item() for p in comp.parameters() if p.grad is not None)
            gn[cname] = math.sqrt(s)
        total_gn = math.sqrt(sum(v ** 2 for v in gn.values()))
        grad_norms.append(total_gn)
        clipped_gn = torch.nn.utils.clip_grad_norm_(agent.parameters(), config.ppo.max_grad_norm).item()
        lrs = [g["lr"] for g in optimizer.param_groups]
        optimizer.step()
        scheduler.step()

        if epoch == 0:
            print(f"\n    --- epoch 1 in full detail ---")
            print(f"    evaluate_actions() re-encodes the SAME window with dropout on:")
            print(f"      old log_prob (stored) = {olp_b.item():.6f}")
            print(f"      new log_prob          = {new_lp.item():.6f}")
            print(f"      ratio = exp(new-old)  = {ratio.item():.6f}")
            print(f"      advantage             = {adv_b.item():.6f}")
            print(f"      surr1 = ratio*A       = {surr1.item():.6f}")
            print(f"      surr2 = clip(ratio,{1-config.ppo.clip_epsilon},{1+config.ppo.clip_epsilon})*A = {surr2.item():.6f}")
            print(f"      policy_loss = -min(surr1,surr2) = {policy_loss.item():.6f}")
            print(f"      V_new = {values_new.item():.6f}, target return = {ret_b.item():.6f} "
                  f"-> value_loss = {value_loss.item():.6f}")
            print(f"      entropy = {ent.item():.6f} -> entropy_loss = {entropy_loss.item():.6f}")
            print(f"      total = {policy_loss.item():.6f} + {config.ppo.value_loss_coeff}*{value_loss.item():.6f} "
                  f"+ {config.ppo.entropy_coeff}*{entropy_loss.item():.6f} = {total_loss.item():.6f}")
            print(f"    backward() gradient norms per component:")
            for k2, v2 in gn.items():
                print(f"      {k2:<16} ||g|| = {v2:.6e}")
            print(f"      TOTAL ||g|| = {total_gn:.6e}  -> clip_grad_norm_(max={config.ppo.max_grad_norm}) "
                  f"returned {clipped_gn:.6e}"
                  f"{'  (CLIPPED, scale=' + format(config.ppo.max_grad_norm/clipped_gn, '.4f') + ')' if clipped_gn > config.ppo.max_grad_norm else '  (no clipping needed)'}")
            print(f"    learning rates at this step: encoder={lrs[0]:.3e} actor={lrs[1]:.3e} critic={lrs[2]:.3e}")
            print(f"      (LambdaLR linear warmup over 1000 steps; scheduler step counter is now "
                  f"{scheduler.last_epoch})")
        else:
            print(f"    epoch {epoch+1:>2}: loss={total_loss.item():9.5f} (pi={policy_loss.item():8.5f} "
                  f"V={value_loss.item():9.5f} H={entropy_loss.item():8.4f}) ratio={ratio.item():.5f} "
                  f"||g||={total_gn:.4e} lr_actor={lrs[1]:.3e}")

    print(f"\n  12d) PARAMETER CHANGE after {config.ppo.n_epochs} epochs on window {w+1}")
    print(f"    {'parameter':<44} {'||before||':>12} {'||delta||':>12} {'max|delta|':>12}")
    for k, v0 in before.items():
        v1 = watch[k].detach()
        d = (v1 - v0)
        print(f"    {k:<44} {v0.norm().item():12.6f} {d.norm().item():12.6e} {d.abs().max().item():12.6e}")
    print(f"    regime_temperature: {before['encoder.regime_temperature'].item():.8f} -> "
          f"{enc.regime_temperature.item():.8f}")
    print(f"    actor.log_std[0]:   {before['actor.log_std'][0].item():.8f} -> "
          f"{agent.actor.log_std[0].item():.8f}")

    # post-update forward on the SAME window, to show the effect of tuning
    agent.eval()
    with torch.no_grad():
        after_out = agent(obs_t, deterministic=True)
    print(f"\n  12e) same window re-run through the TUNED model (deterministic):")
    print(f"    V(s) before update = {value_t.item():.6f}   after = {after_out['value'].item():.6f}   "
          f"(target was {ret_b.item():.6f})")
    print(f"    regime_probs before = {regime_probs[0].numpy()}   after = {after_out['regime_probs'][0].numpy()}")
    print(f"    weights max before = {weights_t.max().item():.6f}  after = {after_out['action'].max().item():.6f}")

    history.append({
        "window": w + 1,
        "date": dec_date.date(),
        "regime": int(regime_probs.argmax()),
        "p_regime": float(regime_probs.max()),
        "w_max": float(weights_t.max()),
        "w_argmax": config.data.asset_names[int(weights_t.argmax())],
        "value": float(value_t),
        "port_return": float(portfolio_return),
        "sharpe_c": float(reward_info["sharpe_component"]),
        "cost_c": float(reward_info["cost_penalty"]),
        "evar_c": float(reward_info["evar_penalty"]),
        "reward": float(reward_info["net_reward"]),
        "advantage": float(adv[0]),
        "pv": float(env.portfolio_value),
    })

    obs = obs_next
    if obs is None:
        print("environment finished")
        break

# ══════════════════════════════════════════════════════════════════════════════
hr("SUMMARY OF THE 5 WINDOWS")
h = pd.DataFrame(history)
print(h.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
print(f"\nportfolio value path: " + " -> ".join(f"{v:,.0f}" for v in env.portfolio_history[:N_WINDOWS + 1]))
print(f"cumulative return over the 5 windows: "
      f"{env.portfolio_history[N_WINDOWS]/env.portfolio_history[0]-1:+.6%}")

hr("SHAPE PIPELINE, END TO END (one window)")
NA, FP = enc.n_assets, enc.n_features_per_asset
print(f"""
  data/raw/*.csv                      836 dates x {NA} banks (close price + share volume)
  engineer_features                   (816, {NA*FP})            15 indicators per bank
  reorder_per_asset                   (816, {NA*FP})            columns grouped per bank
  fit_normalize (train only)          (571, {NA*FP})            z-score
  create_windows                      (542, {W}, {NA*FP})        one row per trading decision
    |
    +-- env.reset()/step() serves     ({W}, {NA*FP})
        unsqueeze(0)                  (1, {W}, {NA*FP})
        encoder view+permute          (1, {NA}, {W}, {FP})       B, N_assets, T, F_per_asset
        reshape (fold assets)         ({NA}, {W}, {FP})
  VSN   selection_grn + softmax       ({NA}, {W}, {FP})         variable importance weights
        var_proj + weighted sum       ({NA}, {W}, 64)
  LSTM  2 layers                      ({NA}, {W}, 128)  (+ h_n, c_n = (2, {NA}, 128))
        LayerNorm + output_proj       ({NA}, {W}, 128)
  MHA   pos-enc, 4 heads              scores/attn ({NA}, 4, {W}, {W}) -> ({NA}, {W}, 128)
        FFN + residual + LayerNorm    ({NA}, {W}, 128)
        take t = T-1                  (1, {NA}, 128)   = H, per-asset temporal embedding
  GAT   node_proj                     (1, {NA}, 64)
        layer 0 (2 heads, concat)     attn (1, 2, {NA}, {NA}) -> (1, {NA}, 128)
        layer 1 (2 heads, mean)       attn (1, 2, {NA}, {NA}) -> (1, {NA}, 64)  = node_embeddings
  FUSE  cat(H, node_emb)              (1, {NA}, 192) -> h_asset (1, {NA}, 128)
        mean over assets              h_market (1, 128)
  REGIME  Linear->GELU->Linear        regime_logits (1, 3) -> regime_probs (1, 3)
  ACTOR   trunk + regime + head       mean (1, {NA}) -> sample raw_action (1, {NA})
          softmax                     action / portfolio weights (1, {NA}), sums to 1
  CRITIC  trunk + regime + head       V(s) scalar
  ENV     clip/renorm, prices t->t+1  portfolio_return scalar, costs in LKR
  REWARD  Sharpe + EVaR + cost        net_reward scalar
  PPO     GAE -> ratio -> clipped L   one joint backward through ALL {sum(agent.count_parameters()[k] for k in ['total']):,} parameters
""")

hr("KEY OBSERVATIONS FROM THIS RUN")
print(f"""
 1. DATA SIZE. n_assets was requested as {N_ASSETS} (main.py's default) but the raw CSVs only contain
    {NA} banking counters, so the model is built with {NA} assets x {FP} features = {NA*FP} inputs,
    not the 705 the --n-assets 47 default implies. The assertion in main.py still passes because it
    compares against the *actual* column count.

 2. LEARNING RATE IS 0 ON THE FIRST UPDATE. LambdaLR computes lr_lambda(0) = 0/1000 = 0 at
    construction, so the very first optimizer.step() applies a zero-size step: gradients are
    computed and thrown away. With total_timesteps=10000 the scheduler's total_steps is
    {total_updates}, far below warmup_steps=1000, so the LR ramps to at most {total_updates/1000:.0%} of base
    ({config.ppo.learning_rate*total_updates/1000:.2e} for actor/critic, {config.ppo.learning_rate*0.5*total_updates/1000:.2e} for the encoder) and the cosine
    decay branch is never reached. That is why the ||delta|| column above is ~1e-4 on norms of ~5-18.

 3. EVERY SINGLE UPDATE HIT THE GRADIENT CLIP. Across the {len(grad_norms)} backward passes traced here the raw
    total ||g|| ranged {min(grad_norms):.3f} .. {max(grad_norms):.1f} (median {sorted(grad_norms)[len(grad_norms)//2]:.2f}) against max_grad_norm=0.5, so the update
    direction was rescaled by {0.5/max(grad_norms):.4f}x .. {0.5/min(grad_norms):.3f}x -- i.e. {min(grad_norms)/0.5:.1f}x to {max(grad_norms)/0.5:.0f}x shrinkage, and the amount of
    shrinkage varies ~100x from step to step. Because clip_grad_norm_ is applied to
    encoder+actor+critic jointly, a large value-loss gradient shrinks the actor's update along with
    its own. This is the effect RewardNormalizer was added to fight, but the rolling-Sharpe reward
    (item 4) reintroduces it.

 4. THE ROLLING SHARPE REWARD IS UNUSABLE FOR THE FIRST ~20 STEPS OF EVERY ROLLOUT.
    compute_sharpe_component divides by the std of the return buffer, which holds only 1, 2, 3 ...
    samples right after reset(). Window 1 returns the raw return ({history[0]['sharpe_c']:+.6f}); window 2 gets
    {history[1]['sharpe_c']:+.4f} from just two observations. The reward therefore swings by 4 orders of magnitude
    between consecutive steps for reasons unrelated to the policy. EVaR is separately inactive for
    the first sharpe_window=20 steps.

 5. THE GAT IS NEAR-UNIFORM AT INIT. build_adjacency() with the default learnable_adj = 0.5
    everywhere yields sigmoid(0.5) = 0.62 > 0, so adj is all-ones: a fully connected graph. The
    resulting attention rows sat in 0.093-0.116 against a uniform 1/{NA} = {1/NA:.3f}, i.e. the graph block
    is currently doing little more than mean-aggregation plus the residual. Also note
    learnable_adj's ||delta|| is exactly 0 every window: build_adjacency() thresholds with
    `(adj > 0).float()`, which is non-differentiable, so that 100-parameter tensor receives no
    gradient and never trains.

 6. info["turnover"] IS ALWAYS 0. In MarketEnvironment.step, self.current_weights is reassigned to
    new_weights before the info dict computes abs(new_weights - self.current_weights). The reward
    path is unaffected (RewardCalculator gets old_weights explicitly from the trainer), but any
    logging or analysis that reads info["turnover"] sees zero.

 7. WINDOW-TO-WINDOW OVERLAP IS {W-1}/{W}. Consecutive observations differ by a single row, so
    consecutive states, values and actions are strongly correlated -- worth remembering when
    reading a 256-step rollout as if it were 256 independent samples.

 8. normalize_advantages() IS UNDEFINED FOR A 1-SAMPLE BATCH (torch std uses ddof=1 -> nan). Not a
    bug at rollout_length=256; it only bites the per-window update used for this trace.
""")
print("\nTRACE COMPLETE")
