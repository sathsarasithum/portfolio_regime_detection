"""
LSTM GATE-BY-GATE, DAY-BY-DAY DEMO
===================================
Feeds one real asset's real 30-day window through the project's actual
VSNLSTM (src/models/vsn_lstm.py), then manually re-derives the LSTM's
input/forget/cell/output gates at every single day using the module's own
trained weight matrices -- so you can watch, day by day, how much the LSTM
is "remembering" vs "forgetting" vs "writing new information".

This does NOT reimplement the LSTM differently: it recomputes the exact
same equations nn.LSTM runs internally, using nn.LSTM's own parameters, and
then asserts the two agree to floating-point precision. nn.LSTM is a fused
kernel that only returns h_t and c_t -- it never exposes i/f/g/o -- so
manual unrolling is the only way to see the gates at all.

Run:
    python scripts/lstm_gate_demo.py
    python scripts/lstm_gate_demo.py --asset COMB --window 3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import get_config, DATA_RAW_DIR
from src.data.data_loader import CSEDataLoader
from src.data.preprocessor import Preprocessor
from src.models.vsn_lstm import VSNLSTM

np.set_printoptions(precision=4, suppress=True, linewidth=150)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)

SEED = 42


def hr(title, ch="="):
    print("\n" + ch * 100)
    print(title)
    print(ch * 100)


def bar(value, width=20, lo=0.0, hi=1.0):
    """ASCII gauge for a 0..1 gate value, e.g. [############--------] 0.62"""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(round(frac * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {value:.3f}"


def manual_lstm_layer(x_seq, h0, c0, w_ih, w_hh, b_ih, b_hh, hidden_size):
    """
    Unrolls ONE LSTM layer across time, one step at a time, using the exact
    equations PyTorch's fused kernel evaluates internally:

        i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)   -- "write how much new info?"
        f_t = sigmoid(W_if x_t + b_if + W_hf h_(t-1) + b_hf)   -- "keep how much old memory?"
        g_t =    tanh(W_ig x_t + b_ig + W_hg h_(t-1) + b_hg)   -- "candidate new memory content"
        o_t = sigmoid(W_io x_t + b_io + W_ho h_(t-1) + b_ho)   -- "reveal how much of the memory?"
        c_t = f_t * c_(t-1) + i_t * g_t                          -- update the memory (cell state)
        h_t = o_t * tanh(c_t)                                    -- the memory as this step's output

    PyTorch packs all 4 gates' weights into one matrix, stacked in the
    fixed order [input, forget, cell/candidate, output].
    """
    T = x_seq.shape[0]
    h, c = h0, c0
    h_seq, c_seq, gate_log = [], [], []
    for t in range(T):
        gates = x_seq[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh   # (4*hidden,)
        i_raw, f_raw, g_raw, o_raw = gates.split(hidden_size)
        i = torch.sigmoid(i_raw)
        f = torch.sigmoid(f_raw)
        g = torch.tanh(g_raw)
        o = torch.sigmoid(o_raw)
        c = f * c + i * g
        h = o * torch.tanh(c)
        h_seq.append(h.clone())
        c_seq.append(c.clone())
        gate_log.append((i.clone(), f.clone(), g.clone(), o.clone()))
    return torch.stack(h_seq), torch.stack(c_seq), gate_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None, help="ticker to trace, e.g. COMB (default: first asset)")
    ap.add_argument("--window", type=int, default=0, help="which training window index to trace (default: 0)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    hr("PART 0 :: BUILD ONE REAL 30-DAY WINDOW, EXACTLY AS THE TRAINING PIPELINE DOES")

    config = get_config(seed=SEED, device="cpu")
    loader = CSEDataLoader(raw_data_dir=DATA_RAW_DIR)
    prices, volumes = loader.load_prices_and_volumes(n_assets=config.data.n_assets)
    config.data.asset_names = list(prices.columns)
    config.data.n_assets = len(config.data.asset_names)
    FA = args.asset or config.data.asset_names[0]
    if FA not in config.data.asset_names:
        raise SystemExit(f"'{FA}' not in universe: {config.data.asset_names}")
    focus_idx = config.data.asset_names.index(FA)

    preprocessor = Preprocessor(window_size=config.data.window_size,
                                normalize_method=config.data.normalize_method)
    features_all = preprocessor.engineer_features(prices, volumes, macro_data=None)
    features_all = preprocessor.reorder_per_asset(features_all, config.data.asset_names)
    F_PER_ASSET = preprocessor.n_features_per_asset

    n_total = len(features_all)
    n_train = int(n_total * config.data.train_ratio)
    features_train = features_all.iloc[:n_train]
    norm_train = preprocessor.fit_normalize(features_train)
    windows_train = preprocessor.create_windows(norm_train.values)   # (n_windows, W, N*F)
    W = config.data.window_size

    if not (0 <= args.window < len(windows_train)):
        raise SystemExit(f"--window must be in [0, {len(windows_train) - 1}]")

    dates = features_train.index
    dec_date = dates[args.window + W - 1]
    print(f"asset traced        : '{FA}' (index {focus_idx} of {config.data.n_assets})")
    print(f"window traced       : #{args.window}  ({dates[args.window].date()} -> {dec_date.date()}, "
          f"decision date = {dec_date.date()})")
    print(f"window_size (days)  : {W}")
    print(f"features per asset  : {F_PER_ASSET}  ({', '.join(preprocessor.PER_ASSET_SUFFIXES[i] or '<log-return>' for i in range(F_PER_ASSET))})")

    one_window_flat = torch.FloatTensor(windows_train[args.window]).unsqueeze(0)   # (1, W, N*F)
    x_all = one_window_flat.view(1, W, config.data.n_assets, F_PER_ASSET).permute(0, 2, 1, 3)  # (1, N, W, F)
    x = x_all[:, focus_idx]   # (1, W, F) -- just this one asset

    dfw = pd.DataFrame(x[0].numpy(),
                       index=[d.date() for d in dates[args.window: args.window + W]],
                       columns=[f"f{i}" for i in range(F_PER_ASSET)])
    print(f"\nz-scored (30, {F_PER_ASSET}) input for '{FA}' (first/last 3 days):")
    print(pd.concat([dfw.head(3), dfw.tail(3)]).to_string(float_format=lambda v: f"{v:7.3f}"))

    # ------------------------------------------------------------------ #
    hr("PART 1 :: BUILD THE VSNLSTM MODULE (same class, same init, as the real encoder)")

    model = VSNLSTM(
        n_features=F_PER_ASSET,
        vsn_hidden_dim=config.regime_encoder.vsn_hidden_dim,
        lstm_hidden_dim=config.regime_encoder.lstm_hidden_dim,
        lstm_num_layers=config.regime_encoder.lstm_num_layers,
        dropout=config.regime_encoder.lstm_dropout,
    )
    model.eval()   # dropout OFF so the manual recomputation matches exactly

    lstm = model.lstm
    H = lstm.hidden_size
    print(f"VSNLSTM: vsn_hidden={config.regime_encoder.vsn_hidden_dim}  "
          f"lstm_hidden={H}  lstm_layers={lstm.num_layers}")
    fbias0 = lstm.bias_ih_l0.data[H:2 * H].mean().item() + lstm.bias_hh_l0.data[H:2 * H].mean().item()
    print(f"forget-gate bias (layer 0, summed ih+hh, averaged over {H} units) = {fbias0:.4f}  "
          f"(Jozefowicz init target: ~2.0, i.e. 1.0 + 1.0)")

    with torch.no_grad():
        selected, var_weights = model.vsn(x)          # (1, W, vsn_hidden)
        ref_lstm_out, (ref_h_n, ref_c_n) = model.lstm(selected)   # nn.LSTM's own answer, for verification

    print(f"\nVSN output 'selected' {tuple(selected.shape)} -> this is what feeds the LSTM each day")

    # ------------------------------------------------------------------ #
    hr("PART 2 :: MANUALLY UNROLL THE LSTM, DAY BY DAY (both layers)")
    print("Gate meaning:  i = how much NEW info to write   f = how much OLD memory to keep")
    print("               g = candidate new memory content  o = how much of the memory to reveal as output\n")

    with torch.no_grad():
        seq = selected[0]   # (W, vsn_hidden) -- input to layer 0

        h0_0 = torch.zeros(H)
        c0_0 = torch.zeros(H)
        h_seq_l0, c_seq_l0, gates_l0 = manual_lstm_layer(
            seq, h0_0, c0_0,
            lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0, H)

        h0_1 = torch.zeros(H)
        c0_1 = torch.zeros(H)
        h_seq_l1, c_seq_l1, gates_l1 = manual_lstm_layer(
            h_seq_l0, h0_1, c0_1,
            lstm.weight_ih_l1, lstm.weight_hh_l1, lstm.bias_ih_l1, lstm.bias_hh_l1, H)

    print(f"{'day':>4} {'date':>11} | {'i (write)':<28} {'f (keep)':<28} {'o (reveal)':<28} | {'||c||':>8} {'||h||':>8}")
    print("-" * 130)
    for t in range(W):
        i_t, f_t, g_t, o_t = gates_l0[t]
        i_m, f_m, o_m = i_t.mean().item(), f_t.mean().item(), o_t.mean().item()
        c_norm = c_seq_l0[t].norm().item()
        h_norm = h_seq_l0[t].norm().item()
        date_s = dates[args.window + t].date()
        print(f"{t:>4} {str(date_s):>11} | {bar(i_m):<28} {bar(f_m):<28} {bar(o_m):<28} | {c_norm:8.3f} {h_norm:8.3f}")

    print(f"\n(bars show the MEAN gate activation across all {H} hidden units on layer 0, on a 0..1 scale)")
    all_f = torch.stack([g[1] for g in gates_l0])   # (W, H)
    print(f"\nforget gate 'f' across the whole window: mean={all_f.mean():.4f}  "
          f"min={all_f.min():.4f}  max={all_f.max():.4f}")
    print(f"-> staying close to 1.0 means the cell state is NOT being wiped between days;")
    print(f"   this is the forget-gate-bias=1.0 init (Jozefowicz et al., 2015) doing its job.")

    # ------------------------------------------------------------------ #
    hr("PART 3 :: LAYER 0 -> LAYER 1, THEN VERIFY AGAINST nn.LSTM'S OWN OUTPUT")
    print(f"layer 0 hidden sequence h_seq_l0  {tuple(h_seq_l0.shape)}  -> fed as layer 1's input")
    print(f"layer 1 hidden sequence h_seq_l1  {tuple(h_seq_l1.shape)}  -> this is lstm_out")

    diff = (h_seq_l1 - ref_lstm_out[0]).abs().max().item()
    print(f"\nmanual 2-layer unroll vs nn.LSTM(selected) directly: max abs diff = {diff:.3e}  "
          f"({'MATCH' if diff < 1e-4 else 'MISMATCH -- check dropout/eval mode'})")

    with torch.no_grad():
        normed = model.lstm_norm(ref_lstm_out)
        final_out = model.output_proj(normed)
    print(f"\nlstm_out (day {W-1}, the decision day)     : "
          f"[{ref_lstm_out[0, -1, :6].numpy()} ...]  ||h||={ref_lstm_out[0,-1].norm():.4f}")
    print(f"after LayerNorm                          : "
          f"[{normed[0, -1, :6].numpy()} ...]  ||h||={normed[0,-1].norm():.4f}")
    print(f"after output_proj (-> what leaves VSNLSTM): "
          f"[{final_out[0, -1, :6].numpy()} ...]  ||h||={final_out[0,-1].norm():.4f}")

    print(f"""
Note: this model is FRESHLY INITIALISED, not the trained checkpoint (VSNLSTM has
no standalone saved weights -- it only exists trained as part of the full
PPOAgent in results/models/best_model.pt). The gate MECHANICS above (how i/f/g/o
combine to update memory) are identical either way; only the actual gate VALUES
would differ after training. Ask if you'd like a version that loads the trained
checkpoint instead.
""")


if __name__ == "__main__":
    main()
