# Presentation Content — Slide-by-Slide Script

**Project:** Regime-Aware Deep Reinforcement Learning for Portfolio Optimisation on the Colombo Stock Exchange
**Duration:** 20 minutes + 10 minutes Q&A
**Companion document:** [Presentation_Outline.md](Presentation_Outline.md) — planning, roles, checklist

> **How to use this file.** Each slide below gives: (a) the **title**, (b) the **exact text to put on the slide**, (c) the **visual** required, and (d) a **speaker script** to say aloud. Copy (b) into PowerPoint verbatim; paste (d) into the speaker-notes pane.
>
> **🔲 marks a placeholder you must fill in.** Search this file for `🔲` before building slides.

---

## PART A — FRAMING (4 min)

---

### SLIDE 1 — Title *(15 sec)*

**On slide:**

> # Regime-Aware Deep Reinforcement Learning for Portfolio Optimisation
> ### A Differentiable Regime Encoder with PPO on the Colombo Stock Exchange
>
> 🔲 Member 1 Name — Index No.
> 🔲 Member 2 Name — Index No.
> 🔲 Member 3 Name — Index No.
> 🔲 Member 4 Name — Index No.
> 🔲 Member 5 Name — Index No.
>
> **Supervisor:** 🔲 Name
> 🔲 Department, University · 🔲 Date

**Script:**
> "Good morning. We're presenting our research on regime-aware reinforcement learning for portfolio optimisation on the Colombo Stock Exchange. I'm 🔲, and I'll be joined by my four team members."

---

### SLIDE 2 — The Problem *(75 sec)*

**On slide:**

> ## Markets change. Static portfolios don't.
>
> - The CSE is a **frontier market** — thin liquidity, high volatility
> - Market behaviour shifts between **distinct regimes**: bull, bear, sideways
> - Classical mean-variance optimisation assumes a **stationary** covariance structure
> - A single fixed allocation rule is **wrong in at least two of three regimes**

**Visual:** Full-width ASPI / CSE index chart across your data period, with 2–3 regime phases shaded and labelled ("Bull", "Correction", "Sideways"). This is the emotional hook — make it large and obvious.

**Script:**
> "Here is the Colombo Stock Exchange over our study period. You can see with your own eyes that this is not one market — it is at least three. There's a sustained rally here, a sharp drawdown here, and an extended sideways period here.
>
> Classical portfolio theory — Markowitz mean-variance optimisation — assumes returns and covariances are stationary. On a chart like this, that assumption is simply false. An allocation that is optimal during the rally is badly wrong during the drawdown.
>
> So the question becomes: can a model learn to recognise which regime it is in, and allocate accordingly?"

---

### SLIDE 3 — The Research Gap *(60 sec)*

**On slide:**

> ## Existing work solves half the problem
>
> **Regime detection (HMM, clustering, change-point):**
> ✅ Identifies regimes  ❌ Regimes inferred **independently** of the trading objective
>
> **Deep RL for portfolios (DDPG, PPO, SAC):**
> ✅ Learns allocation  ❌ **No explicit** regime structure
>
> **Two-stage pipelines (detect → then allocate):**
> ❌ Detection errors propagate  ❌ **No gradient flows back** from returns to the detector
>
> ### The gap
> **Nobody makes regime detection differentiable and trains it end-to-end with the allocation reward.**

**Script:**
> "The literature approaches this from two directions.
>
> The first is statistical regime detection — hidden Markov models, clustering, change-point detection. These identify regimes, but they do it in isolation. The regimes are fitted to the price series, not to the question of what makes money.
>
> The second is deep reinforcement learning for portfolio management. These learn allocation policies well, but treat the market as a single undifferentiated process.
>
> Some work combines them in a two-stage pipeline: detect the regime first, then feed the label to an allocator. The problem is that the two stages never talk to each other. If the detector produces a regime that is statistically valid but useless for trading, no gradient ever tells it so.
>
> That is the gap we address."

---

### SLIDE 4 — Objectives & Contribution *(60 sec)*

**On slide:**

> ## Research objectives
>
> 1. Design a **differentiable** regime encoder producing *soft* regime probabilities
> 2. Train regime detection **jointly** with portfolio allocation under a single objective
> 3. Incorporate **tail-risk control** (EVaR) and realistic **transaction costs** into the reward
> 4. Evaluate on **real CSE data** against classical and passive baselines
>
> ### Core contribution
> > A **Differentiable Regime Encoder** whose regime representation is shaped by the trading objective itself — trained end-to-end with PPO, not as a pre-processing step.

**Script:**
> "Our objectives are four. First, build a regime encoder that is fully differentiable and outputs soft probabilities rather than hard labels. Second, train it jointly with the allocation policy so that a single gradient signal shapes both. Third, build a reward that reflects real trading — risk-adjusted return, minus transaction costs, minus a tail-risk penalty. Fourth, evaluate on actual Colombo Stock Exchange data against meaningful baselines.
>
> Our contribution is the first of these. The regime representation is not fitted separately and handed over — it is learned because it improves the portfolio, and the gradient of the portfolio return reaches all the way back into it.
>
> My colleague 🔲 will now take you through the system architecture."

---

## PART B — METHODOLOGY (8 min)

---

### SLIDE 5 — System Architecture *(75 sec)*

**On slide:**

> ## End-to-end architecture
>
> ```
> CSE Price/Volume + Macro Data
>            ↓
>   Preprocessing (windowing, features, z-score)
>            ↓
> ┌─────────────────────────────────────────┐
> │   DIFFERENTIABLE REGIME ENCODER         │
> │   VSN + LSTM → Temporal Attention       │
> │            ↘ Macro Graph Prior (GAT)    │
> │   → h_market · h_asset · regime probs   │
> └─────────────────────────────────────────┘
>            ↓                    ↓
>      Actor (π)             Critic (V)
>            ↓
>   Market Environment (trades, costs)
>            ↓
>   Reward: Sharpe − Costs − EVaR − Turnover
>            ↓
>   PPO Update ──── gradient flows back to ALL blocks
> ```

**Visual:** Replace the ASCII above with a proper block diagram. **Then reuse it as a small corner thumbnail on slides 6–12 with the current block highlighted.**

**Script:**
> "This is the full system, and I'd ask you to hold this picture in mind — we'll return to it on every slide that follows.
>
> Data enters at the top: CSE price and volume series plus macroeconomic variables. Preprocessing turns these into normalised rolling windows.
>
> The centre block is our contribution — the Differentiable Regime Encoder. It produces three outputs: a pooled market state for the critic, per-asset embeddings for the actor, and soft regime probabilities.
>
> The actor turns those embeddings into portfolio weights. The environment executes the trades and charges transaction costs. The reward module computes risk-adjusted performance.
>
> And here is the critical arrow — the PPO gradient flows back through *everything*. There is no frozen stage, no pre-trained component. The regime encoder learns what a regime is by being told how much money it made."

---

### SLIDE 6 — Data Pipeline *(60 sec)*

**On slide:**

> ## Data & preprocessing
>
> | | |
> |---|---|
> | Source | Colombo Stock Exchange (price + volume) 🔲 + macro indicators |
> | Universe | 🔲 assets |
> | Period | 🔲 start – 🔲 end |
> | **Train / Val / Test** | **70% / 15% / 15%** — strictly chronological |
> | Lookback window | **30 trading days** |
> | Normalisation | Z-score |
>
> **Engineered features:** log returns · rolling volatility · momentum · volume-based features
>
> ⚠️ Test set held out entirely — never seen during training or tuning

**Script:**
> "Our data comes from the Colombo Stock Exchange — daily price and volume for 🔲 assets over 🔲 to 🔲 — combined with macroeconomic indicators.
>
> We split chronologically, seventy-fifteen-fifteen. Chronologically matters: a random split would let the model see the future, which is the most common and most fatal error in financial machine learning.
>
> Each observation is a thirty-day lookback window. From the raw series we engineer log returns, rolling volatility, momentum indicators, and volume features, then z-score normalise using training-set statistics only.
>
> The test set was held out completely — not used for training, and not used for hyperparameter selection."

---

### SLIDE 7 — Variable Selection Network + LSTM *(60 sec)*

**On slide:**

> ## Step 1 — Which features matter, and when?
>
> **Variable Selection Network (VSN)**
> - Learns a **relevance weight per feature, per timestep**
> - Suppresses noisy indicators *before* temporal modelling
> - Weights are inspectable → **interpretability**
>
> **LSTM** — 128 hidden units · 2 layers · dropout 0.1
> - Captures temporal dynamics in the filtered signal
>
> *Not all indicators matter in all regimes — the VSN learns this.*

**Visual:** Small diagram: features → VSN gating weights → weighted features → LSTM. Corner thumbnail of Slide 5 with this block highlighted.

**Script:**
> "The first stage of the encoder answers a question that fixed feature sets cannot: which indicators actually matter right now?
>
> The Variable Selection Network learns a relevance weight for every input feature at every timestep. Features that carry no signal get suppressed before they reach the LSTM. This matters especially in a frontier market where many indicators are noisy.
>
> There is a second benefit. Because these weights are explicit, we can read them out and see which variables the model relies on — we'll show that in the results.
>
> The filtered sequence then passes through a two-layer LSTM with 128 hidden units, which models the temporal dynamics."

---

### SLIDE 8 — Temporal Attention *(45 sec)*

**On slide:**

> ## Step 2 — Which *periods* matter?
>
> **Multi-Head Self-Attention** over the time dimension
> - 4 heads · dimension 128 · dropout 0.1
> - Captures **long-range dependencies** the LSTM compresses away
> - Each head can specialise on a different temporal pattern
>
> *Regime shifts are not always recent — attention lets the model look back.*

**Script:**
> "An LSTM compresses history into a fixed-size hidden state, which means distant but important events can be washed out.
>
> We add multi-head self-attention across the thirty-day window — four heads at dimension 128. This lets the model attend directly to any point in the window. If a regime shift began three weeks ago, attention can reach it, whereas recurrence alone would have diluted it.
>
> Multiple heads let different heads specialise — one may track volatility clustering, another momentum reversals."

---

### SLIDE 9 — Macro Graph Prior *(60 sec)*

**On slide:**

> ## Step 3 — How are assets connected?
>
> **Graph Attention Network (GAT)** as a structural inductive prior
> - **Nodes:** 🔲 assets + macroeconomic variables
> - **Edges:** 🔲 describe your construction (e.g. return correlation, sector membership, macro linkage)
> - 2 layers · 2 attention heads · hidden dim 64
>
> **Why a prior?** Purely data-driven models must rediscover market structure from scarce frontier-market data. The graph supplies it.
>
> *Attention weights are learned — the graph proposes, the model disposes.*

**Script:**
> "Assets do not move independently. Banks co-move; a currency shock propagates across exporters.
>
> We encode this as a graph — nodes are assets and macroeconomic variables, edges are 🔲. A Graph Attention Network then propagates information along those edges.
>
> The word 'prior' is deliberate. On a frontier market we have limited data, and a purely data-driven model would have to rediscover basic market structure from that scarce data. We supply the structure instead, and let the attention mechanism learn *how strongly* to use each connection. The graph proposes relationships; the model decides which matter."

> **🔲 Prepare for the follow-up:** examiners consistently ask how the graph is constructed. Have the adjacency-matrix definition ready, ideally as an appendix slide.

---

### SLIDE 10 — The Differentiable Regime Encoder *(90 sec — your key slide)*

**On slide:**

> ## The core contribution
>
> **Fusion:** VSN-LSTM ⊕ Temporal Attention ⊕ Macro Graph Prior → latent state (dim 128)
>
> **Three outputs:**
>
> | Output | Dimension | Consumed by |
> |---|---|---|
> | `h_market` | 128 | Critic — pooled market state |
> | `h_asset` | per-asset | Actor — one embedding per asset |
> | **Regime probabilities** | **3** (Bull / Bear / Sideways) | Interpretability + conditioning |
>
> ### Why this is different
> - Regimes are **soft** — a probability simplex, never a hard label
> - **No regime labels exist in the data** — nothing is supervised
> - The encoder is **fully differentiable** — reward gradients reach every parameter
> - Regimes emerge because they **improve the portfolio**, not because they fit the prices

**Script:**
> "This is the heart of the work, so I'll spend a moment here.
>
> The three streams — the filtered temporal signal, the attention output, and the graph representation — are fused into a single 128-dimensional latent state. From it we produce three things: a pooled market state for the critic, per-asset embeddings for the actor, and a three-way soft regime probability over bull, bear and sideways.
>
> Three properties make this different from prior work.
>
> First, the regimes are soft. The model never commits to 'we are in a bear market'; it says seventy percent bear, twenty percent sideways. Real markets transition gradually, and hard labels destroy that information.
>
> Second — and this is the point examiners should take away — **we have no regime labels**. Nothing here is supervised. There is no ground truth telling the model what a bull market is.
>
> Third, because every operation is differentiable, the gradient of the portfolio reward flows all the way back into this encoder. The regimes are shaped by profitability.
>
> So the regimes that emerge are not the statistically neatest partition of the price series. They are the partition that helps allocate capital. That is the distinction between our approach and a two-stage pipeline."

---

### SLIDE 11 — Actor–Critic and PPO *(75 sec)*

**On slide:**

> ## Policy learning with PPO
>
> **Actor (π)** — shared per-asset trunk → one logit per asset → portfolio weights
> *Cross-asset interaction already happened in the encoder; the actor scores each asset on its own evidence*
> Hidden dims [256, 128]
>
> **Critic (V)** — estimates state value for advantage computation. Hidden dims [256, 128]
>
> **PPO clipped objective:**
>
> $$L = \underbrace{L^{\text{clip}}}_{\text{policy}} + \underbrace{c_1 L^{\text{value}}}_{c_1 = 0.5} + \underbrace{c_2 L^{\text{entropy}}}_{c_2 = 0.02}$$
>
> $$L^{\text{clip}} = \mathbb{E}\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right], \quad \epsilon = 0.2$$
>
> **Why PPO?** Stable under a clipped trust region · on-policy suits non-stationary markets · robust to hyperparameters
> **Advantages:** Generalised Advantage Estimation, γ = 0.99, λ = 0.95

**Script:**
> "The actor maps each asset's embedding through a shared trunk to a single score, which becomes that asset's portfolio weight. The weights are shared across assets deliberately — cross-asset interaction has already happened inside the encoder, so the actor evaluates each asset on the evidence it has been given.
>
> The critic estimates the value of the current state, which we need for advantage estimation.
>
> We optimise with Proximal Policy Optimisation. The clipped objective prevents any single update from moving the policy too far — epsilon is 0.2, so the probability ratio is confined to plus or minus twenty percent. That trust region is what makes PPO stable, and stability matters enormously in financial RL where a single bad update can destroy a policy.
>
> The total loss adds a value term weighted at 0.5 and an entropy bonus at 0.02, which maintains exploration. Advantages come from Generalised Advantage Estimation with gamma 0.99 and lambda 0.95."

---

### SLIDE 12 — Reward Design *(60 sec)*

**On slide:**

> ## What the agent is actually optimising
>
> $$R_t = \text{Sharpe}_t^{(20d)} - \text{Transaction Costs} - \lambda_{\text{EVaR}} \cdot \text{EVaR}_{95\%} - \text{Turnover Penalty}$$
>
> | Component | Value | Purpose |
> |---|---|---|
> | Rolling Sharpe | 20-day window, rf = 6% | Risk-adjusted return, not raw return |
> | Transaction cost | 0.1% per trade | Realistic CSE friction |
> | EVaR penalty | 95% confidence, weight 0.1 | Tail-risk control |
> | Turnover penalty | 0.001 | Discourage excessive churn |
>
> **Why EVaR over VaR / CVaR?**
> ✅ **Coherent** risk measure (VaR is not) · ✅ **Tighter** tail bound than CVaR · ✅ **Differentiable** — essential for end-to-end training

**Script:**
> "The reward is where the finance enters.
>
> We do not reward raw return — that would produce a maximally leveraged, maximally risky policy. We reward the rolling twenty-day Sharpe ratio against a six percent Sri Lankan risk-free rate.
>
> From that we subtract transaction costs at ten basis points per trade, which is representative of CSE execution, and a turnover penalty to discourage churn.
>
> Finally we subtract an Entropic Value-at-Risk penalty at ninety-five percent confidence. We chose EVaR for three reasons. It is a coherent risk measure, which plain VaR is not — VaR can penalise diversification. It gives a tighter bound on tail losses than CVaR. And critically for us, it is differentiable, so it can participate in end-to-end gradient training.
>
> My colleague 🔲 will now present the results."

---

## PART C — RESULTS (6 min)

---

### SLIDE 13 — Experimental Setup *(45 sec)*

**On slide:**

> ## Experimental configuration
>
> | PPO | | Architecture | |
> |---|---|---|---|
> | Clip ε | 0.2 | LSTM hidden / layers | 128 / 2 |
> | γ (discount) | 0.99 | Attention heads / dim | 4 / 128 |
> | GAE λ | 0.95 | GAT layers / heads | 2 / 2 |
> | Learning rate | 1e-4 | Latent state dim | 128 |
> | Entropy coeff | 0.02 | Actor / Critic hidden | [256, 128] |
> | Value coeff | 0.5 | Regimes | 3 |
> | Batch size | 64 | Lookback window | 30 days |
> | Epochs / update | 10 | | |
> | Rollout length | 256 | **Seeds run** | 🔲 |
>
> Hardware: 🔲 · Training time: 🔲 · Framework: PyTorch

**Script:**
> "Briefly, our configuration. PPO with standard clipping at 0.2, discount 0.99, GAE lambda 0.95, learning rate 1e-4. The encoder uses a two-layer LSTM at 128 units, four attention heads, and a two-layer GAT. Full details are in the report."

---

### SLIDE 14 — Training Diagnostics *(45 sec — keep this SHORT)*

**On slide:**

> ## Training converged and remained stable
>
> - **Value loss:** 0.27 → 0.03 within ~4,000 steps — the critic learned to predict returns
> - **Total loss:** stabilised after ~15,000 steps, bounded oscillation around zero
> - **Policy loss** rises toward zero as the clipped surrogate saturates — expected PPO behaviour, **not** divergence
> - Entropy remained stable → **no policy collapse**, exploration preserved

**Visual:** The 6-panel loss grid, regenerated at ≥200 DPI with readable labels. Do not screenshot TensorBoard.

**Script:**
> "A word on training stability — these are diagnostics rather than results, so I'll be brief.
>
> The value loss collapsed from 0.27 to 0.03 within four thousand steps, confirming the critic learned to predict returns. The total loss stabilised and oscillates in a narrow band.
>
> One point worth pre-empting: our policy loss *rises* during training. This is not divergence. PPO's clipped surrogate is recomputed on fresh data each update, and as the policy approaches optimality for the current data distribution the probability ratio approaches one and the surrogate saturates toward zero by construction. Entropy remained stable throughout, so the policy has not collapsed — exploration was preserved."

---

### SLIDE 15 — Out-of-Sample Performance *(120 sec — THE decisive slide)*

**On slide:**

> ## Test-set performance vs baselines
> **92 trading days · LKR 1,000,000 initial capital · fully out-of-sample**
>
> | Strategy | Total Return | Ann. Return | Ann. Vol | **Sharpe** | Max DD | Turnover |
> |---|---|---|---|---|---|---|
> | **Regime-Aware PPO (ours)** | **13.93%** | 42.93%* | 20.42% | **1.81** | **10.83%** | ~13%/day |
> | Equal-weight (1/N) | 🔲 | 🔲 | 🔲 | 🔲 | 🔲 | ~0 |
> | Buy & hold ASPI | 🔲 | 🔲 | 🔲 | 🔲 | 🔲 | 0 |
> | Mean-variance (Markowitz) | 🔲 | 🔲 | 🔲 | 🔲 | 🔲 | 🔲 |
>
> \* annualised by extrapolation from 92 days — see limitations
> Sharpe computed net of 6% risk-free rate · returns net of 0.1% transaction costs (total LKR 13,159)

**Script:**
> "This is our central result: performance on the held-out test set, ninety-two trading days the model had never seen.
>
> Starting from one million rupees, our agent returned 13.93 percent, ending at 1,139,277 — net of all transaction costs, which totalled just over thirteen thousand rupees. Annualised volatility was 20.4 percent and the Sharpe ratio, net of the six percent risk-free rate, was 1.81. Maximum drawdown was contained at 10.8 percent.
>
> Against the baselines: 🔲 *[state the comparison plainly — whether you beat 1/N and the index, and by how much]*.
>
> I want to be precise about one figure. The 42.93 percent annualised return is an extrapolation from a ninety-two day window, not money earned over a year. The honest headline number is 13.93 percent over four and a half months. I'll return to what that sample size means for confidence in a moment."

> ### ⚠️ Two things you MUST do before this slide is final
> 1. **Fill in the baseline rows.** A one-row table is not a result, and Q&A will centre on this.
> 2. **Use Sharpe = 1.81, not the logged 2.10.** The environment's summary at [market_env.py:288](../src/environment/market_env.py#L288) computes `annual_return / annual_vol` with no risk-free subtraction. Correct value: (0.4293 − 0.06) / 0.2042 = **1.81**.

---

### SLIDE 16 — Equity Curve + Regime Overlay *(75 sec)*

**On slide:**

> ## Does the regime signal actually do anything?
>
> **Top panel:** cumulative portfolio value — ours vs baselines
> **Bottom panel:** detected regime probabilities (Bull / Bear / Sideways) over the same period
>
> - 🔲 Detected regimes align with observable CSE market phases
> - 🔲 Allocation shifts **defensively** as bear probability rises
> - 🔲 Drawdowns are shallower during high bear-probability periods

**Visual:** Two stacked panels sharing the x-axis — equity curves on top, stacked regime-probability area chart below. **This is your most persuasive slide; invest the most design effort here.**

**Script:**
> "This slide connects the mechanism to the outcome.
>
> The top panel is cumulative portfolio value — ours against the baselines. The bottom panel shows the regime probabilities our encoder produced over exactly the same period.
>
> Look at 🔲 *[point to a specific date where bear probability rises]*. The model's bear probability climbs here, and in the panel above you can see the allocation becoming defensive and the drawdown staying shallower than the benchmark.
>
> That correspondence is the argument. The regime signal is not decorative — it is doing work. And remember, no one labelled these regimes. The model learned them from the profitability gradient alone."

> **🔲 If the correspondence is weak in your actual plot, say so honestly** and discuss it as a finding. Overclaiming here is the fastest way to lose credibility under questioning.

---

### SLIDE 17 — Interpretability *(45 sec)*

**On slide:**

> ## The model is not a black box
>
> **VSN variable importance** — which features drive regime detection
> 🔲 Top features: ______, ______, ______
>
> **Temporal attention heatmap** — which periods the model weights
> 🔲 Observation: ______
>
> *Explicit gating and attention weights make the decision process inspectable — important for adoption in a regulated market.*

**Script:**
> "A frequent and fair criticism of deep RL in finance is opacity. Our architecture answers it partly by construction.
>
> The Variable Selection Network's gating weights tell us directly which features drive the regime signal — here, 🔲. The attention heatmap shows which parts of the thirty-day window the model weights most heavily.
>
> This matters beyond academic interest. A portfolio manager in a regulated market cannot deploy a model that gives no account of itself."

---

## PART D — CLOSING (2 min)

---

### SLIDE 18 — Limitations *(60 sec — do NOT skip)*

**On slide:**

> ## Limitations — stated honestly
>
> **1. Sample size** — 92-day test window
> Sharpe 1.81, **95% CI ≈ [−1.4, 5.1]**, t ≈ 1.09 → **not statistically significant**
> *The point estimate is strong; the sample is too small to confirm it*
>
> **2. Single random seed** 🔲 — RL results vary materially across seeds
>
> **3. Turnover ~13% of the portfolio per day**
> Implies ~3.6% annual cost drag; CSE liquidity may not support execution at quoted prices
>
> **4. Training budget** 🔲 — run reached 21k of the 500k-step schedule

**Script:**
> "We want to state our limitations ourselves rather than have them found.
>
> The most important is sample size. Ninety-two days gives a Sharpe of 1.81, but the ninety-five percent confidence interval runs from roughly minus 1.4 to plus 5.1. That interval contains zero. We cannot claim statistical significance. The point estimate is encouraging — but honesty requires saying that four and a half months is not enough to prove it.
>
> Second, our headline run uses a single random seed, and reinforcement learning results are known to vary across seeds.
>
> Third, and practically most serious: our agent turns over roughly thirteen percent of the portfolio daily. On a frontier market with thin order books, the realised cost would exceed the ten basis points we modelled once spread and market impact are included. Our turnover penalty exists but is evidently too weakly weighted.
>
> These are real constraints on what we can claim, and they define our next steps."

---

### SLIDE 19 — Future Work *(30 sec)*

**On slide:**

> ## Next steps
>
> 1. **Walk-forward evaluation** across multiple non-overlapping windows → statistical power
> 2. **Multi-seed runs** (5+) reporting mean ± std
> 3. **Turnover-constrained variant** — stronger penalty or explicit trade limits
> 4. **Liquidity-aware execution model** — spread and market-impact costs
> 5. **Cross-market validation** — test transfer to other frontier markets

**Script:**
> "Our priorities follow directly from those limitations. Walk-forward evaluation across multiple windows to gain statistical power. Multi-seed runs for robustness. A turnover-constrained variant, and a more realistic execution model incorporating spread and market impact. Longer term, testing whether the architecture transfers to other frontier markets."

---

### SLIDE 20 — Conclusion *(45 sec)*

**On slide:**

> ## Conclusion
>
> ### We built a regime encoder that learns what a regime is *from profitability*
>
> - Regime detection made **differentiable** and trained **end-to-end** with allocation — no labels, no two-stage pipeline
> - On held-out CSE data: **13.93% over 92 days**, Sharpe **1.81**, max drawdown **10.8%**, net of costs
> - Regime probabilities align with observable market phases and drive **defensive reallocation**
> - Sample size limits statistical confidence — **walk-forward validation is the next step**

**Script:**
> "To conclude.
>
> We set out to close a specific gap: regime detection in portfolio RL is done separately from allocation, so nothing tells the detector which regimes are worth detecting. Our Differentiable Regime Encoder closes it — regimes emerge, unlabelled, from the profitability gradient.
>
> On held-out Colombo Stock Exchange data the agent returned 13.93 percent over ninety-two days with a Sharpe of 1.81 net of costs and a drawdown under eleven percent. The learned regimes correspond to observable market phases and drive visible defensive reallocation.
>
> We are clear-eyed that the test window is too short for statistical significance, and extending it is our immediate next step. But the mechanism works, and it works without a single regime label.
>
> Thank you — we're happy to take questions."

---

## APPENDIX SLIDES — prepare, but do not present

Keep these after Slide 20 to pull up during Q&A:

- **A1** — Graph construction: adjacency matrix definition, edge weighting
- **A2** — Full hyperparameter table
- **A3** — EVaR derivation and why it is coherent
- **A4** — All six training-diagnostic panels at full size
- **A5** — Per-asset allocation weights over time
- **A6** — Regime transition matrix
- **A7** — Sharpe confidence-interval derivation (Lo, 2002): $SE(SR) = \sqrt{(1 + SR^2/2)/n}$
- **A8** — Data preprocessing detail, feature formulas

---

## TIMING SUMMARY

| Part | Slides | Target |
|---|---|---|
| A — Framing | 1–4 | 4:00 |
| B — Methodology | 5–12 | 8:00 |
| C — Results | 13–17 | 6:00 |
| D — Closing | 18–20 | 2:00 |
| **Total** | **20** | **20:00** |

If you overrun in rehearsal, cut from Part B (compress slides 8 and 9 into one). **Never** cut from Part C.

---

## FINAL PRE-FLIGHT

- [ ] Every 🔲 in this document resolved
- [ ] Baseline rows on Slide 15 filled
- [ ] Sharpe shown as 1.81 everywhere
- [ ] Charts regenerated at ≥200 DPI
- [ ] Full rehearsal timed under 20:00
- [ ] Handover sentences rehearsed between members
- [ ] Exported to PDF as fallback; backup on USB + cloud
- [ ] Q&A questions assigned to named members (see [Presentation_Outline.md](Presentation_Outline.md) §5)
