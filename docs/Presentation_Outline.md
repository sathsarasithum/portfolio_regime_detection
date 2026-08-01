# Research Presentation — Planning Document

**Project:** Regime-Aware Deep Reinforcement Learning for Portfolio Optimisation on the Colombo Stock Exchange
**Format assumed:** 20 minutes presentation + 10 minutes Q&A (adjust slide counts proportionally)
**Target:** ~18–20 content slides

---

## 1. Before you build slides — decide these three things

| Decision | Why it matters | Who decides |
|---|---|---|
| **What is the ONE claim?** | Every slide either supports it or gets cut. Suggested: *"Learning regime detection jointly with the allocation objective outperforms treating regime detection as a separate pre-processing step."* | Whole group |
| **Are baselines ready?** | Without them the results section has no reference point and Q&A will be painful. See §6. | Results owner |
| **Which run are you presenting?** | Currently the training run is ~21k of 500k timesteps and the test window is 92 days. Decide whether to re-run or to present with stated limitations. | Whole group |

**Do not skip the first row.** Groups lose marks by presenting six loosely-connected modules instead of one argument.

---

## 2. Slide-by-slide structure

### Part A — Framing (≈4 min, 4 slides)

**Slide 1 — Title**
Project title, all member names + index numbers, supervisor, institution, date.

**Slide 2 — Problem statement**
- Portfolio allocation on the CSE: a frontier market — thin liquidity, high volatility, strong regime shifts
- Why classical mean-variance optimisation struggles: static covariance assumptions, no regime awareness
- One chart of CSE/ASPI history with visibly different regimes marked. This slide must make the audience *feel* the problem before you show any method.

**Slide 3 — Literature gap**
Keep to 4–5 bullets maximum. The structure that works:
- Prior work A: regime detection via HMM / clustering → *but* regimes are inferred independently of the trading objective
- Prior work B: deep RL for portfolios → *but* no explicit regime structure
- **The gap:** nobody makes regime detection differentiable and trains it end-to-end with the allocation reward

**Slide 4 — Research objectives & contribution**
State 3–4 numbered objectives. Then state the contribution in one sentence — the Differentiable Regime Encoder trained jointly with PPO.

---

### Part B — Methodology (≈8 min, 7–8 slides)

**Slide 5 — System architecture (the anchor slide)**
One full-page block diagram: Data → Preprocessing → Regime Encoder → Actor/Critic → Environment → Reward → PPO update.

> **Presentation tip:** Re-show this diagram in miniature at the corner of each subsequent method slide, with the current block highlighted. This is the single most effective technique for a multi-member technical presentation — the audience never loses track of where they are.

**Slide 6 — Data pipeline**
- Source, universe size, date range, train/validation/test split (with dates)
- Feature engineering and windowing
- Modules: [data_loader.py](../src/data/data_loader.py), [preprocessor.py](../src/data/preprocessor.py)

**Slide 7 — Variable Selection Network + LSTM**
- What the VSN does: learns per-feature relevance weights at each timestep, filtering noise before temporal processing
- Why it matters: interpretability — you can show *which* indicators the model relies on
- Module: [vsn_lstm.py](../src/models/vsn_lstm.py)

**Slide 8 — Temporal Attention**
- Multi-head self-attention over the time dimension for long-range dependency capture
- Module: [temporal_attention.py](../src/models/temporal_attention.py)

**Slide 9 — Macro Graph Prior (GAT)**
- Graph attention over inter-asset and macroeconomic relationships as a structural inductive prior
- Explain how the graph is constructed (nodes, edges, edge weights) — examiners ask this
- Module: [macro_graph_prior.py](../src/models/macro_graph_prior.py)

**Slide 10 — Differentiable Regime Encoder (your core contribution — give it the most time)**
- Fuses VSN-LSTM + attention + graph prior
- Three outputs: `h_market` (→ Critic), `h_asset` (→ Actor), soft regime probabilities (Bull/Bear/Sideways)
- **Emphasise:** regimes are *soft* and *learned end-to-end*, never hard-labelled
- Module: [regime_encoder.py](../src/models/regime_encoder.py)

**Slide 11 — Actor–Critic and PPO**
- Actor: shared per-asset trunk → softmax for long-only allocation ([actor.py](../src/models/actor.py))
- Critic: state-value estimation ([critic.py](../src/models/critic.py))
- PPO clipped objective — show the equation:

  $$L = L^{\text{clip}} + c_1 L^{\text{value}} + c_2 L^{\text{entropy}}$$

**Slide 12 — Reward design**
$$R_t = \text{Sharpe}_t - \text{Transaction Costs} - \lambda \cdot \text{EVaR Penalty} - \text{Turnover Penalty}$$
- Justify EVaR over VaR/CVaR: coherent risk measure, tail-sensitive
- Give the parameter values (0.1% cost, 95% confidence, 0.1 penalty weight)
- Module: [reward.py](../src/environment/reward.py)

---

### Part C — Results (≈6 min, 5 slides) — *the part that decides your grade*

**Slide 13 — Experimental setup**
Hyperparameter table from [config.py](../src/config/config.py), hardware, training duration, number of seeds.

**Slide 14 — Training diagnostics**
The 6-panel loss grid. **Spend ≤60 seconds here.** Say only:
- Value loss converged (0.27 → 0.03) — the critic learned
- Total loss stabilised — training is stable
- One sentence acknowledging the rising policy loss and what you concluded from `approx_kl` / `clip_fraction`

Do not walk through all six panels. These are diagnostics, not results.

**Slide 15 — Out-of-sample performance vs baselines** ⚠️ **THE MOST IMPORTANT SLIDE**

| Strategy | Total Return | Ann. Return | Ann. Vol | **Sharpe** | Max DD | Turnover |
|---|---|---|---|---|---|---|
| **Our model** | 13.93% | 42.93% | 20.42% | **1.81** | 10.83% | ~13%/day |
| Equal-weight (1/N) | ? | ? | ? | ? | ? | ~0 |
| Buy & hold ASPI | ? | ? | ? | ? | ? | 0 |
| Mean-variance | ? | ? | ? | ? | ? | ? |

**Fill the question marks before this presentation happens.** A single-row table is not a result.

> **Use Sharpe = 1.81, not 2.10.** The logged 2.10 comes from [market_env.py:288](../src/environment/market_env.py#L288), which omits the risk-free subtraction. Correct value: (0.4293 − 0.06) / 0.2042 = 1.81. Fix this before the slide is made — an examiner who recomputes it will find the error.

**Slide 16 — Equity curve + regime overlay**
Cumulative portfolio value vs baselines, with detected regime probabilities shaded underneath. This is your most persuasive visual — it shows the strategy *and* the regime mechanism in one frame.

**Slide 17 — Interpretability**
VSN variable-importance chart and/or attention heatmap. Answers "is this a black box?" pre-emptively.

---

### Part D — Closing (≈2 min, 2–3 slides)

**Slide 18 — Limitations (do not skip this)**
Stating limitations yourself converts your weakest points from attacks into evidence of rigour:
- Test window of 92 days → Sharpe 1.81 with 95% CI ≈ [−1.4, 5.1]; **not statistically significant** (t ≈ 1.09)
- Single random seed — RL results vary across seeds
- Turnover ~13%/day implies costs beyond the modelled 0.1%; CSE liquidity may not support execution
- Training run truncated relative to the 500k-step budget *(remove if you re-run to completion)*

**Slide 19 — Future work**
Extended walk-forward testing, multi-seed evaluation, turnover-constrained variants, live paper-trading.

**Slide 20 — Conclusion**
Restate the one claim from §1 and the three strongest supporting numbers. End on the contribution, not on a "Thank you" slide.

---

## 3. Group role allocation

Map presenters to the modules they built. Suggested split for five members:

| Member | Slides | Modules owned |
|---|---|---|
| 1 | 1–4 (framing) | Problem framing, literature review |
| 2 | 5–6 (pipeline) | `data_loader.py`, `preprocessor.py`, `dataset.py` |
| 3 | 7–9 (encoder components) | `vsn_lstm.py`, `temporal_attention.py`, `macro_graph_prior.py` |
| 4 | 10–12 (agent + reward) | `regime_encoder.py`, `actor.py`, `critic.py`, `ppo_agent.py`, `reward.py` |
| 5 | 13–20 (results + closing) | `trainer.py`, `metrics.py`, `visualization.py`, baselines |

**Rules that prevent the common failures:**
- Every member presents; every member must be able to answer questions on **any** slide, not just their own
- Rehearse handovers explicitly — one sentence linking your last slide to the next presenter's first
- Uniform template: same fonts, same colours, same chart styling across all members' slides. Mismatched slide design is the most visible sign of a group that didn't integrate.

---

## 4. Design guidelines

- **Font ≥ 24pt.** Anything smaller is unreadable from the back.
- **One idea per slide.** If a slide needs two headings, it's two slides.
- **No paragraphs.** Bullets ≤ 8 words. The detail goes in speaker notes.
- **Regenerate all charts at high DPI** with readable axis labels — do not screenshot TensorBoard. Use [visualization.py](../src/utils/visualization.py) and export at `dpi=200`.
- **Consistent colour coding:** assign one colour per strategy (e.g. your model = blue, baselines = grey) and keep it identical in every chart.
- **Number every slide** — essential for Q&A ("could you go back to slide 15?").

---

## 5. Q&A preparation — the questions you WILL be asked

Assign each question to a specific member. Rehearse the answers aloud.

| # | Likely question | Your answer must cover |
|---|---|---|
| 1 | *"Is Sharpe 1.81 statistically significant over 92 days?"* | **No** — CI spans zero, t ≈ 1.09. Say so directly, then explain the walk-forward plan. Honesty scores higher than bluffing. |
| 2 | *"Does it beat a simple equal-weight portfolio?"* | Requires Slide 15 to be complete. There is no acceptable answer without baselines. |
| 3 | *"13% daily turnover on the CSE — is that executable?"* | Acknowledge the liquidity constraint; note the turnover penalty exists but is weakly weighted (0.001); propose stronger constraints as future work. |
| 4 | *"How do you know the regimes are real and not arbitrary clusters?"* | Point to Slide 16 — regime probabilities aligning with observable market phases. Consider adding a regime-transition matrix. |
| 5 | *"Why PPO rather than DDPG/SAC/A2C?"* | Stability under a clipped objective, on-policy suitability for non-stationary financial data, robustness to hyperparameters. |
| 6 | *"Why EVaR instead of CVaR?"* | Coherent risk measure, tighter tail bound, differentiable — critical for end-to-end training. |
| 7 | *"Did you tune hyperparameters on the test set?"* | Must be **no**, and you must be able to describe the validation protocol. |
| 8 | *"What's the contribution over existing regime-detection RL work?"* | The differentiability and joint optimisation — regimes shaped by the trading objective itself. |
| 9 | *"Your policy loss rises during training — is that divergence?"* | No. Explain the clipped surrogate saturating as the ratio → 1; cite `clip_fraction` and `approx_kl`. |
| 10 | *"How many random seeds?"* | Answer truthfully. If one, name it as a limitation before they do. |

---

## 6. Pre-presentation checklist

**Blocking — the presentation is not ready without these:**
- [ ] Baselines implemented (1/N, buy-and-hold ASPI, mean-variance) and Slide 15 filled in
- [ ] Sharpe corrected to 1.81 everywhere, or [market_env.py:288](../src/environment/market_env.py#L288) fixed to route through `PortfolioMetrics`
- [ ] Train/validation/test split dates stated explicitly on Slide 6
- [ ] Confidence interval on the Sharpe reported on Slide 18

**Strongly recommended:**
- [ ] Training run extended beyond 21k timesteps
- [ ] Test window extended beyond 92 days, or walk-forward across multiple windows
- [ ] 3–5 seeds with mean ± std on headline metrics
- [ ] All charts regenerated at presentation resolution

**Logistics:**
- [ ] Full rehearsal with timing — cut content if over 20 min, do not speed up
- [ ] Handover sentences rehearsed between members
- [ ] Slides exported to PDF as a fallback (fonts/animations break across machines)
- [ ] Backup copy on USB and cloud
- [ ] Appendix slides prepared for anticipated questions (extra charts, full hyperparameter table, derivations)

---

## 7. The three things that most affect your grade

1. **Baselines.** A result without a comparison is not a result. This is the highest-priority outstanding task.
2. **Honest treatment of statistical significance.** Presenting 1.81 [−1.4, 5.1] and explaining it demonstrates more competence than presenting 2.10 unqualified — and it is defensible under questioning.
3. **One coherent story.** Five members presenting five disconnected modules reads as five mini-projects. The architecture diagram recurring on every method slide is what binds them into one system.
