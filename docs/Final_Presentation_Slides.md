# Final Presentation Slide Deck Draft

Project: Regime-Aware Deep Reinforcement Learning for Portfolio Optimisation on the Colombo Stock Exchange

## Slide 1 — Title Slide
**Title:** Regime-Aware Deep Reinforcement Learning for Portfolio Optimisation on the Colombo Stock Exchange

**Add on the slide:**
- Your full names and index numbers
- Supervisor name
- Institution name
- Date
- Optional: a small visual of the CSE market chart or portfolio icon

**Speaker note:**
Introduce the topic briefly and state the main idea: the model learns market regimes and uses them to make portfolio decisions.

---

## Slide 2 — Problem Statement
**Title:** Why portfolio optimisation in the CSE is difficult

**Add on the slide:**
- CSE is a frontier market with high volatility and regime shifts
- Traditional portfolio methods such as mean-variance often assume fixed market behaviour
- Market conditions change over time, so static models can fail
- A portfolio manager needs a method that adapts to changing regimes

**Visual suggestion:**
- One chart of the CSE / ASPI index showing periods of bullish, bearish, and sideways movement

**Speaker note:**
Explain that the market does not behave the same way all the time, so a one-size-fits-all strategy is weak.

---

## Slide 3 — Literature Gap
**Title:** What is missing in existing research?

**Add on the slide:**
- Existing regime detection methods often separate regime detection from trading
- Standard deep RL portfolio methods usually do not explicitly learn regimes
- Most approaches treat regimes as fixed or pre-processed labels
- The gap: regimes should be learned jointly with portfolio decision-making

**Visual suggestion:**
- A simple comparison table: “Regime detection only” vs “RL only” vs “Proposed joint approach”

**Speaker note:**
State clearly that the novelty of your work is the joint learning of regime detection and portfolio optimisation.

---

## Slide 4 — Research Objectives and Contribution
**Title:** Objectives and contribution

**Add on the slide:**
- Objective 1: Build a regime-aware portfolio optimisation framework
- Objective 2: Learn market regimes directly from financial data
- Objective 3: Train the regime encoder jointly with the PPO portfolio policy
- Objective 4: Evaluate the strategy against strong baselines

**Contribution statement:**
- A differentiable regime encoder is trained end-to-end with the PPO agent so that regime detection is shaped by the portfolio objective.

**Speaker note:**
Keep this slide short and strong. Your contribution should be the main message of the whole presentation.

---

## Slide 5 — System Architecture
**Title:** End-to-end system overview

**Add on the slide:**
- A block diagram showing:
  - Raw data
  - Preprocessing
  - Regime encoder
  - Actor/Critic
  - Environment
  - Reward
  - PPO update

**Visual suggestion:**
- One complete architecture diagram covering the full pipeline

**Speaker note:**
This is your anchor slide. Keep it visible throughout the presentation and refer back to it when moving to each component.

---

## Slide 6 — Data Pipeline
**Title:** Data collection, preprocessing, and windowing

**Add on the slide:**
- Raw CSE banking-sector price and volume data
- Feature engineering: returns, volatility, momentum, RSI, MACD, Bollinger bands, volume features
- Normalisation and sliding window creation
- Train / validation / test split details

**Visual suggestion:**
- A simple pipeline flowchart
- Small table with dataset split sizes and dates

**Speaker note:**
Explain that the model does not see raw prices directly; it receives structured temporal windows of engineered features.

---

## Slide 7 — Regime Encoder Part 1: VSN + LSTM
**Title:** Variable Selection Network and LSTM

**Add on the slide:**
- VSN learns which input features are most useful at each time step
- It filters noise before temporal modelling
- LSTM captures temporal dependencies across the window
- This helps the model focus on relevant signals instead of all features equally

**Visual suggestion:**
- A small schematic of VSN → LSTM
- Optionally, a simple feature importance bar chart

**Speaker note:**
Explain that this part improves robustness and interpretability by deciding which features matter most.

---

## Slide 8 — Regime Encoder Part 2: Temporal Attention
**Title:** Temporal attention for long-range patterns

**Add on the slide:**
- Multi-head self-attention is applied over the time dimension
- The model can attend to important historical periods in the window
- It helps capture long-range dependencies rather than only short-term motion

**Visual suggestion:**
- A heatmap showing attention weights across the time window

**Speaker note:**
Explain that the model is not only looking at the latest points; it can focus on the most informative past moments.

---

## Slide 9 — Regime Encoder Part 3: Macro Graph Prior
**Title:** Graph-based structural prior over assets

**Add on the slide:**
- A graph structure is used to model relationships between assets
- The graph attention mechanism allows the model to share information across related nodes
- This gives the encoder a structural inductive bias
- It helps capture sector-wise or cross-asset dependencies

**Visual suggestion:**
- A simple graph diagram showing nodes and edges between assets

**Speaker note:**
Mention that the graph prior acts like an inductive bias, helping the model reason about inter-asset structure.

---

## Slide 10 — Differentiable Regime Encoder
**Title:** The core contribution

**Add on the slide:**
- The encoder combines VSN, LSTM, temporal attention, and graph prior
- It outputs:
  - h_market for the critic
  - h_asset for the actor
  - soft regime probabilities (Bull / Bear / Sideways)
- Regimes are learned end-to-end, not hard-coded or separately clustered

**Visual suggestion:**
- A compact diagram of the regime encoder block

**Speaker note:**
This is the most important technical slide. Emphasise that the regime signal is soft and learned jointly from the portfolio objective.

---

## Slide 11 — Actor-Critic and PPO
**Title:** Policy learning with PPO

**Add on the slide:**
- Actor network outputs portfolio weights
- Critic estimates the value of the current market state
- PPO updates the policy using clipped objective
- The training objective combines:
  - policy loss
  - value loss
  - entropy bonus

**Equation to include:**
- $L = L^{clip} + c_1 L^{value} + c_2 L^{entropy}$

**Visual suggestion:**
- A small actor-critic diagram

**Speaker note:**
Explain that the policy is improved by observing rewards from the environment and updating through PPO.

---

## Slide 12 — Reward Design
**Title:** Reward function for portfolio learning

**Add on the slide:**
- Reward combines:
  - Sharpe-like return signal
  - transaction cost penalty
  - turnover penalty
  - downside-risk penalty using EVaR
- The objective is to maximise return while controlling risk and overtrading

**Equation to include:**
- $R_t = Sharpe_t - TransactionCosts - TurnoverPenalty - EVaRPenalty$

**Visual suggestion:**
- A small formula box and a short explanation of each component

**Speaker note:**
Explain that the reward is not only about high return; it also discourages excessive trading and severe downside risk.

---

## Slide 13 — Experimental Setup
**Title:** Experimental design

**Add on the slide:**
- Model hyperparameters used
- Hardware / training environment
- Training duration and number of timesteps
- Train / validation / test split used
- Number of random seeds (if available)

**Visual suggestion:**
- A small table of hyperparameters

**Speaker note:**
Be clear about what was trained, tested, and compared.

---

## Slide 14 — Training Diagnostics
**Title:** How training progressed

**Add on the slide:**
- Training loss curves
- Value loss convergence
- Policy loss behaviour
- Entropy and KL/clip statistics if available

**Important message:**
- Keep this slide brief
- Do not spend too much time here
- Focus on whether the network learned stably

**Speaker note:**
Explain that training remained stable and the critic learned a reasonable value representation.

---

## Slide 15 — Results: Out-of-Sample Performance
**Title:** Performance against baselines

**Add on the slide:**
- A comparison table with at least these strategies:
  - Proposed model
  - Equal-weight portfolio
  - Buy-and-hold ASPI
  - Mean-variance baseline

**Suggested metrics:**
- Total return
- Annual return
- Annual volatility
- Sharpe ratio
- Max drawdown
- Turnover

**Example values to include if available:**
- Proposed model: Total return 13.93%, Annual return 42.93%, Volatility 20.42%, Sharpe 1.81, Max drawdown 10.83%

**Visual suggestion:**
- Bar chart for Sharpe and return
- Table for all key metrics

**Speaker note:**
This is one of the most important slides. Show that your method outperforms the baselines on risk-adjusted return.

---

## Slide 16 — Equity Curve and Regime Overlay
**Title:** Portfolio growth and regime behaviour

**Add on the slide:**
- Cumulative portfolio value over time
- Baseline comparison curves
- Regime probabilities under the equity curve

**Visual suggestion:**
- One main figure with two panels:
  - portfolio value curve
  - regime probability shading

**Speaker note:**
Use this to show that the model’s decisions align with changing market regimes rather than acting randomly.

---

## Slide 17 — Interpretability
**Title:** Is the model a black box?

**Add on the slide:**
- VSN feature importance results
- Attention heatmap over the window
- Optional: regime probability examples for different market phases

**Visual suggestion:**
- Feature importance chart
- Attention heatmap

**Speaker note:**
Explain that the model is not completely opaque; it highlights which features and time steps drive its decisions.

---

## Slide 18 — Limitations
**Title:** Limitations and careful interpretation

**Add on the slide:**
- Test window is relatively short
- Results may not be statistically significant over a small sample
- Turnover may be high for real-world execution on the CSE
- Single-seed results may not be robust enough

**Speaker note:**
This slide improves credibility. Show that you understand the weaknesses and are aware of how to improve the method.

---

## Slide 19 — Future Work
**Title:** What comes next?

**Add on the slide:**
- Longer walk-forward evaluation
- Multi-seed experiments
- Stronger turnover constraints
- More realistic execution costs and liquidity handling
- Potential deployment to live or paper-trading settings

**Speaker note:**
Frame this as a strong next step rather than a failure.

---

## Slide 20 — Conclusion
**Title:** Conclusion

**Add on the slide:**
- Restate the main claim
- Mention the strongest result: the model learns regimes jointly with the portfolio objective and improves risk-adjusted performance
- End with the main research contribution

**Suggested closing sentence:**
- “This work shows that regime-aware reinforcement learning can improve portfolio optimisation by learning market regimes directly from the trading objective.”

**Speaker note:**
Finish confidently and leave the audience with the core message.

---

## Extra Tips for the Presentation
- Use one consistent colour theme throughout
- Keep text short; use bullets instead of paragraphs
- Use large fonts (at least 24 pt)
- Avoid too many equations; keep only the most important one
- Reuse the architecture diagram on several slides
- Practice your speaking time so the whole presentation fits in 20 minutes
