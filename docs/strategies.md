# Strategies

PaperRL uses two layers. Deterministic rule strategies live in `paperrl-sim`, and a PPO policy in `paperrl-rl` implements the same `Strategy` trait. The Signals tab combines rule votes with a PPO overlay.

## Signal votes

`paperrl_trade::recommend` scores a ticker from its daily closes. Each factor is normalized by the move it should react to, so small noise stays near zero. The weighted factors sum to a score in `[-1, 1]`.

| Factor | Normalization | Weight |
|---|---|---:|
| Price vs EMA(50) | `(price / EMA50 - 1) / 2%`, clamped to `[-1, 1]` | 0.25 |
| EMA(20) vs EMA(50) | `(EMA20 / EMA50 - 1) / 1%`, clamped | 0.20 |
| 20-day return | `ret20 / 10%`, clamped | 0.25 |
| MACD histogram | `hist / (1% of price)`, clamped | 0.15 |
| RSI(14) | overbought or oversold distance, clamped | 0.15 |
| PPO overlay | target weight from the newest checkpoint | +0.35 or -0.35 |

The verdict uses fixed thresholds: Buy at score `>= 0.3`, Sell at `<= -0.3`, Hold otherwise. Confidence is `0.5 + |score| / 2`, so it runs from 50% to 100%. The reason list includes each factor that crossed its reporting threshold, plus the holding size when a position exists.

The PPO overlay adds `+0.35` when the policy targets at least 60% equity and subtracts `0.35` when it targets 20% or less. Between those levels the model is reported as neutral but does not change the score. The reason string names the checkpoint seed.

A ticker with fewer than 60 daily bars returns Hold at 50% confidence with a not-enough-history reason, rather than guessing.

## Baselines

All baselines are long-only and run through the same simulator, with the same fees, slippage and starting cash.

| Strategy | Rule |
|---|---|
| Buy & hold | target 100% equity at the first close and hold |
| EMA 10/50 | long when EMA(10) is above EMA(50), flat below; no position during the first 50 bars |
| Momentum 10 | long while the trailing 10-bar close-to-close return is positive |
| Mean reversion 10 | long when the close is below its 10-bar simple moving average |
| Volatility targeting | weight `min(1, 15% / realized annualized volatility)` over a 10-bar window |

These are reference points for the PPO policy, not claims of profitable trading. The 27-stock results appear in [Benchmarks](benchmarks.md).

## PPO agent

The policy chooses one of five target equity weights: 0%, 25%, 50%, 75% or 100%. It cannot short or use leverage.

### Observation

The observation has 16 values with the default 10-bar window:

- the last 10 one-bar close-to-close returns;
- current position weight;
- current drawdown from peak equity;
- volatility, the population standard deviation of those 10 returns;
- volume ratio, current volume divided by the mean of the prior 10 bars;
- MACD(12, 26, 9) histogram divided by close;
- RSI(14) divided by 100 and centered on zero.

`RollingObs` produces these values for both training and live inference, so the feature set cannot drift between the two.

### Reward

For each step the environment pays:

```
log(equity_t / equity_t-1)
  - dd_penalty * increase in drawdown beyond dd_threshold
  - turnover_penalty * traded notional / prior equity
  - vol_penalty * position weight * volatility
```

| Parameter | Default |
|---|---:|
| `dd_penalty` | 0.5 |
| `dd_threshold` | 0.15 |
| `vol_penalty` | 0.01 |
| `turnover_penalty` | 0.005 |
| `fee_bps` | 1 |
| `slippage_bps` | 5 |
| `initial_cash` | 100,000 |

The penalty weights came from calibration. A diagnostic required a full-position hold on an uptrending stock to show positive mean reward and a losing stock to show negative mean reward. With the defaults, AAPL scored `+0.000282` per step and PEP scored `-0.001559` per step. Earlier defaults failed that check and the agent learned to never invest.

### Training

| Setting | Value |
|---|---|
| Implementation | PPO on Burn 0.22, flex CPU backend, autodiff |
| Network | two hidden layers of 32, shared backbone, actor and critic heads |
| Clip epsilon | 0.2 |
| Discount (`gamma`) | 0.99 |
| GAE lambda | 0.95 |
| Learning rate | 3e-4 |
| Epochs per iteration | 4 |
| Rollout length | 512 steps |
| Value and entropy coefficients | 0.5, 0.01 |
| Default iterations | 50 (300 for the published benchmark) |
| Default seed | 42 |

Each training run saves a Burn checkpoint (`.bpk`) and a JSON sidecar with the symbol, timeframe, window, environment shaping, PPO configuration and the per-iteration reward curve. The backtester loads the policy once and clones it for every evaluation, and rejects a checkpoint whose recorded symbol does not match the bar series.

### Inference

`PpoStrategy` rebuilds the rolling observation at each close and takes the highest-probability action. It returns the corresponding target weight to the shared portfolio accounting, so backtests use the same fill path as every other strategy.

## Known limits

- A checkpoint is trained per symbol and timeframe. There is no portfolio-level policy.
- Crypto pairs use the rule vote only; no PPO checkpoint has been trained on crypto bars.
- Results vary with the seed. The published spread across five stocks is recorded in [Benchmarks](benchmarks.md#seed-variation).
- The published PPO benchmark is not strictly out-of-sample because the checkpoints trained over the full range being evaluated. See [Backtesting](backtesting.md#out-of-sample-caveat).
