# Benchmarks

Every number here comes from the deterministic simulator described in [Backtesting](backtesting.md). None of it is a claim about future returns.

## Latest run: six months versus SPY

Dates: 2026-05-14 to 2026-11-13. Universe: 27 large-cap US stocks. Costs: 1 bp fee and 5 bp slippage on every trade. PPO checkpoints: seed 42, 300 iterations, 10-bar observation window. Values are means across the 27 stocks; SPY is measured over the same dates.

| Strategy | Net return | Max drawdown | Exposure |
|---|---:|---:|---:|
| Volatility targeting | +5.49% | 10.91% | 54.9% |
| Buy & hold, 27 stocks equal weight | +5.30% | 21.07% | 99.2% |
| Mean reversion | +4.44% | 15.45% | 43.5% |
| **Risk-aware PPO** | **+2.43%** | **7.06%** | **34.1%** |
| **SPY buy & hold** | **+1.61%** | **5.86%** | ~100% |
| EMA 10/50 | +1.33% | 11.24% | 36.8% |
| Momentum | -0.83% | 15.64% | 48.4% |

Risk-aware PPO beat SPY on net return in 17 of 27 names and on return per unit of drawdown in 16 of 27. The lower exposure shows up in the worst names: NVDA's buy-and-hold lost 29.2% with a 31.9% drawdown, while the agent lost 6.2% with an 8.6% drawdown. The agent also trailed buy-and-hold on the strongest names, a consequence of averaging about a third of the exposure.

The 27-stock equal-weight row is not an index. It beat SPY on return and carried 3.6 times SPY's drawdown.

Dispersion was wide: ABBV returned +38.7%, MSFT +26.4%, INTC -45.7%, NVDA -29.2% and GOOGL -25.3%.

## Three-year comparison

Roughly 750 daily bars per stock through 2026-11-13. Same 27 stocks, costs and seed.

| Strategy | Net return | Max drawdown | Exposure |
|---|---:|---:|---:|
| Buy & hold | 78.82% | 33.89% | 99.9% |
| Volatility targeting | 40.50% | n/r | n/r |
| Risk-aware PPO | 31.44% | 15.48% | 42.6% |
| Momentum | 30.20% | n/r | n/r |
| EMA 10/50 | 28.28% | n/r | n/r |
| Mean reversion | 24.71% | n/r | n/r |

`n/r` means the metric was not recorded for that strategy in this run. Drawdown and exposure were captured only for buy-and-hold and PPO. PPO traded 26 of the 27 stocks.

This sample covers a bull-heavy period, which favors buy-and-hold. The PPO policy gave up about 47 points of mean return against buy-and-hold while taking less than half the drawdown and exposure.

## Walk-forward

Six consecutive six-month windows per stock, 27 stocks, 162 full windows per strategy. The figures average the per-stock aggregates from `experiments/walkforward-27stocks-v2.csv`.

| Strategy | Mean window return | Window stdev | Worst window | Positive windows |
|---|---:|---:|---:|---:|
| Buy & hold | +11.29% | 22.14% | -14.18% | 105 / 162 |
| Volatility targeting | +4.52% | 11.98% | -10.44% | 100 / 162 |
| Risk-aware PPO | +3.15% | 6.71% | -5.26% | 100 / 162 |
| Momentum | +3.24% | 15.09% | -14.72% | 84 / 162 |

The PPO policy produced the lowest dispersion and the mildest worst window, at about a quarter of buy-and-hold's mean return. These windows are not out-of-sample for PPO; see the caveat below.

## Seed variation

Three seeds on five stocks, three-year sample, otherwise identical settings. Values are PPO net return.

| Stock | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| NVDA | 129.00% | 193.49% | 193.49% |
| AAPL | 43.63% | 43.63% | 65.44% |
| JPM | 60.10% | 60.10% | 90.16% |
| WMT | 111.76% | 74.51% | 74.51% |
| KO | 34.71% | 34.71% | 34.71% |

Report the spread across seeds rather than one run before drawing conclusions about the policy.

## Earlier artifacts

`experiments/benchmark-27stocks-300iters.csv` is the earlier benchmark, generated before the risk-aware reward shaping. It is kept for history and is superseded by the tables above.

## Caveats

- **The PPO results are not strictly out-of-sample.** Checkpoints trained on the full three-year range, which contains both evaluation periods. Walk-forward windows for PPO are also in-sample for the same reason.
- Six months is a short, single-regime sample.
- The 27 stocks are liquid large caps chosen today, not point-in-time index members. Selection and survivorship bias apply.
- The equal-weight 27-stock series is not SPY; SPY is cap-weighted.
- Fixed fees and slippage only. No market impact, borrow costs or taxes.
- Results show what the simulator did with these inputs. They do not show that the policy generalizes.

## Data

| Artifact | Contents |
|---|---|
| `experiments/benchmark-6mo-vs-spy.csv` | per-stock six-month returns, drawdowns, exposure and PPO trades |
| `experiments/benchmark-27stocks-v2-risk-aware-ppo.csv` | per-stock three-year comparison |
| `experiments/walkforward-27stocks-v2.csv` | per-stock window aggregates |
| `experiments/seeds-5stocks-v2.csv` | seed variation |
| `scripts/benchmark_6mo_vs_spy.py` | six-month sweep used for the first table |

The methodology and reproduction commands are in [Backtesting](backtesting.md#benchmark-protocol).
