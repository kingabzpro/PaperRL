# Backtesting

The simulator in `paperrl-sim` is the only execution path used by research. Every strategy, including the PPO policy, emits a target equity weight at a bar's close, and the portfolio fills it at the next bar's open.

## Simulator semantics

- The target weight is clamped to `[0, 1]`. No leverage, no shorting.
- Buys cannot spend more cash than the target allocates.
- The execution price includes slippage in basis points against the trade.
- A fee in basis points is charged on traded notional.
- Rebalances below $1 of notional are skipped as dust.
- Identical inputs produce identical outputs. There is no randomness in the simulator.

Defaults are $100,000 of starting cash, 1 bp fee and 5 bp slippage on every trade.

## Metrics

| Metric | Definition |
|---|---|
| Net return | final equity divided by initial cash, minus one |
| Max drawdown | worst peak-to-trough loss of the marked-to-close equity curve |
| Sharpe | annualized from per-bar equity returns, sample standard deviation, 252 bars per year for daily data |
| Win rate | fraction of bars with a positive equity return |
| Exposure | mean position weight across bars; this is not leverage |
| Trades | number of fills |
| Turnover | absolute traded notional in dollars |
| Fees | total fees paid in dollars |

## Walk-forward

`--window-months N` splits the stored history into consecutive calendar windows of N months, measured from the first bar. Each window starts a fresh strategy with the configured starting cash, so results do not compound across windows.

The last window may be shorter than N months. It appears in the per-window table but is excluded from the aggregate mean, standard deviation, worst window and positive-window count. The standard deviation uses the sample formula with n-1.

Example:

```bash
paperrl backtest --symbol AAPL --strategies buyhold,voltarget,ppo \
  --model models/ppo-AAPL-42.bpk --window-months 6
```

## Benchmark protocol

The latest published comparison evaluates 27 large-cap US stocks over 2026-05-14 to 2026-11-13:

```
AAPL  MSFT  NVDA  AMZN  GOOGL META  TSLA  BRK.B JPM   V
UNH   XOM   JNJ   WMT   PG    MA    HD    CVX   ABBV  KO
PEP   CSCO  ORCL  AMD   NFLX  INTC  DIS
```

The run uses daily bars from Alpaca's IEX feed, a 1 bp fee, 5 bp slippage, $100,000 of starting cash per symbol, seed 42 and 300 PPO iterations per stock. SPY buy-and-hold is evaluated with the same costs over the same dates. Strategy rows are means across the 27 stocks. SPY is one index position, so its exposure is about 100% by construction.

Reproduce one symbol from a fresh checkout:

```bash
cargo build --release -p paperrl-cli
target/release/paperrl data sync --symbols AAPL --years 3
target/release/paperrl train --symbol AAPL --iters 300 --seed 42 --window 10
target/release/paperrl backtest --symbol AAPL --strategies buyhold,ema,momentum,meanrev,voltarget,ppo \
  --model models/ppo-AAPL-42.bpk --start 2026-05-14 --end 2026-11-13
```

On Windows, the binary is `target\release\paperrl.exe`. The full sweep is in `scripts/benchmark_6mo_vs_spy.py`; it hardcodes the repository root and the release binary path, so adjust `ROOT` before running it elsewhere. The per-stock output is `experiments/benchmark-6mo-vs-spy.csv`.

The three-year comparison uses the same costs over roughly 750 daily bars per symbol, ending 2026-11-13.

## Out-of-sample caveat

The PPO checkpoints trained on the full three-year range, which contains both the six-month and the three-year evaluation periods. The PPO rows are therefore **not strictly out-of-sample**. The walk-forward windows for PPO are not out-of-sample either, because the policy saw those dates during training.

The rule baselines do not train, so their results are unaffected by this specific issue. They are still in-sample in the wider sense: the universe was chosen from stocks that are liquid today, which introduces selection and survivorship bias.

## Speed

Measured on the developer's Windows machine with a release build and the CPU flex backend. Treat these as order-of-magnitude figures.

| Operation | Wall time | Notes |
|---|---:|---|
| Sync 25 symbols, 3 years of daily bars | ~7 s | one paginated request per symbol; Alpaca rate limits dominate |
| Train PPO, 300 iterations | 9.7 s | about 155k environment steps and 1,200 minibatch updates |
| Backtest one stock, six strategies, 750 bars | 32 ms | deterministic |
| Walk-forward one stock, four strategies, six windows | 30 ms | per-window metrics and aggregates |
| Full 27-stock sweep, train plus backtest | ~4.5 min | the run behind the published table |

## Limits

- Daily bars only in the published runs. The CLI accepts 1m, 5m, 15m, 1h and 1d, but the benchmark data is daily.
- Fixed basis-point costs. There is no order-book, market-impact or borrow model.
- Long-only, no leverage, no options, no forex.
- One PPO policy per symbol. The benchmark does not model portfolio-level allocation across the 27 names.
- Single-regime samples. The six-month window covers one market regime and says little about robustness across regimes.
- Alpaca's free IEX feed is a subset of consolidated volume and can differ from other vendors.

Results are research artifacts, not evidence of future returns. See [Benchmarks](benchmarks.md) for the tables and [Configuration](configuration.md) for the CLI flags.
