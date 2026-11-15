# PaperRL

A local-first trading research platform in Rust: download market data from Alpaca, train PPO agents on it, backtest them against Buy & Hold and EMA crossover, and inspect the results. The trading engine, RL training, backtesting, storage, and CLI are all Rust; there is no Python or JavaScript in the application.

## Status

Phases 1 to 3 of the roadmap are built and tested (54 tests, all passing):

- Alpaca market data and paper account adapter (alpacars 0.8.0, pinned; paper API only)
- Parquet bar storage with merge-on-write sync and schema validation
- Deterministic simulator: signal at close, fill at next open, fees in basis points, slippage, no leverage
- Strategies: Buy & Hold, EMA crossover, and a trained PPO policy behind one `Strategy` trait
- Walk-forward evaluation in month-based windows with per-window and aggregate metrics
- PPO training on Burn 0.22 (flex CPU backend), seeded and reproducible, checkpoints in burnpack with JSON metadata

Not built yet: the risk engine and Barter integration, the experiment manager with redb storage, the Dioxus desktop UI, and Alpaca paper-trading order deployment.

## Quickstart

Set your Alpaca paper keys in the environment (they are only needed for data sync and account commands):

```
setx APCA_API_KEY_ID "your_paper_key"
setx APCA_API_SECRET "your_paper_secret"
```

Download, train, and compare:

```
paperrl data sync --symbols AAPL,MSFT --years 3
paperrl data show --symbol AAPL --limit 5
paperrl train --symbol AAPL --iters 50 --seed 42 --window 10
paperrl backtest --symbol AAPL --strategies buyhold,ema,ppo --model models/ppo-AAPL-42.bpk
paperrl backtest --symbol AAPL --strategies buyhold,ppo --model models/ppo-AAPL-42.bpk --window-months 6
paperrl alpaca account
```

Training and backtesting work offline from the cached Parquet datasets; only sync and account commands touch the network. The trading client is hardcoded to Alpaca's paper endpoint, so the live-money API is unreachable by construction.

## Workspace

| Crate | What it holds |
| --- | --- |
| paperrl-core | Bar, Symbol, Timeframe, shared error type |
| paperrl-alpaca | Alpaca REST adapter (data + paper account) |
| paperrl-data | BarStore: one Parquet file per (timeframe, symbol) |
| paperrl-sim | Simulator, Portfolio accounting, strategies, walk-forward |
| paperrl-rl | TradingEnv and the PPO agent |
| paperrl-cli | The `paperrl` binary |

Datasets land in `datasets/`, model checkpoints and their JSON metadata in `models/`. Both are gitignored.

## Design notes

The simulator and the RL environment share one fill-accounting implementation (`paperrl_sim::Portfolio`), so a policy's training rewards and its backtest equity agree bar for bar; a test pins this invariant. Model metadata records the symbol, timeframe, window, seed, and hyperparameters, and the backtester refuses a model trained on a different symbol. Split-adjusted bars are the default; re-sync the full stored range when widening windows so adjustment epochs stay consistent.
