# PaperRL

**A local-first RL trading desk in Rust: reasoned signals, reproducible backtests, paper trades only.**

PaperRL turns a list of tickers into Buy, Sell or Hold verdicts, shows the evidence behind each one, and lets a risk-gated bot act on your watchlist through Alpaca's paper API. The learning, simulator and desktop app are one Rust workspace, with no Python or JavaScript application code.

[![Rust 2024 edition](https://img.shields.io/badge/rust-2024%20edition-DEA584?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![MIT license](https://img.shields.io/badge/license-MIT-3DA639?style=flat-square)](LICENSE)
[![123 tests passing](https://img.shields.io/badge/tests-123%20passing-08B98B?style=flat-square)](docs/architecture.md#testing)
[![Alpaca paper API](https://img.shields.io/badge/Alpaca-paper%20API-F5B94C?style=flat-square)](https://alpaca.markets)

![PaperRL Signals tab with a Buy verdict for AAPL and a Sell for BTC/USD, showing confidence, price context and model provenance where available](screenshots/signals-analyzed.png)

- [Why PaperRL?](#why-paperrl)
- [Quick start](#quick-start)
- [User guide](#user-guide)
  - [Keys and the paper account](#keys-and-the-paper-account)
  - [A first paper session](#a-first-paper-session)
  - [Research: market data, backtests, checkpoints](#research-market-data-backtests-checkpoints)
  - [Command line](#command-line)
  - [Where the data lives](#where-the-data-lives)
- [Latest benchmark](#latest-benchmark)
- [Documentation](#documentation)

## Why PaperRL?

**Verdicts you can audit.** Five factors vote on a weighted score, and an optional PPO checkpoint shifts it. The Signals tab lists every reason, the confidence, the data staleness and the seed of the model that contributed.

**One engine from training to paper.** The RL environment and the backtester share feature code and fill accounting, so a policy is evaluated on the observations it learned from. The workspace has 123 tests, including ones that pin both invariants.

**Paper-only by construction.** The Alpaca client is hardcoded to the paper endpoint. A deterministic risk gate caps position count, position size, order notional and spend, and Stop cancels only orders PaperRL placed.

**Local and reproducible.** Bars live in a Parquet cache, checkpoints save with JSON metadata, and the simulator is deterministic. The same bars and seed give the same result, offline.

**Nothing to script.** Keys, market-data sync, signals, PPO training and paper orders are all dashboard actions. A launch syncs the saved watchlist and shows verdicts without opening a terminal; the CLI stays available for CI and unattended runs.

## Quick start

Requires a recent stable Rust toolchain (2024 edition). Linux builds also need GTK3, WebKitGTK 4.1, libssl and libxdo development packages; the [release workflow](.github/workflows/release.yml) lists the exact set.

```bash
git clone https://github.com/kingabzpro/PaperRL.git
cd PaperRL

cargo run -p paperrl-desktop
```

The app opens on the Dashboard. Paste your Alpaca paper key id and secret in **Settings → Alpaca paper keys**, press **Save keys**, and it fetches bars for your watchlist and fills the Signals tab straight away. Backtests, the Strategy Lab and PPO training run offline from the local Parquet cache.

## User guide

### Keys and the paper account

PaperRL only ever calls Alpaca's paper endpoint; the live-money endpoint is not reachable from this build. Keys are resolved in a fixed order:

1. the environment: `APCA_API_KEY_ID` and `APCA_API_SECRET`
2. `<data dir>/credentials.json`, written when you press **Save keys**
3. neither: Signals, Portfolio and Automation show a prompt pointing at Settings, and the offline tabs keep working

```bash
# bash
export APCA_API_KEY_ID="your-paper-key-id"
export APCA_API_SECRET="your-paper-secret"

# PowerShell
$env:APCA_API_KEY_ID="your-paper-key-id"
$env:APCA_API_SECRET="your-paper-secret"
```

At launch the app checks the environment first, syncs daily bars for the saved watchlist and fills Signals with verdicts. The status strip above the page names the key source and points at Settings, which shows the same state as a tag. **Test connection** reads the paper account without changing anything, and **Remove saved keys** hands control back to the environment.

### A first paper session

1. **Signals** — type tickers such as `AAPL, MSFT, BTC/USD` and press **Analyze**. Each row carries a verdict, confidence, price context and a `cached` chip when the newest bar is more than two days old.
2. Press **why** on a row to see every factor's reason plus EMA, RSI, MACD and 5/20-day returns.
3. Press **Save as watchlist** so launches, the bot and the next analysis reuse the same tickers.
4. **Buy** or **Sell** opens a confirm box: enter the dollar amount and press **Confirm paper order**. Sells close the exact held quantity.
5. **Portfolio** — account totals, holdings with average entry and P&L, the equity curve with a drawdown line, and recent fills.
6. **Automation** — set an interval and press **Start bot**. It trades the saved watchlist, defers equity symbols until the market opens, and writes one line per pass to `logs/passes.jsonl`. **Stop bot** cancels only orders whose client id starts with `paperrl-`.

### Research: market data, backtests, checkpoints

**Markets** fetches history: type tickers, pick a timeframe and year span, then press **Fetch market data**. The table below lists every cached dataset; click a row to chart it.

**Strategy Lab** runs the deterministic baselines on the selected dataset, plus the symbol's PPO checkpoint when one exists. Edit cash, fees and slippage, press **Run comparison**, and set walk-forward months to get per-window aggregates.

**Experiments** trains a PPO checkpoint on a cached dataset: choose the dataset, iterations, seed and observation window, then press **Train checkpoint**. The table shows each checkpoint's seed, iterations, window and final reward, with its training curve and a one-click backtest against the baselines.

<p align="center">
  <img src="screenshots/markets.png" width="49%" alt="Markets tab with the fetch market data card and cached datasets">
  <img src="screenshots/experiments.png" width="49%" alt="Experiments tab with the train checkpoint card and PPO checkpoints">
</p>
<p align="center"><sub>Markets: fetch and chart cached datasets. Experiments: train and inspect checkpoints.</sub></p>

### Command line

Build the binary once:

```bash
cargo build --release -p paperrl-cli
```

It lands at `target/release/paperrl` on Linux and macOS, and `target\release\paperrl.exe` on Windows. Substitute `cargo run -p paperrl-cli --` for `paperrl` if you would rather not build. The CLI resolves keys exactly like the app; its directory flags default to `datasets` and `models` relative to the working directory, so run it from the repository root (or pass `--out` / `--datasets`) to reuse the app's cache.

| Command | What it does |
|---|---|
| `paperrl data sync --symbols AAPL,MSFT --years 3` | Fetch and merge bars into the Parquet cache; crypto pairs are daily only and capped at 150 days |
| `paperrl data show --symbol AAPL --limit 5` | Print the newest cached bars |
| `paperrl alpaca account` | Show the paper account summary |
| `paperrl alpaca positions` | List open paper positions |
| `paperrl alpaca bars --symbol AAPL --days 5` | Fetch bars without storing them |
| `paperrl train --symbol AAPL --iters 300 --seed 42` | Train a PPO checkpoint into `models/` with its JSON metadata |
| `paperrl backtest --symbol AAPL --strategies buyhold,voltarget,ppo --model models/ppo-AAPL-42.bpk` | Compare strategies on identical bars and costs; add `--window-months 6` for walk-forward |
| `paperrl trade scan --top 100 --train-top 3 --since 2026-01-01` | Rank the most active names by momentum and optionally train this year's leaders |
| `paperrl trade auto --top 100 --long 3` | Plan one automated pass without submitting anything |
| `paperrl trade auto --top 100 --long 3 --execute --interval 300` | Submit that plan to the paper account and repeat every five minutes |

A few worked examples:

```bash
# Cache three years of AAPL and MSFT, then inspect the newest rows
paperrl data sync --symbols AAPL,MSFT --years 3
paperrl data show --symbol AAPL --limit 3

# Train one checkpoint and evaluate it against the baselines
paperrl train --symbol AAPL --iters 300 --seed 42 --window 10
paperrl backtest --symbol AAPL --strategies buyhold,voltarget,ppo \
  --model models/ppo-AAPL-42.bpk --window-months 6

# Plan a pass, then run the same pass for real on the paper account
paperrl trade auto --top 100 --long 3
paperrl trade auto --top 100 --long 3 --execute
```

`trade auto` without `--execute` only prints the plan. With it, sells are submitted before buys, every order passes the risk gate (position count, per-position size, notional and available cash), and orders carry client ids prefixed `paperrl-`. `paperrl --help` and `paperrl <command> --help` list every flag.

### Where the data lives

The data directory is `PAPERRL_HOME` if set, otherwise a repository checkout (detected by a `datasets` folder beside the working directory), otherwise `~/paperrl`.

```
<data dir>/
├── config.json          display currency, watchlist, risk limits
├── credentials.json     optional saved keys, written by Settings
├── datasets/1day/       <SYMBOL>.parquet, one file per symbol and timeframe
├── models/              ppo-<SYMBOL>-<SEED>.bpk plus .json metadata
└── logs/
    ├── app.log          console and file log
    ├── passes.jsonl     one line per bot or manual pass
    └── orders.csv       every submitted paper order
```

Datasets, models, logs, `config.json` and `credentials.json` are gitignored.

## Latest benchmark

Trailing six months (2026-05-14 to 2026-11-13), 27 large-cap US stocks, 1 bp fee and 5 bp slippage per trade, seed 42, 300 PPO iterations per stock. Values are means across stocks.

| Strategy | Net return | Max drawdown | Exposure |
|---|---:|---:|---:|
| Volatility targeting | +5.49% | 10.91% | 54.9% |
| Buy & hold, 27 stocks equal weight | +5.30% | 21.07% | 99.2% |
| Mean reversion | +4.44% | 15.45% | 43.5% |
| **Risk-aware PPO** | **+2.43%** | **7.06%** | **34.1%** |
| **SPY buy & hold** | **+1.61%** | **5.86%** | ~100% |
| EMA 10/50 | +1.33% | 11.24% | 36.8% |
| Momentum | -0.83% | 15.64% | 48.4% |

PPO beat SPY on net return in 17 of 27 names and on return per unit of drawdown in 16, with about a third of the exposure. Treat this as a regime check, not a track record: the checkpoints trained on the full three-year range that contains this window, so the comparison is **not strictly out-of-sample**. Six months of one market also says little on its own, and the equal-weight 27-stock row is not SPY.

[Full results and methodology](docs/benchmarks.md) · [CSV](experiments/benchmark-6mo-vs-spy.csv) · [reproduction script](scripts/benchmark_6mo_vs_spy.py)

## Documentation

| Guide | What it covers |
|---|---|
| [Architecture](docs/architecture.md) | crates, data flow, desktop tabs, safety model |
| [Strategies](docs/strategies.md) | signal votes, baselines, PPO environment and reward |
| [Backtesting](docs/backtesting.md) | simulator semantics, metrics, walk-forward methodology |
| [Benchmarks](docs/benchmarks.md) | current and historical results, caveats, per-stock data |
| [Configuration](docs/configuration.md) | saved keys, config file, environment variables, CLI reference, bot behavior |

## License

MIT. See [LICENSE](LICENSE). Copyright 2026 Abid Ali Awan.

PaperRL is research software. It places orders only on Alpaca paper accounts and is not investment advice.
