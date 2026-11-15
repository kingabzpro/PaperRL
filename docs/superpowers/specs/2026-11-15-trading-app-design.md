# PaperRL trading app design

Date: 2026-11-15
Status: approved by conversation, pending build

## Intent

PaperRL has been an experiment workbench: a desktop dashboard that reads local parquet files, a paper-trading pass triggered by button or CLI. This change turns it into a personal trading app. The user opens the app, types tickers (stocks and crypto), gets a Buy / Sell / Hold verdict per ticker with the reasoning shown, manages their paper portfolio in a real tab, and runs the trading bot from inside the app on a schedule. All trading stays on Alpaca paper.

Why now: the dashboard already exists and the momentum screen, risk gate, and PPO models work end to end. The missing pieces are per-ticker recommendations, portfolio visibility, a managed scheduler, and packaging.

## Decisions already made (with the user)

- Currency scope: crypto assets (BTC/USD style pairs) become first-class, plus a base-currency setting that converts display values through Alpaca FX rates. Alpaca cannot trade forex, so no forex pairs.
- Automation runs in-app: a scheduler owned by the desktop process, started and stopped from the Automation tab. No daemon.
- Ticker data comes live from Alpaca market data and is cached to parquet. The user no longer needs to pre-sync datasets.
- Packaging is a portable exe that resolves a user-chosen data directory. No installer.
- The scheduled bot trades the user's saved watchlist, stocks and crypto. Equities wait for market hours, crypto entries run any time. The screener path stays available through the CLI.
- Signal verdicts are actionable: each row has Buy and Sell buttons with a notional input and a confirm step that places a paper market order.
- Stack stays: Rust workspace, alpacars pinned =0.8.0 (isolated behind paperrl-alpaca), Dioxus 0.7 desktop, Burn PPO for model overlays.

## Builder-check calls (settled defaults)

- Stop cancels only bot-owned orders: those whose client order id starts with "paperrl-". Orders placed outside PaperRL are never touched.
- When several PPO checkpoints exist for a symbol, the newest by file modification time biases the verdict, and the reason string shows its seed.
- After a relaunch the bot always starts Idle. Nothing trades until Start is pressed in that session.
- If the FX rates call fails, the app keeps the last good rate for the session; with no rate at all it displays USD with an "FX unavailable" note.
- Data directory resolution is shared by the desktop app and the CLI: if a datasets folder exists next to the working directory it is used (repo checkout), otherwise $HOME/paperrl is created. PAPERRL_HOME overrides both.
- The market-hours gate applies to equity symbols only; crypto symbols pass at any hour.

## Goals

1. Signals: type one or many tickers, get a verdict table. Each row shows action (Buy / Sell / Hold), confidence, price, day change, and the indicator state behind it. An expandable row shows the full reasoning. A trained PPO model biases the verdict when one exists, labeled with its provenance.
2. Portfolio: holdings table (qty, avg entry, current price, market value, unrealized P&L and %, weight), allocation view, equity curve from Alpaca portfolio history, recent fills, currency exposure line.
3. Automation: start/stop the bot from the tab, configurable interval and risk limits, market-hours gate (skip passes while the equity market is closed), pass history persisted locally, stop cancels open orders.
4. Crypto: parse and trade BTC/USD pairs on paper, fetch crypto bars through CryptoHistoricalDataClient, cache alongside stock bars.
5. Base currency: a setting (USD, EUR, GBP, ...), converts every money display via Alpaca FX rates, exposure line splits USD equities from crypto.
6. Production base: portable exe with a data directory resolved at first launch, config file, file logging, tolerant error paths replaced with surfaced errors, tests for new logic, README and screenshots updated.

## Non-goals

- Live trading endpoint. The paper flag stays hardcoded true.
- Forex trading, shorting, options, bracket orders.
- Websockets / streaming. Polling at 30s or slower is enough for this app.
- Alerts and notifications.
- Multi-monitor layouts, command palette, cross-broker import.

## How it fits together

Data flow for a signal: user types "AAPL, BTC/USD, TSLA" and presses Analyze. For each symbol: parse (crypto-aware) then ensure bars (parquet cache; live fetch from the stock or crypto feed when missing or stale) then compute indicators then check models dir for a PPO checkpoint then emit a Recommendation. The table renders it.

The bot loop: a tokio task spawned by the desktop app, owned through a handle. Each tick checks the Alpaca clock for equity symbols and skips them while the market is closed (crypto symbols are not gated), then plans and executes against the saved watchlist with UI-configured limits. run_once gains a universe parameter so the watchlist replaces the screener when supplied. A watch channel carries state (Idle / Running / Stopping) so the tab renders live status. Stop sets the channel, cancels bot-owned open orders, and joins the task. Pass results append to a JSONL history file next to the config.

Config: one JSON file in the data directory (base currency, risk limits, bot interval, watchlist of user tickers, data dir paths). Loaded at startup, written on change. Env vars still supply API keys only.

## Changes per crate

paperrl-core
- AssetClass enum (UsEquity, Crypto).
- Symbol accepts crypto pairs: letters, then /, then USD, USDC, USDT, or BTC as quote. Max length raised for pairs like BTC/USD. Existing equity rules unchanged (1-6 letters, . and - allowed).
- file_slug() for filesystem-safe names (BTC/USD becomes BTC-USD) used by parquet cache and model lookup.
- CurrencyCode type and money formatting helper with per-code symbols.

paperrl-alpaca (all new calls go through the pinned SDK, no new dependencies)
- assets(): get_all_assets filtered to active and tradable, for symbol validation and asset class detection.
- crypto_bars(): CryptoHistoricalDataClient::get_crypto_bars mapped to the shared Bar type.
- portfolio_history(period): SDK get_portfolio_history, returned as an equity curve view.
- activities(): FILL activities, paginated once (first page of 100).
- clock(): SDK get_clock mapped to a small view.
- cancel_all_open_orders(): SDK cancel_orders.
- Sell-side orders: remove the hardcoded OrderSide::Buy in submit_market_notional_order and pass the side through.
- fx_rate(base): RestClient::get on the v1beta1 forex latest rates path, returns rate for base/USD.
- The SDK's RestClient already retries 429 and 504 three times with a 3s wait, so no extra rate limiter is built.

paperrl-trade
- Recommendation struct: action (Buy/Sell/Hold), confidence 0-1, indicators (ema20, ema50, rsi14, macd_hist, ret5, ret20, score), reasons (Vec of short strings), model provenance when a PPO checkpoint contributed.
- recommend(symbol, bars, model_weight, held_qty): rule votes from trend (close vs EMA50, EMA20 vs EMA50), momentum (existing momentum score), RSI extremes (>70 sell bias, <30 buy bias), MACD histogram sign. Weighted vote maps to action and confidence. PPO weight above 0.6 pushes toward Buy, below 0.2 toward Sell, and is reported as a reason with its seed. A held position makes Sell mean exit sizing in the plan, and adds position P&L to reasons.
- BotHandle: spawn(interval, cfg, client, paths) returning a handle with state(), stop(), and force_pass(). Market-hours gate before each pass. Pass history JSONL in the data dir.
- AutoConfig gains interval and enabled fields so the UI and the config file drive the same struct.

paperrl-desktop
- Data directory: PAPERRL_HOME env override, else $USERPROFILE/paperrl (or $HOME/paperrl). First-launch creates it. Settings tab shows and can change it.
- Tabs: Dashboard (account card with 30s refresh, market clock banner, top three signals summary), Signals (new), Portfolio (new), Automation (upgraded), Markets, Lab, Experiments unchanged, Settings (expanded).
- Signals tab: text input accepting comma or space separated tickers, saved watchlist from config, verdict table with expandable reasons, per-symbol crypto support, refresh button, stale-data indicator per row, and Buy/Sell buttons per row that open a confirm dialog with a notional input and place a paper market order through the existing submission path.
- Portfolio tab: holdings table, SVG allocation bars per asset class, SVG line chart for equity curve (reuse the hand-rolled SVG approach from CandleChart), recent fills table, currency exposure row.
- Automation tab: run state from the handle, interval and risk limit inputs wired to config, pass history table from JSONL, market-closed notice, dry-run toggle kept.
- Currency: all money display goes through one formatter that converts from USD to the base currency with the cached FX rate when base is not USD.
- Logging: tracing subscriber writing to the data dir logs file plus console.
- Lab page: validate SimParams inputs before constructing Portfolio so bad numbers surface as UI errors instead of asserts.

## Production details

- Silent failure removal: position and account parses in the alpaca crate stop using unwrap_or(0.0) for critical fields and log at warn with the raw string when parsing fails; the UI shows an error state rather than zeros for the account card when the API call fails.
- Order log write failures stop being swallowed; they append to the pass history entry and show in the Automation tab.
- BarStore stays single-writer by convention; the bot and manual refresh route through one helper so concurrent desktop + CLI use is a documented limitation, marked with a ponytail comment.
- Tests: Symbol crypto parsing and file_slug, recommend verdicts (uptrend, downtrend, chop), PPO overlay provenance, bot state machine including market-closed skip, FX conversion math, config round-trip, view mapping for new alpaca endpoints against captured JSON fixtures.
- Verification before completion: cargo test across the workspace, cargo clippy, then the app is launched and the Signals, Portfolio, Automation, and Settings flows are exercised in the real window with screenshots.

## Risks and calls

- Crypto PPO models do not exist yet; crypto signals are rules-only until a model is trained on crypto bars. The rules path carries it; no new work is added for crypto models.
- The free IEX feed delays and caps real-time data. Signals run on daily bars, so this does not matter for verdicts; it matters for the portfolio tab's current price, which Alpaca supplies per position anyway.
- FX rates only affect display. Cost basis stays in USD because Alpaca settles in USD. The portfolio tab labels values as "in base currency" when base is not USD.
- The SDK's crypto quote currency set (USD, USDC, USDT, BTC) is allowed in the parser; anything else is rejected with a clear error, and the assets() call validates tradability for the account.
