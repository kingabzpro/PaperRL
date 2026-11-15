# PaperRL Trading App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task inline, or dispatch tasks to subagents with a review gate per task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Dioxus dashboard into a personal trading app: multi-ticker Buy/Sell/Hold signals with reasoning and actionable paper orders, a real portfolio tab, an in-app scheduled bot trading the user's watchlist, crypto support, base-currency display, and a portable-exe production base.

**Architecture:** Bottom-up across the existing 8-crate workspace. paperrl-core gains crypto-aware symbols and currency helpers; paperrl-alpaca exposes the rest of the pinned alpacars SDK (clock, portfolio history, fills, assets, crypto bars, FX, cancel, sell orders); paperrl-trade gains a recommendation engine, a tokio bot loop with market-hours gate, and the app config; paperrl-desktop gains Signals and Portfolio tabs and owns the bot handle. No new external dependencies.

**Tech Stack:** Rust 2024 workspace, alpacars =0.8.0 (isolated in paperrl-alpaca), Dioxus 0.7 desktop, Burn 0.22 flex (untouched), tokio (workspace dep, add sync/time features), serde_json.

**Spec:** docs/superpowers/specs/2026-11-15-trading-app-design.md

## Global Constraints

- alpacars stays pinned `=0.8.0` and is depended on only by paperrl-alpaca.
- No new external crates except enabling tokio features `sync` and `time` in the workspace.
- Trading client stays paper-only (`paper: true` hardcoded).
- No em dashes anywhere (code, comments, docs, commit messages); use " - " or commas.
- Money rendering goes through one formatter; internal money is always USD; conversion happens at display time.
- Existing 62 tests must keep passing; every task leaves `cargo test --workspace` green before its commit.
- Deliberate shortcuts get a `ponytail:` comment naming the ceiling.
- UI keeps the existing emoji-led styling conventions of `crates/paperrl-desktop/src/theme.rs`.

## Review Focus

Failure modes a real user hits that no task's tests fully pin; each is owned by the task named.

1. A typed ticker that fails parsing or is untradable (lowercase "btc", "BTC/EUR", "AAPLX!") must show a per-row error message, never crash or silently vanish. Owner: Task 7 (UI renders error rows; parse rules tested in Task 1).
2. Missing API keys must leave the app usable: pages render, actions show a setup hint, no panic on any tab. Owner: Task 10 (launch with unset keys during live verification).
3. Bot started while the equity market is closed must skip equity symbols, still enter crypto, and record the skip in pass history. Owner: Task 5 (test_should_run) plus live check in Task 10.
4. Stop mid-pass must end the task cleanly and cancel only orders whose client order id starts with "paperrl-". Owner: Task 2 (cancel filter test) and Task 5 (loop wiring).
5. FX rate call failure must fall back to USD display with a note, never show converted-looking zeros. Owner: Task 6 (fallback test), wired in Task 7/8.

---

### Task 1: paperrl-core crypto symbols and currency helpers

**Files:**
- Modify: `crates/paperrl-core/src/lib.rs`
- Test: unit tests in the same file (existing convention)

**Interfaces:**
- Produces: `pub enum AssetClass { UsEquity, Crypto }`; `Symbol::asset_class(&self) -> AssetClass`; `Symbol::file_slug(&self) -> String`; `pub fn normalize_currency(code: &str) -> Option<&'static str>`; `pub fn fmt_money(amount: f64, currency: &str) -> String`.

- [ ] **Step 1: Write the failing tests**

```rust
// crypto pairs: uppercase-normalized, letters before "/", quote in USD|USDC|USDT|BTC
#[test] fn parse_crypto_pair_ok() { let s = Symbol::parse("btc/usd").unwrap(); assert_eq!(s.as_str(), "BTC/USD"); assert!(matches!(s.asset_class(), AssetClass::Crypto)); }
#[test] fn parse_crypto_bad_quote_rejected() { assert!(Symbol::parse("BTC/EUR").is_err()); }
#[test] fn parse_crypto_base_length() { assert!(Symbol::parse("PEPEUSD").is_ok()); assert!(Symbol::parse("LONGCOIN/USD").is_err()); }
#[test] fn equity_rules_unchanged() { assert!(Symbol::parse("AAPL").is_ok()); assert!(Symbol::parse("BRK.B").is_ok()); assert!(Symbol::parse("BTC/USD").is_ok()); assert!(Symbol::parse("123456").is_err()); }
#[test] fn file_slug_is_fs_safe() { assert_eq!(Symbol::parse("BTC/USD").unwrap().file_slug(), "BTC-USD"); assert_eq!(Symbol::parse("AAPL").unwrap().file_slug(), "AAPL"); }
#[test] fn currency_normalize() { assert_eq!(normalize_currency("eur"), Some("EUR")); assert_eq!(normalize_currency("XYZ"), None); }
#[test] fn money_formatting() { assert_eq!(fmt_money(1234.5, "USD"), "$1,234.50"); assert_eq!(fmt_money(1234.5, "EUR"), "\u{20ac}1,234.50"); assert_eq!(fmt_money(1234.5, "XYZ"), "XYZ 1,234.50"); }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p paperrl-core`
Expected: FAIL (methods and functions not defined)

- [ ] **Step 3: Implement**

Extend the existing `Symbol` validation: after uppercasing, if the string contains `/`, require 2-8 letters before it and one of USD|USDC|USDT|BTC after; otherwise keep the current 1-6 letter + `.`/`-` equity rule. `asset_class()` returns Crypto iff the canonical string contains `/`. `file_slug()` replaces `/` with `-`. `normalize_currency` uppercases and matches against a const list (USD, EUR, GBP, JPY, CAD, AUD, CHF, SEK, NOK, DKK, NZD, HKD, SGD, MXN, PLN, CZK, HUF, TRY). `fmt_money` groups thousands and picks a symbol per code (`$` default, `€`, `£`, `¥`, `C$`, `A$`, `Fr`, `kr`, `NZ$`, `HK$`, `S$`, `Mex$`, `zł`, `Kč`, `Ft`, `₺`), unknown code prefixes the code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p paperrl-core && cargo test --workspace`
Expected: PASS (workspace green confirms no caller broke)

- [ ] **Step 5: Commit**

```bash
git add crates/paperrl-core && git commit -m "feat: crypto symbols and currency helpers in core"
```

---

### Task 2: paperrl-alpaca expanded API surface

**Files:**
- Modify: `crates/paperrl-alpaca/src/lib.rs`
- Modify callers of the changed order signature: `crates/paperrl-trade/src/lib.rs`, `crates/paperrl-cli/src/main.rs`

**Interfaces:**
- Consumes: `Symbol`, `AssetClass`, `Bar` from paperrl-core; `BarStore` from paperrl-data is NOT touched here.
- Produces: `pub enum Side { Buy, Sell }`; `AlpacaClient::submit_market_notional_order(&self, symbol: &str, notional: f64, side: Side)` (breaking change, all callers updated in this task); `PositionView` gains `avg_entry_price: f64, cost_basis: f64, current_price: f64, unrealized_plpc: f64, asset_class: String`; new views `ClockView { timestamp, is_open, next_open, next_close }`, `EquityPoint { timestamp, equity, profit_loss, profit_loss_pct }`, `FillView { id, symbol, side, qty, price, time }`, `AssetView { symbol, asset_class, tradable, active }`; methods `clock() -> ClockView`, `portfolio_history(period: &str) -> Vec<EquityPoint>`, `fills(limit: u32) -> Vec<FillView>`, `assets() -> Vec<AssetView>`, `crypto_bars(&Symbol, days: u32) -> Vec<Bar>` (daily bars, `CryptoFeed::Us`), `fx_rate(base: &str) -> f64` (units of base per 1 USD, via `RestClient::get` on `/v1beta1/forex/latest/rates`), `cancel_bot_orders(prefix: &str) -> usize` (cancels open orders whose client_order_id starts with prefix).

- [ ] **Step 1: Write the failing tests** (mapping functions are pure; test them, not the network)

```rust
// build an alpacars Position with String fields "150.25", cost_basis "1502.50", asset_class Crypto; map through the existing positions() mapper; assert f64 fields and asset_class "crypto"
#[test] fn position_view_maps_full_fields() { /* ... */ }
// Side::Sell must land as OrderSide::Sell in the OrderRequest the builder produces
#[test] fn notional_order_carries_side() { /* ... */ }
// fx JSON body {"base":"EUR","rates":{"USD":1.0865}} maps to 1/1.0865 units of EUR per USD
#[test] fn fx_rate_inverts_usd_pair() { /* ... */ }
// clock view from SDK Clock
#[test] fn clock_view_maps() { /* ... */ }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p paperrl-alpaca`
Expected: FAIL (types and methods not defined)

- [ ] **Step 3: Implement**

Thin wrappers over the pinned SDK as listed. `cancel_bot_orders` iterates `open_orders()`, filters by client_order_id prefix, calls `cancel_order_by_id` per match, returns the count. Crypto parse of `String` amount fields keeps the existing tolerant pattern but logs at warn on parse failure instead of silently returning 0. `fx_rate` builds a dedicated `RestClient` with the data base URL and version `v1beta1`, deserializes `{"base": String, "rates": HashMap<String, f64>}`, returns `1.0 / rates["USD"]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/paperrl-alpaca crates/paperrl-trade crates/paperrl-cli && git commit -m "feat: alpaca surface for portfolio, crypto, clock, fills, fx, sell orders"
```

---

### Task 3: shared data directory resolution

**Files:**
- Modify: `crates/paperrl-data/src/lib.rs`

**Interfaces:**
- Produces: `pub fn paperrl_home_in(cwd: &Path) -> PathBuf` (pure) and `pub fn paperrl_home() -> PathBuf` (env then cwd). Order: `PAPERRL_HOME` env var; else if `<cwd>/datasets` exists, cwd; else `$HOME/paperrl` (Windows: `%USERPROFILE%/paperrl`), created on demand by callers.

- [ ] **Step 1: Write the failing tests**

```rust
#[test] fn env_var_wins() { /* set PAPERRL_HOME to a tmp path (guard with a mutex, restore after), call paperrl_home_in(tmp_cwd), assert equals env path */ }
#[test] fn cwd_with_datasets_wins_over_home() { /* tmp dir with datasets/ inside; assert paperrl_home_in(that dir) == that dir */ }
#[test] fn falls_back_to_home_dir() { /* empty tmp cwd, env unset; assert result == std::env::home_dir()/paperrl */ }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p paperrl-data`
Expected: FAIL

- [ ] **Step 3: Implement** (both functions; the env-backed one delegates to the pure one)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p paperrl-data`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/paperrl-data && git commit -m "feat: shared paperrl home resolution"
```

---

### Task 4: recommendation engine in paperrl-trade

**Files:**
- Modify: `crates/paperrl-trade/src/lib.rs`
- Modify: `crates/paperrl-trade/Cargo.toml` (no new deps expected; add tokio in Task 5 only)

**Interfaces:**
- Consumes: `Bar`, `Symbol`, `AssetClass` from core; `momentum` and `ScreenFilter` already in this crate.
- Produces: `pub enum Action { Buy, Sell, Hold }`; `pub struct Indicators { price, change_pct, ema20, ema50, rsi14, macd_hist, ret5, ret20: f64 }`; `pub struct ModelProvenance { seed: u64, weight: f64 }`; `pub struct Recommendation { symbol: Symbol, action: Action, confidence: f64, indicators: Indicators, reasons: Vec<String>, model: Option<ModelProvenance> }`; `pub fn recommend(symbol: &Symbol, bars: &[Bar], model_weight: Option<f64>, held_qty: f64) -> Recommendation`. Internal helpers (private, still tested through the module): `ema(closes: &[f64], n: usize) -> Vec<f64>`, `rsi(closes: &[f64], n: usize) -> f64`, `macd_hist(closes: &[f64]) -> f64`. Also: `ensure_bars` learns crypto routing (calls `client.crypto_bars` when `symbol.asset_class() == Crypto`, cache filename via `file_slug()`), and `latest_model_weight` picks the newest `ppo-<slug>-*.bpk` by mtime and returns its seed.

- [ ] **Step 1: Write the failing tests**

```rust
// bar builders: fn rising(n) / falling(n) / flat(n) produce daily Bars with drifting closes
#[test] fn ema_trends() { assert!(ema(&rising(60), 20).last().unwrap() > ema(&rising(60), 50).last().unwrap()); }
#[test] fn rsi_bounds_and_direction() { rsi(&rising(30),14) > 70.0; rsi(&falling(30),14) < 30.0; }
#[test] fn macd_hist_sign_matches_trend() { macd_hist(&rising(60)) > 0.0; macd_hist(&falling(60)) < 0.0; }
#[test] fn recommend_buy_on_uptrend() { let r = recommend(&"AAPL".parse().unwrap(), &rising(120), None, 0.0); assert!(matches!(r.action, Action::Buy)); assert!(r.confidence > 0.5); assert!(!r.reasons.is_empty()); }
#[test] fn recommend_sell_on_downtrend() { /* falling(120), assert Sell */ }
#[test] fn recommend_hold_on_chop() { /* alternating closes, assert Hold */ }
#[test] fn ppo_weight_biases_verdict() { /* flat(120) + Some(0.9) -> Buy with model reason containing seed; flat + Some(0.05) -> Sell */ }
#[test] fn held_position_exit_reason() { /* falling(120), held_qty 2.0 -> Sell and a reason mentions exiting the position */ }
#[test] fn needs_minimum_history() { /* 10 bars -> Hold with a "not enough history" reason, no panic */ }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p paperrl-trade`
Expected: FAIL

- [ ] **Step 3: Implement**

Indicator helpers first (EMA is the standard smoothing over closes; RSI is Wilder's; MACD hist = EMA12 - EMA26 minus its own 9-EMA signal). `recommend` computes all indicators from the last 120 closes, builds votes: trend (price > ema50, ema20 > ema50), momentum (reuse `momentum()` sign thresholds), RSI bands, MACD sign; weights trend and momentum highest. Score maps to Action with confidence = |score| normalized to 0.5..1.0. Reasons are short human strings naming each contributing factor with its value. PPO weight >= 0.6 adds a Buy vote, <= 0.2 a Sell vote, with the model reason; held_qty > 0 adds the unrealized context line and makes Sell reasons say exit. Fewer than 60 bars returns Hold with a history reason.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/paperrl-trade && git commit -m "feat: per-ticker buy/sell/hold recommendation engine"
```

---

### Task 5: bot scheduler, watchlist universe, and app config

**Files:**
- Modify: `crates/paperrl-trade/src/lib.rs`
- Modify: `crates/paperrl-trade/Cargo.toml` (add tokio with `sync`, `time`, `macros`, `rt` from workspace deps)

**Interfaces:**
- Consumes: everything from Tasks 1-4; `AutoConfig`.
- Produces: `AutoConfig` gains `pub universe: Option<Vec<Symbol>>` (Some skips `collect_universe`; crypto symbols bypass the market-hours gate inside the pass) and `pub interval_secs: u64`; `run_once` plans sells for watchlist symbols that the recommendation engine says exit (reuse `plan_orders` unchanged; the universe input is the change). New: `pub enum BotState { Idle, Running, Stopping }`; `pub struct BotHandle` with `state() -> BotState`, `async stop()`, `async join()`, `last_pass_at() -> Option<std::time::Instant>`; `pub fn spawn_bot(client: AlpacaClient, cfg: AutoConfig, home: &Path, interval: std::time::Duration) -> BotHandle` (tokio task: loop sleeps interval, fetches `clock()` when any equity symbol is in the universe, skips the pass if closed and logs the reason, else runs `run_once`, appends a pass entry, and exits on the stop signal after `cancel_bot_orders(client, "paperrl-")`); `pub fn should_run(asset: AssetClass, market_open: bool) -> bool`; pass history appended by `pub fn append_pass_entry(home: &Path, entry: &PassEntry) -> std::io::Result<()>` writing one JSON line to `<home>/logs/passes.jsonl`; `pub struct PassEntry { time: String, mode: String, planned: usize, executed: usize, skipped: Option<String>, errors: Vec<String> }` (serde Serialize+Deserialize). New in this crate: `pub struct AppConfig { base_currency: String, interval_secs: u64, watchlist: Vec<String>, risk: RiskLimits }` with `load(home) -> Result<Self>` (missing file = defaults), `save(home) -> Result<()>`, `Default` impl (USD, 3600, empty watchlist, RiskLimits defaults), all serde.

- [ ] **Step 1: Write the failing tests**

```rust
#[test] fn should_run_respects_market_hours() { assert!(!should_run(AssetClass::UsEquity, false)); assert!(should_run(AssetClass::UsEquity, true)); assert!(should_run(AssetClass::Crypto, false)); }
#[test] fn config_roundtrip() { /* tmp home: AppConfig { base_currency "EUR", watchlist vec, risk tweaked } -> save -> load -> assert equal */ }
#[test] fn config_missing_file_gives_defaults() { /* tmp home with no file -> load -> Default::default() */ }
#[test] fn pass_entry_jsonl_roundtrip() { /* append two entries to tmp home, read file, deserialize both */ }
#[test] fn universe_override_skips_screener() { /* pure-planning test: rank_filtered over bars supplied for exactly the universe symbols [AAPL, MSFT], then plan_orders from those candidates; intents reference only universe symbols and no collect_universe call happens (it stays network-only and is not invoked) */ }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p paperrl-trade`
Expected: FAIL

- [ ] **Step 3: Implement**

State carried in `tokio::sync::watch`; stop flips to Stopping and the loop breaks at the next boundary, cancelling through `cancel_bot_orders(client, "paperrl-")` before exiting. The loop is thin plumbing around `run_once`; the gate and history are the tested units. Mark the loop as `// ponytail: no pause state, stop+start covers it; add pause if users ask`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/paperrl-trade && git commit -m "feat: watchlist bot scheduler with market-hours gate and app config"
```

---

### Task 6: desktop display services (currency, watchlist parsing)

**Files:**
- Modify: `crates/paperrl-desktop/src/lib.rs`

**Interfaces:**
- Consumes: `fmt_money`, `normalize_currency` from core; `AppConfig` from trade.
- Produces: `pub fn fmt_money_display(amount_usd: f64, base: &str, fx: Option<f64>) -> String` (fx = units of base per USD; None or base == USD renders plain USD); `pub fn parse_watchlist_input(s: &str) -> Vec<String>` (split on commas, spaces, and newlines; uppercase; dedupe; drop empties); `pub fn load_config_with_fx(home: &Path) -> (AppConfig, Option<f64>)` used at startup.

- [ ] **Step 1: Write the failing tests**

```rust
#[test] fn display_converts_with_fx() { assert_eq!(fmt_money_display(100.0, "EUR", Some(0.9)), "\u{20ac}90.00"); }
#[test] fn display_usd_ignores_fx() { assert_eq!(fmt_money_display(100.0, "USD", Some(0.9)), "$100.00"); }
#[test] fn display_falls_back_to_usd_without_fx() { assert_eq!(fmt_money_display(100.0, "EUR", None), "$100.00"); }
#[test] fn watchlist_input_parses() { assert_eq!(parse_watchlist_input("aapl, btc/usd  TSLA\nmsft"), vec!["AAPL", "BTC/USD", "TSLA", "MSFT"]); }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p paperrl-desktop`
Expected: FAIL

- [ ] **Step 3: Implement** (thin wrappers; conversion multiplies by fx and delegates to `fmt_money`)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p paperrl-desktop`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/paperrl-desktop && git commit -m "feat: currency display and watchlist input helpers"
```

---

### Task 7: Signals tab (multi-ticker verdicts and manual paper orders)

**Files:**
- Modify: `crates/paperrl-desktop/src/main.rs` (new `SignalsPage`, nav entry, Page enum)
- Modify: `crates/paperrl-desktop/src/theme.rs` (badge and expandable-row styles)
- Modify: `crates/paperrl-desktop/src/lib.rs` (async helpers the page calls)

**Interfaces:**
- Consumes: `recommend`, `ensure_bars`, `latest_model_weight`, `AlpacaClient` order methods, `fmt_money_display`, `parse_watchlist_input`, config watchlist.
- Produces: rendering only; the page persists its ticker list back to `AppConfig.watchlist` on change.

- [ ] **Step 1: Implement the page**

Input row (text field + 📊 Analyze button + 💾 Save to watchlist). Rows render per ticker: emoji badge (🟢 Buy / 🔴 Sell / 🟡 Hold), confidence, price with day change, indicators summary, provenance chip when a PPO model contributed ("🤖 PPO seed 42 w=0.75"), stale-data chip when bars came from cache older than 2 days, expandable "Why" section listing reasons verbatim, and Buy / Sell buttons opening a confirm dialog with a notional input (default = min(20% equity, max order) from config risk) that calls `submit_market_notional_order` and shows the submitted order or the error. Rows that failed parse or fetch show the error inline in red and never panic.

- [ ] **Step 2: Build**

Run: `cargo build -p paperrl-desktop`
Expected: compiles clean, no warnings from the new code

- [ ] **Step 3: Unit-check the pure parts**

Run: `cargo test -p paperrl-desktop`
Expected: PASS (existing + Task 6 tests)

- [ ] **Step 4: Commit**

```bash
git add crates/paperrl-desktop && git commit -m "feat: signals tab with multi-ticker verdicts and manual paper orders"
```

---

### Task 8: Portfolio tab (holdings, allocation, equity curve, fills)

**Files:**
- Modify: `crates/paperrl-desktop/src/main.rs` (new `PortfolioPage`, nav entry)
- Modify: `crates/paperrl-desktop/src/theme.rs` (styles)

**Interfaces:**
- Consumes: `positions()` (extended view), `portfolio_history()`, `fills()`, `account()`, `fx_rate()`, `fmt_money_display`, the SVG chart pattern from `CandleChart`.

- [ ] **Step 1: Implement the page**

Holdings table: 📎 symbol, qty, avg entry, current price, market value, unrealized P&L (colored), P&L %, weight % (computed against total equity). Allocation block: two SVG bars, USD equities vs crypto, with the currency-exposure label. Equity curve: SVG line over `EquityPoint`s with min/max axis labels and drawdown shading between running max and equity (reuse the SVG helper style; a polyline + one path is enough). Recent fills table (last 20). Period selector for the curve (1W / 1M / 3M / 1A buttons). All money through `fmt_money_display`; when FX is unavailable and base is not USD, a single "FX unavailable, showing USD" note renders once instead of per-row.

- [ ] **Step 2: Build**

Run: `cargo build -p paperrl-desktop`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add crates/paperrl-desktop && git commit -m "feat: portfolio tab with holdings, allocation, equity curve, fills"
```

---

### Task 9: Automation tab upgrade, Dashboard refresh, Settings

**Files:**
- Modify: `crates/paperrl-desktop/src/main.rs` (`AutomationPage`, `DashboardPage`, `SettingsPage`, App state)
- Modify: `crates/paperrl-desktop/src/lib.rs`

**Interfaces:**
- Consumes: `BotHandle`, `BotState`, `PassEntry` history reader, `AppConfig` save, `clock()` for the market banner, `recommend` for dashboard top signals.

- [ ] **Step 1: Implement**

Automation: ▶️ Start / ⏹ Stop buttons bound to the `BotHandle` (owned in a `use_signal<Option<BotHandle>>`), state chip, interval input (seconds, min 60), risk-limit inputs (max positions, position pct, max order) wired to `AutoConfig` built from `AppConfig`, pass history table read from `passes.jsonl` (newest first, capped at 50), market-closed notice from `clock()`. Dashboard: market clock banner (🟢 Open / 🔴 Closed, next open), account card refreshes every 30s while the bot runs (single `spawn` + `async_std`-style interval via tokio, replacing the fetch-once effect), top three watchlist signals row. Settings: base currency dropdown (from `normalize_currency` list), data dir display from `paperrl_home()`, credentials status line, config save button, and the existing env-var help.

- [ ] **Step 2: Build and run existing tests**

Run: `cargo build -p paperrl-desktop && cargo test --workspace`
Expected: clean and green

- [ ] **Step 3: Commit**

```bash
git add crates/paperrl-desktop && git commit -m "feat: in-app bot controls, live dashboard, settings with base currency"
```

---

### Task 10: hardening, logging, portable build, docs, live verification

**Files:**
- Modify: `crates/paperrl-desktop/src/main.rs` (tracing init, error states)
- Modify: `crates/paperrl-cli/src/main.rs` (use `paperrl_home()` as default `--datasets`/`--model-dir`)
- Modify: `README.md`, screenshots
- Create: release build artifact check

**Interfaces:**
- Consumes: everything built so far.

- [ ] **Step 1: Logging and error surfacing**

Initialize `tracing_subscriber` in desktop `main()` writing to console and `<home>/logs/app.log` (env filter from `RUST_LOG`, default `info`). Account card and portfolio page render an error state with the message instead of zeros on API failure. Lab page validates `SimParams` numerics before constructing `Portfolio`, showing the validation error in the card. Grep the workspace for remaining `unwrap_or(0.0)` on money paths and surface or log each.

- [ ] **Step 2: Full test + clippy gate**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets`
Expected: green, no new clippy warnings

- [ ] **Step 3: Portable release build**

Run: `cargo build --release -p paperrl-desktop`
Expected: `target/release/paperrl-desktop.exe` exists. Launch it once from a non-repo working directory (no `datasets/` nearby) and confirm it creates `$HOME/paperrl`, shows Settings with that path, and starts with an empty watchlist and no panic (keys unset path included: tabs render, actions show the setup hint).

- [ ] **Step 4: Live UI verification**

Dispatch the live-tester subagent: run the app with real paper keys, exercise Signals (multi-ticker input incl. one crypto pair, expand reasons, place one confirmed paper order, one bad ticker error row), Portfolio (holdings, equity curve, fills render), Automation (start with 60s interval, see a pass row, stop cleanly), Settings (currency switch changes money strings), capture screenshots into `screenshots/`.

- [ ] **Step 5: Docs**

Update README (Status, feature table, quickstart for the portable exe, PAPERRL_HOME note, crypto and currency caveats). Apply the humanizer voice. Refresh the five-page table to eight tabs.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: production hardening, portable build, docs and screenshots"
```
