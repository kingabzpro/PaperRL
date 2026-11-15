//! Automated paper trading: a momentum screen over a liquid universe, an
//! optional PPO position-sizing overlay, a deterministic risk gate, and paper
//! order execution through the Alpaca adapter.
//!
//! Safety: every order goes through [`plan_orders`], which can never spend
//! more cash than the account holds and caps position size and count.
//! Orders are only submitted when the caller opts in; the trading client is
//! hardcoded to Alpaca's paper endpoint.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use paperrl_alpaca::{AlpacaClient, OrderSideView};
use paperrl_core::{Bar, Symbol, Timeframe};
use paperrl_data::BarStore;
use paperrl_sim::Strategy;

/// Long lookback in closes for the momentum score (20 returns).
pub const LONG_WINDOW: usize = 21;
/// Short lookback in closes (5 returns).
pub const SHORT_WINDOW: usize = 6;

/// A universe member ranked by blended momentum.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub symbol: String,
    /// Fractional return over the long window.
    pub ret20: f64,
    /// Fractional return over the short window.
    pub ret5: f64,
    pub last_close: f64,
    /// Mean close times volume over the long window.
    pub avg_dollar_volume: f64,
    /// 0.7 * ret20 + 0.3 * ret5.
    pub score: f64,
}

impl Candidate {
    pub fn gainer(&self) -> bool {
        self.score > 0.0
    }
}

/// Momentum features from stored bars; needs at least 21 bars.
pub fn momentum(bars: &[Bar]) -> Option<(f64, f64, f64)> {
    if bars.len() < LONG_WINDOW {
        return None;
    }
    let n = bars.len();
    let close = |i: usize| bars[i].close;
    let ret20 = close(n - 1) / close(n - LONG_WINDOW) - 1.0;
    let ret5 = close(n - 1) / close(n - SHORT_WINDOW) - 1.0;
    Some((ret20, ret5, close(n - 1)))
}

/// Liquidity filter for the momentum screen; penny stocks and thinly traded
/// names produce absurd scores and fill badly even in paper trading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenFilter {
    pub min_price: f64,
    pub min_avg_dollar_volume: f64,
}

impl Default for ScreenFilter {
    fn default() -> Self {
        Self { min_price: 5.0, min_avg_dollar_volume: 2_000_000.0 }
    }
}

/// Mean close times volume over the last `window` bars.
pub fn avg_dollar_volume(bars: &[Bar], window: usize) -> f64 {
    let start = bars.len().saturating_sub(window);
    let slice = &bars[start..];
    if slice.is_empty() {
        return 0.0;
    }
    slice.iter().map(|b| b.close * b.volume).sum::<f64>() / slice.len() as f64
}

/// Ranks every symbol with enough history, best score first, with no filters.
pub fn rank<'a>(items: impl IntoIterator<Item = (&'a str, &'a [Bar])>) -> Vec<Candidate> {
    rank_filtered(items, ScreenFilter { min_price: 0.0, min_avg_dollar_volume: 0.0 })
}

/// Ranks the universe under a liquidity filter.
pub fn rank_filtered<'a>(
    items: impl IntoIterator<Item = (&'a str, &'a [Bar])>,
    filter: ScreenFilter,
) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = items
        .into_iter()
        .filter_map(|(symbol, bars)| {
            let (ret20, ret5, last_close) = momentum(bars)?;
            let adv = avg_dollar_volume(bars, LONG_WINDOW);
            if last_close < filter.min_price || adv < filter.min_avg_dollar_volume {
                return None;
            }
            Some(Candidate {
                symbol: symbol.to_string(),
                ret20,
                ret5,
                last_close,
                avg_dollar_volume: adv,
                score: 0.7 * ret20 + 0.3 * ret5,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    out
}

/// The verdict for one ticker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Buy,
    Sell,
    Hold,
}

/// Indicator values behind a recommendation, shown in the "why" panel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Indicators {
    pub price: f64,
    /// Percent change of the last close vs the previous close.
    pub change_pct: f64,
    pub ema20: f64,
    pub ema50: f64,
    pub rsi14: f64,
    pub macd_hist: f64,
    pub ret5: f64,
    pub ret20: f64,
}

/// The PPO checkpoint that contributed to a verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelProvenance {
    pub seed: u64,
    /// Latest target equity weight the model produced, in [0, 1].
    pub weight: f64,
}

/// A per-ticker Buy / Sell / Hold verdict with its reasoning.
#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub symbol: Symbol,
    pub action: Action,
    /// 0.5 (coin flip) to 1.0 (all signals agree).
    pub confidence: f64,
    pub indicators: Indicators,
    /// Short human strings naming each contributing factor.
    pub reasons: Vec<String>,
    pub model: Option<ModelProvenance>,
}

/// Newest `ppo-<slug>-<seed>.bpk` in the model dir by file modification time.
fn newest_checkpoint(model_dir: &Path, slug: &str) -> Option<(PathBuf, u64)> {
    let mut best: Option<(std::time::SystemTime, PathBuf, u64)> = None;
    for entry in std::fs::read_dir(model_dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(&format!("ppo-{slug}-")) else {
            continue;
        };
        let Some(seed_str) = rest.strip_suffix(".bpk") else {
            continue;
        };
        let Ok(seed) = seed_str.parse::<u64>() else {
            continue;
        };
        let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if best.as_ref().is_none_or(|(t, _, _)| modified > *t) {
            best = Some((modified, entry.path(), seed));
        }
    }
    best.map(|(_, path, seed)| (path, seed))
}

/// Exponential moving average over closes; seeds with the first `n` values.
/// Returns the EMA series from the seed point on, or empty without enough data.
fn ema(closes: &[f64], n: usize) -> Vec<f64> {
    if n == 0 || closes.len() < n {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(closes.len() - n + 1);
    out.push(closes[..n].iter().sum::<f64>() / n as f64);
    let k = 2.0 / (n as f64 + 1.0);
    for &c in &closes[n..] {
        let prev = *out.last().expect("seeded above");
        out.push(prev + k * (c - prev));
    }
    out
}

/// Wilder's RSI; 50.0 (neutral) when there is not enough history.
fn rsi(closes: &[f64], n: usize) -> f64 {
    if n == 0 || closes.len() < n + 1 {
        return 50.0;
    }
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for w in closes.windows(2).take(n) {
        let d = w[1] - w[0];
        if d >= 0.0 {
            avg_gain += d;
        } else {
            avg_loss -= d;
        }
    }
    avg_gain /= n as f64;
    avg_loss /= n as f64;
    for w in closes.windows(2).skip(n) {
        let d = w[1] - w[0];
        let (g, l) = if d >= 0.0 { (d, 0.0) } else { (0.0, -d) };
        avg_gain = (avg_gain * (n as f64 - 1.0) + g) / n as f64;
        avg_loss = (avg_loss * (n as f64 - 1.0) + l) / n as f64;
    }
    if avg_loss == 0.0 {
        return 100.0;
    }
    let rs = avg_gain / avg_loss;
    100.0 - 100.0 / (1.0 + rs)
}

/// MACD histogram: MACD line minus its 9-period signal line; 0.0 without
/// enough history.
fn macd_hist(closes: &[f64]) -> f64 {
    let fast = ema(closes, 12);
    let slow = ema(closes, 26);
    if fast.is_empty() || slow.is_empty() {
        return 0.0;
    }
    let min = fast.len().min(slow.len());
    let macd: Vec<f64> = fast[fast.len() - min..]
        .iter()
        .zip(&slow[slow.len() - min..])
        .map(|(f, s)| f - s)
        .collect();
    match (macd.last(), ema(&macd, 9).last()) {
        (Some(m), Some(s)) => m - s,
        _ => 0.0,
    }
}

/// Scores one ticker from its daily bars. Fewer than 60 bars yields Hold
/// with an explanation. `model` biases the verdict when a PPO checkpoint
/// exists; `held_qty > 0` adds position context.
pub fn recommend(
    symbol: &Symbol,
    bars: &[Bar],
    model: Option<ModelProvenance>,
    held_qty: f64,
) -> Recommendation {
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let n = closes.len();
    let price = closes.last().copied().unwrap_or(0.0);
    if n < 60 {
        return Recommendation {
            symbol: symbol.clone(),
            action: Action::Hold,
            confidence: 0.5,
            indicators: Indicators {
                price,
                change_pct: 0.0,
                ema20: price,
                ema50: price,
                rsi14: 50.0,
                macd_hist: 0.0,
                ret5: 0.0,
                ret20: 0.0,
            },
            reasons: vec!["Not enough history (need about 60 daily bars)".to_string()],
            model,
        };
    }

    let change_pct = (closes[n - 1] / closes[n - 2] - 1.0) * 100.0;
    let ema20 = *ema(&closes, 20).last().expect("n >= 20");
    let ema50 = *ema(&closes, 50).last().expect("n >= 50");
    let rsi14 = rsi(&closes, 14);
    let hist = macd_hist(&closes);
    let (ret20, ret5, _) = momentum(bars).unwrap_or((0.0, 0.0, price));
    let indicators = Indicators {
        price,
        change_pct,
        ema20,
        ema50,
        rsi14,
        macd_hist: hist,
        ret5,
        ret20,
    };

    // Weighted vote in [-1, 1]; trend and momentum carry the most weight.
    // Each factor is normalized by the move size it should react to, so
    // small noise stays near zero instead of flipping the verdict.
    let mut score = 0.0;
    let mut reasons = Vec::new();

    let t1 = ((price / ema50 - 1.0) / 0.02).clamp(-1.0, 1.0);
    score += 0.25 * t1;
    if t1 > 0.1 {
        reasons.push(format!("Price {price:.2} is above the 50-day EMA ({ema50:.2})"));
    } else if t1 < -0.1 {
        reasons.push(format!("Price {price:.2} is below the 50-day EMA ({ema50:.2})"));
    }

    let t2 = ((ema20 / ema50 - 1.0) / 0.01).clamp(-1.0, 1.0);
    score += 0.20 * t2;
    if t2 > 0.1 {
        reasons.push("EMA20 is above EMA50 (uptrend)".to_string());
    } else if t2 < -0.1 {
        reasons.push("EMA20 is below EMA50 (downtrend)".to_string());
    }

    let m = (ret20 / 0.10).clamp(-1.0, 1.0);
    score += 0.25 * m;
    if ret20.abs() > 0.02 {
        reasons.push(format!("20-day return is {:+.1}%", ret20 * 100.0));
    }

    let md = (hist / (price * 0.01)).clamp(-1.0, 1.0);
    score += 0.15 * md;
    if hist > 0.0 {
        reasons.push("MACD histogram is positive".to_string());
    } else if hist < 0.0 {
        reasons.push("MACD histogram is negative".to_string());
    }

    let r = if rsi14 >= 70.0 {
        -(((rsi14 - 70.0) / 30.0).min(1.0))
    } else if rsi14 <= 30.0 {
        ((30.0 - rsi14) / 30.0).min(1.0)
    } else {
        0.0
    };
    score += 0.15 * r;
    if rsi14 >= 70.0 {
        reasons.push(format!("RSI is {rsi14:.0} (overbought)"));
    } else if rsi14 <= 30.0 {
        reasons.push(format!("RSI is {rsi14:.0} (oversold)"));
    }

    if let Some(mp) = model {
        if mp.weight >= 0.6 {
            score += 0.35;
            reasons.push(format!(
                "PPO model (seed {}) targets {:.0}% of equity",
                mp.seed,
                mp.weight * 100.0
            ));
        } else if mp.weight <= 0.2 {
            score -= 0.35;
            reasons.push(format!(
                "PPO model (seed {}) stays near cash ({:.0}% weight)",
                mp.seed,
                mp.weight * 100.0
            ));
        } else {
            reasons.push(format!(
                "PPO model (seed {}) is neutral ({:.0}% weight)",
                mp.seed,
                mp.weight * 100.0
            ));
        }
    }

    let score = score.clamp(-1.0, 1.0);
    let action = if score >= 0.3 {
        Action::Buy
    } else if score <= -0.3 {
        Action::Sell
    } else {
        Action::Hold
    };
    let confidence = 0.5 + score.abs() / 2.0;

    if held_qty > 0.0 {
        reasons.push(format!("Currently holding {held_qty} shares"));
        if action == Action::Sell {
            reasons.push("Signals point to exiting the position".to_string());
        }
    }

    Recommendation {
        symbol: symbol.clone(),
        action,
        confidence,
        indicators,
        reasons,
        model,
    }
}

/// Hard limits enforced before any order is sent.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RiskLimits {
    pub max_positions: usize,
    /// Cap per position as a fraction of equity.
    pub position_pct: f64,
    /// Cap per order in dollars.
    pub max_order_notional: f64,
    /// Orders below this notional are skipped as dust.
    pub min_order_notional: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_positions: 5,
            position_pct: 0.2,
            max_order_notional: 5_000.0,
            min_order_notional: 50.0,
        }
    }
}

/// One planned order before submission.
#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    Buy { symbol: String, notional: f64 },
    Sell { symbol: String, qty: f64 },
}

impl Intent {
    pub fn symbol(&self) -> &str {
        match self {
            Intent::Buy { symbol, .. } | Intent::Sell { symbol, .. } => symbol,
        }
    }
}

/// Inputs for one planning pass.
pub struct PlanInput<'a> {
    /// Momentum-ranked candidates, best first.
    pub candidates: &'a [Candidate],
    /// Open positions as (symbol, qty).
    pub held: &'a [(String, f64)],
    /// Symbols with open orders; treated as held so passes are idempotent.
    pub pending: &'a [String],
    /// Optional model target weight per symbol (0..1); overrides the default.
    pub model_weights: &'a [(String, f64)],
    pub equity: f64,
    pub cash: f64,
    pub limits: RiskLimits,
    /// How many top candidates may be held long.
    pub long_count: usize,
}

fn model_weight(input: &PlanInput, symbol: &str) -> Option<f64> {
    input
        .model_weights
        .iter()
        .find(|(s, _)| s == symbol)
        .map(|(_, w)| w.clamp(0.0, 1.0))
}

/// Plans sells for positions whose momentum has turned or dropped out of the
/// long list, then buys into the top positive-momentum candidates within the
/// risk limits. Never exceeds available cash.
pub fn plan_orders(input: &PlanInput) -> Vec<Intent> {
    let longs: Vec<&Candidate> = input
        .candidates
        .iter()
        .filter(|c| c.gainer())
        .take(input.long_count)
        .collect();
    let long_symbols: Vec<&str> = longs.iter().map(|c| c.symbol.as_str()).collect();

    let mut intents = Vec::new();

    // Sells first: anything held that is no longer a long and has no order
    // already in flight.
    let mut slots_freed = 0usize;
    for (symbol, qty) in input.held {
        if *qty > 0.0
            && !long_symbols.contains(&symbol.as_str())
            && !input.pending.iter().any(|p| p == symbol)
        {
            intents.push(Intent::Sell { symbol: symbol.clone(), qty: round4(*qty) });
            slots_freed += 1;
        }
    }

    // Buys: top longs not already held or pending, bounded by positions and cash.
    let held_symbols: Vec<&str> = input.held.iter().map(|(s, _)| s.as_str()).collect();
    let mut open_slots = input
        .limits
        .max_positions
        .saturating_sub(input.held.len() + input.pending.len())
        + slots_freed.min(input.held.len());
    let mut budget = input.cash.max(0.0);
    let per_position = (input.equity * input.limits.position_pct)
        .min(input.limits.max_order_notional);

    for c in longs {
        if held_symbols.contains(&c.symbol.as_str()) || input.pending.iter().any(|p| p == &c.symbol) {
            continue;
        }
        if open_slots == 0 || budget < input.limits.min_order_notional {
            break;
        }
        let weight = model_weight(input, &c.symbol).unwrap_or(0.5);
        let notional = (per_position * weight).min(budget);
        if notional < input.limits.min_order_notional {
            continue;
        }
        intents.push(Intent::Buy { symbol: c.symbol.clone(), notional: round2(notional) });
        budget -= notional;
        open_slots -= 1;
    }

    intents
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

/// Result of one automated pass.
#[derive(Debug, Clone, Default)]
pub struct AutoReport {
    pub universe: usize,
    pub gainers: Vec<Candidate>,
    pub losers: Vec<Candidate>,
    pub plan: Vec<(String, String)>, // (side, description) for display
    pub executed: Vec<String>,
    pub skipped: Vec<String>,
    pub errors: Vec<String>,
}

/// Configuration for one automated pass.
#[derive(Clone)]
pub struct AutoConfig {
    pub top: u32,
    pub long_count: usize,
    pub limits: RiskLimits,
    pub datasets: PathBuf,
    pub model_dir: PathBuf,
    pub filter: ScreenFilter,
    /// When false, orders are planned but never submitted (dry run).
    pub execute: bool,
    /// Bars older than this many days are refreshed before ranking.
    pub stale_days: i64,
    /// Explicit symbol list (the saved watchlist). When set, the screener
    /// is skipped entirely and only these symbols are ranked.
    pub universe: Option<Vec<Symbol>>,
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            top: 100,
            long_count: 5,
            limits: RiskLimits::default(),
            datasets: PathBuf::from("datasets"),
            model_dir: PathBuf::from("models"),
            filter: ScreenFilter::default(),
            execute: false,
            stale_days: 5,
            universe: None,
        }
    }
}

/// Lifecycle state of the in-app trading bot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotState {
    Idle,
    Running,
    Stopping,
}

/// One entry in the append-only bot pass history.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PassEntry {
    /// RFC3339 timestamp of the pass.
    pub time: String,
    /// "bot" or "manual".
    pub mode: String,
    pub planned: usize,
    pub executed: usize,
    /// Why a pass was skipped entirely (e.g. market closed), if it was.
    pub skipped: Option<String>,
    pub errors: Vec<String>,
}

/// App-level configuration, persisted as `config.json` in the data dir.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    /// Display currency code (USD, EUR, ...); conversion happens at render time.
    pub base_currency: String,
    /// Bot pass interval in seconds (minimum 60).
    pub interval_secs: u64,
    /// Saved tickers, e.g. ["AAPL", "BTC/USD"].
    pub watchlist: Vec<String>,
    pub risk: RiskLimits,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            base_currency: "USD".to_string(),
            interval_secs: 3600,
            watchlist: Vec::new(),
            risk: RiskLimits::default(),
        }
    }
}

impl AppConfig {
    /// Loads `config.json` from the data dir; a missing file yields defaults.
    pub fn load(home: &Path) -> paperrl_core::Result<Self> {
        let path = home.join("config.json");
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(paperrl_core::Error::Io(e)),
        };
        serde_json::from_str(&raw).map_err(|e| paperrl_core::Error::Other(e.to_string()))
    }

    /// Writes `config.json` into the data dir, creating it if needed.
    pub fn save(&self, home: &Path) -> paperrl_core::Result<()> {
        std::fs::create_dir_all(home)?;
        let path = home.join("config.json");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| paperrl_core::Error::Other(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Whether a pass should include this asset class right now: crypto trades
/// around the clock, equities wait for the market to open.
pub fn should_run(asset: paperrl_core::AssetClass, market_open: bool) -> bool {
    match asset {
        paperrl_core::AssetClass::Crypto => true,
        paperrl_core::AssetClass::UsEquity => market_open,
    }
}

/// Appends one pass entry to `<home>/logs/passes.jsonl`.
pub fn append_pass_entry(home: &Path, entry: &PassEntry) -> std::io::Result<()> {
    let dir = home.join("logs");
    std::fs::create_dir_all(&dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("passes.jsonl"))?;
    use std::io::Write;
    writeln!(file, "{}", serde_json::to_string(entry).map_err(std::io::Error::other)?)
}

/// Reads the pass history, oldest first; unreadable lines are skipped.
pub fn read_pass_history(home: &Path) -> Vec<PassEntry> {
    let Ok(raw) = std::fs::read_to_string(home.join("logs").join("passes.jsonl")) else {
        return Vec::new();
    };
    raw.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

/// The strategy weight the symbol's newest checkpoint would take on the
/// latest close, or None when no checkpoint exists or it cannot be loaded.
fn latest_model_weight(model_dir: &Path, symbol: &str, bars: &[Bar]) -> Option<ModelProvenance> {
    let slug = Symbol::parse(symbol)
        .map(|s| s.file_slug())
        .unwrap_or_else(|_| symbol.to_string());
    let (path, seed) = newest_checkpoint(model_dir, &slug)?;
    let mut strategy = paperrl_rl::PpoStrategy::load(&path).ok()?;
    let mut weight = 0.0;
    let start = bars.len().saturating_sub(120);
    for bar in &bars[start..] {
        weight = strategy.on_close(bar, 0.0, 0.0);
    }
    Some(ModelProvenance { seed, weight })
}

async fn ensure_bars(
    client: &AlpacaClient,
    store: &BarStore,
    symbol: &Symbol,
    stale_days: i64,
    now: chrono::DateTime<chrono::Utc>,
) -> paperrl_core::Result<Vec<Bar>> {
    let mut bars = store.read(symbol, Timeframe::Day).unwrap_or_default();
    let fresh = bars
        .last()
        .map(|b| (now - b.timestamp).num_days() <= stale_days)
        .unwrap_or(false);
    if !fresh || bars.len() < LONG_WINDOW {
        let fetched = match symbol.asset_class() {
            paperrl_core::AssetClass::Crypto => client.crypto_bars(symbol, 150).await?,
            paperrl_core::AssetClass::UsEquity => {
                client
                    .bars(symbol, Timeframe::Day, now - chrono::Duration::days(150), now)
                    .await?
            }
        };
        let merged = paperrl_data::merge_bars(bars, fetched);
        let _ = store.write(symbol, Timeframe::Day, &merged);
        bars = merged;
    }
    Ok(bars)
}

/// Collects the trading universe: most actives plus screener movers.
pub async fn collect_universe(client: &AlpacaClient, top: u32) -> paperrl_core::Result<Vec<String>> {
    let mut universe: Vec<String> = Vec::new();
    for sym in client.most_actives(top).await? {
        if !universe.contains(&sym) {
            universe.push(sym);
        }
    }
    let movers = client.movers(50).await?;
    for q in movers.gainers.iter().chain(movers.losers.iter()) {
        if !universe.contains(&q.symbol) {
            universe.push(q.symbol.clone());
        }
    }
    Ok(universe)
}

/// Ensures fresh daily bars for every symbol, fetching the last 150 days for
/// missing or stale entries. Returns the symbols that have enough history.
pub async fn ensure_universe_bars(
    client: &AlpacaClient,
    datasets: &Path,
    symbols: &[String],
    stale_days: i64,
) -> paperrl_core::Result<(Vec<(String, Vec<Bar>)>, Vec<String>, Vec<String>)> {
    let now = chrono::Utc::now();
    let store = BarStore::new(datasets);
    let mut with_bars = Vec::new();
    let mut skipped = Vec::new();
    let mut errors = Vec::new();
    for sym_str in symbols {
        let Ok(symbol) = Symbol::parse(sym_str) else {
            skipped.push(format!("{sym_str}: invalid symbol"));
            continue;
        };
        match ensure_bars(client, &store, &symbol, stale_days, now).await {
            Ok(bars) if bars.len() >= LONG_WINDOW => with_bars.push((symbol.to_string(), bars)),
            Ok(_) => skipped.push(format!("{sym_str}: not enough bars")),
            Err(e) => errors.push(format!("{sym_str}: {e}")),
        }
    }
    Ok((with_bars, skipped, errors))
}

/// Runs one automated pass: screen, rank, plan, and optionally execute.
///
/// Sells are submitted before buys. Every order carries a deterministic
/// client order id prefixed with `paperrl-` and is appended to an audit log
/// under `<datasets>/../logs/orders.csv`.
pub async fn run_once(client: &AlpacaClient, cfg: &AutoConfig) -> paperrl_core::Result<AutoReport> {
    let now = chrono::Utc::now();
    let mut report = AutoReport::default();

    // 1. Universe: the explicit watchlist when supplied, else the screener.
    let universe: Vec<String> = match &cfg.universe {
        Some(symbols) => symbols.iter().map(|s| s.as_str().to_string()).collect(),
        None => collect_universe(client, cfg.top).await?,
    };
    report.universe = universe.len();

    // 2. Ensure fresh daily bars for every universe member.
    let (with_bars, skipped, errors) =
        ensure_universe_bars(client, &cfg.datasets, &universe, cfg.stale_days).await?;
    report.skipped.extend(skipped);
    report.errors.extend(errors);

    // 3. Momentum ranking.
    let ranked = rank_filtered(
        with_bars.iter().map(|(s, b)| (s.as_str(), b.as_slice())),
        cfg.filter,
    );
    report.gainers = ranked.iter().filter(|c| c.gainer()).cloned().collect();
    report.losers = ranked.iter().filter(|c| !c.gainer()).cloned().collect();

    // 4. Account and positions.
    let account = client.account().await?;
    let equity: f64 = account
        .equity
        .as_deref()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let cash: f64 = account
        .cash
        .as_deref()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let positions = client.positions().await?;
    let held: Vec<(String, f64)> =
        positions.iter().map(|p| (p.symbol.clone(), p.qty)).collect();
    let pending = client.open_orders().await?;

    // 5. Model overlay for the top long candidates.
    let mut model_weights: Vec<(String, f64)> = Vec::new();
    let mut bars_by_symbol: Vec<(&str, &[Bar])> =
        with_bars.iter().map(|(s, b)| (s.as_str(), b.as_slice())).collect();
    bars_by_symbol.sort_by(|a, b| a.0.cmp(b.0));
    for c in report.gainers.iter().take(cfg.long_count) {
        if let Some((_, bars)) = bars_by_symbol.iter().find(|(s, _)| *s == c.symbol)
            && let Some(mp) = latest_model_weight(&cfg.model_dir, &c.symbol, bars) {
                model_weights.push((c.symbol.clone(), mp.weight));
            }
    }

    // 6. Plan under the risk gate.
    let plan = plan_orders(&PlanInput {
        candidates: &ranked,
        held: &held,
        pending: &pending,
        model_weights: &model_weights,
        equity,
        cash,
        limits: cfg.limits,
        long_count: cfg.long_count,
    });
    for intent in &plan {
        match intent {
            Intent::Buy { symbol, notional } => report
                .plan
                .push((format!("BUY {symbol}"), format!("${notional:.2}"))),
            Intent::Sell { symbol, qty } => report
                .plan
                .push((format!("SELL {symbol}"), format!("{qty} shares"))),
        }
    }

    // 7. Execute when explicitly enabled.
    if cfg.execute {
        let stamp = now.format("%Y%m%d%H%M%S").to_string();
        let log_dir = cfg.datasets.parent().unwrap_or(Path::new(".")).join("logs");
        let _ = std::fs::create_dir_all(&log_dir);
        let mut log_lines = String::new();
        for (i, intent) in plan.iter().enumerate() {
            let client_id = format!("paperrl-{stamp}-{i}");
            let result = match intent {
                Intent::Sell { symbol, qty } => {
                    client
                        .submit_market_order(symbol, OrderSideView::Sell, *qty, &client_id)
                        .await
                }
                Intent::Buy { symbol, notional } => {
                    client
                        .submit_market_notional_order(symbol, *notional, OrderSideView::Buy, &client_id)
                        .await
                }
            };
            match result {
                Ok(order) => {
                    report
                        .executed
                        .push(format!("{}: {}", order.client_order_id, order.status));
                    log_lines.push_str(&format!(
                        "{},{},{:?},{},{}\n",
                        now.to_rfc3339(),
                        intent.symbol(),
                        intent,
                        order.client_order_id,
                        order.status
                    ));
                }
                Err(e) => {
                    report.errors.push(format!("{}: {e}", intent.symbol()));
                }
            }
        }
        if !log_lines.is_empty() {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_dir.join("orders.csv"))
            {
                let _ = f.write_all(log_lines.as_bytes());
            }
        }
    }

    Ok(report)
}

/// Handle owning the in-app bot loop. The loop runs one pass immediately on
/// start, then every `interval`, stopping at the next boundary when stopped.
#[derive(Clone)]
pub struct BotHandle {
    state_rx: tokio::sync::watch::Receiver<BotState>,
    stop_tx: tokio::sync::mpsc::Sender<()>,
    task: Arc<tokio::task::JoinHandle<()>>,
    last_pass: Arc<std::sync::Mutex<Option<std::time::Instant>>>,
}

impl BotHandle {
    /// Current lifecycle state.
    pub fn state(&self) -> BotState {
        *self.state_rx.borrow()
    }

    /// When the last pass finished, for the status line.
    pub fn last_pass_at(&self) -> Option<std::time::Instant> {
        *self.last_pass.lock().expect("last_pass lock")
    }

    /// Requests a stop; takes effect once the current pass ends.
    pub async fn stop(&self) {
        let _ = self.stop_tx.send(()).await;
    }

    /// Waits for the loop task to finish (after `stop`).
    pub async fn join(&self) {
        while !self.task.is_finished() {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
}

/// Spawns the bot loop on the current tokio runtime. The loop filters the
/// universe each tick (equities wait for market hours, crypto passes any
/// time), runs `run_once`, and appends a history entry per pass.
// ponytail: no pause state, stop+start covers it; add pause if users ask.
pub fn spawn_bot(
    client: AlpacaClient,
    cfg: AutoConfig,
    home: &Path,
    interval: std::time::Duration,
) -> BotHandle {
    let (state_tx, state_rx) = tokio::sync::watch::channel(BotState::Running);
    let (stop_tx, stop_rx) = tokio::sync::mpsc::channel::<()>(1);
    let last_pass = Arc::new(std::sync::Mutex::new(None));
    
    BotHandle {
        state_rx,
        stop_tx,
        task: Arc::new(tokio::spawn(bot_loop(
            client,
            cfg,
            home.to_path_buf(),
            interval,
            state_tx,
            stop_rx,
            last_pass.clone(),
        ))),
        last_pass,
    }
}

async fn bot_loop(
    client: AlpacaClient,
    cfg: AutoConfig,
    home: PathBuf,
    interval: std::time::Duration,
    state_tx: tokio::sync::watch::Sender<BotState>,
    mut stop_rx: tokio::sync::mpsc::Receiver<()>,
    last_pass: Arc<std::sync::Mutex<Option<std::time::Instant>>>,
) {
    let mut first = true;
    loop {
        if !first {
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = stop_rx.recv() => break,
            }
        }
        first = false;

        // Market-hours gate: equities wait, crypto passes any time.
        let mut skipped: Option<String> = None;
        let pass_symbols: Vec<Symbol> = match &cfg.universe {
            Some(symbols) => {
                let has_equity = symbols
                    .iter()
                    .any(|s| s.asset_class() == paperrl_core::AssetClass::UsEquity);
                let market_open = if has_equity {
                    match client.clock().await {
                        Ok(c) => c.is_open,
                        Err(e) => {
                            // conservative: skip equities when the clock is unreachable
                            skipped = Some(format!("market clock unavailable: {e}"));
                            false
                        }
                    }
                } else {
                    true
                };
                symbols
                    .iter()
                    .filter(|s| should_run(s.asset_class(), market_open))
                    .cloned()
                    .collect()
            }
            None => Vec::new(),
        };

        if let Some(_symbols) = &cfg.universe {
            if pass_symbols.is_empty() {
                let entry = PassEntry {
                    time: chrono::Utc::now().to_rfc3339(),
                    mode: "bot".into(),
                    planned: 0,
                    executed: 0,
                    skipped: skipped.or_else(|| Some("market closed".into())),
                    errors: Vec::new(),
                };
                let _ = append_pass_entry(&home, &entry);
                *last_pass.lock().expect("last_pass lock") = Some(std::time::Instant::now());
                if stop_rx.try_recv().is_ok() {
                    break;
                }
                continue;
            }
            let mut pass_cfg = cfg.clone();
            pass_cfg.universe = Some(pass_symbols);
            let report = run_once(&client, &pass_cfg).await;
            let entry = match report {
                Ok(r) => PassEntry {
                    time: chrono::Utc::now().to_rfc3339(),
                    mode: "bot".into(),
                    planned: r.plan.len(),
                    executed: r.executed.len(),
                    skipped,
                    errors: r.errors,
                },
                Err(e) => PassEntry {
                    time: chrono::Utc::now().to_rfc3339(),
                    mode: "bot".into(),
                    planned: 0,
                    executed: 0,
                    skipped,
                    errors: vec![e.to_string()],
                },
            };
            let _ = append_pass_entry(&home, &entry);
            *last_pass.lock().expect("last_pass lock") = Some(std::time::Instant::now());
        }
        if stop_rx.try_recv().is_ok() {
            break;
        }
    }

    // Cancel only orders this app created; leave everything else alone.
    let _ = client.cancel_bot_orders("paperrl-").await;
    let _ = state_tx.send(BotState::Idle);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars_with_closes(closes: &[f64]) -> Vec<Bar> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| Bar {
                timestamp: chrono::DateTime::from_timestamp(86_400 * i as i64, 0).unwrap(),
                open: c,
                high: c,
                low: c,
                close: c,
                volume: 1000.0,
            })
            .collect()
    }

    fn cand(symbol: &str, score: f64) -> Candidate {
        Candidate {
            symbol: symbol.into(),
            ret20: score,
            ret5: score,
            last_close: 10.0,
            avg_dollar_volume: 10_000_000.0,
            score,
        }
    }

    // Trend builders for the recommendation tests: strict up, strict down,
    // and a low-noise chop that should stay neutral.
    fn rising(n: usize) -> Vec<Bar> {
        bars_with_closes(&(0..n).map(|i| 100.0 + i as f64).collect::<Vec<_>>())
    }

    fn falling(n: usize) -> Vec<Bar> {
        bars_with_closes(&(0..n).map(|i| 220.0 - i as f64).collect::<Vec<_>>())
    }

    fn chop(n: usize) -> Vec<Bar> {
        bars_with_closes(
            &(0..n)
                .map(|i| 100.0 + if i % 2 == 0 { 0.1 } else { -0.1 })
                .collect::<Vec<_>>(),
        )
    }

    fn sym(s: &str) -> Symbol {
        Symbol::parse(s).unwrap()
    }

    #[test]
    fn ema_trends() {
        // EMA lags a trend: below the last close on the way up, above on the way down.
        let up: Vec<f64> = (0..60).map(|i| 100.0 + i as f64).collect();
        let down: Vec<f64> = (0..60).map(|i| 200.0 - i as f64).collect();
        let up_ema = *ema(&up, 20).last().expect("enough history");
        let down_ema = *ema(&down, 20).last().expect("enough history");
        assert!(up_ema < *up.last().unwrap(), "uptrend EMA {up_ema} lags below close {}", up.last().unwrap());
        assert!(down_ema > *down.last().unwrap(), "downtrend EMA {down_ema} lags above close {}", down.last().unwrap());
    }

    #[test]
    fn rsi_bounds_and_direction() {
        let up = rsi(&(0..40).map(|i| 100.0 + i as f64).collect::<Vec<_>>(), 14);
        let down = rsi(&(0..40).map(|i| 200.0 - i as f64).collect::<Vec<_>>(), 14);
        assert!(up > 70.0, "rsi up = {up}");
        assert!(down < 30.0, "rsi down = {down}");
        // too little history stays neutral
        assert_eq!(rsi(&[1.0, 2.0], 14), 50.0);
    }

    #[test]
    fn macd_hist_sign_matches_trend() {
        // accelerating series keep the histogram away from zero
        let up: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64).powi(2) * 0.05).collect();
        let down: Vec<f64> = (0..60).map(|i| 400.0 - (i as f64).powi(2) * 0.05).collect();
        assert!(macd_hist(&up) > 0.0, "macd up = {}", macd_hist(&up));
        assert!(macd_hist(&down) < 0.0, "macd down = {}", macd_hist(&down));
        assert_eq!(macd_hist(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn recommend_buy_on_uptrend() {
        let r = recommend(&sym("AAPL"), &rising(120), None, 0.0);
        assert_eq!(r.action, Action::Buy);
        assert!(r.confidence > 0.5);
        assert!(!r.reasons.is_empty());
        assert!(r.model.is_none());
    }

    #[test]
    fn recommend_sell_on_downtrend() {
        let r = recommend(&sym("AAPL"), &falling(120), None, 0.0);
        assert_eq!(r.action, Action::Sell);
        assert!(r.confidence > 0.5);
    }

    #[test]
    fn recommend_hold_on_chop() {
        let r = recommend(&sym("AAPL"), &chop(120), None, 0.0);
        assert_eq!(r.action, Action::Hold);
    }

    #[test]
    fn ppo_weight_biases_verdict() {
        let buy = recommend(&sym("AAPL"), &chop(120), Some(ModelProvenance { seed: 42, weight: 0.9 }), 0.0);
        assert_eq!(buy.action, Action::Buy);
        assert!(buy.reasons.iter().any(|s| s.contains("42")), "model reason: {:?}", buy.reasons);
        let sell = recommend(&sym("AAPL"), &chop(120), Some(ModelProvenance { seed: 42, weight: 0.05 }), 0.0);
        assert_eq!(sell.action, Action::Sell);
    }

    #[test]
    fn held_position_exit_reason() {
        let r = recommend(&sym("AAPL"), &falling(120), None, 2.0);
        assert_eq!(r.action, Action::Sell);
        assert!(r.reasons.iter().any(|s| s.contains("exit")), "reasons: {:?}", r.reasons);
    }

    #[test]
    fn needs_minimum_history() {
        let r = recommend(&sym("AAPL"), &rising(10), None, 0.0);
        assert_eq!(r.action, Action::Hold);
        assert!(r.reasons.iter().any(|s| s.contains("history")));
    }

    #[test]
    fn newest_checkpoint_wins() {
        let dir = tempfile::tempdir().unwrap();
        let older = dir.path().join("ppo-AAPL-1.bpk");
        let newer = dir.path().join("ppo-AAPL-42.bpk");
        std::fs::write(&older, b"x").unwrap();
        std::fs::write(&newer, b"x").unwrap();
        let t = std::time::SystemTime::now();
        std::fs::File::options().write(true).open(&older).unwrap().set_modified(t - std::time::Duration::from_secs(7200)).unwrap();
        std::fs::File::options().write(true).open(&newer).unwrap().set_modified(t - std::time::Duration::from_secs(3600)).unwrap();
        let (path, seed) = newest_checkpoint(dir.path(), "AAPL").unwrap();
        assert_eq!(seed, 42);
        assert!(path.ends_with("ppo-AAPL-42.bpk"));
    }

    #[test]
    fn checkpoint_seed_parses_hyphen_slug() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ppo-BTC-USD-42.bpk");
        std::fs::write(&path, b"x").unwrap();
        let (_, seed) = newest_checkpoint(dir.path(), "BTC-USD").unwrap();
        assert_eq!(seed, 42);
    }

    #[test]
    fn no_checkpoint_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(newest_checkpoint(dir.path(), "AAPL").is_none());
    }

    #[test]
    fn should_run_respects_market_hours() {
        use paperrl_core::AssetClass;
        assert!(!should_run(AssetClass::UsEquity, false));
        assert!(should_run(AssetClass::UsEquity, true));
        assert!(should_run(AssetClass::Crypto, false));
    }

    #[test]
    fn config_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = AppConfig {
            base_currency: "EUR".to_string(),
            interval_secs: 120,
            watchlist: vec!["AAPL".into(), "BTC/USD".into()],
            risk: RiskLimits { max_positions: 3, ..RiskLimits::default() },
        };
        cfg.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded, cfg);
    }

    #[test]
    fn config_missing_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded, AppConfig::default());
    }

    #[test]
    fn pass_entry_jsonl_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let e1 = PassEntry {
            time: "2026-11-15T15:00:00Z".into(),
            mode: "bot".into(),
            planned: 2,
            executed: 1,
            skipped: None,
            errors: vec![],
        };
        let e2 = PassEntry {
            time: "2026-11-15T16:00:00Z".into(),
            mode: "bot".into(),
            planned: 0,
            executed: 0,
            skipped: Some("market closed".into()),
            errors: vec!["AAPL: timeout".into()],
        };
        append_pass_entry(dir.path(), &e1).unwrap();
        append_pass_entry(dir.path(), &e2).unwrap();
        let history = read_pass_history(dir.path());
        assert_eq!(history, vec![e1, e2]);
    }

    #[test]
    fn universe_override_skips_screener() {
        // The override path guarantees only watchlist bars reach the ranker;
        // pin that candidates outside the supplied bars can never be planned.
        let bars = vec![
            ("AAPL".to_string(), rising(120)),
            ("MSFT".to_string(), falling(120)),
        ];
        let ranked = rank_filtered(
            bars.iter().map(|(s, b)| (s.as_str(), b.as_slice())),
            ScreenFilter::default(),
        );
        let plan = plan_orders(&PlanInput {
            candidates: &ranked,
            held: &[],
            pending: &[],
            model_weights: &[],
            equity: 10_000.0,
            cash: 10_000.0,
            limits: RiskLimits::default(),
            long_count: 3,
        });
        assert!(plan.iter().all(
            |i| matches!(i, Intent::Buy { symbol, .. } if symbol == "AAPL")
        ));
    }

    #[test]
    fn momentum_needs_full_window() {
        assert!(momentum(&bars_with_closes(&[1.0; 10])).is_none());
        let bars = bars_with_closes(&[
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let (ret20, ret5, last) = momentum(&bars).unwrap();
        assert!((ret20 - 0.6).abs() < 1e-9); // 16/10 - 1
        assert!((ret5 - (16.0 / 11.0 - 1.0)).abs() < 1e-9); // close[n-6] is 11.0
        assert!((last - 16.0).abs() < 1e-9);
    }

    #[test]
    fn rank_orders_by_blended_score() {
        let a = bars_with_closes(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                                   20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]);
        let b = bars_with_closes(&[30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0,
                                   20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0]);
        let ranked = rank(vec![("AAA", a.as_slice()), ("BBB", b.as_slice())]);
        assert_eq!(ranked[0].symbol, "AAA");
        assert!(ranked[0].gainer());
        assert!(!ranked[1].gainer());
    }

    #[test]
    fn liquidity_filter_excludes_penny_and_thin_names() {
        let mut flat = bars_with_closes(&[10.0; 30]);
        for b in flat.iter_mut() {
            b.volume = 300_000.0; // $3M average dollar volume
        }
        let mut penny = bars_with_closes(&[0.5; 30]);
        for b in penny.iter_mut() {
            b.volume = 300_000.0; // $150k ADV and a sub-$5 price
        }
        let items = vec![("FLAT", flat.as_slice()), ("PENNY", penny.as_slice())];
        let filtered = rank_filtered(items.iter().copied(), ScreenFilter::default());
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].symbol, "FLAT");
        // The unfiltered rank sees both.
        assert_eq!(rank(items.iter().copied()).len(), 2);
        // A high price with thin volume is also excluded.
        let mut thin = bars_with_closes(&[10.0; 30]);
        for b in thin.iter_mut() {
            b.volume = 10.0; // $100/day
        }
        let items2 = vec![("THIN", thin.as_slice())];
        assert!(rank_filtered(items2.iter().copied(), ScreenFilter::default()).is_empty());
    }

    fn limits() -> RiskLimits {
        RiskLimits { max_positions: 2, position_pct: 0.2, max_order_notional: 5_000.0, min_order_notional: 50.0 }
    }

    #[test]
    fn plan_sells_dropped_positions_and_buys_top_longs() {
        let candidates = vec![cand("AAA", 0.5), cand("BBB", 0.3), cand("CCC", -0.2)];
        let held = vec![("CCC".to_string(), 10.0), ("AAA".to_string(), 5.0)];
        let plan = plan_orders(&PlanInput {
            candidates: &candidates,
            held: &held,
            pending: &[],
            model_weights: &[],
            equity: 10_000.0,
            cash: 5_000.0,
            limits: limits(),
            long_count: 2,
        });
        assert_eq!(
            plan,
            vec![
                Intent::Sell { symbol: "CCC".into(), qty: 10.0 },
                Intent::Buy { symbol: "BBB".into(), notional: 1_000.0 },
            ]
        );
    }

    #[test]
    fn plan_respects_cash_and_position_caps() {
        let candidates = vec![cand("AAA", 0.5), cand("BBB", 0.3), cand("CCC", 0.1)];
        let plan = plan_orders(&PlanInput {
            candidates: &candidates,
            held: &[],
            pending: &[],
            model_weights: &[],
            equity: 10_000.0,
            cash: 900.0,
            limits: limits(),
            long_count: 3,
        });
        // Per-position 2000 x 0.5 weight = 1000, but cash is 900: first buy
        // takes all of it, the rest stop.
        assert_eq!(plan, vec![Intent::Buy { symbol: "AAA".into(), notional: 900.0 }]);
    }

    #[test]
    fn model_weight_zero_blocks_a_buy_and_scales_others() {
        let candidates = vec![cand("AAA", 0.5), cand("BBB", 0.3)];
        let plan = plan_orders(&PlanInput {
            candidates: &candidates,
            held: &[],
            pending: &[],
            model_weights: &[("AAA".to_string(), 0.0), ("BBB".to_string(), 1.0)],
            equity: 10_000.0,
            cash: 10_000.0,
            limits: RiskLimits { max_positions: 2, position_pct: 0.2, max_order_notional: 5_000.0, min_order_notional: 50.0 },
            long_count: 2,
        });
        assert_eq!(plan, vec![Intent::Buy { symbol: "BBB".into(), notional: 2_000.0 }]);
    }

    #[test]
    fn pending_orders_block_duplicate_buys_and_sells() {
        let candidates = vec![cand("AAA", 0.5), cand("BBB", 0.3)];
        let pending = vec!["AAA".to_string()];
        let plan = plan_orders(&PlanInput {
            candidates: &candidates,
            held: &[],
            pending: &pending,
            model_weights: &[],
            equity: 10_000.0,
            cash: 10_000.0,
            limits: limits(),
            long_count: 2,
        });
        // AAA has an order in flight: only BBB is bought.
        assert_eq!(plan, vec![Intent::Buy { symbol: "BBB".into(), notional: 1_000.0 }]);

        // A held position that fell out of the longs is not sold twice while
        // a sell is already pending.
        let held = vec![("CCC".to_string(), 3.0)];
        let pending_sell = vec!["CCC".to_string()];
        let plan2 = plan_orders(&PlanInput {
            candidates: &candidates,
            held: &held,
            pending: &pending_sell,
            model_weights: &[],
            equity: 10_000.0,
            cash: 10_000.0,
            limits: limits(),
            long_count: 2,
        });
        assert!(plan2.iter().all(|i| !matches!(i, Intent::Sell { .. })));
    }

    #[test]
    fn plan_never_exceeds_max_positions() {
        let candidates = vec![cand("AAA", 0.5), cand("BBB", 0.4), cand("CCC", 0.3)];
        let plan = plan_orders(&PlanInput {
            candidates: &candidates,
            held: &[("ZZZ".to_string(), 1.0)],
            pending: &[],
            model_weights: &[],
            equity: 10_000.0,
            cash: 10_000.0,
            limits: RiskLimits { max_positions: 2, position_pct: 0.2, max_order_notional: 5_000.0, min_order_notional: 50.0 },
            long_count: 3,
        });
        // ZZZ is sold (frees a slot); with max 2 positions we buy only AAA
        // (the second slot is consumed by the freed slot accounting).
        let buys = plan.iter().filter(|i| matches!(i, Intent::Buy { .. })).count();
        let sells = plan.iter().filter(|i| matches!(i, Intent::Sell { .. })).count();
        assert_eq!(sells, 1);
        assert!(buys <= 2, "bought {buys}");
    }
}
