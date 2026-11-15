//! Automated paper trading: a momentum screen over a liquid universe, an
//! optional PPO position-sizing overlay, a deterministic risk gate, and paper
//! order execution through the Alpaca adapter.
//!
//! Safety: every order goes through [`plan_orders`], which can never spend
//! more cash than the account holds and caps position size and count.
//! Orders are only submitted when the caller opts in; the trading client is
//! hardcoded to Alpaca's paper endpoint.

use std::path::{Path, PathBuf};

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

/// Hard limits enforced before any order is sent.
#[derive(Debug, Clone, Copy, PartialEq)]
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
        }
    }
}

/// The strategy weight a symbol's checkpoint would take on the latest close,
/// or None when no checkpoint exists or it cannot be loaded.
fn latest_model_weight(model_dir: &Path, symbol: &str, bars: &[Bar]) -> Option<f64> {
    let path = model_dir.join(format!("ppo-{symbol}-42.bpk"));
    if !path.exists() {
        return None;
    }
    let mut strategy = paperrl_rl::PpoStrategy::load(&path).ok()?;
    let mut weight = 0.0;
    let start = bars.len().saturating_sub(120);
    for bar in &bars[start..] {
        weight = strategy.on_close(bar, 0.0, 0.0);
    }
    Some(weight)
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
        let fetched = client
            .bars(symbol, Timeframe::Day, now - chrono::Duration::days(150), now)
            .await?;
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

    // 1. Universe: most active plus screener movers, deduplicated.
    let universe = collect_universe(client, cfg.top).await?;
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
        if let Some((_, bars)) = bars_by_symbol.iter().find(|(s, _)| *s == c.symbol) {
            if let Some(w) = latest_model_weight(&cfg.model_dir, &c.symbol, *bars) {
                model_weights.push((c.symbol.clone(), w));
            }
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
