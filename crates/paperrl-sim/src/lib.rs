//! Deterministic one-symbol execution simulator.
//!
//! Semantics: a strategy emits a target equity weight at a bar's close; the
//! resulting rebalance fills at the next bar's open with slippage applied to
//! the execution price and a basis-point fee on traded notional. No leverage:
//! targets are clamped to [0, 1] and buys never spend more cash than they are
//! allocated. Same inputs always produce identical outputs.

use chrono::{DateTime, Datelike, Utc};
use paperrl_core::Bar;

/// Rebalances smaller than this notional (USD) are skipped as dust.
const MIN_TRADE_USD: f64 = 1.0;

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub initial_cash: f64,
    /// Fee in basis points of traded notional.
    pub fee_bps: f64,
    /// Slippage in basis points applied against you on the execution price.
    pub slippage_bps: f64,
    /// Bar frequency per year, used to annualize Sharpe (252 for daily bars).
    pub periods_per_year: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            initial_cash: 100_000.0,
            fee_bps: 1.0,
            slippage_bps: 5.0,
            periods_per_year: 252.0,
        }
    }
}

/// A strategy maps each bar's close to a target equity weight in [0, 1].
/// It sees the position weight and drawdown as of that close, which
/// portfolio-aware strategies (like RL policies) use as context.
pub trait Strategy {
    fn name(&self) -> &str;
    fn on_close(&mut self, bar: &Bar, position_weight: f64, drawdown: f64) -> f64;
}

/// Buy on the first fill opportunity, then hold forever.
pub struct BuyHold;

impl Strategy for BuyHold {
    fn name(&self) -> &str {
        "buyhold"
    }
    fn on_close(&mut self, _bar: &Bar, _position_weight: f64, _drawdown: f64) -> f64 {
        1.0
    }
}

/// Enter when the fast EMA is above the slow EMA, exit when it is below.
/// Both EMAs are seeded with the first close seen; no position during warmup.
pub struct EmaCross {
    fast: usize,
    slow: usize,
    fast_ema: Option<f64>,
    slow_ema: Option<f64>,
    seen: usize,
}

impl EmaCross {
    pub fn new(fast: usize, slow: usize) -> Self {
        assert!(fast > 0 && fast < slow, "need 0 < fast < slow");
        Self { fast, slow, fast_ema: None, slow_ema: None, seen: 0 }
    }
}

impl Strategy for EmaCross {
    fn name(&self) -> &str {
        "ema"
    }
    fn on_close(&mut self, bar: &Bar, _position_weight: f64, _drawdown: f64) -> f64 {
        let k_fast = 2.0 / (self.fast as f64 + 1.0);
        let k_slow = 2.0 / (self.slow as f64 + 1.0);
        self.fast_ema = Some(match self.fast_ema {
            Some(e) => bar.close * k_fast + e * (1.0 - k_fast),
            None => bar.close,
        });
        self.slow_ema = Some(match self.slow_ema {
            Some(e) => bar.close * k_slow + e * (1.0 - k_slow),
            None => bar.close,
        });
        self.seen += 1;
        if self.seen < self.slow {
            return 0.0;
        }
        if self.fast_ema.unwrap() > self.slow_ema.unwrap() {
            1.0
        } else {
            0.0
        }
    }
}

/// Time-series momentum: long while the trailing `window`-bar return is positive.
pub struct Momentum {
    window: usize,
    closes: std::collections::VecDeque<f64>,
}

impl Momentum {
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window must be positive");
        Self { window, closes: std::collections::VecDeque::new() }
    }
}

impl Strategy for Momentum {
    fn name(&self) -> &str {
        "momentum"
    }
    fn on_close(&mut self, bar: &Bar, _position_weight: f64, _drawdown: f64) -> f64 {
        self.closes.push_back(bar.close);
        if self.closes.len() > self.window + 1 {
            self.closes.pop_front();
        }
        if self.closes.len() < self.window + 1 {
            return 0.0;
        }
        if bar.close > self.closes[0] {
            1.0
        } else {
            0.0
        }
    }
}

/// Mean reversion: long when the close is below its `window`-bar SMA.
pub struct MeanReversion {
    window: usize,
    closes: std::collections::VecDeque<f64>,
}

impl MeanReversion {
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window must be positive");
        Self { window, closes: std::collections::VecDeque::new() }
    }
}

impl Strategy for MeanReversion {
    fn name(&self) -> &str {
        "meanrev"
    }
    fn on_close(&mut self, bar: &Bar, _position_weight: f64, _drawdown: f64) -> f64 {
        self.closes.push_back(bar.close);
        if self.closes.len() > self.window {
            self.closes.pop_front();
        }
        if self.closes.len() < self.window {
            return 0.0;
        }
        let sma = self.closes.iter().sum::<f64>() / self.closes.len() as f64;
        if bar.close < sma {
            1.0
        } else {
            0.0
        }
    }
}

/// Volatility targeting: weight = min(1, target_vol / realized_vol), long only.
pub struct VolTarget {
    window: usize,
    target_vol: f64,
    closes: std::collections::VecDeque<f64>,
}

impl VolTarget {
    /// `target_vol` is annualized (0.15 = 15%).
    pub fn new(window: usize, target_vol: f64) -> Self {
        assert!(window > 1 && target_vol > 0.0, "need window > 1 and target_vol > 0");
        Self { window, target_vol, closes: std::collections::VecDeque::new() }
    }
}

impl Strategy for VolTarget {
    fn name(&self) -> &str {
        "voltarget"
    }
    fn on_close(&mut self, bar: &Bar, _position_weight: f64, _drawdown: f64) -> f64 {
        self.closes.push_back(bar.close);
        if self.closes.len() > self.window + 1 {
            self.closes.pop_front();
        }
        if self.closes.len() < self.window + 1 {
            return 0.0;
        }
        let closes: Vec<f64> = self.closes.iter().copied().collect();
        let returns: Vec<f64> = closes
            .windows(2)
            .map(|w| w[1] / w[0] - 1.0)
            .collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let var =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let ann_vol = var.sqrt() * 252f64.sqrt();
        if ann_vol < 1e-9 {
            return 1.0;
        }
        (self.target_vol / ann_vol).clamp(0.0, 1.0)
    }
}

/// Performance metrics for one simulated run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub bars: usize,
    pub final_equity: f64,
    /// final_equity / initial_cash - 1.
    pub net_return: f64,
    /// Annualized Sharpe of per-bar equity returns (sample stdev, n-1).
    pub sharpe: f64,
    /// Worst peak-to-trough loss of the equity curve, in [0, 1].
    pub max_drawdown: f64,
    /// Fraction of bars with a positive equity return.
    pub win_rate: f64,
    pub trades: usize,
    /// Absolute traded notional in USD.
    pub turnover: f64,
    pub total_fees: f64,
    /// Mean position weight over the bars (exposure, not leverage).
    pub avg_exposure: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimResult {
    /// Marked-to-close equity after each bar, aligned with the input bars.
    pub equity: Vec<f64>,
    pub metrics: Metrics,
}

/// Incremental portfolio accounting shared by the batch simulator and the RL
/// environment: fills pending target changes at bar opens with slippage and
/// fees. Targets are clamped to [0, 1]; sells never exceed held shares.
pub struct Portfolio {
    pub cash: f64,
    pub shares: f64,
    fee: f64,
    slip: f64,
    target: f64,
    pending: Option<f64>,
    pub trades: usize,
    pub turnover: f64,
    pub total_fees: f64,
}

impl Portfolio {
    pub fn new(initial_cash: f64, fee_bps: f64, slippage_bps: f64) -> Self {
        assert!(initial_cash > 0.0, "initial_cash must be positive");
        assert!(fee_bps >= 0.0, "fee_bps must be >= 0");
        assert!(
            (0.0..10_000.0).contains(&slippage_bps),
            "slippage_bps must be in [0, 10000)"
        );
        Self {
            cash: initial_cash,
            shares: 0.0,
            fee: fee_bps / 10_000.0,
            slip: slippage_bps / 10_000.0,
            target: 0.0,
            pending: None,
            trades: 0,
            turnover: 0.0,
            total_fees: 0.0,
        }
    }

    /// Portfolio value at `price`.
    pub fn equity(&self, price: f64) -> f64 {
        self.cash + self.shares * price
    }

    /// Current position weight at `price`.
    pub fn position_weight(&self, price: f64) -> f64 {
        let equity = self.equity(price);
        if equity > 0.0 {
            self.shares * price / equity
        } else {
            0.0
        }
    }

    /// Records a target weight emitted at a close; it fills at the next open.
    pub fn on_close(&mut self, weight: f64) {
        let w = weight.clamp(0.0, 1.0);
        if (w - self.target).abs() > 1e-9 {
            self.pending = Some(w);
        }
    }

    /// Fills any pending target change at `bar.open`.
    pub fn fill_at_open(&mut self, bar: &Bar) {
        let Some(w) = self.pending.take() else { return };
        let position_value = self.shares * bar.open;
        let delta_value = w * (self.cash + position_value) - position_value;
        if delta_value > MIN_TRADE_USD {
            // Buy: spend exactly delta_value including fees.
            let exec = bar.open * (1.0 + self.slip);
            let cost = delta_value / (1.0 + self.fee);
            if cost <= self.cash + 1e-9 {
                let f = cost * self.fee;
                self.cash -= cost + f;
                self.shares += cost / exec;
                self.turnover += cost;
                self.total_fees += f;
                self.trades += 1;
                self.target = w;
            }
        } else if delta_value < -MIN_TRADE_USD {
            // Sell: liquidate delta_value worth at the discounted price.
            let exec = bar.open * (1.0 - self.slip);
            let requested = -delta_value / exec;
            let qty = requested.min(self.shares);
            let proceeds = qty * exec;
            let f = proceeds * self.fee;
            self.cash += proceeds - f;
            self.shares -= qty;
            self.turnover += proceeds;
            self.total_fees += f;
            self.trades += 1;
            // If the clamp bound the sell, record the achieved position
            // weight so a repeated signal can still rebalance.
            self.target = if qty < requested {
                self.shares * bar.open / (self.cash + self.shares * bar.open).max(1e-12)
            } else {
                w
            };
        }
    }
}

pub fn run(bars: &[Bar], strategy: &mut dyn Strategy, cfg: &SimConfig) -> SimResult {
    if bars.is_empty() {
        return SimResult {
            equity: Vec::new(),
            metrics: Metrics {
                bars: 0,
                final_equity: 0.0,
                net_return: 0.0,
                sharpe: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                trades: 0,
                turnover: 0.0,
                total_fees: 0.0,
                avg_exposure: 0.0,
            },
        };
    }

    let mut pf = Portfolio::new(cfg.initial_cash, cfg.fee_bps, cfg.slippage_bps);
    let mut equity = Vec::with_capacity(bars.len());
    let mut peak = cfg.initial_cash;
    let mut exposure_sum = 0.0;
    for bar in bars {
        pf.fill_at_open(bar);
        let eq = pf.equity(bar.close);
        equity.push(eq);
        peak = peak.max(eq);
        let pw = pf.position_weight(bar.close);
        exposure_sum += pw;
        let dd = (1.0 - eq / peak).max(0.0);
        pf.on_close(strategy.on_close(bar, pw, dd));
    }

    let final_equity = equity[equity.len() - 1];
    let rets: Vec<f64> = equity.windows(2).map(|w| w[1] / w[0] - 1.0).collect();
    let mean = if rets.is_empty() { 0.0 } else { rets.iter().sum::<f64>() / rets.len() as f64 };
    let sharpe = if rets.len() < 2 {
        0.0
    } else {
        let var =
            rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
        let sd = var.sqrt();
        if sd > 0.0 {
            mean / sd * cfg.periods_per_year.sqrt()
        } else {
            0.0
        }
    };
    let mut peak: f64 = equity[0];
    let mut max_drawdown: f64 = 0.0;
    for &e in &equity {
        peak = peak.max(e);
        max_drawdown = max_drawdown.max(1.0 - e / peak);
    }
    let win_rate = if rets.is_empty() {
        0.0
    } else {
        rets.iter().filter(|r| **r > 0.0).count() as f64 / rets.len() as f64
    };

    SimResult {
        equity,
        metrics: Metrics {
            bars: bars.len(),
            final_equity,
            net_return: final_equity / cfg.initial_cash - 1.0,
            sharpe,
            max_drawdown,
            win_rate,
            trades: pf.trades,
            turnover: pf.turnover,
            total_fees: pf.total_fees,
            avg_exposure: exposure_sum / bars.len() as f64,
        },
    }
}

/// One evaluation window's outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowResult {
    /// First and last bar timestamps inside the window.
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    /// True for a trailing window shorter than `window_months`.
    pub partial: bool,
    pub metrics: Metrics,
}

/// Segmented evaluation over consecutive `window_months` windows anchored at
/// the first bar's month. Each window runs independently: fresh strategy
/// state and reset capital, so a window's return measures that segment alone.
/// A trailing window shorter than `window_months` is reported with
/// `partial: true` and excluded from the aggregate statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct WalkForwardResult {
    pub windows: Vec<WindowResult>,
    /// Mean net return over the full-length windows.
    pub mean_return: f64,
    /// Sample stdev of full-window net returns (0 with fewer than 2).
    pub return_stdev: f64,
    /// Worst net return among full-length windows.
    pub worst_return: f64,
    /// Full-length windows with a positive net return.
    pub positive_windows: usize,
}

pub fn walk_forward(
    bars: &[Bar],
    make_strategy: &mut dyn FnMut() -> Box<dyn Strategy>,
    cfg: &SimConfig,
    window_months: u32,
) -> WalkForwardResult {
    assert!(window_months > 0, "window_months must be positive");
    debug_assert!(
        bars.windows(2).all(|w| w[0].timestamp <= w[1].timestamp),
        "bars must be sorted by timestamp"
    );
    if bars.is_empty() {
        return WalkForwardResult {
            windows: Vec::new(),
            mean_return: 0.0,
            return_stdev: 0.0,
            worst_return: 0.0,
            positive_windows: 0,
        };
    }
    let first = month_index(bars[0].timestamp);
    let mut groups: Vec<Vec<Bar>> = Vec::new();
    for &bar in bars {
        let k = ((month_index(bar.timestamp) - first) as i64 / window_months as i64) as usize;
        if k >= groups.len() {
            groups.resize_with(k + 1, Vec::new);
        }
        groups[k].push(bar);
    }
    let last_group = groups
        .iter()
        .rposition(|g| !g.is_empty())
        .expect("non-empty bars produce at least one non-empty group");
    let mut windows = Vec::new();
    for (gi, group) in groups.into_iter().enumerate() {
        if group.is_empty() {
            continue;
        }
        let mut strategy = make_strategy();
        let metrics = run(&group, strategy.as_mut(), cfg).metrics;
        let start = group[0].timestamp;
        let end = group[group.len() - 1].timestamp;
        let span = month_index(end) - month_index(start) + 1;
        windows.push(WindowResult {
            start,
            end,
            partial: gi == last_group && span < window_months as i32,
            metrics,
        });
    }
    let rets: Vec<f64> = windows
        .iter()
        .filter(|w| !w.partial)
        .map(|w| w.metrics.net_return)
        .collect();
    let (mean, stdev, worst, positive) = if rets.is_empty() {
        (0.0, 0.0, 0.0, 0)
    } else {
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let stdev = if rets.len() < 2 {
            0.0
        } else {
            (rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64)
                .sqrt()
        };
        let worst = rets.iter().copied().fold(f64::INFINITY, f64::min);
        let positive = rets.iter().filter(|r| **r > 0.0).count();
        (mean, stdev, worst, positive)
    };
    WalkForwardResult {
        positive_windows: positive,
        worst_return: worst,
        mean_return: mean,
        return_stdev: stdev,
        windows,
    }
}

fn month_index(t: DateTime<Utc>) -> i32 {
    t.year() * 12 + t.month0() as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone};

    /// open_i and close_i per bar.
    fn bars(opens: &[f64], closes: &[f64]) -> Vec<Bar> {
        assert_eq!(opens.len(), closes.len());
        opens
            .iter()
            .zip(closes)
            .enumerate()
            .map(|(i, (&o, &c))| Bar {
                timestamp: DateTime::from_timestamp(60 * i as i64, 0).unwrap(),
                open: o,
                high: o.max(c),
                low: o.min(c),
                close: c,
                volume: 1000.0,
            })
            .collect()
    }

    fn cfg(cash: f64, fee_bps: f64, slip_bps: f64) -> SimConfig {
        SimConfig { initial_cash: cash, fee_bps, slippage_bps: slip_bps, periods_per_year: 252.0 }
    }

    /// Fixed target schedule, one value per bar close.
    struct Scripted {
        targets: Vec<f64>,
        i: usize,
    }
    impl Scripted {
        fn new(targets: Vec<f64>) -> Self {
            Self { targets, i: 0 }
        }
    }
    impl Strategy for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }
        fn on_close(&mut self, _bar: &Bar, _position_weight: f64, _drawdown: f64) -> f64 {
            let t = self.targets[self.i.min(self.targets.len() - 1)];
            self.i += 1;
            t
        }
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "left {a} vs right {b}");
    }

    fn ts(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn bar_at(t: DateTime<Utc>, open: f64, close: f64) -> Bar {
        Bar {
            timestamp: t,
            open,
            high: open.max(close),
            low: open.min(close),
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn empty_bars_give_zero_metrics() {
        let res = run(&[], &mut BuyHold, &cfg(10_000.0, 0.0, 0.0));
        assert_eq!(res.equity, Vec::<f64>::new());
        assert_eq!(res.metrics.bars, 0);
        assert_eq!(res.metrics.net_return, 0.0);
    }

    #[test]
    fn buyhold_fills_once_with_fee_math() {
        let bs = bars(&[99.0, 100.0, 101.0], &[100.0, 101.0, 102.0]);
        let res = run(&bs, &mut BuyHold, &cfg(10_000.0, 10.0, 0.0));
        // Fill at bar 1 open (100.0): shares = cash / (price * (1 + fee)).
        let shares = 10_000.0 / (100.0 * 1.001);
        approx(res.equity[0], 10_000.0);
        approx(res.equity[1], shares * 101.0);
        approx(res.equity[2], shares * 102.0);
        assert_eq!(res.metrics.trades, 1);
        let cost = shares * 100.0;
        approx(res.metrics.total_fees, cost * 0.001);
        approx(res.metrics.turnover, cost);
        // Flat at the first close, fully invested after: 2 of 3 bars.
        approx(res.metrics.avg_exposure, 2.0 / 3.0);
        approx(res.metrics.max_drawdown, 0.0);
        approx(res.metrics.net_return, shares * 102.0 / 10_000.0 - 1.0);
    }

    #[test]
    fn signal_at_close_fills_at_next_open() {
        let bs = bars(&[10.0, 20.0], &[15.0, 25.0]);
        let res = run(&bs, &mut Scripted::new(vec![1.0]), &cfg(10_000.0, 0.0, 0.0));
        // No position at the first close.
        approx(res.equity[0], 10_000.0);
        // Order placed at close of bar 0 fills at bar 1 open (20.0), not its close.
        approx(res.equity[1], (10_000.0 / 20.0) * 25.0);
        assert_eq!(res.metrics.trades, 1);
    }

    #[test]
    fn reducing_target_sells_half() {
        let bs = bars(&[10.0, 20.0, 30.0], &[15.0, 25.0, 35.0]);
        let res = run(&bs, &mut Scripted::new(vec![1.0, 0.5]), &cfg(10_000.0, 0.0, 0.0));
        // Bar 1: 500 shares at open 20, cash 0. Bar 1 close: target 0.5.
        // Bar 2 open: sell 250 shares at 30, cash 7500, equity 7500 + 250*35.
        approx(res.equity[2], 7_500.0 + 250.0 * 35.0);
        assert_eq!(res.metrics.trades, 2);
        approx(res.metrics.turnover, 10_000.0 + 7_500.0);
    }

    #[test]
    fn slippage_hits_buy_and_sell_sides() {
        // Buy at next-open*(1+slip), sell at open*(1-slip).
        let bs = bars(&[10.0, 20.0, 30.0], &[15.0, 25.0, 35.0]);
        let slip = 0.01; // 100 bps
        let res = run(&bs, &mut Scripted::new(vec![1.0, 0.0]), &cfg(10_000.0, 0.0, 100.0));
        // Signal at bar 0 close fills at bar 1 open (20.0), executed at 20.2.
        let shares = 10_000.0 / (20.0 * (1.0 + slip));
        // Sold at bar 2 open 30 * 0.99 = 29.7; all cash, no shares left.
        approx(res.equity[2], shares * 29.7);
        assert_eq!(res.metrics.trades, 2);
    }

    #[test]
    fn buyhold_never_going_negative_cash_or_shares() {
        let bs = bars(&[10.0, 11.0, 12.0], &[10.5, 11.5, 12.5]);
        let res = run(&bs, &mut BuyHold, &cfg(10_000.0, 50.0, 100.0));
        approx(res.metrics.final_equity / 10_000.0 - 1.0, res.metrics.net_return);
        assert!(res.metrics.net_return > 0.0);
        assert_eq!(res.metrics.trades, 1);
    }

    #[test]
    fn metrics_capture_drawdown_winrate_sharpe() {
        // Buy at open 10 (fill after first close), closes 12, 6, 11.
        let bs = bars(&[10.0, 10.0, 10.0, 10.0], &[10.0, 12.0, 6.0, 11.0]);
        let res = run(&bs, &mut Scripted::new(vec![1.0]), &cfg(10_000.0, 0.0, 0.0));
        let shares = 10_000.0 / 10.0;
        approx(res.metrics.max_drawdown, 1.0 - 6_000.0 / 12_000.0);
        approx(res.metrics.net_return, 11.0 * shares / 10_000.0 - 1.0);
        approx(res.metrics.win_rate, 2.0 / 3.0);
        assert!(res.metrics.sharpe > 4.0 && res.metrics.sharpe < 4.5);
    }

    #[test]
    fn ema_cross_ignores_flat_series() {
        let closes = [5.0; 10];
        let opens = [5.0; 10];
        let bs = bars(&opens, &closes);
        let res = run(&bs, &mut EmaCross::new(2, 3), &cfg(10_000.0, 0.0, 0.0));
        assert_eq!(res.metrics.trades, 0);
        approx(res.metrics.final_equity, 10_000.0);
    }

    #[test]
    fn ema_cross_enters_on_uptrend() {
        let up = bars(
            &[1.0, 1.5, 2.5, 3.5, 4.5, 5.5],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let res = run(&up, &mut EmaCross::new(2, 3), &cfg(10_000.0, 0.0, 0.0));
        assert!(res.metrics.trades >= 1);
        assert!(res.metrics.final_equity > 10_000.0);
    }

    #[test]
    fn ema_cross_exits_when_trend_turns() {
        // Rise then fall: entry during the rise, exit once the fast EMA
        // crosses back below the slow one (a strict decline never triggers an
        // entry, so the series must turn over).
        let bs = bars(
            &[1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.8],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        );
        let res = run(&bs, &mut EmaCross::new(2, 3), &cfg(10_000.0, 0.0, 0.0));
        assert_eq!(res.metrics.trades, 2, "expected entry and exit");
    }

    #[test]
    fn clamped_sell_lets_a_repeated_signal_rebalance() {
        // Target 0.5% with 100 bps slippage cannot be partially held from a
        // full position: the sell clamp liquidates everything. A repeated
        // 0.005 signal must then re-trigger a buy instead of being suppressed.
        let bs = bars(&[10.0, 20.0, 30.0, 40.0], &[15.0, 25.0, 35.0, 45.0]);
        let res = run(
            &bs,
            &mut Scripted::new(vec![1.0, 0.005, 0.005]),
            &cfg(10_000.0, 0.0, 100.0),
        );
        assert_eq!(res.metrics.trades, 3, "buy, clamped sell, then re-buy");
    }

    #[test]
    fn walk_forward_splits_by_month_windows() {
        let bars = vec![
            bar_at(ts(2024, 1, 15), 10.0, 10.0),
            bar_at(ts(2024, 3, 15), 10.0, 11.0),
            bar_at(ts(2024, 7, 15), 11.0, 12.0),
            bar_at(ts(2024, 9, 15), 12.0, 13.0),
            bar_at(ts(2025, 1, 15), 13.0, 14.0),
            bar_at(ts(2025, 3, 15), 14.0, 15.0),
        ];
        let res = walk_forward(&bars, &mut || Box::new(BuyHold), &cfg(10_000.0, 0.0, 0.0), 6);
        assert_eq!(res.windows.len(), 3);
        let counts: Vec<usize> =
            res.windows.iter().map(|w| w.metrics.bars).collect();
        assert_eq!(counts, vec![2, 2, 2]);
        assert_eq!(res.windows[0].start, ts(2024, 1, 15));
        assert_eq!(res.windows[1].start, ts(2024, 7, 15));
        assert_eq!(res.windows[2].end, ts(2025, 3, 15));
    }

    #[test]
    fn walk_forward_windows_run_independently() {
        // Window 1 rises, window 2 falls; each window's return is its own,
        // measured from reset capital. Both buckets span a full 6 months.
        let bars = vec![
            bar_at(ts(2024, 1, 15), 10.0, 10.0),
            bar_at(ts(2024, 6, 15), 10.0, 20.0),
            bar_at(ts(2024, 7, 15), 20.0, 20.0),
            bar_at(ts(2024, 12, 15), 20.0, 10.0),
        ];
        let res = walk_forward(&bars, &mut || Box::new(BuyHold), &cfg(10_000.0, 0.0, 0.0), 6);
        assert_eq!(res.windows.len(), 2);
        approx(res.windows[0].metrics.net_return, 1.0);
        approx(res.windows[1].metrics.net_return, -0.5);
        approx(res.mean_return, 0.25);
        approx(res.return_stdev, 1.06066017);
        approx(res.worst_return, -0.5);
        assert_eq!(res.positive_windows, 1);
    }

    #[test]
    fn walk_forward_excludes_partial_tail_from_aggregates() {
        let bars = vec![
            bar_at(ts(2024, 1, 15), 10.0, 10.0),
            bar_at(ts(2024, 6, 15), 10.0, 20.0), // full bucket Jan..Jun
            bar_at(ts(2024, 7, 15), 20.0, 20.0),
            bar_at(ts(2024, 12, 15), 20.0, 10.0), // full bucket Jul..Dec
            bar_at(ts(2025, 1, 15), 10.0, 10.0),  // partial tail, single bar
        ];
        let res = walk_forward(&bars, &mut || Box::new(BuyHold), &cfg(10_000.0, 0.0, 0.0), 6);
        assert_eq!(res.windows.len(), 3);
        assert!(!res.windows[0].partial);
        assert!(!res.windows[1].partial);
        assert!(res.windows[2].partial);
        // Aggregates come from the two full windows only.
        approx(res.mean_return, 0.25);
        approx(res.worst_return, -0.5);
        assert_eq!(res.positive_windows, 1);
    }

    #[test]
    fn walk_forward_empty_bars_give_empty_windows() {
        let res = walk_forward(&[], &mut || Box::new(BuyHold), &cfg(10_000.0, 0.0, 0.0), 6);
        assert!(res.windows.is_empty());
        assert_eq!(res.mean_return, 0.0);
        assert_eq!(res.return_stdev, 0.0);
    }

    #[test]
    #[should_panic(expected = "initial_cash")]
    fn portfolio_rejects_nonpositive_cash() {
        let _ = Portfolio::new(0.0, 1.0, 5.0);
    }

    #[test]
    #[should_panic(expected = "slippage_bps")]
    fn portfolio_rejects_extreme_slippage() {
        let _ = Portfolio::new(100.0, 1.0, 10_000.0);
    }

    #[test]
    #[should_panic(expected = "window_months")]
    fn walk_forward_rejects_zero_window() {
        let bs = vec![bar_at(ts(2024, 1, 15), 10.0, 10.0)];
        let _ = walk_forward(&bs, &mut || Box::new(BuyHold), &cfg(10_000.0, 0.0, 0.0), 0);
    }

    #[test]
    fn momentum_follows_trend_sign() {
        let mut up = Momentum::new(3);
        let rising: Vec<Bar> = (0..6).map(|i| bar_at(ts(2024, 1, 1 + i), 10.0 + i as f64, 10.0 + i as f64)).collect();
        let mut last = 0.0;
        for b in &rising {
            last = up.on_close(b, 0.0, 0.0);
        }
        approx(last, 1.0);

        let mut down = Momentum::new(3);
        let falling: Vec<Bar> = (0..6).map(|i| bar_at(ts(2024, 1, 1 + i), 20.0 - i as f64, 20.0 - i as f64)).collect();
        for b in &falling {
            last = down.on_close(b, 0.0, 0.0);
        }
        approx(last, 0.0);
    }

    #[test]
    fn mean_reversion_buys_below_sma() {
        let mut mr = MeanReversion::new(3);
        let mut last = 0.0;
        for c in [10.0, 10.0, 10.0, 8.0] {
            last = mr.on_close(&bar_at(ts(2024, 1, 1), 10.0, c), 0.0, 0.0);
        }
        approx(last, 1.0);
        last = mr.on_close(&bar_at(ts(2024, 1, 2), 10.0, 12.0), 0.0, 0.0);
        approx(last, 0.0);
    }

    #[test]
    fn vol_target_scales_with_realized_volatility() {
        // Alternating +2% / -2% moves -> annualized vol well above 15%, so the
        // weight must be scaled down below 1.
        let mut vt = VolTarget::new(4, 0.15);
        let mut last = 0.0;
        for (i, c) in [100.0, 102.0, 100.0, 102.0, 100.0, 102.0].iter().enumerate() {
            last = vt.on_close(&bar_at(ts(2024, 1, 1 + i as u32), *c, *c), 0.0, 0.0);
        }
        assert!(last > 0.0 && last < 1.0, "got {last}");
        // Flat prices -> zero vol -> fully invested.
        let mut flat = VolTarget::new(4, 0.15);
        for i in 0..6 {
            last = flat.on_close(&bar_at(ts(2024, 1, 1 + i as u32), 10.0, 10.0), 0.0, 0.0);
        }
        approx(last, 1.0);
    }

    #[test]
    fn same_inputs_same_outputs() {
        let bs = bars(&[10.0, 11.0, 12.0, 11.5], &[10.5, 11.5, 12.0, 11.8]);
        let a = run(&bs, &mut EmaCross::new(2, 3), &cfg(10_000.0, 1.0, 5.0));
        let b = run(&bs, &mut EmaCross::new(2, 3), &cfg(10_000.0, 1.0, 5.0));
        assert_eq!(a, b);
    }
}
