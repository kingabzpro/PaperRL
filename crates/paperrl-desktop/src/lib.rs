//! Data access and presentation helpers for the PaperRL desktop dashboard.
//! All heavy lifting goes through the same crates the CLI uses, so the UI and
//! the CLI share one engine.

pub mod theme;

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use paperrl_core::{Bar, Symbol, Timeframe};
use paperrl_data::BarStore;
use paperrl_rl::{ModelMeta, PpoStrategy};
use paperrl_sim::{
    BuyHold, EmaCross, MeanReversion, Momentum, SimConfig, Strategy, VolTarget, WalkForwardResult,
};

/// One stored dataset discovered on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub symbol: String,
    pub timeframe: String,
    pub bars: usize,
    pub first: DateTime<Utc>,
    pub last: DateTime<Utc>,
}

fn tf_from_slug(slug: &str) -> Option<Timeframe> {
    match slug {
        "1min" => Some(Timeframe::Min1),
        "5min" => Some(Timeframe::Min5),
        "15min" => Some(Timeframe::Min15),
        "1hour" => Some(Timeframe::Hour1),
        "1day" => Some(Timeframe::Day),
        _ => None,
    }
}

/// Parses a symbol tag from a filename: equities parse directly; crypto
/// caches are stored as slugs (BTC-USD), so retry with `-` back to `/`.
fn parse_symbol_flexible(tag: &str) -> Result<Symbol, paperrl_core::Error> {
    if let Ok(s) = Symbol::parse(tag) {
        return Ok(s);
    }
    Symbol::parse(&tag.replace('-', "/"))
}

/// Scans `<root>/<timeframe>/<SYMBOL>.parquet` files and summarises them.
pub fn list_datasets(root: &Path) -> Vec<Dataset> {
    let mut out = Vec::new();
    let Ok(tfs) = std::fs::read_dir(root) else {
        return out;
    };
    for tf_entry in tfs.flatten() {
        if !tf_entry.path().is_dir() {
            continue;
        }
        let tf_name = tf_entry.file_name().to_string_lossy().to_string();
        let Some(tf) = tf_from_slug(&tf_name) else {
            continue;
        };
        let Ok(files) = std::fs::read_dir(tf_entry.path()) else {
            continue;
        };
        for f in files.flatten() {
            let p = f.path();
            if p.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }
            let sym = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
            let Ok(symbol) = parse_symbol_flexible(&sym) else {
                continue;
            };
            if let Ok(bars) = BarStore::new(root).read(&symbol, tf)
                && let (Some(first), Some(last)) = (bars.first(), bars.last()) {
                    out.push(Dataset {
                        symbol: symbol.to_string(),
                        timeframe: tf_name.clone(),
                        bars: bars.len(),
                        first: first.timestamp,
                        last: last.timestamp,
                    });
                }
        }
    }
    out.sort_by(|a, b| a.symbol.cmp(&b.symbol).then(a.timeframe.cmp(&b.timeframe)));
    out
}

pub fn load_bars(root: &Path, symbol: &str, tf_slug: &str) -> Result<Vec<Bar>, String> {
    let symbol = parse_symbol_flexible(symbol).map_err(|e| e.to_string())?;
    let tf = tf_from_slug(tf_slug).ok_or_else(|| format!("unknown timeframe {tf_slug}"))?;
    BarStore::new(root).read(&symbol, tf).map_err(|e| e.to_string())
}

pub fn periods_per_year(tf: Timeframe) -> f64 {
    match tf {
        Timeframe::Min1 => 252.0 * 390.0,
        Timeframe::Min5 => 252.0 * 78.0,
        Timeframe::Min15 => 252.0 * 26.0,
        Timeframe::Hour1 => 252.0 * 6.5,
        Timeframe::Day => 252.0,
    }
}

/// Simulator parameters exposed in the Strategy Lab.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimParams {
    pub cash: f64,
    pub fee_bps: f64,
    pub slippage_bps: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        Self { cash: 100_000.0, fee_bps: 1.0, slippage_bps: 5.0 }
    }
}

impl SimParams {
    pub fn to_sim_config(self, tf: Timeframe) -> SimConfig {
        SimConfig {
            initial_cash: self.cash,
            fee_bps: self.fee_bps,
            slippage_bps: self.slippage_bps,
            periods_per_year: periods_per_year(tf),
        }
    }
}

/// One row of the strategy comparison table.
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestRow {
    pub name: String,
    pub net_return: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub exposure: f64,
    pub trades: usize,
    pub fees: f64,
}

/// Builds the deterministic baseline strategies plus, when supplied, a
/// trained PPO model.
pub fn default_strategies(ppo_model: Option<&Path>) -> Vec<Box<dyn Strategy>> {
    let mut strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(BuyHold),
        Box::new(EmaCross::new(10, 50)),
        Box::new(Momentum::new(10)),
        Box::new(MeanReversion::new(10)),
        Box::new(VolTarget::new(10, 0.15)),
    ];
    if let Some(path) = ppo_model
        && let Ok(strategy) = PpoStrategy::load(path) {
            strategies.push(Box::new(strategy));
        }
    strategies
}

/// Runs every strategy on the same bars with identical costs.
pub fn run_comparison(
    bars: &[Bar],
    tf: Timeframe,
    params: SimParams,
    ppo_model: Option<&Path>,
) -> Vec<BacktestRow> {
    let cfg = params.to_sim_config(tf);
    default_strategies(ppo_model)
        .into_iter()
        .map(|mut s| {
            let m = paperrl_sim::run(bars, s.as_mut(), &cfg).metrics;
            BacktestRow {
                name: s.name().to_string(),
                net_return: m.net_return,
                sharpe: m.sharpe,
                max_drawdown: m.max_drawdown,
                win_rate: m.win_rate,
                exposure: m.avg_exposure,
                trades: m.trades,
                fees: m.total_fees,
            }
        })
        .collect()
}

/// Walk-forward results for the deterministic baselines.
pub fn run_walk_forward(
    bars: &[Bar],
    tf: Timeframe,
    params: SimParams,
    months: u32,
) -> Vec<(String, WalkForwardResult)> {
    let cfg = params.to_sim_config(tf);
    let mut out = Vec::new();
    for name in ["buyhold", "ema", "momentum", "meanrev", "voltarget"] {
        let res = paperrl_sim::walk_forward(
            bars,
            &mut || -> Box<dyn Strategy> {
                match name {
                    "buyhold" => Box::new(BuyHold),
                    "ema" => Box::new(EmaCross::new(10, 50)),
                    "momentum" => Box::new(Momentum::new(10)),
                    "meanrev" => Box::new(MeanReversion::new(10)),
                    _ => Box::new(VolTarget::new(10, 0.15)),
                }
            },
            &cfg,
            months,
        );
        out.push((name.to_string(), res));
    }
    out
}

/// A model checkpoint with its sidecar metadata.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub model_path: PathBuf,
    pub meta: ModelMeta,
}

/// Lists `*.json` sidecars in the models directory; the matching `.bpk` sits
/// beside each sidecar.
pub fn list_models(dir: &Path) -> Vec<ModelEntry> {
    let mut out = Vec::new();
    let Ok(files) = std::fs::read_dir(dir) else {
        return out;
    };
    for f in files.flatten() {
        let p = f.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        let Ok(meta) = serde_json::from_str::<ModelMeta>(&text) else {
            continue;
        };
        out.push(ModelEntry { model_path: p.with_extension("bpk"), meta });
    }
    out.sort_by(|a, b| {
        a.meta
            .symbol
            .cmp(&b.meta.symbol)
            .then(a.meta.config.seed.cmp(&b.meta.config.seed))
    });
    out
}

/// Geometry for one candlestick in SVG coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Candle {
    pub x: f64,
    pub center: f64,
    pub width: f64,
    pub wick_top: f64,
    pub wick_bottom: f64,
    pub body_top: f64,
    pub body_bottom: f64,
    pub up: bool,
    pub volume_top: f64,
    pub volume_height: f64,
}

/// Full chart geometry for a slice of bars.
#[derive(Debug, Clone, PartialEq)]
pub struct ChartGeom {
    pub candles: Vec<Candle>,
    pub width: f64,
    pub height: f64,
    pub min: f64,
    pub max: f64,
    /// Top of the price area and bottom of the price area.
    pub price_top: f64,
    pub price_bottom: f64,
    /// Top and height of the volume strip.
    pub volume_top: f64,
    pub volume_height: f64,
}

/// Maps bars into SVG candle coordinates with a price area and a volume strip.
pub fn chart_geometry(bars: &[Bar], width: f64, height: f64) -> ChartGeom {
    const LEFT_PAD: f64 = 8.0;
    const RIGHT_PAD: f64 = 8.0;
    const TOP_PAD: f64 = 8.0;
    let price_top = TOP_PAD;
    let price_bottom = height * 0.74;
    let volume_top = height * 0.78;
    let volume_height = height * 0.18;

    let min = bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
    let max = bars.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(1e-9);
    let max_volume = bars.iter().map(|b| b.volume).fold(0.0_f64, f64::max).max(1e-9);
    let plot_width = (width - LEFT_PAD - RIGHT_PAD).max(1.0);
    let slot = plot_width / bars.len().max(1) as f64;
    let candle_width = (slot * 0.62).max(0.5);

    let y_price = |price: f64| -> f64 {
        price_top + (max - price) / span * (price_bottom - price_top)
    };

    let candles = bars
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let center = LEFT_PAD + slot * (i as f64 + 0.5);
            let vol_h = b.volume / max_volume * volume_height;
            Candle {
                x: center - candle_width / 2.0,
                center,
                width: candle_width,
                wick_top: y_price(b.high),
                wick_bottom: y_price(b.low),
                body_top: y_price(b.open.max(b.close)),
                body_bottom: y_price(b.open.min(b.close)),
                up: b.close >= b.open,
                volume_top: volume_top + volume_height - vol_h,
                volume_height: vol_h,
            }
        })
        .collect();

    ChartGeom {
        candles,
        width,
        height,
        min,
        max,
        price_top,
        price_bottom,
        volume_top,
        volume_height,
    }
}

/// SVG polyline points for a sparkline over `values`, oldest first.
pub fn sparkline_points(values: &[f64], width: f64, height: f64) -> String {
    if values.is_empty() {
        return String::new();
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(1e-9);
    let n = values.len();
    values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = if n == 1 { width / 2.0 } else { i as f64 / (n - 1) as f64 * width };
            let y = height - (v - min) / span * height;
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Formats a dollar value with thousands separators.
pub fn group_thousands(v: f64, decimals: usize) -> String {
    let s = format!("{:.*}", decimals, v);
    let (int_part, frac) = match s.split_once('.') {
        Some((i, f)) => (i.to_string(), Some(f.to_string())),
        None => (s, None),
    };
    let (sign, digits) = match int_part.strip_prefix('-') {
        Some(rest) => ("-", rest.to_string()),
        None => ("", int_part),
    };
    let mut grouped = String::new();
    for (i, ch) in digits.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            grouped.insert(0, ',');
        }
        grouped.insert(0, ch);
    }
    match frac {
        Some(f) => format!("{sign}${grouped}.{f}"),
        None => format!("{sign}${grouped}"),
    }
}

pub fn fmt_money(v: f64) -> String {
    group_thousands(v, 0)
}

pub fn fmt_money_cents(v: f64) -> String {
    group_thousands(v, 2)
}

pub fn fmt_pct(v: f64) -> String {
    format!("{:+.2}%", v * 100.0)
}

pub fn fmt_price(v: f64) -> String {
    format!("{:.2}", v)
}

/// Renders a USD amount in the base currency. `fx` is units of base per
/// 1 USD; without a rate (or for USD itself) the value stays in dollars.
pub fn fmt_money_display(amount_usd: f64, base: &str, fx: Option<f64>) -> String {
    let base = paperrl_core::normalize_currency(base).unwrap_or("USD");
    match (base, fx) {
        ("USD", _) => paperrl_core::fmt_money(amount_usd, "USD"),
        (_, Some(rate)) if rate > 0.0 => paperrl_core::fmt_money(amount_usd * rate, base),
        _ => paperrl_core::fmt_money(amount_usd, "USD"),
    }
}

/// Splits ticker input on commas, spaces, and newlines; uppercases, drops
/// empties, and dedupes while preserving order.
pub fn parse_watchlist_input(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in s.split([',', ' ', '\n', '\r', '\t']) {
        let tag = raw.trim().to_uppercase();
        if !tag.is_empty() && !out.contains(&tag) {
            out.push(tag);
        }
    }
    out
}

/// Like [`fmt_money_display`] but without cents, for stat cards where cent
/// digits would crowd the box.
pub fn fmt_money_display_short(amount_usd: f64, base: &str, fx: Option<f64>) -> String {
    let base = paperrl_core::normalize_currency(base).unwrap_or("USD");
    match (base, fx) {
        ("USD", _) => paperrl_core::fmt_money_dp(amount_usd, "USD", 0),
        (_, Some(rate)) if rate > 0.0 => paperrl_core::fmt_money_dp(amount_usd * rate, base, 0),
        _ => paperrl_core::fmt_money_dp(amount_usd, "USD", 0),
    }
}

pub fn fmt_date(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%d").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Bar {
        Bar {
            timestamp: DateTime::from_timestamp(0, 0).unwrap(),
            open,
            high,
            low,
            close,
            volume,
        }
    }

    #[test]
    fn chart_geometry_scales_prices_and_volumes() {
        let bars = vec![
            bar(10.0, 11.0, 9.0, 10.5, 100.0),
            bar(10.5, 12.0, 10.0, 11.0, 200.0),
            bar(11.0, 11.5, 8.0, 9.0, 100.0),
        ];
        let g = chart_geometry(&bars, 300.0, 200.0);
        assert_eq!(g.candles.len(), 3);
        assert_eq!(g.min, 8.0);
        assert_eq!(g.max, 12.0);
        // Candles are ordered left to right and never overlap.
        assert!(g.candles[0].center < g.candles[1].center);
        assert!(g.candles[1].center < g.candles[2].center);
        // The highest wick maps to the top of the price area.
        assert!((g.candles[1].wick_top - g.price_top).abs() < 1e-6);
        // The lowest low maps to the bottom of the price area.
        assert!((g.candles[2].wick_bottom - g.price_bottom).abs() < 1e-6);
        // Up bars: body top above body bottom.
        assert!(g.candles[0].up);
        assert!(!g.candles[2].up);
        // Volume 200 is twice volume 100.
        let h1 = g.candles[0].volume_height;
        let h2 = g.candles[1].volume_height;
        assert!((h2 / h1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn money_grouping_adds_separators() {
        assert_eq!(group_thousands(1234567.0, 0), "$1,234,567");
        assert_eq!(group_thousands(999.0, 0), "$999");
        assert_eq!(group_thousands(12.5, 2), "$12.50");
        assert_eq!(group_thousands(-1000.25, 2), "-$1,000.25");
    }

    #[test]
    fn sparkline_points_order_and_extremes() {
        let pts = sparkline_points(&[1.0, 2.0, 3.0], 100.0, 50.0);
        let parts: Vec<&str> = pts.split(' ').collect();
        assert_eq!(parts.len(), 3);
        let ys: Vec<f64> = parts
            .iter()
            .map(|p| p.split(',').nth(1).unwrap().parse().unwrap())
            .collect();
        // Rising values plot downward (SVG y grows down): y decreases here
        // because the first value is the minimum and maps to the bottom.
        assert!(ys[0] > ys[1] && ys[1] > ys[2]);
        assert_eq!(sparkline_points(&[], 100.0, 50.0), "");
    }

    #[test]
    fn list_datasets_survives_missing_root() {
        assert!(list_datasets(Path::new("Z:/definitely/not/here")).is_empty());
    }

    #[test]
    fn symbol_parse_flexible_handles_slugs() {
        // crypto slug round-trips through the filename form
        let s = parse_symbol_flexible("BTC-USD").unwrap();
        assert_eq!(s.as_str(), "BTC/USD");
        // real equity hyphen tags parse directly first
        let s = parse_symbol_flexible("BF-B").unwrap();
        assert_eq!(s.as_str(), "BF-B");
        assert!(parse_symbol_flexible("NOT-A-TICKER").is_err());
    }

    #[test]
    fn display_converts_with_fx() {
        assert_eq!(fmt_money_display(100.0, "EUR", Some(0.9)), "\u{20ac}90.00");
    }

    #[test]
    fn display_usd_ignores_fx() {
        assert_eq!(fmt_money_display(100.0, "USD", Some(0.9)), "$100.00");
    }

    #[test]
    fn display_falls_back_to_usd_without_fx() {
        assert_eq!(fmt_money_display(100.0, "EUR", None), "$100.00");
    }

    #[test]
    fn display_short_drops_cents() {
        assert_eq!(fmt_money_display_short(20000.0, "USD", None), "$20,000");
        assert_eq!(fmt_money_display_short(100.0, "EUR", Some(0.9)), "€90");
        assert_eq!(fmt_money_display_short(100.0, "EUR", None), "$100");
    }

    #[test]
    fn watchlist_input_parses() {
        assert_eq!(
            parse_watchlist_input("aapl, btc/usd  TSLA\nmsft"),
            vec!["AAPL", "BTC/USD", "TSLA", "MSFT"]
        );
        assert_eq!(parse_watchlist_input("AAPL,,AAPL"), vec!["AAPL"]);
        assert_eq!(parse_watchlist_input("   "), Vec::<String>::new());
    }
}
