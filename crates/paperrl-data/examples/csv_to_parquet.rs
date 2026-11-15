//! Import a CSV of daily bars into the local Parquet store.
//!
//! Lets the CLI run without Alpaca credentials, and exercises the same
//! `BarStore::write` path that `paperrl data sync` uses. The CSV header must be
//! `timestamp,open,high,low,close,volume` where `timestamp` is either an
//! ISO-8601 date (`2024-01-31`) or Unix seconds.
//!
//! Usage:
//!     cargo run -p paperrl-data --example csv_to_parquet -- \
//!         datasets/csv/SPY.csv SPY 1d datasets

use std::path::PathBuf;

use chrono::DateTime;
use paperrl_core::{Bar, Symbol, Timeframe};
use paperrl_data::BarStore;

fn timestamp(raw: &str) -> Result<DateTime<chrono::Utc>, String> {
    let raw = raw.trim();
    if let Ok(secs) = raw.parse::<i64>() {
        return DateTime::from_timestamp(secs, 0).ok_or_else(|| format!("bad unix ts {raw}"));
    }
    let date = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d")
        .map_err(|e| format!("bad date {raw}: {e}"))?;
    Ok(date.and_hms_opt(0, 0, 0).expect("midnight is valid").and_utc())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: csv_to_parquet <csv> <symbol> <timeframe> <dataset-root>");
        std::process::exit(2);
    }
    let csv_path = PathBuf::from(&args[1]);
    let symbol = Symbol::parse(&args[2])?;
    let tf = Timeframe::parse(&args[3])?;
    let store = BarStore::new(PathBuf::from(&args[4]));

    let text = std::fs::read_to_string(&csv_path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or("empty csv")?;
    let cols: Vec<&str> = header.split(',').map(|c| c.trim()).collect();
    let expected = ["timestamp", "open", "high", "low", "close", "volume"];
    if cols != expected {
        return Err(format!("header {cols:?} does not match {expected:?}").into());
    }

    let mut bars = Vec::new();
    for (n, line) in lines.enumerate() {
        let f: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        if f.len() != 6 {
            return Err(format!("line {}: expected 6 columns, got {}", n + 2, f.len()).into());
        }
        let num = |i: usize| -> Result<f64, String> {
            f[i].parse::<f64>().map_err(|e| format!("line {}: {}: {e}", n + 2, f[i]))
        };
        let bar = Bar {
            timestamp: timestamp(f[0])?,
            open: num(1)?,
            high: num(2)?,
            low: num(3)?,
            close: num(4)?,
            volume: num(5)?,
        };
        if bar.high < bar.low {
            return Err(format!("line {}: high below low", n + 2).into());
        }
        bars.push(bar);
    }
    if bars.windows(2).any(|w| w[1].timestamp < w[0].timestamp) {
        return Err("csv rows are not in ascending timestamp order".into());
    }

    let (stored, added) = store.write(&symbol, tf, &bars)?;
    println!(
        "{symbol} {tf}: imported {} bars ({added} added), {stored} stored at {}",
        bars.len(),
        store.path(&symbol, tf).display()
    );
    Ok(())
}
