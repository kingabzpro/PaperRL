//! History sync shared by the CLI and the desktop dashboard's data button.
//!
//! One symbol at a time: fetch, merge into the Parquet store, report what
//! happened. Failures are collected instead of aborting the run, so a single
//! bad ticker does not stop a watchlist sync.

use std::path::Path;

use chrono::{Duration, Utc};
use paperrl_alpaca::AlpacaClient;
use paperrl_core::{AssetClass, Symbol, Timeframe};
use paperrl_data::BarStore;

/// The Alpaca crypto feed caps a single request at 150 days and only serves
/// daily bars, so crypto history ignores the requested span beyond that.
pub const CRYPTO_MAX_DAYS: u32 = 150;

/// What one symbol's sync did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncedSymbol {
    pub symbol: String,
    pub fetched: usize,
    pub added: usize,
    pub stored: usize,
}

/// Result of a full sync run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncReport {
    pub synced: Vec<SyncedSymbol>,
    pub failed: Vec<(String, String)>,
}

impl SyncReport {
    /// Bars added across every symbol.
    pub fn bars_added(&self) -> usize {
        self.synced.iter().map(|s| s.added).sum()
    }

    /// True when nothing was attempted or every symbol failed.
    pub fn is_empty(&self) -> bool {
        self.synced.is_empty() && self.failed.is_empty()
    }

    /// One line for a banner: counts plus the first failure, if any.
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "Nothing to sync: add tickers first.".to_string();
        }
        let symbols = self.synced.len();
        let failed = self.failed.len();
        let mut out = format!(
            "{symbols} symbol{} synced, {} new bar{}",
            if symbols == 1 { "" } else { "s" },
            self.bars_added(),
            if self.bars_added() == 1 { "" } else { "s" }
        );
        if failed > 0 {
            out.push_str(&format!(" · {failed} failed"));
        }
        out
    }
}

/// Fetches history for each tag and merges it into `<datasets>`.
///
/// Equities honour the requested timeframe and year span. Crypto pairs are
/// daily only and capped at [`CRYPTO_MAX_DAYS`]; a non-daily timeframe for a
/// crypto pair is reported as a failure instead of silently storing nothing.
pub async fn sync_history(
    client: &AlpacaClient,
    datasets: &Path,
    symbols: &[String],
    tf: Timeframe,
    years: u32,
) -> SyncReport {
    let store = BarStore::new(datasets);
    let end = Utc::now();
    let start = end - Duration::days(365 * i64::from(years.clamp(1, 20)));
    let mut report = SyncReport::default();

    for raw in symbols {
        let Ok(symbol) = Symbol::parse(raw) else {
            report.failed.push((
                raw.clone(),
                "not a valid ticker: use AAPL style or BTC/USD".into(),
            ));
            continue;
        };
        let fetched = match symbol.asset_class() {
            AssetClass::UsEquity => client.bars(&symbol, tf, start, end).await,
            AssetClass::Crypto => {
                if tf != Timeframe::Day {
                    report.failed.push((
                        symbol.to_string(),
                        "crypto history is daily only; pick the 1day timeframe".into(),
                    ));
                    continue;
                }
                let days = (years.clamp(1, 20) * 365).min(CRYPTO_MAX_DAYS);
                client.crypto_bars(&symbol, days).await
            }
        };
        match fetched {
            Ok(bars) => match store.write(&symbol, tf, &bars) {
                Ok((stored, added)) => report.synced.push(SyncedSymbol {
                    symbol: symbol.to_string(),
                    fetched: bars.len(),
                    added,
                    stored,
                }),
                Err(e) => report.failed.push((symbol.to_string(), e.to_string())),
            },
            Err(e) => report.failed.push((symbol.to_string(), e.to_string())),
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_counts_symbols_and_failures() {
        let report = SyncReport {
            synced: vec![
                SyncedSymbol {
                    symbol: "AAPL".into(),
                    fetched: 500,
                    added: 500,
                    stored: 500,
                },
                SyncedSymbol {
                    symbol: "MSFT".into(),
                    fetched: 500,
                    added: 1,
                    stored: 600,
                },
            ],
            failed: vec![("BAD".into(), "not a valid ticker".into())],
        };
        assert_eq!(
            report.summary(),
            "2 symbols synced, 501 new bars · 1 failed"
        );
        assert_eq!(report.bars_added(), 501);
    }

    #[test]
    fn empty_report_has_a_usable_summary() {
        let report = SyncReport::default();
        assert!(report.is_empty());
        assert_eq!(report.summary(), "Nothing to sync: add tickers first.");
    }
}
