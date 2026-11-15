//! Local bar storage: one Parquet file per (timeframe, symbol) under a dataset root.

use arrow::array::{Float64Array, RecordBatch, TimestampMillisecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use chrono::DateTime;
use paperrl_core::{Bar, Error, Result, Symbol, Timeframe};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

/// Stringifies arrow/parquet errors into the shared error type.
// ponytail: SDK errors become Error::Other strings; add typed variants when a caller needs to match on them.
fn other<E: std::fmt::Display>(e: E) -> Error {
    Error::Other(e.to_string())
}

/// App data directory for a given working directory, given the injected
/// `PAPERRL_HOME` value and user home. Order: env override, then a repo
/// checkout (a `datasets` folder beside the working directory), then
/// `<user home>/paperrl`.
pub fn paperrl_home_from(
    cwd: &std::path::Path,
    paperrl_home_env: Option<&str>,
    user_home: Option<&std::path::Path>,
) -> PathBuf {
    if let Some(home) = paperrl_home_env.map(str::trim).filter(|h| !h.is_empty()) {
        return PathBuf::from(home);
    }
    if cwd.join("datasets").is_dir() {
        return cwd.to_path_buf();
    }
    user_home
        .map(|h| h.join("paperrl"))
        .unwrap_or_else(|| PathBuf::from("paperrl"))
}

/// App data directory for the current process: reads `PAPERRL_HOME` and
/// the platform user-home variable, then applies [`paperrl_home_from`].
pub fn paperrl_home() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let env = std::env::var("PAPERRL_HOME").ok();
    let user_home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from);
    paperrl_home_from(&cwd, env.as_deref(), user_home.as_deref())
}

fn bar_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into())),
            false,
        ),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ])
}

fn to_batch(bars: &[Bar]) -> Result<RecordBatch> {
    let ts = TimestampMillisecondArray::from_iter_values(bars.iter().map(|b| b.timestamp.timestamp_millis()))
        .with_timezone("+00:00");
    let open = Float64Array::from_iter_values(bars.iter().map(|b| b.open));
    let high = Float64Array::from_iter_values(bars.iter().map(|b| b.high));
    let low = Float64Array::from_iter_values(bars.iter().map(|b| b.low));
    let close = Float64Array::from_iter_values(bars.iter().map(|b| b.close));
    let volume = Float64Array::from_iter_values(bars.iter().map(|b| b.volume));
    RecordBatch::try_new(
        Arc::new(bar_schema()),
        vec![
            Arc::new(ts),
            Arc::new(open),
            Arc::new(high),
            Arc::new(low),
            Arc::new(close),
            Arc::new(volume),
        ],
    )
    .map_err(other)
}

fn batch_to_bars(batch: &RecordBatch) -> Result<Vec<Bar>> {
    let ts = batch
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .expect("timestamp column");
    let open = batch.column(1).as_any().downcast_ref::<Float64Array>().expect("open column");
    let high = batch.column(2).as_any().downcast_ref::<Float64Array>().expect("high column");
    let low = batch.column(3).as_any().downcast_ref::<Float64Array>().expect("low column");
    let close = batch.column(4).as_any().downcast_ref::<Float64Array>().expect("close column");
    let volume = batch.column(5).as_any().downcast_ref::<Float64Array>().expect("volume column");
    (0..batch.num_rows())
        .map(|i| {
            Ok(Bar {
                timestamp: DateTime::from_timestamp_millis(ts.value(i))
                    .ok_or_else(|| Error::Other("invalid bar timestamp".into()))?,
                open: open.value(i),
                high: high.value(i),
                low: low.value(i),
                close: close.value(i),
                volume: volume.value(i),
            })
        })
        .collect()
}

/// Dataset root: `<root>/<timeframe-slug>/<SYMBOL>.parquet`.
// ponytail: no locking; a single writer process is assumed. Add a lockfile if the CLI runs concurrently.
pub struct BarStore {
    root: PathBuf,
}

impl BarStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Dataset file for one symbol and timeframe.
    pub fn path(&self, symbol: &Symbol, tf: Timeframe) -> PathBuf {
        self.root.join(tf.slug()).join(format!("{symbol}.parquet"))
    }

    /// Writes bars, merging with anything already stored: fetched values win on
    /// timestamp collisions, output is sorted and deduplicated. Returns
    /// `(stored, added)` bar counts. The file is replaced atomically, so a
    /// crash mid-write leaves the previous dataset intact.
    // ponytail: full rewrite per sync; add range appends if minute bars get slow.
    // When adjustment windows shrink, re-sync the full stored range: split-adjusted
    // prices are only consistent across one adjustment epoch.
    pub fn write(&self, symbol: &Symbol, tf: Timeframe, bars: &[Bar]) -> Result<(usize, usize)> {
        let existing = self.read(symbol, tf)?;
        let existing_n = existing.len();
        let merged = merge_bars(existing, bars.to_vec());
        let path = self.path(symbol, tf);
        std::fs::create_dir_all(path.parent().expect("path has a parent"))?;
        let tmp = path.with_extension("parquet.tmp");
        let mut writer =
            ArrowWriter::try_new(File::create(&tmp)?, Arc::new(bar_schema()), None).map_err(other)?;
        writer.write(&to_batch(&merged)?).map_err(other)?;
        writer.close().map_err(other)?;
        std::fs::rename(&tmp, &path)?;
        Ok((merged.len(), merged.len() - existing_n))
    }

    /// Reads all stored bars, oldest first. Empty vec when nothing is stored yet.
    pub fn read(&self, symbol: &Symbol, tf: Timeframe) -> Result<Vec<Bar>> {
        let path = self.path(symbol, tf);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?).map_err(other)?;
        let actual: Vec<&str> =
            builder.schema().fields().iter().map(|f| f.name().as_str()).collect();
        let schema = bar_schema();
        let expected: Vec<&str> =
            schema.fields().iter().map(|f| f.name().as_str()).collect();
        if actual != expected {
            return Err(Error::Other(format!(
                "{}: dataset columns {actual:?} do not match expected {expected:?}",
                path.display()
            )));
        }
        let mut bars = Vec::new();
        for batch in builder.build().map_err(other)? {
            bars.extend(batch_to_bars(&batch.map_err(other)?)?);
        }
        if bars.windows(2).any(|w| w[1].timestamp < w[0].timestamp) {
            return Err(Error::Other(format!(
                "{}: bars are not sorted by timestamp",
                path.display()
            )));
        }
        Ok(bars)
    }
}

/// Merges fetched bars into stored ones: fetched wins on timestamp collisions,
/// result sorted oldest first with no duplicate timestamps. Keys are
/// millisecond-precision so sub-second bars stay distinct.
pub fn merge_bars(existing: Vec<Bar>, fetched: Vec<Bar>) -> Vec<Bar> {
    let mut by_ts: BTreeMap<i64, Bar> = existing
        .into_iter()
        .map(|b| (b.timestamp.timestamp_millis(), b))
        .collect();
    by_ts.extend(fetched.into_iter().map(|b| (b.timestamp.timestamp_millis(), b)));
    by_ts.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts: i64, close: f64) -> Bar {
        Bar {
            timestamp: DateTime::from_timestamp(ts, 0).unwrap(),
            open: close,
            high: close,
            low: close,
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn merge_bars_prefers_fetched_and_sorts() {
        let existing = vec![bar(1, 10.0), bar(2, 11.0)];
        let fetched = vec![bar(2, 12.0), bar(3, 13.0)];
        let merged = merge_bars(existing, fetched);
        let ts: Vec<i64> = merged.iter().map(|b| b.timestamp.timestamp()).collect();
        assert_eq!(ts, vec![1, 2, 3]);
        assert_eq!(merged[1].close, 12.0);
    }

    #[test]
    fn merge_bars_handles_empty_sides() {
        assert!(merge_bars(vec![], vec![]).is_empty());
        assert_eq!(merge_bars(vec![bar(1, 10.0)], vec![]).len(), 1);
        assert_eq!(merge_bars(vec![], vec![bar(1, 10.0)]).len(), 1);
    }

    #[test]
    fn merge_bars_keeps_distinct_subsecond_bars() {
        let a = Bar { timestamp: DateTime::from_timestamp_millis(1000).unwrap(), ..bar(1, 10.0) };
        let b = Bar { timestamp: DateTime::from_timestamp_millis(1500).unwrap(), ..bar(1, 11.0) };
        assert_eq!(merge_bars(vec![a], vec![b]).len(), 2);
    }

    #[test]
    fn dataset_path_uses_slug_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = BarStore::new(dir.path());
        let sym = Symbol::parse("aapl").unwrap();
        let p = store.path(&sym, Timeframe::Day);
        assert_eq!(p.file_name().unwrap().to_str().unwrap(), "AAPL.parquet");
        assert!(p.to_string_lossy().contains("1day"));
    }

    #[test]
    fn parquet_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = BarStore::new(dir.path());
        let sym = Symbol::parse("aapl").unwrap();
        let bars = vec![bar(1, 10.0), bar(2, 11.0)];
        store.write(&sym, Timeframe::Day, &bars).unwrap();
        assert_eq!(store.read(&sym, Timeframe::Day).unwrap(), bars);
        assert!(!dir.path().join("1day").join("AAPL.parquet.tmp").exists());
    }

    #[test]
    fn read_missing_dataset_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = BarStore::new(dir.path());
        let sym = Symbol::parse("nvda").unwrap();
        assert_eq!(store.read(&sym, Timeframe::Day).unwrap(), vec![]);
    }

    #[test]
    fn read_rejects_unsorted_bars() {
        let dir = tempfile::tempdir().unwrap();
        let store = BarStore::new(dir.path());
        let sym = Symbol::parse("bad").unwrap();
        let path = store.path(&sym, Timeframe::Day);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Hand-write an unsorted file the same way `write` would.
        let mut w = parquet::arrow::ArrowWriter::try_new(
            std::fs::File::create(&path).unwrap(),
            std::sync::Arc::new(bar_schema()),
            None,
        )
        .unwrap();
        w.write(&to_batch(&[bar(2, 10.0), bar(1, 11.0)]).unwrap()).unwrap();
        w.close().unwrap();
        assert!(store.read(&sym, Timeframe::Day).is_err());
    }

    #[test]
    fn write_merges_with_existing() {
        let dir = tempfile::tempdir().unwrap();
        let store = BarStore::new(dir.path());
        let sym = Symbol::parse("msft").unwrap();
        let (stored, added) = store
            .write(&sym, Timeframe::Day, &[bar(1, 10.0), bar(2, 11.0)])
            .unwrap();
        assert_eq!((stored, added), (2, 2));
        let (stored, added) = store
            .write(&sym, Timeframe::Day, &[bar(2, 12.0), bar(3, 13.0)])
            .unwrap();
        assert_eq!((stored, added), (3, 1));
        let back = store.read(&sym, Timeframe::Day).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back[1].close, 12.0);
    }

    #[test]
    fn home_env_var_wins() {
        let cwd = tempfile::tempdir().unwrap();
        let env = tempfile::tempdir().unwrap();
        assert_eq!(
            paperrl_home_from(cwd.path(), Some(env.path().to_str().unwrap()), None),
            env.path().to_path_buf()
        );
    }

    #[test]
    fn home_blank_env_ignored() {
        let cwd = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        assert_eq!(
            paperrl_home_from(cwd.path(), Some("   "), Some(home.path())),
            home.path().join("paperrl")
        );
    }

    #[test]
    fn home_repo_checkout_wins_over_user_home() {
        let cwd = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(cwd.path().join("datasets")).unwrap();
        let home = tempfile::tempdir().unwrap();
        assert_eq!(
            paperrl_home_from(cwd.path(), None, Some(home.path())),
            cwd.path().to_path_buf()
        );
    }

    #[test]
    fn home_falls_back_to_user_dir() {
        let cwd = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        assert_eq!(
            paperrl_home_from(cwd.path(), None, Some(home.path())),
            home.path().join("paperrl")
        );
    }
}
