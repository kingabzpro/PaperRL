//! Domain types shared by every PaperRL crate: bars, symbols, timeframes, errors.

use chrono::{DateTime, Utc};
use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid symbol: {0}")]
    InvalidSymbol(String),
    #[error("invalid timeframe: {0}")]
    InvalidTimeframe(String),
    #[error("missing credentials: {0}")]
    MissingCredentials(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One OHLCV bar at a fixed timestamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// A US equity symbol: 1-6 uppercase letters plus `.`/`-` (Alpaca style, e.g. BRK.B).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(String);

impl Symbol {
    /// Normalizes case and surrounding whitespace; rejects malformed tags.
    /// The first and last character must be a letter, so `AAPL.` and `-AAPL` fail.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_uppercase();
        let chars: Vec<char> = s.chars().collect();
        let valid = !chars.is_empty()
            && chars.len() <= 6
            && chars.iter().all(|c| c.is_ascii_uppercase() || *c == '.' || *c == '-')
            && chars.first().is_some_and(|c| c.is_ascii_uppercase())
            && chars.last().is_some_and(|c| c.is_ascii_uppercase());
        if !valid {
            return Err(Error::InvalidSymbol(s));
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Bar granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Timeframe {
    Min1,
    Min5,
    Min15,
    Hour1,
    Day,
}

impl Timeframe {
    /// Parses user-facing tags: 1m, 5m, 15m, 1h, 1d.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim() {
            "1m" => Ok(Self::Min1),
            "5m" => Ok(Self::Min5),
            "15m" => Ok(Self::Min15),
            "1h" => Ok(Self::Hour1),
            "1d" => Ok(Self::Day),
            other => Err(Error::InvalidTimeframe(other.to_string())),
        }
    }

    /// Filesystem-safe name for dataset directories (1min, 5min, 15min, 1hour, 1day).
    pub fn slug(self) -> &'static str {
        match self {
            Self::Min1 => "1min",
            Self::Min5 => "5min",
            Self::Min15 => "15min",
            Self::Hour1 => "1hour",
            Self::Day => "1day",
        }
    }
}

impl fmt::Display for Timeframe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_parse_normalizes_case_and_whitespace() {
        assert_eq!(Symbol::parse("aapl").unwrap().as_str(), "AAPL");
        assert_eq!(Symbol::parse(" msft ").unwrap().as_str(), "MSFT");
        assert_eq!(Symbol::parse("brk.b").unwrap().as_str(), "BRK.B");
    }

    #[test]
    fn symbol_parse_rejects_malformed() {
        assert!(Symbol::parse("").is_err());
        assert!(Symbol::parse("AA PL").is_err());
        assert!(Symbol::parse("WAYTOOLONG").is_err());
        assert!(Symbol::parse("AAP$L").is_err());
        assert!(Symbol::parse(".").is_err());
        assert!(Symbol::parse("-").is_err());
        assert!(Symbol::parse("AAPL.").is_err());
        assert!(Symbol::parse("-AAPL").is_err());
    }

    #[test]
    fn symbol_display_shows_normalized_tag() {
        assert_eq!(format!("{}", Symbol::parse("aapl").unwrap()), "AAPL");
    }

    #[test]
    fn timeframe_parse_maps_tags() {
        assert_eq!(Timeframe::parse("1d").unwrap(), Timeframe::Day);
        assert_eq!(Timeframe::parse("1h").unwrap(), Timeframe::Hour1);
        assert_eq!(Timeframe::parse("15m").unwrap(), Timeframe::Min15);
        assert_eq!(Timeframe::parse("5m").unwrap(), Timeframe::Min5);
        assert_eq!(Timeframe::parse("1m").unwrap(), Timeframe::Min1);
        assert_eq!(Timeframe::Day.slug(), "1day");
        assert_eq!(Timeframe::Min1.slug(), "1min");
        assert_eq!(Timeframe::Hour1.slug(), "1hour");
    }

    #[test]
    fn timeframe_parse_rejects_unknown() {
        assert!(Timeframe::parse("7x").is_err());
        assert!(Timeframe::parse("1w").is_err());
        assert!(Timeframe::parse("").is_err());
    }

    #[test]
    fn timeframe_display_uses_slug() {
        assert_eq!(format!("{}", Timeframe::Day), "1day");
    }
}
