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

/// Which market a symbol belongs to. Crypto pairs use Alpaca's `BASE/QUOTE` tag style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetClass {
    UsEquity,
    Crypto,
}

/// A US equity symbol (1-6 uppercase letters plus `.`/`-`, e.g. BRK.B)
/// or a crypto pair (`BTC/USD`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(String);

impl Symbol {
    /// Normalizes case and surrounding whitespace; rejects malformed tags.
    /// Equities: 1-6 letters plus `.`/`-`, first and last a letter. Crypto: 2-8
    /// letters, a `/`, and a quote in USD | USDC | USDT | BTC.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_uppercase();
        if let Some((base, quote)) = s.split_once('/') {
            let b = base.as_bytes();
            let base_ok = (2..=8).contains(&b.len()) && b.iter().all(u8::is_ascii_uppercase);
            let quote_ok = matches!(quote, "USD" | "USDC" | "USDT" | "BTC");
            if base_ok && quote_ok {
                return Ok(Self(s));
            }
            return Err(Error::InvalidSymbol(s));
        }
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

    /// Equity tag or crypto pair.
    pub fn asset_class(&self) -> AssetClass {
        if self.0.contains('/') {
            AssetClass::Crypto
        } else {
            AssetClass::UsEquity
        }
    }

    /// Filesystem-safe name: crypto pairs replace `/` with `-` (BTC/USD -> BTC-USD).
    pub fn file_slug(&self) -> String {
        self.0.replace('/', "-")
    }
}

/// The currencies the app can display, as (code, symbol) pairs.
pub const CURRENCIES: &[(&str, &str)] = &[
    ("USD", "$"),
    ("EUR", "\u{20ac}"),
    ("GBP", "\u{a3}"),
    ("JPY", "\u{a5}"),
    ("CAD", "C$"),
    ("AUD", "A$"),
    ("CHF", "Fr"),
    ("SEK", "kr"),
    ("NOK", "kr"),
    ("DKK", "kr"),
    ("NZD", "NZ$"),
    ("HKD", "HK$"),
    ("SGD", "S$"),
    ("MXN", "Mex$"),
    ("PLN", "z\u{142}"),
    ("CZK", "K\u{10d}"),
    ("HUF", "Ft"),
    ("TRY", "\u{20ba}"),
];

/// Currency codes the app can display. Returns the canonical uppercase code.
pub fn normalize_currency(code: &str) -> Option<&'static str> {
    let up = code.trim().to_uppercase();
    CURRENCIES.iter().find(|(c, _)| *c == up).map(|(c, _)| *c)
}

fn group_thousands(digits: &str) -> String {
    let n = digits.len();
    let mut out = String::with_capacity(n + n / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (n - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Renders an amount with the currency's symbol; unknown codes prefix the code instead.
pub fn fmt_money(amount: f64, currency: &str) -> String {
    fmt_money_dp(amount, currency, 2)
}

/// Same as [`fmt_money`] with a chosen number of decimal places.
pub fn fmt_money_dp(amount: f64, currency: &str, decimals: usize) -> String {
    let up = currency.trim().to_uppercase();
    let known = CURRENCIES.iter().find(|(c, _)| *c == up);
    let (label, symbol) = match known {
        Some((c, sym)) => (*c, *sym),
        None => (up.as_str(), ""),
    };
    let neg = amount < 0.0;
    let fixed = if decimals == 0 {
        format!("{:.0}", amount.abs())
    } else {
        format!("{:.*}", decimals, amount.abs())
    };
    let body = match fixed.split_once('.') {
        Some((int, frac)) => format!("{}.{}", group_thousands(int), frac),
        None => group_thousands(&fixed),
    };
    if symbol.is_empty() {
        format!("{}{} {}", if neg { "-" } else { "" }, label, body)
    } else {
        format!("{}{}{}", if neg { "-" } else { "" }, symbol, body)
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

    #[test]
    fn symbol_parse_accepts_crypto_pairs() {
        let s = Symbol::parse("btc/usd").unwrap();
        assert_eq!(s.as_str(), "BTC/USD");
        assert!(matches!(s.asset_class(), AssetClass::Crypto));
        assert!(Symbol::parse("BTC/USDC").is_ok());
        assert!(Symbol::parse("ETH/USDT").is_ok());
        assert!(Symbol::parse("PEPE/USD").is_ok());
        assert!(Symbol::parse("WBTC/BTC").is_ok());
    }

    #[test]
    fn symbol_parse_rejects_bad_crypto() {
        assert!(Symbol::parse("BTC/EUR").is_err());
        assert!(Symbol::parse("BTC/USD/USDT").is_err());
        assert!(Symbol::parse("/USD").is_err());
        assert!(Symbol::parse("BTC/").is_err());
        assert!(Symbol::parse("LONGCOINN/USD").is_err());
        assert!(Symbol::parse("B/USD").is_err());
        assert!(Symbol::parse("BTC/US DX").is_err());
    }

    #[test]
    fn symbol_equity_rules_unchanged() {
        assert!(matches!(
            Symbol::parse("AAPL").unwrap().asset_class(),
            AssetClass::UsEquity
        ));
        assert!(Symbol::parse("BRK.B").is_ok());
        assert!(Symbol::parse("123456").is_err());
        assert!(Symbol::parse("AAP$L").is_err());
    }

    #[test]
    fn symbol_file_slug_is_fs_safe() {
        assert_eq!(Symbol::parse("BTC/USD").unwrap().file_slug(), "BTC-USD");
        assert_eq!(Symbol::parse("AAPL").unwrap().file_slug(), "AAPL");
    }

    #[test]
    fn normalize_currency_uppercases_known_codes() {
        assert_eq!(normalize_currency("eur"), Some("EUR"));
        assert_eq!(normalize_currency(" USD "), Some("USD"));
        assert_eq!(normalize_currency("xyz"), None);
        assert_eq!(normalize_currency(""), None);
    }

    #[test]
    fn fmt_money_dp_controls_decimals() {
        assert_eq!(fmt_money_dp(1234.5, "USD", 0), "$1,234");
        assert_eq!(fmt_money_dp(1234.5, "EUR", 0), "€1,234");
        assert_eq!(fmt_money_dp(1234.5, "USD", 2), "$1,234.50");
    }

    #[test]
    fn fmt_money_uses_currency_symbol() {
        assert_eq!(fmt_money(1234.5, "USD"), "$1,234.50");
        assert_eq!(fmt_money(1234.5, "EUR"), "\u{20ac}1,234.50");
        assert_eq!(fmt_money(1234.5, "CAD"), "C$1,234.50");
        assert_eq!(fmt_money(1234.5, "XYZ"), "XYZ 1,234.50");
        assert_eq!(fmt_money(-42.25, "USD"), "-$42.25");
        assert_eq!(fmt_money(0.0, "USD"), "$0.00");
    }
}
