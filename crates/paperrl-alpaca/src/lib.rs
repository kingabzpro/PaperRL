//! Alpaca adapter: market data and the paper-trading account.
//!
//! The trading client is always constructed against the paper API: PaperRL
//! never talks to the live-money endpoint.

use alpacars::data::enums::{Adjustment, DataFeed, MostActivesBy, TimeFrame, TimeFrameUnit};
use alpacars::data::historical::screener::{
    MarketMoversRequest, MostActivesRequest, ScreenerClient,
};
use alpacars::data::historical::stock::{StockBarsRequest, StockHistoricalDataClient};
use alpacars::trading::client::TradingClient;
use alpacars::trading::enums::{OrderSide, QueryOrderStatus};
use alpacars::trading::requests::{GetOrdersRequest, OrderRequest};
use chrono::{DateTime, Utc};
use paperrl_core::{Bar, Error, Result, Symbol, Timeframe};

/// Stringifies SDK errors into the shared error type.
// ponytail: SDK errors become Error::Other strings; add typed variants when a caller needs to match on them.
fn other<E: std::fmt::Display>(e: E) -> Error {
    Error::Other(e.to_string())
}

/// Alpaca API credentials from the standard environment variables.
pub struct Credentials {
    pub key_id: String,
    pub secret_key: String,
}

impl Credentials {
    /// Reads APCA_API_KEY_ID and APCA_API_SECRET.
    pub fn from_env() -> Result<Self> {
        Self::from_parts(
            std::env::var("APCA_API_KEY_ID").ok(),
            std::env::var("APCA_API_SECRET").ok(),
        )
    }

    pub fn from_parts(key_id: Option<String>, secret_key: Option<String>) -> Result<Self> {
        match (key_id, secret_key) {
            (Some(key_id), Some(secret_key)) => Ok(Self { key_id, secret_key }),
            (None, _) => Err(Error::MissingCredentials("APCA_API_KEY_ID is not set".into())),
            (_, None) => Err(Error::MissingCredentials("APCA_API_SECRET is not set".into())),
        }
    }
}

/// The part of the account worth showing on screen.
#[derive(Debug)]
pub struct AccountSummary {
    pub status: String,
    pub equity: Option<String>,
    pub cash: Option<String>,
    pub buying_power: Option<String>,
    pub portfolio_value: Option<String>,
}

/// One screener quote with its percent move as a fraction.
#[derive(Debug, Clone, PartialEq)]
pub struct MoverQuote {
    pub symbol: String,
    pub pct_change: f64,
    pub price: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MoversView {
    pub gainers: Vec<MoverQuote>,
    pub losers: Vec<MoverQuote>,
}

/// Paper account position summary.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionView {
    pub symbol: String,
    pub qty: f64,
    pub market_value: f64,
    pub unrealized_pl: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSideView {
    Buy,
    Sell,
}

/// Result of a submitted paper order.
#[derive(Debug, Clone, PartialEq)]
pub struct SubmittedOrder {
    pub client_order_id: String,
    pub status: String,
}

/// Client for Alpaca market data and the paper account.
pub struct AlpacaClient {
    trading: TradingClient,
    data: StockHistoricalDataClient,
    screener: ScreenerClient,
}

impl AlpacaClient {
    pub fn new(creds: &Credentials) -> Result<Self> {
        Ok(Self {
            // Paper API only: PaperRL never submits live-money orders.
            trading: TradingClient::new(&creds.key_id, &creds.secret_key, true).map_err(other)?,
            data: StockHistoricalDataClient::new(Some(&creds.key_id), Some(&creds.secret_key), false)
                .map_err(other)?,
            screener: ScreenerClient::new(Some(&creds.key_id), Some(&creds.secret_key))
                .map_err(other)?,
        })
    }

    /// Fetches the paper account summary.
    pub async fn account(&self) -> Result<AccountSummary> {
        let acct = self.trading.get_account().await.map_err(other)?;
        Ok(AccountSummary {
            status: format!("{:?}", acct.status),
            equity: acct.equity,
            cash: acct.cash,
            buying_power: acct.buying_power,
            portfolio_value: acct.portfolio_value,
        })
    }

    /// Fetches bars for one symbol over [start, end]; alpacars paginates automatically.
    // ponytail: Split-only adjustment; switch to Adjustment::All (adds dividend effects) when research demands total-return series.
    pub async fn bars(
        &self,
        symbol: &Symbol,
        tf: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Bar>> {
        let req = StockBarsRequest {
            symbols: vec![symbol.as_str().to_string()],
            timeframe: Some(to_alpaca_timeframe(tf)),
            start: Some(start),
            end: Some(end),
            adjustment: Some(Adjustment::Split),
            // The free Basic plan only serves the IEX feed; the API defaults
            // to SIP and returns 403 without this.
            // ponytail: make the feed configurable when a paid data plan is used.
            feed: Some(DataFeed::Iex),
            ..Default::default()
        };
        let set = self.data.get_stock_bars(&req).await.map_err(other)?;
        Ok(set
            .get(symbol.as_str())
            .into_iter()
            .flatten()
            .map(|b| Bar {
                timestamp: b.timestamp,
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close,
                volume: b.volume,
            })
            .collect())
    }
}

impl AlpacaClient {
    /// Today's market movers (gainers and losers) from the screener.
    pub async fn movers(&self, top: u32) -> Result<MoversView> {
        let req = MarketMoversRequest { market_type: None, top: Some(top) };
        let movers = self.screener.get_market_movers(&req).await.map_err(other)?;
        let quote = |m: &alpacars::data::models::Mover| MoverQuote {
            symbol: m.symbol.clone(),
            pct_change: m.percent_change.unwrap_or(0.0) / 100.0,
            price: m.price.unwrap_or(0.0),
        };
        Ok(MoversView {
            gainers: movers.gainers.iter().map(quote).collect(),
            losers: movers.losers.iter().map(quote).collect(),
        })
    }

    /// Most active symbols by volume.
    pub async fn most_actives(&self, top: u32) -> Result<Vec<String>> {
        let req = MostActivesRequest { by: Some(MostActivesBy::Volume), top: Some(top) };
        let actives = self.screener.get_most_actives(&req).await.map_err(other)?;
        Ok(actives.most_actives.into_iter().map(|a| a.symbol).collect())
    }

    /// Open paper positions.
    pub async fn positions(&self) -> Result<Vec<PositionView>> {
        let positions = self.trading.get_all_positions().await.map_err(other)?;
        Ok(positions
            .into_iter()
            .map(|p| PositionView {
                symbol: p.symbol,
                qty: p.qty.parse().unwrap_or(0.0),
                market_value: p
                    .market_value
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0),
                unrealized_pl: p.unrealized_pl.and_then(|v| v.parse().ok()).unwrap_or(0.0),
            })
            .collect())
    }

    /// Symbols with open (unfilled or partially filled) paper orders.
    pub async fn open_orders(&self) -> Result<Vec<String>> {
        let req = GetOrdersRequest { status: Some(QueryOrderStatus::Open), ..Default::default() };
        let orders = self.trading.get_orders(Some(&req)).await.map_err(other)?;
        Ok(orders.into_iter().filter_map(|o| o.symbol).collect())
    }

    /// Submits a market order for a share quantity. Only the paper API is
    /// constructed, so this can never reach live-money endpoints.
    pub async fn submit_market_order(
        &self,
        symbol: &str,
        side: OrderSideView,
        qty: f64,
        client_order_id: &str,
    ) -> Result<SubmittedOrder> {
        let side = match side {
            OrderSideView::Buy => OrderSide::Buy,
            OrderSideView::Sell => OrderSide::Sell,
        };
        let mut req = OrderRequest::market(symbol, side, format!("{qty:.4}"));
        req.client_order_id = Some(client_order_id.to_string());
        let order = self.trading.submit_order(&req).await.map_err(other)?;
        Ok(SubmittedOrder {
            client_order_id: order.client_order_id,
            status: format!("{:?}", order.status),
        })
    }

    /// Submits a market order sized in dollars (fractional shares).
    pub async fn submit_market_notional_order(
        &self,
        symbol: &str,
        notional: f64,
        client_order_id: &str,
    ) -> Result<SubmittedOrder> {
        let mut req = OrderRequest::market_notional(symbol, OrderSide::Buy, format!("{notional:.2}"));
        req.client_order_id = Some(client_order_id.to_string());
        let order = self.trading.submit_order(&req).await.map_err(other)?;
        Ok(SubmittedOrder {
            client_order_id: order.client_order_id,
            status: format!("{:?}", order.status),
        })
    }
}

/// Maps PaperRL timeframes to Alpaca wire values (1Day, 5Min, 1Hour, ...).
fn to_alpaca_timeframe(tf: Timeframe) -> TimeFrame {
    match tf {
        Timeframe::Min1 => TimeFrame::minute(),
        Timeframe::Min5 => TimeFrame::new(5, TimeFrameUnit::Minute),
        Timeframe::Min15 => TimeFrame::new(15, TimeFrameUnit::Minute),
        Timeframe::Hour1 => TimeFrame::hour(),
        Timeframe::Day => TimeFrame::day(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_reject_missing_parts() {
        assert!(Credentials::from_parts(None, Some("s".into())).is_err());
        assert!(Credentials::from_parts(Some("k".into()), None).is_err());
        assert!(Credentials::from_parts(Some("k".into()), Some("s".into())).is_ok());
    }

    #[test]
    fn timeframes_map_to_alpaca_wire_values() {
        assert_eq!(to_alpaca_timeframe(Timeframe::Day), TimeFrame::day());
        assert_eq!(to_alpaca_timeframe(Timeframe::Min1).value(), "1Min");
        assert_eq!(to_alpaca_timeframe(Timeframe::Min5).value(), "5Min");
        assert_eq!(to_alpaca_timeframe(Timeframe::Min15).value(), "15Min");
        assert_eq!(to_alpaca_timeframe(Timeframe::Hour1).value(), "1Hour");
    }
}
