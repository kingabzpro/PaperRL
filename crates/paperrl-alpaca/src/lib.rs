//! Alpaca adapter: market data and the paper-trading account.
//!
//! The trading client is always constructed against the paper API: PaperRL
//! never talks to the live-money endpoint.

use alpacars::common::client::RestClient;
use alpacars::data::enums::{Adjustment, CryptoFeed, DataFeed, MostActivesBy, TimeFrame, TimeFrameUnit};
use alpacars::data::historical::crypto::{CryptoBarsRequest, CryptoHistoricalDataClient};
use alpacars::data::historical::screener::{
    MarketMoversRequest, MostActivesRequest, ScreenerClient,
};
use alpacars::data::historical::stock::{StockBarsRequest, StockHistoricalDataClient};
use alpacars::trading::client::TradingClient;
use alpacars::trading::enums::{ActivityType, AssetStatus, OrderSide, QueryOrderStatus};
use alpacars::trading::requests::{
    GetAccountActivitiesRequest, GetAssetsRequest, GetOrdersRequest, GetPortfolioHistoryRequest,
    OrderRequest,
};
use chrono::{DateTime, Utc};
use paperrl_core::{Bar, Error, Result, Symbol, Timeframe};
use serde_json::Value;

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
#[derive(Debug, Clone)]
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
    pub avg_entry_price: f64,
    pub cost_basis: f64,
    pub current_price: f64,
    pub unrealized_plpc: f64,
    pub asset_class: String,
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

/// Market clock: whether the US equity market is open right now.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockView {
    pub timestamp: DateTime<Utc>,
    pub is_open: bool,
    pub next_open: DateTime<Utc>,
    pub next_close: DateTime<Utc>,
}

/// One point of the account equity curve.
#[derive(Debug, Clone, PartialEq)]
pub struct EquityPoint {
    pub timestamp: DateTime<Utc>,
    pub equity: f64,
    pub profit_loss: f64,
    pub profit_loss_pct: f64,
}

/// One order fill from the account activity feed.
#[derive(Debug, Clone, PartialEq)]
pub struct FillView {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub qty: f64,
    pub price: f64,
    pub time: DateTime<Utc>,
}

/// One tradable asset known to the account.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetView {
    pub symbol: String,
    pub asset_class: String,
    pub tradable: bool,
    pub active: bool,
}

/// Client for Alpaca market data and the paper account.
#[derive(Clone)]
pub struct AlpacaClient {
    trading: TradingClient,
    data: StockHistoricalDataClient,
    screener: ScreenerClient,
    crypto: CryptoHistoricalDataClient,
    fx: RestClient,
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
            crypto: CryptoHistoricalDataClient::new(Some(&creds.key_id), Some(&creds.secret_key))
                .map_err(other)?,
            fx: RestClient::new(
                Some(creds.key_id.clone()),
                Some(creds.secret_key.clone()),
                None,
                alpacars::common::client::base_url::DATA.to_string(),
                "v1beta1".to_string(),
                false,
            )
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
            .map(|p| {
                position_from_parts(
                    &p.symbol,
                    &p.qty,
                    p.market_value.as_deref(),
                    p.unrealized_pl.as_deref(),
                    &p.cost_basis,
                    &p.avg_entry_price,
                    p.current_price.as_deref(),
                    p.unrealized_plpc.as_deref(),
                    asset_class_str(p.asset_class),
                )
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
        // Lossless quantity: shortest representation that round-trips the f64, so a
        // fractional position (notional buys) is never oversold by formatting.
        let mut req = OrderRequest::market(symbol, side, format!("{qty}"));
        req.client_order_id = Some(client_order_id.to_string());
        let order = self.trading.submit_order(&req).await.map_err(other)?;
        Ok(SubmittedOrder {
            client_order_id: order.client_order_id,
            status: format!("{:?}", order.status),
        })
    }

    /// Submits a market order sized in dollars (fractional shares, either side).
    pub async fn submit_market_notional_order(
        &self,
        symbol: &str,
        notional: f64,
        side: OrderSideView,
        client_order_id: &str,
    ) -> Result<SubmittedOrder> {
        let req = notional_request(symbol, notional, side, client_order_id);
        let order = self.trading.submit_order(&req).await.map_err(other)?;
        Ok(SubmittedOrder {
            client_order_id: order.client_order_id,
            status: format!("{:?}", order.status),
        })
    }
}

impl AlpacaClient {
    /// US equity market clock.
    pub async fn clock(&self) -> Result<ClockView> {
        let c = self.trading.get_clock().await.map_err(other)?;
        Ok(ClockView {
            timestamp: c.timestamp,
            is_open: c.is_open,
            next_open: c.next_open,
            next_close: c.next_close,
        })
    }

    /// Account equity curve over a period (1D, 1W, 1M, 3M, 1A).
    pub async fn portfolio_history(&self, period: &str) -> Result<Vec<EquityPoint>> {
        let filter = GetPortfolioHistoryRequest {
            period: Some(period.to_string()),
            ..Default::default()
        };
        let h = self.trading.get_portfolio_history(Some(&filter)).await.map_err(other)?;
        let point = |i: usize| -> Option<EquityPoint> {
            Some(EquityPoint {
                timestamp: DateTime::from_timestamp(h.timestamp.get(i).copied()?, 0)?,
                equity: h.equity.get(i).copied().flatten()?,
                profit_loss: h.profit_loss.get(i).copied().flatten()?,
                profit_loss_pct: h.profit_loss_pct.get(i).copied().flatten()?,
            })
        };
        Ok((0..h.timestamp.len()).filter_map(point).collect())
    }

    /// Recent fill activities, newest first.
    pub async fn fills(&self, limit: u32) -> Result<Vec<FillView>> {
        let req = GetAccountActivitiesRequest {
            page_size: Some(limit),
            ..Default::default()
        };
        let vals = self
            .trading
            .get_account_activities_by_type(&ActivityType::Fill, Some(&req))
            .await
            .map_err(other)?;
        Ok(vals.into_iter().filter_map(fill_from_value).collect())
    }

    /// Tradable assets known to the account (both equities and crypto).
    pub async fn assets(&self) -> Result<Vec<AssetView>> {
        let list = self
            .trading
            .get_all_assets(None::<&GetAssetsRequest>)
            .await
            .map_err(other)?;
        Ok(list
            .into_iter()
            .map(|a| AssetView {
                symbol: a.symbol,
                asset_class: asset_class_str(a.asset_class).to_string(),
                tradable: a.tradable,
                active: a.status == AssetStatus::Active,
            })
            .collect())
    }

    /// Daily bars for a crypto pair (BTC/USD), direct from the crypto feed.
    pub async fn crypto_bars(&self, symbol: &Symbol, days: u32) -> Result<Vec<Bar>> {
        let end = Utc::now();
        let start = end - chrono::Duration::days(days as i64);
        let req = CryptoBarsRequest {
            symbols: vec![symbol.as_str().to_string()],
            timeframe: Some(TimeFrame::day()),
            start: Some(start),
            end: Some(end),
            loc: Some(CryptoFeed::Us),
            ..Default::default()
        };
        let set = self.crypto.get_crypto_bars(&req).await.map_err(other)?;
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

    /// Units of `base` per 1 USD from the FX rates feed (display only).
    pub async fn fx_rate(&self, base: &str) -> Result<f64> {
        #[derive(serde::Serialize)]
        struct Q<'a> {
            base: &'a str,
            currencies: &'a str,
        }
        let v: Value = self
            .fx
            .get("/forex/latest/rates", Some(&Q { base, currencies: "USD" }))
            .await
            .map_err(other)?;
        fx_from_value(v)
    }

    /// Cancels open orders this app created (client order ids starting with
    /// `prefix`); orders placed elsewhere are never touched.
    pub async fn cancel_bot_orders(&self, prefix: &str) -> Result<usize> {
        let req = GetOrdersRequest { status: Some(QueryOrderStatus::Open), ..Default::default() };
        let orders = self.trading.get_orders(Some(&req)).await.map_err(other)?;
        let mut n = 0;
        for o in orders {
            if o.client_order_id.starts_with(prefix) {
                self.trading.cancel_order_by_id(&o.id).await.map_err(other)?;
                n += 1;
            }
        }
        Ok(n)
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

/// Parses a decimal-string field from the API, warning when the value is unreadable
/// so silent zeros never reach the portfolio view.
fn parse_amount(value: Option<&str>, field: &str, symbol: &str) -> f64 {
    match value.and_then(|v| v.parse().ok()) {
        Some(x) => x,
        None => {
            tracing::warn!("position {symbol}: unreadable {field}");
            0.0
        }
    }
}

#[allow(clippy::too_many_arguments)] // mirrors the nine API string fields it maps
fn position_from_parts(
    symbol: &str,
    qty: &str,
    market_value: Option<&str>,
    unrealized_pl: Option<&str>,
    cost_basis: &str,
    avg_entry_price: &str,
    current_price: Option<&str>,
    unrealized_plpc: Option<&str>,
    asset_class: &str,
) -> PositionView {
    PositionView {
        symbol: symbol.to_string(),
        qty: parse_amount(Some(qty), "qty", symbol),
        market_value: parse_amount(market_value, "market_value", symbol),
        unrealized_pl: parse_amount(unrealized_pl, "unrealized_pl", symbol),
        cost_basis: parse_amount(Some(cost_basis), "cost_basis", symbol),
        avg_entry_price: parse_amount(Some(avg_entry_price), "avg_entry_price", symbol),
        current_price: parse_amount(current_price, "current_price", symbol),
        unrealized_plpc: parse_amount(unrealized_plpc, "unrealized_plpc", symbol),
        asset_class: asset_class.to_string(),
    }
}

/// Stable lowercase wire names for SDK asset classes ("us_equity", "crypto").
fn asset_class_str(class: alpacars::trading::enums::AssetClass) -> &'static str {
    use alpacars::trading::enums::AssetClass as C;
    match class {
        C::UsEquity => "us_equity",
        C::Crypto | C::CryptoPerp => "crypto",
        _ => "other",
    }
}

fn notional_request(
    symbol: &str,
    notional: f64,
    side: OrderSideView,
    client_order_id: &str,
) -> OrderRequest {
    let side = match side {
        OrderSideView::Buy => OrderSide::Buy,
        OrderSideView::Sell => OrderSide::Sell,
    };
    let mut req = OrderRequest::market_notional(symbol, side, format!("{notional:.2}"));
    req.client_order_id = Some(client_order_id.to_string());
    req
}

/// Converts the FX response into units of the base currency per 1 USD
/// (the API serves USD units per 1 base).
fn fx_from_value(v: serde_json::Value) -> Result<f64> {
    let usd = v
        .get("rates")
        .and_then(|r| r.get("USD"))
        .and_then(|x| x.as_f64())
        .ok_or_else(|| Error::Other("fx response missing USD rate".into()))?;
    if usd <= 0.0 {
        return Err(Error::Other("fx USD rate not positive".into()));
    }
    Ok(1.0 / usd)
}

/// Reads a fill from the raw activity JSON; non-FILL activities and
/// malformed entries return None so one bad line cannot break the page.
fn fill_from_value(v: serde_json::Value) -> Option<FillView> {
    if v.get("activity_type").and_then(|t| t.as_str()) != Some("FILL") {
        return None;
    }
    let num = |key: &str| -> Option<f64> {
        v.get(key).and_then(|x| {
            x.as_str()
                .and_then(|s| s.parse().ok())
                .or_else(|| x.as_f64())
        })
    };
    Some(FillView {
        id: v.get("id")?.as_str()?.to_string(),
        symbol: v.get("symbol")?.as_str()?.to_string(),
        side: v.get("side")?.as_str()?.to_string(),
        qty: num("qty")?,
        price: num("price")?,
        time: v.get("transaction_time")?.as_str()?.parse().ok()?,
    })
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

    #[test]
    fn position_view_maps_full_fields() {
        let p = position_from_parts(
            "BTC/USD",
            "0.25",
            Some("27500.00"),
            Some("2500.00"),
            "25000.00",
            "100000.00",
            Some("110000.00"),
            Some("0.10"),
            "crypto",
        );
        assert_eq!(p.symbol, "BTC/USD");
        assert_eq!(p.qty, 0.25);
        assert_eq!(p.market_value, 27500.0);
        assert_eq!(p.unrealized_pl, 2500.0);
        assert_eq!(p.cost_basis, 25000.0);
        assert_eq!(p.avg_entry_price, 100000.0);
        assert_eq!(p.current_price, 110000.0);
        assert_eq!(p.unrealized_plpc, 0.10);
        assert_eq!(p.asset_class, "crypto");
    }

    #[test]
    fn position_view_tolerates_missing_optionals() {
        let p = position_from_parts(
            "AAPL", "2", None, None, "500.00", "250.00", None, None, "us_equity",
        );
        assert_eq!(p.market_value, 0.0);
        assert_eq!(p.unrealized_pl, 0.0);
        assert_eq!(p.current_price, 0.0);
        assert_eq!(p.unrealized_plpc, 0.0);
        assert_eq!(p.asset_class, "us_equity");
    }

    #[test]
    fn notional_order_carries_side() {
        let req = notional_request("AAPL", 500.0, OrderSideView::Sell, "paperrl-x-0");
        assert_eq!(req.symbol, "AAPL");
        assert!(matches!(req.side, OrderSide::Sell));
        assert_eq!(req.notional.as_deref(), Some("500.00"));
        assert_eq!(req.client_order_id.as_deref(), Some("paperrl-x-0"));
    }

    #[test]
    fn fx_rate_inverts_usd_pair() {
        let v = serde_json::json!({"base": "EUR", "rates": {"USD": 1.0865}});
        let rate = fx_from_value(v).unwrap();
        assert!((rate - 1.0 / 1.0865).abs() < 1e-12);
    }

    #[test]
    fn fx_rate_rejects_missing_usd() {
        let v = serde_json::json!({"base": "EUR", "rates": {}});
        assert!(fx_from_value(v).is_err());
    }

    #[test]
    fn fill_json_maps_to_view() {
        let v = serde_json::json!({
            "id": "2024-01-01T00:00:00Z::abc",
            "activity_type": "FILL",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "1.5",
            "price": "189.43",
            "transaction_time": "2024-01-02T15:30:00Z"
        });
        let f = fill_from_value(v).unwrap();
        assert_eq!(f.symbol, "AAPL");
        assert_eq!(f.side, "buy");
        assert_eq!(f.qty, 1.5);
        assert_eq!(f.price, 189.43);
    }

    #[test]
    fn fill_json_skips_non_fill() {
        let v = serde_json::json!({"activity_type": "DIV", "symbol": "AAPL"});
        assert!(fill_from_value(v).is_none());
    }
}
