//! PaperRL CLI: market data sync, local dataset inspection, Alpaca paper account.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use paperrl_alpaca::{AlpacaClient, Credentials};
use paperrl_core::{Bar, Symbol, Timeframe};
use paperrl_data::BarStore;
use paperrl_rl::{EnvConfig, PpoConfig, PpoStrategy, train_ppo};
use paperrl_trade::{
    AutoConfig, RiskLimits, ScreenFilter, collect_universe, ensure_universe_bars, rank_filtered,
    run_once,
};
use paperrl_sim::{
    BuyHold, EmaCross, MeanReversion, Momentum, SimConfig, Strategy, VolTarget,
};

#[derive(Parser)]
#[command(name = "paperrl", version, about = "Local-first RL trading research platform")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Local historical datasets
    Data {
        #[command(subcommand)]
        cmd: DataCmd,
    },
    /// Alpaca paper account and market data
    Alpaca {
        #[command(subcommand)]
        cmd: AlpacaCmd,
    },
    /// Automated paper trading: momentum screen plus risk-gated orders
    Trade {
        #[command(subcommand)]
        cmd: TradeCmd,
    },
    /// Train an RL agent locally on stored bars
    Train {
        #[arg(long)]
        symbol: String,
        /// Timeframe tag: 1m, 5m, 15m, 1h, 1d
        #[arg(long, default_value = "1d")]
        timeframe: String,
        /// PPO training iterations
        #[arg(long, default_value_t = 50)]
        iters: usize,
        /// Random seed for env sampling and reproducibility metadata
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Observation lookback in bars (must match backtests of this model)
        #[arg(long, default_value_t = 10)]
        window: usize,
        /// Output directory for the model checkpoint
        #[arg(long, default_value = "models")]
        out: PathBuf,
        /// Dataset root directory
        #[arg(long, default_value = "datasets")]
        datasets: PathBuf,
    },
    /// Backtest strategies on stored bars and compare metrics
    Backtest {
        #[arg(long)]
        symbol: String,
        /// Timeframe tag: 1m, 5m, 15m, 1h, 1d
        #[arg(long, default_value = "1d")]
        timeframe: String,
        /// Comma-separated strategies: buyhold, ema
        #[arg(long, value_delimiter = ',', default_values_t = vec![Strat::BuyHold, Strat::Ema])]
        strategies: Vec<Strat>,
        /// Fast EMA period (ema strategy)
        #[arg(long, default_value_t = 10)]
        fast: usize,
        /// Slow EMA period (ema strategy)
        #[arg(long, default_value_t = 50)]
        slow: usize,
        /// Starting cash in USD
        #[arg(long, default_value_t = 100_000.0)]
        cash: f64,
        /// Fee in basis points of traded notional
        #[arg(long, default_value_t = 1.0)]
        fee_bps: f64,
        /// Slippage in basis points on execution price
        #[arg(long, default_value_t = 5.0)]
        slippage_bps: f64,
        /// If > 0, evaluate in consecutive walk-forward windows of this many months
        #[arg(long, default_value_t = 0)]
        window_months: u32,
        /// Start date YYYY-MM-DD (inclusive)
        #[arg(long)]
        start: Option<String>,
        /// End date YYYY-MM-DD (inclusive)
        #[arg(long)]
        end: Option<String>,
        /// Model checkpoint (.bpk) for the ppo strategy
        #[arg(long)]
        model: Option<PathBuf>,
        /// Dataset root directory
        #[arg(long, default_value = "datasets")]
        out: PathBuf,
    },
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Strat {
    /// Buy at the first opportunity and hold
    #[value(name = "buyhold")]
    BuyHold,
    /// Enter when the fast EMA is above the slow one, exit below
    #[value(name = "ema")]
    Ema,
    /// Long while the trailing --fast-bar return is positive
    #[value(name = "momentum")]
    Momentum,
    /// Long when the close is below its --fast-bar SMA
    #[value(name = "meanrev")]
    MeanRev,
    /// Long with weight scaled to a 15% annualized volatility target
    #[value(name = "voltarget")]
    VolTarget,
    /// A trained PPO policy (requires --model)
    #[value(name = "ppo")]
    Ppo,
}

impl std::fmt::Display for Strat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_possible_value().expect("named variant").get_name())
    }
}

#[derive(Subcommand)]
enum TradeCmd {
    /// Screen the most active universe, rank momentum, optionally train this year's PPO models
    Scan {
        /// Universe size (most active symbols), 25 to 500
        #[arg(long, default_value_t = 100)]
        top: u32,
        /// Train PPO models for this many top gainers on bars since --since
        #[arg(long, default_value_t = 0)]
        train_top: usize,
        /// Training iterations per model
        #[arg(long, default_value_t = 150)]
        iters: usize,
        /// Seed for trained models (model files are ppo-SYMBOL-SEED.bpk)
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Training data start (YYYY-MM-DD); default is January 1 of the current year
        #[arg(long)]
        since: Option<String>,
        /// Minimum last price for the screen
        #[arg(long, default_value_t = 5.0)]
        min_price: f64,
        /// Minimum average daily dollar volume for the screen
        #[arg(long, default_value_t = 2_000_000.0)]
        min_dollar_vol: f64,
        #[arg(long, default_value = "datasets")]
        datasets: PathBuf,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
    },
    /// One automated pass: sell losing positions, buy top momentum within the risk gate
    Auto {
        #[arg(long, default_value_t = 100)]
        top: u32,
        /// How many top momentum candidates may be held long
        #[arg(long, default_value_t = 5)]
        long: usize,
        #[arg(long, default_value_t = 5)]
        max_positions: usize,
        /// Per-position cap as a fraction of equity
        #[arg(long, default_value_t = 0.2)]
        position_pct: f64,
        #[arg(long, default_value_t = 5000.0)]
        max_order: f64,
        /// Submit paper orders; without this flag the pass is a dry run
        #[arg(long)]
        execute: bool,
        /// Minimum last price for the screen
        #[arg(long, default_value_t = 5.0)]
        min_price: f64,
        /// Minimum average daily dollar volume for the screen
        #[arg(long, default_value_t = 2_000_000.0)]
        min_dollar_vol: f64,
        /// Seconds between passes; 0 runs a single pass
        #[arg(long, default_value_t = 0)]
        interval: u64,
        #[arg(long, default_value = "datasets")]
        datasets: PathBuf,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum DataCmd {
    /// Sync bars from Alpaca into the local Parquet store
    Sync {
        /// Comma-separated symbols, e.g. AAPL,MSFT
        #[arg(long, value_delimiter = ',')]
        symbols: Vec<String>,
        /// Timeframe tag: 1m, 5m, 15m, 1h, 1d
        #[arg(long, default_value = "1d")]
        timeframe: String,
        /// How many years of history to fetch
        #[arg(long, default_value_t = 3)]
        years: u32,
        /// Dataset root directory
        #[arg(long, default_value = "datasets")]
        out: PathBuf,
    },
    /// Print the newest stored bars for a symbol
    Show {
        #[arg(long)]
        symbol: String,
        #[arg(long, default_value = "1d")]
        timeframe: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long, default_value = "datasets")]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum AlpacaCmd {
    /// Show the paper trading account summary
    Account,
    /// Show open paper positions
    Positions,
    /// Fetch recent bars straight from the API without storing them
    Bars {
        #[arg(long)]
        symbol: String,
        #[arg(long, default_value = "1d")]
        timeframe: String,
        #[arg(long, default_value_t = 5)]
        days: u32,
    },
}

fn parse_symbols(raw: &[String]) -> Result<Vec<Symbol>> {
    raw.iter()
        .map(|s| Symbol::parse(s).map_err(anyhow::Error::from))
        .collect()
}

fn creds() -> Result<Credentials> {
    Credentials::from_env().context("set APCA_API_KEY_ID and APCA_API_SECRET to reach Alpaca")
}

fn print_candidates(header: &str, list: &[paperrl_trade::Candidate], worst_first: bool, limit: usize) {
    println!("{header}");
    println!("{:<8} {:>9} {:>8} {:>9} {:>10}", "symbol", "20d", "5d", "score", "last");
    let iter: Box<dyn Iterator<Item = &paperrl_trade::Candidate>> = if worst_first {
        Box::new(list.iter().rev())
    } else {
        Box::new(list.iter())
    };
    for c in iter.take(limit) {
        println!(
            "{:<8} {:>8.2}% {:>7.2}% {:>8.2}% {:>10.2}",
            c.symbol,
            c.ret20 * 100.0,
            c.ret5 * 100.0,
            c.score * 100.0,
            c.last_close
        );
    }
}

fn print_one(bar: &Bar) {
    println!(
        "{}  O {:.2}  H {:.2}  L {:.2}  C {:.2}  V {:.0}",
        bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume
    );
}

/// The `limit` newest bars, oldest first.
fn newest(bars: &[Bar], limit: usize) -> Vec<Bar> {
    bars.iter().rev().take(limit).rev().copied().collect()
}

/// Fetches and stores bars for one symbol; returns (fetched, added, stored).
async fn sync_one(
    client: &AlpacaClient,
    store: &BarStore,
    symbol: &Symbol,
    tf: Timeframe,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<(usize, usize, usize)> {
    let fetched = client.bars(symbol, tf, start, end).await?;
    let fetched_n = fetched.len();
    let (stored, added) = store.write(symbol, tf, &fetched)?;
    Ok((fetched_n, added, stored))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match Cli::parse().cmd {
        Cmd::Trade { cmd } => match cmd {
            TradeCmd::Scan {
                top,
                train_top,
                iters,
                seed,
                since,
                min_price,
                min_dollar_vol,
                datasets,
                model_dir,
            } => {
                anyhow::ensure!((25..=500).contains(&top), "--top must be in [25, 500]");
                let client = AlpacaClient::new(&creds()?)?;
                let universe = collect_universe(&client, top).await?;
                let (bars_by_symbol, skipped, errors) =
                    ensure_universe_bars(&client, &datasets, &universe, 5).await?;
                println!(
                    "universe {} symbols: {} with usable bars, {} skipped, {} errors",
                    universe.len(),
                    bars_by_symbol.len(),
                    skipped.len(),
                    errors.len()
                );
                let ranked = rank_filtered(
                    bars_by_symbol.iter().map(|(s, b)| (s.as_str(), b.as_slice())),
                    ScreenFilter { min_price, min_avg_dollar_volume: min_dollar_vol },
                );
                print_candidates("gaining momentum:", &ranked, false, 12);
                print_candidates("losing momentum (worst 10):", &ranked, true, 10);

                if train_top > 0 {
                    let since_dt = match since.as_deref() {
                        Some(s) => NaiveDate::parse_from_str(s, "%Y-%m-%d")?
                            .and_hms_opt(0, 0, 0)
                            .context("bad --since date")?
                            .and_utc(),
                        None => NaiveDate::from_ymd_opt(Utc::now().year(), 1, 1)
                            .expect("valid year")
                            .and_hms_opt(0, 0, 0)
                            .expect("valid time")
                            .and_utc(),
                    };
                    println!(
                        "training PPO on bars since {} for the top {train_top} gainers",
                        since_dt.date_naive()
                    );
                    std::fs::create_dir_all(&model_dir)?;
                    let mut trained = 0usize;
                    for c in ranked.iter().filter(|c| c.gainer()).take(train_top) {
                        let Some((_, all_bars)) =
                            bars_by_symbol.iter().find(|(s, _)| *s == c.symbol)
                        else {
                            continue;
                        };
                        let subset: Vec<Bar> =
                            all_bars.iter().filter(|b| b.timestamp >= since_dt).cloned().collect();
                        if subset.len() <= 12 {
                            println!("skip {}: only {} bars this year", c.symbol, subset.len());
                            continue;
                        }
                        let env_cfg = EnvConfig { window: 10, ..EnvConfig::default() };
                        let ppo_cfg = PpoConfig { iters, seed, ..PpoConfig::default() };
                        let trained_model =
                            train_ppo(&subset, env_cfg, ppo_cfg, &c.symbol, "1day");
                        let path = model_dir.join(format!("ppo-{}-{}.bpk", c.symbol, seed));
                        trained_model.save(&path)?;
                        println!(
                            "{}: {} bars this year -> {}",
                            c.symbol,
                            subset.len(),
                            path.display()
                        );
                        trained += 1;
                    }
                    println!("trained {trained} models");
                }
            }
            TradeCmd::Auto {
                top,
                long,
                max_positions,
                position_pct,
                max_order,
                execute,
                min_price,
                min_dollar_vol,
                interval,
                datasets,
                model_dir,
            } => {
                anyhow::ensure!((25..=500).contains(&top), "--top must be in [25, 500]");
                anyhow::ensure!(
                    (1..=100).contains(&long),
                    "--long must be in [1, 100]"
                );
                let client = AlpacaClient::new(&creds()?)?;
                let cfg = AutoConfig {
                    top,
                    long_count: long,
                    limits: RiskLimits {
                        max_positions,
                        position_pct,
                        max_order_notional: max_order,
                        min_order_notional: 50.0,
                    },
                    datasets,
                    model_dir,
                    filter: ScreenFilter { min_price, min_avg_dollar_volume: min_dollar_vol },
                    execute,
                    stale_days: 5,
                    universe: None,
                };
                loop {
                    let report = run_once(&client, &cfg).await?;
                    println!("=== automated pass: universe {} symbols ===", report.universe);
                    print_candidates("gaining momentum:", &report.gainers, false, 10);
                    print_candidates("losing momentum:", &report.losers, true, 10);
                    if report.plan.is_empty() {
                        println!("plan: no orders (nothing to buy or sell under the risk gate)");
                    } else {
                        println!("plan:");
                        for (side, detail) in &report.plan {
                            println!("  {:<16} {}", side, detail);
                        }
                    }
                    if cfg.execute {
                        println!("executed (paper):");
                        for e in &report.executed {
                            println!("  {e}");
                        }
                    } else {
                        println!("dry run: pass --execute to submit paper orders");
                    }
                    for e in &report.errors {
                        println!("error: {e}");
                    }
                    if interval == 0 {
                        break;
                    }
                    tracing::info!("sleeping {interval}s before the next pass");
                    tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
                }
            }
        },
        Cmd::Train {
            symbol,
            timeframe,
            iters,
            seed,
            window,
            out,
            datasets,
        } => {
            let symbol = Symbol::parse(&symbol)?;
            let tf = Timeframe::parse(&timeframe)?;
            let bars = BarStore::new(datasets).read(&symbol, tf)?;
            anyhow::ensure!(
                !bars.is_empty(),
                "no stored bars for {symbol} {tf}; run `paperrl data sync` first"
            );
            anyhow::ensure!(window >= 1, "--window must be >= 1");
            anyhow::ensure!(iters >= 1, "--iters must be >= 1");
            anyhow::ensure!(
                bars.len() > window + 1,
                "need more than {} bars for --window {window}; got {}",
                window + 1,
                bars.len()
            );
            let env_cfg = EnvConfig {
                window,
                ..EnvConfig::default()
            };
            let ppo_cfg = PpoConfig {
                iters,
                seed,
                ..PpoConfig::default()
            };
            tracing::info!(
                "training ppo on {symbol} {tf}: {} bars, {iters} iters, seed {seed}",
                bars.len()
            );
            let trained = train_ppo(&bars, env_cfg, ppo_cfg, &symbol.to_string(), &tf.to_string());
            std::fs::create_dir_all(&out)?;
            let path = out.join(format!("ppo-{symbol}-{seed}.bpk"));
            trained.save(&path)?;
            let first = trained.stats.first().map(|s| s.mean_reward).unwrap_or(0.0);
            let last = trained.stats.last().map(|s| s.mean_reward).unwrap_or(0.0);
            println!("model saved: {}", path.display());
            println!("metadata:    {}", path.with_extension("json").display());
            println!("mean step reward: first iter {first:+.4}, last iter {last:+.4}");
        }
        Cmd::Data { cmd } => match cmd {
            DataCmd::Sync {
                symbols,
                timeframe,
                years,
                out,
            } => {
                let symbols = parse_symbols(&symbols)?;
                let tf = Timeframe::parse(&timeframe)?;
                let client = AlpacaClient::new(&creds()?)?;
                let store = BarStore::new(out);
                let end = Utc::now();
                let start = end - Duration::days(365 * i64::from(years));
                let mut failed = Vec::new();
                for symbol in &symbols {
                    match sync_one(&client, &store, symbol, tf, start, end).await {
                        Ok((fetched, added, stored)) => {
                            tracing::info!("{symbol} {tf}: {fetched} fetched, {added} added, {stored} stored");
                        }
                        Err(e) => {
                            tracing::error!("{symbol}: {e:#}");
                            failed.push(symbol.to_string());
                        }
                    }
                }
                if !failed.is_empty() {
                    anyhow::bail!("sync failed for: {}", failed.join(", "));
                }
            }
            DataCmd::Show {
                symbol,
                timeframe,
                limit,
                out,
            } => {
                let symbol = Symbol::parse(&symbol)?;
                let tf = Timeframe::parse(&timeframe)?;
                let bars = BarStore::new(out).read(&symbol, tf)?;
                if bars.is_empty() {
                    println!("{symbol} {tf}: no bars stored; run `paperrl data sync` first");
                    return Ok(());
                }
                println!(
                    "{symbol} {tf}: {} bars stored, newest {}:",
                    bars.len(),
                    bars[bars.len() - 1].timestamp
                );
                for bar in &newest(&bars, limit) {
                    print_one(bar);
                }
            }
        },
        Cmd::Alpaca { cmd } => match cmd {
            AlpacaCmd::Account => {
                let summary = AlpacaClient::new(&creds()?)?.account().await?;
                println!("status           {}", summary.status);
                println!("equity           {}", summary.equity.as_deref().unwrap_or("-"));
                println!("cash             {}", summary.cash.as_deref().unwrap_or("-"));
                println!("buying power     {}", summary.buying_power.as_deref().unwrap_or("-"));
                println!("portfolio value  {}", summary.portfolio_value.as_deref().unwrap_or("-"));
            }
            AlpacaCmd::Positions => {
                let positions = AlpacaClient::new(&creds()?)?.positions().await?;
                if positions.is_empty() {
                    println!("no open positions");
                } else {
                    println!(
                        "{:<8} {:>10} {:>14} {:>12}",
                        "symbol", "qty", "market value", "unrealized P/L"
                    );
                    for p in positions {
                        println!(
                            "{:<8} {:>10.4} {:>14.2} {:>12.2}",
                            p.symbol, p.qty, p.market_value, p.unrealized_pl
                        );
                    }
                }
            }
            AlpacaCmd::Bars {
                symbol,
                timeframe,
                days,
            } => {
                let symbol = Symbol::parse(&symbol)?;
                let tf = Timeframe::parse(&timeframe)?;
                let client = AlpacaClient::new(&creds()?)?;
                let end = Utc::now();
                let start = end - Duration::days(i64::from(days));
                let bars = client.bars(&symbol, tf, start, end).await?;
                println!("{symbol} {tf}: {} bars in the last {days} days, newest:", bars.len());
                for bar in &newest(&bars, 10) {
                    print_one(bar);
                }
            }
        },
        Cmd::Backtest {
            symbol,
            timeframe,
            strategies,
            fast,
            slow,
            cash,
            fee_bps,
            slippage_bps,
            window_months,
            start,
            end,
            model,
            out,
        } => {
            let symbol = Symbol::parse(&symbol)?;
            let tf = Timeframe::parse(&timeframe)?;
            anyhow::ensure!(cash > 0.0, "--cash must be positive");
            anyhow::ensure!(fee_bps >= 0.0, "--fee-bps must be >= 0");
            anyhow::ensure!(
                (0.0..10_000.0).contains(&slippage_bps),
                "--slippage-bps must be in [0, 10000)"
            );
            anyhow::ensure!(fast > 0 && fast < slow, "--fast must be > 0 and < --slow");
            let mut bars = BarStore::new(out).read(&symbol, tf)?;
            let stored_n = bars.len();
            if let Some(s) = start.as_deref() {
                let t = NaiveDate::parse_from_str(s, "%Y-%m-%d")?
                    .and_hms_opt(0, 0, 0)
                    .context("bad start date")?
                    .and_utc();
                bars.retain(|b| b.timestamp >= t);
            }
            if let Some(e) = end.as_deref() {
                let t = NaiveDate::parse_from_str(e, "%Y-%m-%d")?
                    .and_hms_opt(23, 59, 59)
                    .context("bad end date")?
                    .and_utc();
                bars.retain(|b| b.timestamp <= t);
            }
            anyhow::ensure!(
                !bars.is_empty(),
                "{}",
                if stored_n == bars.len() {
                    format!("no stored bars for {symbol} {tf}; run `paperrl data sync` first")
                } else {
                    format!(
                        "{stored_n} bars stored for {symbol} {tf}, none within [{}, {}]",
                        start.as_deref().unwrap_or("-"),
                        end.as_deref().unwrap_or("-")
                    )
                }
            );
            let cfg = SimConfig {
                initial_cash: cash,
                fee_bps,
                slippage_bps,
                periods_per_year: periods_per_year(tf),
            };
            // The PPO policy is loaded once; each evaluation clones it for a
            // fresh rolling-history state.
            if model.is_some() && !strategies.contains(&Strat::Ppo) {
                tracing::warn!("--model ignored: the ppo strategy is not selected");
            }
            let ppo = if strategies.contains(&Strat::Ppo) {
                let strat = PpoStrategy::load(model.as_ref().context(
                    "--model is required for the ppo strategy",
                )?)?;
                anyhow::ensure!(
                    strat.meta().symbol == symbol.to_string(),
                    "model was trained on {} but backtesting {}",
                    strat.meta().symbol,
                    symbol
                );
                Some(strat)
            } else {
                None
            };
            let build = |s: &Strat| -> anyhow::Result<Box<dyn Strategy>> {
                Ok(match s {
                    Strat::BuyHold => Box::new(BuyHold),
                    Strat::Ema => Box::new(EmaCross::new(fast, slow)),
                    Strat::Momentum => Box::new(Momentum::new(fast)),
                    Strat::MeanRev => Box::new(MeanReversion::new(fast)),
                    Strat::VolTarget => Box::new(VolTarget::new(fast, 0.15)),
                    Strat::Ppo => Box::new(ppo.as_ref().context(
                        "--model is required for the ppo strategy",
                    )?.clone()),
                })
            };
            if window_months > 0 {
                println!(
                    "{symbol} {tf}: {} bars, {window_months}-month walk-forward windows, cash ${cash:.0}, fee {fee_bps:.0} bps, slippage {slippage_bps:.0} bps",
                    bars.len()
                );
                println!(
                    "{:<10} {:<24} {:>5} {:>9} {:>8} {:>8} {:>7}",
                    "strategy", "window", "bars", "net ret", "max dd", "sharpe", "trades"
                );
                let print_wf =
                    |s: &Strat, res: &paperrl_sim::WalkForwardResult| {
                        for w in &res.windows {
                            println!(
                                "{:<10} {:<24} {:>5} {:>8.2}% {:>7.2}% {:>8.2} {:>7}",
                                s,
                                format!("{}..{}", w.start.date_naive(), w.end.date_naive()),
                                w.metrics.bars,
                                w.metrics.net_return * 100.0,
                                w.metrics.max_drawdown * 100.0,
                                w.metrics.sharpe,
                                w.metrics.trades
                            );
                        }
                        let full_windows =
                            res.windows.iter().filter(|w| !w.partial).count();
                        println!(
                            "{:<10} aggregate over {} full windows: mean {:+.2}% ± {:.2}%, worst {:+.2}%, {}/{} positive",
                            s,
                            full_windows,
                            res.mean_return * 100.0,
                            res.return_stdev * 100.0,
                            res.worst_return * 100.0,
                            res.positive_windows,
                            full_windows
                        );
                    };
                for s in &strategies {
                    let res = match s {
                        Strat::Ppo => {
                            let ppo = ppo.as_ref().context(
                                "--model is required for the ppo strategy",
                            )?.clone();
                            paperrl_sim::walk_forward(
                                &bars,
                                &mut || Box::new(ppo.clone()),
                                &cfg,
                                window_months,
                            )
                        }
                        other => paperrl_sim::walk_forward(
                            &bars,
                            &mut || match other {
                                Strat::BuyHold => Box::new(BuyHold),
                                Strat::Ema => Box::new(EmaCross::new(fast, slow)),
                                Strat::Momentum => Box::new(Momentum::new(fast)),
                                Strat::MeanRev => Box::new(MeanReversion::new(fast)),
                                Strat::VolTarget => Box::new(VolTarget::new(fast, 0.15)),
                                Strat::Ppo => unreachable!("handled by the ppo arm"),
                            },
                            &cfg,
                            window_months,
                        ),
                    };
                    print_wf(s, &res);
                }
                return Ok(());
            }
            println!(
                "{symbol} {tf}: {} bars, cash ${cash:.0}, fee {fee_bps:.0} bps, slippage {slippage_bps:.0} bps",
                bars.len()
            );
            println!(
                "{:<10} {:>10} {:>8} {:>9} {:>8} {:>7} {:>7} {:>14} {:>10}",
                "strategy", "net ret", "sharpe", "max dd", "win", "exp", "trades", "turnover $",
                "fees $"
            );
            for s in &strategies {
                let mut strat: Box<dyn Strategy> = build(s)?;
                let m = paperrl_sim::run(&bars, strat.as_mut(), &cfg).metrics;
                println!(
                    "{:<10} {:>9.2}% {:>8.2} {:>8.2}% {:>7.1}% {:>6.1}% {:>7} {:>14.0} {:>10.2}",
                    strat.name(),
                    m.net_return * 100.0,
                    m.sharpe,
                    m.max_drawdown * 100.0,
                    m.win_rate * 100.0,
                    m.avg_exposure * 100.0,
                    m.trades,
                    m.turnover,
                    m.total_fees
                );
            }
        }
    }
    Ok(())
}

/// Bars per year for Sharpe annualization, from US equity session lengths.
fn periods_per_year(tf: Timeframe) -> f64 {
    match tf {
        Timeframe::Min1 => 252.0 * 390.0,
        Timeframe::Min5 => 252.0 * 78.0,
        Timeframe::Min15 => 252.0 * 26.0,
        Timeframe::Hour1 => 252.0 * 6.5,
        Timeframe::Day => 252.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periods_per_year_matches_bar_frequency() {
        assert_eq!(periods_per_year(Timeframe::Min1), 252.0 * 390.0);
        assert_eq!(periods_per_year(Timeframe::Min5), 252.0 * 78.0);
        assert_eq!(periods_per_year(Timeframe::Min15), 252.0 * 26.0);
        assert_eq!(periods_per_year(Timeframe::Hour1), 252.0 * 6.5);
        assert_eq!(periods_per_year(Timeframe::Day), 252.0);
    }

    fn bar(ts: i64) -> Bar {
        Bar {
            timestamp: chrono::DateTime::from_timestamp(ts, 0).unwrap(),
            open: 1.0,
            high: 1.0,
            low: 1.0,
            close: 1.0,
            volume: 1.0,
        }
    }

    #[test]
    fn newest_returns_last_n_oldest_first() {
        let bars: Vec<Bar> = (0..15).map(bar).collect();
        let got = newest(&bars, 3);
        let ts: Vec<i64> = got.iter().map(|b| b.timestamp.timestamp()).collect();
        assert_eq!(ts, vec![12, 13, 14]);
    }

    #[test]
    fn newest_limit_over_length_returns_all() {
        let bars: Vec<Bar> = (0..2).map(bar).collect();
        assert_eq!(newest(&bars, 10).len(), 2);
    }
}
