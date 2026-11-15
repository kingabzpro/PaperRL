//! PaperRL desktop dashboard: a local research desk over the same engine the
//! CLI uses (Alpaca data cache, simulator, PPO checkpoints).

use std::path::PathBuf;
use std::sync::OnceLock;

use dioxus::prelude::*;
use paperrl_desktop::{
    chart_geometry, fmt_date, fmt_money_cents, fmt_money_display, fmt_money_display_short, fmt_pct, fmt_price,
    list_datasets, list_models, load_bars, parse_watchlist_input, run_comparison,
    run_walk_forward, sparkline_points, BacktestRow, Dataset, ModelEntry, SimParams,
};
use paperrl_trade::{AppConfig, CredentialSource, SyncReport};

fn app_icon() -> dioxus::desktop::tao::window::Icon {
    // Raw RGBA generated from assets/logo.svg (64x64), so no runtime decoder
    // and no extra dependency is needed.
    dioxus::desktop::tao::window::Icon::from_rgba(
        include_bytes!("../../../assets/icon.rgba").to_vec(),
        64,
        64,
    )
    .expect("valid icon data")
}

/// Writes every log record to the append-only file behind the mutex.
struct FileMakeWriter(std::sync::Mutex<std::fs::File>);

struct FileGuard<'a>(&'a std::sync::Mutex<std::fs::File>);

impl std::io::Write for FileGuard<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log file lock").write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().expect("log file lock").flush()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for FileMakeWriter {
    type Writer = FileGuard<'a>;
    fn make_writer(&'a self) -> Self::Writer {
        FileGuard(&self.0)
    }
}

/// Console and `<home>/logs/app.log` logging; RUST_LOG picks the level.
fn init_logging() {
    use tracing_subscriber::prelude::*;
    let logs = home().join("logs");
    let _ = std::fs::create_dir_all(&logs);
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(logs.join("app.log"));
    let file_layer = file.ok().map(|f| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(FileMakeWriter(std::sync::Mutex::new(f)))
    });
    match file_layer {
        Some(file_layer) => {
            tracing_subscriber::registry()
                .with(filter.clone())
                .with(tracing_subscriber::fmt::layer())
                .with(file_layer)
                .init();
        }
        None => {
            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer())
                .init();
        }
    }
}

fn main() {
    init_logging();
    // PAPERRL_START_PAGE lets screenshots and demos open a specific page.
    // Retry once with a clear message: on Windows, WebView2 cannot create
    // its composition controller while the session is locked or the desktop
    // is not rendering (LogonUI active), and it fails with an opaque panic.
    for attempt in 1..=2 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dioxus::LaunchBuilder::desktop()
                .with_cfg(
                    dioxus::desktop::Config::new()
                        // No menu bar; the native title bar follows the dark theme.
                        .with_menu(None)
                        .with_window(
                            dioxus::desktop::WindowBuilder::new()
                                .with_title("PaperRL")
                                .with_theme(Some(dioxus::desktop::tao::window::Theme::Dark))
                                .with_inner_size(dioxus::desktop::LogicalSize::new(
                                    1280.0, 800.0,
                                )),
                        )
                        // The icon goes on the builder AFTER with_window: with_window
                        // replaces the inner builder wholesale, so an earlier
                        // Config::with_icon call is silently dropped.
                        .with_icon(app_icon()),
                )
                .launch(App);
        }));
        if result.is_ok() {
            return;
        }
        eprintln!(
            "PaperRL could not open its window (attempt {attempt} of 2).
             On Windows this usually means the desktop session is locked or not rendering:
             WebView2 refuses to start while the logon screen is active.
             Unlock the machine and run `cargo run -p paperrl-desktop` again."
        );
        if attempt == 1 {
            std::thread::sleep(std::time::Duration::from_millis(800));
        }
    }
    std::process::exit(1);
}

fn initial_page() -> Page {
    match std::env::var("PAPERRL_START_PAGE").as_deref() {
        Ok("markets") => Page::Markets,
        Ok("lab") => Page::Lab,
        Ok("automation") => Page::Automation,
        Ok("experiments") => Page::Experiments,
        Ok("settings") => Page::Settings,
        Ok("signals") => Page::Signals,
        Ok("portfolio") => Page::Portfolio,
        _ => Page::Dashboard,
    }
}

/// App data directory, resolved once per process: PAPERRL_HOME, a repo
/// checkout (datasets/ beside the working directory), or ~/paperrl.
fn home() -> &'static PathBuf {
    static HOME: OnceLock<PathBuf> = OnceLock::new();
    HOME.get_or_init(paperrl_data::paperrl_home)
}

/// One Signals row: either a verdict or the reason the ticker failed.
#[derive(Clone)]
struct SignalRow {
    symbol: String,
    result: Result<paperrl_trade::Recommendation, String>,
    stale: bool,
}

/// Launch-time work reported in the status strip above the page.
#[derive(Clone, PartialEq)]
enum BootKind {
    Working,
    Done,
    Failed,
}

#[derive(Clone, PartialEq)]
struct BootStatus {
    kind: BootKind,
    text: String,
}

impl BootStatus {
    fn cleared() -> Self {
        Self { kind: BootKind::Done, text: String::new() }
    }
    fn working(text: impl Into<String>) -> Self {
        Self { kind: BootKind::Working, text: text.into() }
    }
    fn done(text: impl Into<String>) -> Self {
        Self { kind: BootKind::Done, text: text.into() }
    }
    fn failed(text: impl Into<String>) -> Self {
        Self { kind: BootKind::Failed, text: text.into() }
    }
}

/// Shared app state passed through Dioxus context.
#[derive(Clone, Copy)]
struct AppCtx {
    config: Signal<AppConfig>,
    /// Units of the base currency per 1 USD; None means USD or FX unavailable.
    fx: Signal<Option<f64>>,
    bot: Signal<Option<paperrl_trade::BotHandle>>,
    /// Verdicts shared by the Signals tab and the launch pass.
    signals: Signal<Vec<SignalRow>>,
    signals_busy: Signal<bool>,
    signals_note: Signal<String>,
    boot: Signal<BootStatus>,
}

/// Builds an Alpaca paper client, or an error string for the UI.
async fn alpaca_client() -> Result<paperrl_alpaca::AlpacaClient, String> {
    let creds = paperrl_trade::resolve_credentials(home()).map_err(|e| e.to_string())?;
    paperrl_alpaca::AlpacaClient::new(&creds).map_err(|e| e.to_string())
}

/// AutoConfig for a watchlist pass from the saved config.
fn watchlist_config(cfg: &AppConfig, execute: bool) -> Result<paperrl_trade::AutoConfig, String> {
    let symbols: Vec<paperrl_core::Symbol> = cfg
        .watchlist
        .iter()
        .filter_map(|t| paperrl_core::Symbol::parse(t).ok())
        .collect();
    if symbols.is_empty() {
        return Err("the watchlist is empty; add tickers in the Signals tab first".into());
    }
    Ok(paperrl_trade::AutoConfig {
        top: 100,
        long_count: cfg.risk.max_positions,
        limits: cfg.risk,
        datasets: home().join("datasets"),
        model_dir: home().join("models"),
        filter: paperrl_trade::ScreenFilter::default(),
        execute,
        stale_days: 5,
        universe: Some(symbols),
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Page {
    Dashboard,
    Signals,
    Portfolio,
    Markets,
    Lab,
    Automation,
    Experiments,
    Settings,
}

#[allow(non_snake_case)]
#[component]
fn Logo() -> Element {
    rsx! {
        // Monochrome shredding mark: a sheet is shredded into falling strips
        // (theme off-white, single color).
        svg {
            width: "30",
            height: "30",
            view_box: "0 0 64 64",
            path { d: "M12 5 H32 L40 13 V25 H12 Z", fill: "#EAF2F7" }
            path { d: "M32 5 V13 H40 Z", fill: "#EAF2F7", fill_opacity: "0.55" }
            rect { x: "12", y: "29", width: "28", height: "4.5", rx: "1.75", fill: "#EAF2F7" }
            rect { x: "14", y: "36.5", width: "3.2", height: "8", rx: "1.6", fill: "#EAF2F7" }
            rect { x: "19.3", y: "36.5", width: "3.2", height: "12", rx: "1.6", fill: "#EAF2F7" }
            rect { x: "24.6", y: "36.5", width: "3.2", height: "6", rx: "1.6", fill: "#EAF2F7" }
            rect { x: "29.9", y: "36.5", width: "3.2", height: "13", rx: "1.6", fill: "#EAF2F7" }
            rect { x: "14", y: "36.5", width: "3.2", height: "10", rx: "1.6", fill: "#EAF2F7" }
            rect { x: "19.3", y: "36.5", width: "3.2", height: "15", rx: "1.6", fill: "#EAF2F7" }
            rect { x: "24.6", y: "36.5", width: "3.2", height: "8", rx: "1.6", fill: "#EAF2F7" }
            rect { x: "29.9", y: "36.5", width: "3.2", height: "16", rx: "1.6", fill: "#EAF2F7" }
            rect { x: "35.2", y: "36.5", width: "3.2", height: "9", rx: "1.6", fill: "#EAF2F7" }
        }
    }
}

/// Monochrome line icons shared by the sidebar and the page titles: one 24x24
/// stroke set drawn in `currentColor`, so active and hover states tint them
/// like text instead of pulling in colour emoji.
#[allow(non_snake_case)]
#[component]
fn NavIcon(page: Page) -> Element {
    let glyph = match page {
        Page::Dashboard => rsx! {
            rect { x: "3", y: "3", width: "7.5", height: "7.5", rx: "1.6" }
            rect { x: "13.5", y: "3", width: "7.5", height: "7.5", rx: "1.6" }
            rect { x: "3", y: "13.5", width: "7.5", height: "7.5", rx: "1.6" }
            rect { x: "13.5", y: "13.5", width: "7.5", height: "7.5", rx: "1.6" }
        },
        Page::Signals => rsx! {
            polyline { points: "22 12 18 12 15 21 9 3 6 12 2 12" }
        },
        Page::Portfolio => rsx! {
            rect { x: "3", y: "7.5", width: "18", height: "13", rx: "2.2" }
            path { d: "M8.5 7.5V5.8a2 2 0 0 1 2-2h3a2 2 0 0 1 2 2v1.7" }
            line { x1: "3", y1: "13.5", x2: "21", y2: "13.5" }
        },
        Page::Markets => rsx! {
            path { d: "M3 3v18h18" }
            polyline { points: "7 16 11 11 14 14 20 6.5" }
        },
        Page::Lab => rsx! {
            path { d: "M9.5 3h5" }
            path { d: "M10.5 3v6.2L4.9 18a2.1 2.1 0 0 0 1.7 3.2h10.8a2.1 2.1 0 0 0 1.7-3.2l-5.6-8.8V3" }
            line { x1: "8", y1: "15.5", x2: "16", y2: "15.5" }
        },
        Page::Automation => rsx! {
            polyline { points: "16.5 2.5 20.5 6.5 16.5 10.5" }
            path { d: "M3 11.5V9.5A4 4 0 0 1 7 5.5h13.5" }
            polyline { points: "7.5 21.5 3.5 17.5 7.5 13.5" }
            path { d: "M21 12.5v2a4 4 0 0 1-4 4H3.5" }
        },
        Page::Experiments => rsx! {
            path { d: "M12 2.8 21.2 12 12 21.2 2.8 12Z" }
            circle { cx: "12", cy: "12", r: "2.4" }
        },
        Page::Settings => rsx! {
            circle { cx: "12", cy: "12", r: "3.2" }
            path { d: "M12 2.6v2.5M12 18.9v2.5M2.6 12h2.5M18.9 12h2.5M5.4 5.4l1.8 1.8M16.8 16.8l1.8 1.8M18.6 5.4l-1.8 1.8M7.2 16.8l-1.8 1.8" }
        },
    };
    rsx! {
        svg {
            class: "icon",
            view_box: "0 0 24 24",
            width: "18",
            height: "18",
            fill: "none",
            stroke: "currentColor",
            stroke_width: "1.7",
            stroke_linecap: "round",
            stroke_linejoin: "round",
            {glyph}
        }
    }
}

/// Page heading with the same monochrome icon as its sidebar entry.
#[allow(non_snake_case)]
#[component]
fn PageTitle(page: Page, title: String) -> Element {
    rsx! {
        div { class: "page-title title-row",
            span { class: "title-icon", NavIcon { page } }
            span { "{title}" }
        }
    }
}

#[allow(non_snake_case)]
fn App() -> Element {
    let mut page = use_signal(initial_page);
    let datasets = use_signal(|| list_datasets(&home().join("datasets")));
    let sel = use_signal(|| {
        // PAPERRL_START_SYMBOL preselects a dataset for screenshots and demos.
        let list = datasets.read();
        match std::env::var("PAPERRL_START_SYMBOL") {
            Ok(sym) => {
                let sym = sym.to_uppercase();
                list.iter().position(|d| d.symbol == sym).unwrap_or(0)
            }
            Err(_) => 0,
        }
    });

    use_context_provider(|| AppCtx {
        config: Signal::new(AppConfig::load(home()).unwrap_or_default()),
        fx: Signal::new(None),
        bot: Signal::new(None),
        signals: Signal::new(Vec::new()),
        signals_busy: Signal::new(false),
        signals_note: Signal::new(String::new()),
        boot: Signal::new(BootStatus::cleared()),
    });
    let mut ctx = use_context::<AppCtx>();

    // Launch automation, once per process: fetch the watchlist and show
    // verdicts so a fresh start already has data on screen.
    let mut boot_started = use_signal(|| false);
    use_effect(move || {
        if boot_started() {
            return;
        }
        boot_started.set(true);
        run_startup(ctx, datasets);
    });

    // Load the FX rate for the base currency; re-fetch when it changes.
    use_effect(move || {
        let base = ctx.config.cloned().base_currency;
        spawn(async move {
            if base == "USD" {
                ctx.fx.set(None);
                return;
            }
            let rate = async {
                let client = alpaca_client().await?;
                client.fx_rate(&base).await.map_err(|e| e.to_string())
            }
            .await;
            match rate {
                Ok(r) => ctx.fx.set(Some(r)),
                Err(_) => ctx.fx.set(None),
            }
        });
    });

    let body = match *page.read() {
        Page::Dashboard => rsx! { DashboardPage { datasets, sel } },
        Page::Signals => rsx! { SignalsPage {} },
        Page::Portfolio => rsx! { PortfolioPage {} },
        Page::Markets => rsx! { MarketsPage { datasets, sel } },
        Page::Lab => rsx! { LabPage { datasets, sel } },
        Page::Automation => rsx! { AutomationPage {} },
        Page::Experiments => rsx! { ExperimentsPage { datasets } },
        Page::Settings => rsx! { SettingsPage { datasets } },
    };

    let boot_status = ctx.boot.cloned();
    let mut boot = ctx.boot;
    let boot_class = match boot_status.kind {
        BootKind::Working => "status-strip working",
        BootKind::Done => "status-strip done",
        BootKind::Failed => "status-strip failed",
    };

    rsx! {
        style { dangerous_inner_html: paperrl_desktop::theme::CSS.to_string() }
        div { class: "app",
            aside { class: "sidebar",
                div { class: "brand",
                    Logo {}
                    div {
                        div { class: "brand-name", "PaperRL" }
                        div { class: "brand-sub", "trading desk" }
                    }
                }
                for (p, label) in [
                    (Page::Dashboard, "Dashboard"),
                    (Page::Signals, "Signals"),
                    (Page::Portfolio, "Portfolio"),
                    (Page::Markets, "Markets"),
                    (Page::Lab, "Strategy Lab"),
                    (Page::Automation, "Automation"),
                    (Page::Experiments, "Experiments"),
                    (Page::Settings, "Settings"),
                ] {
                    div {
                        class: if *page.read() == p { "nav-item active" } else { "nav-item" },
                        onclick: move |_| page.set(p),
                        span { class: "nav-icon", NavIcon { page: p } }
                        span { "{label}" }
                    }
                }
                div { class: "sidebar-foot",
                    div { "Paper trading only" }
                    span { class: "safety-pill", "no live orders" }
                }
            }
            main { class: "content",
                if !boot_status.text.is_empty() {
                    div { class: "{boot_class}",
                        span { "{boot_status.text}" }
                        if boot_status.kind != BootKind::Working {
                            button {
                                class: "ghost",
                                onclick: move |_| boot.set(BootStatus::cleared()),
                                "Dismiss"
                            }
                        }
                    }
                }
                div { class: "page", {body} }
            }
        }
    }
}

/// Collapses whitespace and cuts a long provider error down to one readable
/// line: Alpaca 401s arrive as a whole nginx HTML page.
fn short_error(raw: &str, max: usize) -> String {
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = collapsed.chars();
    let truncated: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        collapsed
    }
}

/// Launch automation: with saved keys and a watchlist, fetch daily bars for the
/// watchlist, then fill the Signals tab with verdicts. Nothing here needs a
/// terminal: the same work the `paperrl data sync` command used to require.
fn run_startup(ctx: AppCtx, mut datasets: Signal<Vec<Dataset>>) {
    // A launch pass may still be running when keys get saved; one sync at a
    // time keeps the single-writer Parquet store honest.
    if ctx.boot.cloned().kind == BootKind::Working {
        return;
    }
    let mut boot = ctx.boot;
    let mut signals = ctx.signals;
    let mut busy = ctx.signals_busy;
    spawn(async move {
        let tags = ctx.config.cloned().watchlist;
        // Check the environment first: exported keys win, and the launch pass
        // says so instead of leaving the source implicit.
        let source = paperrl_trade::credential_source(home());
        if source == CredentialSource::Missing {
            boot.set(BootStatus::failed(
                "No Alpaca paper keys yet: open Settings, paste your key id and secret, then press Save keys.",
            ));
            return;
        }
        let source_note = match source {
            CredentialSource::Environment => {
                "Using the Alpaca keys in APCA_API_KEY_ID / APCA_API_SECRET (Settings shows the source)"
            }
            _ => "Using the Alpaca keys saved in Settings",
        };
        if tags.is_empty() {
            boot.set(BootStatus::done(format!(
                "{source_note}. Add tickers in Signals and press Analyze; the app fetches their data for you."
            )));
            return;
        }

        boot.set(BootStatus::working(format!(
            "{source_note} · fetching market data for {}...",
            tags.join(", ")
        )));
        let report = match alpaca_client().await {
            Ok(client) => {
                let report = paperrl_trade::sync_history(
                    &client,
                    &home().join("datasets"),
                    &tags,
                    paperrl_core::Timeframe::Day,
                    2,
                )
                .await;
                datasets.set(list_datasets(&home().join("datasets")));
                report
            }
            Err(e) => {
                boot.set(BootStatus::failed(format!(
                    "Could not reach Alpaca: {e} · {source_note}"
                )));
                return;
            }
        };

        busy.set(true);
        signals.set(
            tags.iter()
                .map(|t| SignalRow { symbol: t.clone(), result: Err("loading...".into()), stale: false })
                .collect(),
        );
        analyze_symbols(ctx, tags).await;

        let summary = report.summary();
        if report.synced.is_empty() {
            boot.set(BootStatus::failed(format!("{summary} · {source_note}")));
        } else if report.failed.is_empty() {
            boot.set(BootStatus::done(format!("{summary} · Signals updated · {source_note}")));
        } else {
            let failed = report
                .failed
                .iter()
                .map(|(symbol, error)| format!("{symbol}: {}", short_error(error, 140)))
                .collect::<Vec<_>>()
                .join("; ");
            boot.set(BootStatus::done(format!(
                "{summary} · Signals updated · {failed} · {source_note}"
            )));
        }
    });
}

/// Computes verdicts for every tag and writes them into the shared state.
/// The caller owns the busy flag and the placeholder rows; this always clears
/// the flag when it finishes, so a failed key cannot leave the UI spinning.
async fn analyze_symbols(ctx: AppCtx, tags: Vec<String>) {
    let mut signals = ctx.signals;
    let mut busy = ctx.signals_busy;
    let mut note = ctx.signals_note;

    let client = match alpaca_client().await {
        Ok(c) => Some(c),
        Err(e) => {
            note.set(format!("Error: {e}"));
            None
        }
    };
    let positions: Vec<paperrl_alpaca::PositionView> = match &client {
        Some(c) => c.positions().await.unwrap_or_default(),
        None => Vec::new(),
    };
    let mut out: Vec<SignalRow> = Vec::new();
    for tag in tags {
        let symbol = match paperrl_core::Symbol::parse(&tag) {
            Ok(s) => s,
            Err(_) => {
                out.push(SignalRow {
                    symbol: tag,
                    result: Err("not a valid ticker: use AAPL style or BTC/USD".into()),
                    stale: false,
                });
                continue;
            }
        };
        let Some(client) = &client else {
            out.push(SignalRow {
                symbol: tag,
                result: Err("Alpaca keys not set; add them in Settings".into()),
                stale: false,
            });
            continue;
        };
        let bars = match paperrl_trade::ensure_universe_bars(
            client,
            &home().join("datasets"),
            std::slice::from_ref(&tag),
            5,
        )
        .await
        {
            Ok((with_bars, _, _)) => with_bars.into_iter().next().map(|(_, b)| b).unwrap_or_default(),
            Err(e) => {
                out.push(SignalRow { symbol: tag, result: Err(format!("data fetch failed: {e}")), stale: false });
                continue;
            }
        };
        if bars.is_empty() {
            out.push(SignalRow {
                symbol: tag,
                result: Err("no market data returned for this ticker".into()),
                stale: false,
            });
            continue;
        }
        let stale = bars
            .last()
            .map(|b| (chrono::Utc::now() - b.timestamp).num_days() > 2)
            .unwrap_or(false);
        let model = paperrl_trade::latest_model_weight(&home().join("models"), symbol.as_str(), &bars);
        let held = positions
            .iter()
            .find(|p| p.symbol == symbol.as_str())
            .map(|p| p.qty)
            .unwrap_or(0.0);
        out.push(SignalRow {
            symbol: tag,
            result: Ok(paperrl_trade::recommend(&symbol, &bars, model, held)),
            stale,
        });
    }
    signals.set(out);
    busy.set(false);
}

fn selected_dataset(datasets: &Signal<Vec<Dataset>>, sel: &Signal<usize>) -> Option<Dataset> {
    datasets.read().get(*sel.read()).cloned()
}

#[component]
fn SymbolSelector(datasets: Signal<Vec<Dataset>>, sel: Signal<usize>) -> Element {
    let mut sel = sel;
    let list = datasets.read();
    if list.is_empty() {
        return rsx! {
            span { class: "dim", "no datasets yet: open Markets and fetch market data" }
        };
    }
    rsx! {
        select {
            onchange: move |evt| {
                if let Ok(i) = evt.value().parse::<usize>() {
                    sel.set(i);
                }
            },
            for (i, ds) in list.iter().enumerate() {
                option { value: "{i}", selected: i == *sel.read(), "{ds.symbol} · {ds.timeframe}" }
            }
        }
    }
}

fn bars_for(datasets: &Signal<Vec<Dataset>>, sel: &Signal<usize>) -> Vec<paperrl_core::Bar> {
    match selected_dataset(datasets, sel) {
        Some(ds) => load_bars(&home().join("datasets"), &ds.symbol, &ds.timeframe).unwrap_or_default(),
        None => Vec::new(),
    }
}

fn tf_of(ds: &Dataset) -> paperrl_core::Timeframe {
    match ds.timeframe.as_str() {
        "1min" => paperrl_core::Timeframe::Min1,
        "5min" => paperrl_core::Timeframe::Min5,
        "15min" => paperrl_core::Timeframe::Min15,
        "1hour" => paperrl_core::Timeframe::Hour1,
        _ => paperrl_core::Timeframe::Day,
    }
}

fn ppo_model_for(symbol: &str) -> Option<PathBuf> {
    let p = home().join("models").join(format!("ppo-{}-42.bpk", paperrl_core::Symbol::parse(symbol).map(|s| s.file_slug()).unwrap_or_else(|_| symbol.to_string())));
    p.exists().then_some(p)
}

#[component]
fn DashboardPage(datasets: Signal<Vec<Dataset>>, sel: Signal<usize>) -> Element {
    let ds = selected_dataset(&datasets, &sel);
    let bars = use_memo(move || bars_for(&datasets, &sel));
    let rows = use_memo(move || {
        let Some(ds) = selected_dataset(&datasets, &sel) else {
            return Vec::new();
        };
        let bars = bars_for(&datasets, &sel);
        let model = ppo_model_for(&ds.symbol);
        run_comparison(&bars, tf_of(&ds), SimParams::default(), model.as_deref())
    });

    let bars_ro = bars();
    let last_close = bars_ro.last().map(|b| b.close);
    let period_return = match (bars_ro.first(), bars_ro.last()) {
        (Some(f), Some(l)) if f.close > 0.0 => Some(l.close / f.close - 1.0),
        _ => None,
    };
    let chart_bars: Vec<paperrl_core::Bar> = bars_ro.iter().rev().take(120).rev().cloned().collect();

    rsx! {
        div { class: "page-head",
            div {
                PageTitle { page: Page::Dashboard, title: "Dashboard" }
                div { class: "page-sub",
                    if let Some(ds) = ds.as_ref() {
                        "{ds.symbol} · {ds.timeframe} · {ds.bars} bars · {fmt_date(ds.first)} to {fmt_date(ds.last)}"
                    } else {
                        "local-first RL trading research"
                    }
                }
            }
            div { class: "head-actions", MarketClock {}, SymbolSelector { datasets, sel } }
        }

        div { class: "grid-4",
            Stat { label: "Last close".to_string(), value: last_close.map(fmt_price).unwrap_or_else(|| String::from("-")), sub: "mark price, daily bars".to_string(), class: String::new() }
            Stat { label: "Period return".to_string(), value: period_return.map(fmt_pct).unwrap_or_else(|| String::from("-")), sub: "close over close, full history".to_string(), class: if period_return.unwrap_or(0.0) >= 0.0 { "pos".to_string() } else { "neg".to_string() } }
            Stat { label: "Baselines".to_string(), value: format!("{}", rows().len()), sub: "strategies on this dataset".to_string(), class: String::new() }
            Stat { label: "Execution".to_string(), value: String::from("Local sim"), sub: "paper API read-only".to_string(), class: String::new() }
        }

        div { class: "split",
            div { class: "card",
                div { class: "card-title", "Price action, last 120 bars" }
                CandleChart { bars: chart_bars }
            }
            div {
                AccountCard {}
                div { class: "card",
                    div { class: "card-title", "Engine" }
                    div { class: "muted-note",
                        "Same Rust core as the CLI: Parquet store, deterministic simulator with fees and slippage, and the PPO environment. The agent is evaluated on this page's dataset with the checkpoint in models/ when one exists."
                    }
                }
            }
        }

        div { class: "card",
            div { class: "card-title", "Strategy comparison (same costs for every row)" }
            if rows().is_empty() {
                div { class: "muted-note", "No bars available for the selected dataset." }
            } else {
                ComparisonTable { rows: rows() }
            }
        }
    }
}

#[component]
fn Stat(
    label: String,
    value: String,
    sub: String,
    class: String,
    #[props(default)]
    compact: bool,
) -> Element {
    rsx! {
        div { class: if compact { "stat compact" } else { "stat" },
            div { class: "stat-label", "{label}" }
            div { class: "stat-value {class}", "{value}" }
            div { class: "stat-sub", "{sub}" }
        }
    }
}

#[component]
fn AccountCard() -> Element {
    let ctx = use_context::<AppCtx>();
    let mut account = use_signal(|| None::<Result<String, String>>);
    use_effect(move || {
        spawn(async move {
            loop {
                let result = async {
                    let client = alpaca_client().await?;
                    let a = client.account().await.map_err(|e| e.to_string())?;
                    Ok::<_, String>(format!(
                        "{}|{}|{}|{}",
                        a.status,
                        a.equity.unwrap_or_else(|| "-".into()),
                        a.cash.unwrap_or_else(|| "-".into()),
                        a.buying_power.unwrap_or_else(|| "-".into())
                    ))
                }
                .await;
                account.set(Some(result));
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            }
        });
    });

    let value = account.read().clone();
    rsx! {
        div { class: "card",
            div { class: "card-title", "Alpaca paper account" }
            match value {
                None => rsx! { div { class: "muted-note", "Loading account..." } },
                Some(Err(e)) => rsx! { div { class: "error-box", "{e}" } },
                Some(Ok(text)) => {
                    let parts: Vec<&str> = text.split('|').collect();
                    let fx = ctx.fx.cloned();
                    let base = ctx.config.cloned().base_currency;
                    let money_or_raw = |v: &str| match v.parse::<f64>() {
                        Ok(n) => fmt_money_display_short(n, &base, fx),
                        Err(_) => v.to_string(),
                    };
                    let equity = parts.get(1).map(|v| money_or_raw(v)).unwrap_or_default();
                    let cash = parts.get(2).map(|v| money_or_raw(v)).unwrap_or_default();
                    let bp = parts.get(3).map(|v| money_or_raw(v)).unwrap_or_default();
                    rsx! {
                        div { class: "row", style: "justify-content: space-between;",
                            span { class: "dim", "status" }
                            span { class: "tag", "{parts.first().copied().unwrap_or(\"-\")}" }
                        }
                        div { class: "grid-4", style: "grid-template-columns: repeat(3, 1fr); margin-top: 12px; margin-bottom: 0;",
                            Stat { label: "Equity".to_string(), value: equity, sub: "paper".to_string(), class: String::new(), compact: true }
                            Stat { label: "Cash".to_string(), value: cash, sub: "available".to_string(), class: String::new(), compact: true }
                            Stat { label: "Buying power".to_string(), value: bp, sub: "margin".to_string(), class: String::new(), compact: true }
                        }
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn ActionBadge(action: paperrl_trade::Action) -> Element {
    match action {
        paperrl_trade::Action::Buy => rsx! { span { class: "badge badge-buy", "BUY" } },
        paperrl_trade::Action::Sell => rsx! { span { class: "badge badge-sell", "SELL" } },
        paperrl_trade::Action::Hold => rsx! { span { class: "badge badge-hold", "HOLD" } },
    }
}

#[component]
fn SignalsPage() -> Element {
    let mut ctx = use_context::<AppCtx>();
    let mut input = use_signal(String::new);
    let mut expanded = use_signal(|| None::<usize>);
    // (symbol, is_buy) awaiting confirmation
    let mut pending = use_signal(|| None::<(String, bool)>);
    let mut notional = use_signal(String::new);
    let config = ctx.config.cloned();
    // Shared with the launch pass, so a verdict computed at startup is already
    // here when this tab opens.
    let mut rows = ctx.signals;
    let mut busy = ctx.signals_busy;
    let mut note = ctx.signals_note;

    // Start from the saved watchlist so one click re-runs it.
    let mut seeded = use_signal(|| false);
    use_effect(move || {
        if seeded() {
            return;
        }
        seeded.set(true);
        let watchlist = ctx.config.cloned().watchlist;
        if input().trim().is_empty() && !watchlist.is_empty() {
            input.set(watchlist.join(", "));
        }
    });

    let analyze = move |_| {
        let tags = parse_watchlist_input(&input());
        if tags.is_empty() {
            note.set("Type one or more tickers first, e.g. \"AAPL, BTC/USD\".".into());
            return;
        }
        busy.set(true);
        note.set(String::new());
        rows.set(
            tags.iter()
                .map(|t| SignalRow { symbol: t.clone(), result: Err("loading...".into()), stale: false })
                .collect(),
        );
        spawn(async move {
            analyze_symbols(ctx, tags).await;
        });
    };

    let save_watchlist = move |_| {
        let tags = parse_watchlist_input(&input());
        if tags.is_empty() {
            note.set("Nothing to save; type tickers first.".into());
            return;
        }
        let mut cfg = ctx.config.cloned();
        cfg.watchlist = tags.clone();
        ctx.config.set(cfg.clone());
        match cfg.save(home()) {
            Ok(_) => note.set(format!("watchlist saved: {} tickers", cfg.watchlist.len())),
            Err(e) => note.set(format!("Could not save: {e}")),
        }
    };

    let load_watchlist = move |_| {
        input.set(ctx.config.cloned().watchlist.join(", "));
    };

    let confirm = move |_| {
        let Some((sym, is_buy)) = pending() else { return };
        let amount: f64 = notional().trim().parse().unwrap_or(0.0);
        if amount <= 0.0 {
            note.set("Enter an order amount in dollars first.".into());
            return;
        }
        let client_id = format!(
            "paperrl-manual-{}",
            chrono::Utc::now().format("%Y%m%d%H%M%S%3f")
        );
        spawn(async move {
            let result = async {
                let client = alpaca_client().await?;
                if is_buy {
                    // Notional buys work for equities and crypto alike.
                    let order = client
                        .submit_market_notional_order(&sym, amount, paperrl_alpaca::OrderSideView::Buy, &client_id)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>(format!("{} · {}", order.status, order.client_order_id))
                } else {
                    // Sells close the exact held quantity: crypto rejects notional
                    // sells, and a qty order cannot exceed the position.
                    let positions = client.positions().await.map_err(|e| e.to_string())?;
                    let qty = positions
                        .iter()
                        .find(|p| p.symbol == sym)
                        .map(|p| p.qty)
                        .unwrap_or(0.0);
                    if qty <= 0.0 {
                        return Err(format!("no open position in {sym} to sell"));
                    }
                    let order = client
                        .submit_market_order(&sym, paperrl_alpaca::OrderSideView::Sell, qty, &client_id)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>(format!("{} · {}", order.status, order.client_order_id))
                }
            }
            .await;
            match result {
                Ok(s) => note.set(format!("paper order submitted ({s})")),
                Err(e) => note.set(format!("Order failed: {e}")),
            }
            pending.set(None);
        });
    };

    // Plain-owned views so element closures capture only owned data.
    struct SignalRowView {
        symbol: String,
        stale: bool,
        err: Option<String>,
        action: paperrl_trade::Action,
        conf: String,
        conf_frac: f64,
        price: String,
        day: String,
        ret20: String,
        model: Option<u64>,
        reasons: Vec<String>,
        ema20: String,
        ema50: String,
        rsi: String,
        macd: String,
        ret5: String,
        why_title: String,
    }

    let row_views: Vec<SignalRowView> = rows()
        .iter()
        .map(|row| match &row.result {
            Err(e) => SignalRowView {
                symbol: row.symbol.clone(),
                stale: row.stale,
                err: Some(e.clone()),
                action: paperrl_trade::Action::Hold,
                conf: String::new(),
                conf_frac: 0.0,
                price: String::new(),
                day: String::new(),
                ret20: String::new(),
                model: None,
                reasons: Vec::new(),
                ema20: String::new(),
                ema50: String::new(),
                rsi: String::new(),
                macd: String::new(),
                ret5: String::new(),
                why_title: String::new(),
            },
            Ok(r) => SignalRowView {
                symbol: row.symbol.clone(),
                stale: row.stale,
                err: None,
                action: r.action,
                conf: format!("{:.0}%", r.confidence * 100.0),
                conf_frac: r.confidence,
                price: fmt_price(r.indicators.price),
                day: format!("{:+.2}%", r.indicators.change_pct),
                ret20: fmt_pct(r.indicators.ret20),
                model: r.model.map(|m| m.seed),
                reasons: r.reasons.clone(),
                ema20: fmt_price(r.indicators.ema20),
                ema50: fmt_price(r.indicators.ema50),
                rsi: format!("{:.0}", r.indicators.rsi14),
                macd: format!("{:.3}", r.indicators.macd_hist),
                ret5: fmt_pct(r.indicators.ret5),
                why_title: format!(
                    "Why {}? {:?} at {:.0}% confidence",
                    row.symbol, r.action, r.confidence * 100.0
                ),
            },
        })
        .collect();

    let table_rows: Vec<Element> = row_views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let buy_sym = v.symbol.clone();
            let sell_sym = v.symbol.clone();
            let max_order = ctx.config.cloned().risk.max_order_notional as i64;
            let buy = move |_| {
                notional.set(format!("{max_order}"));
                pending.set(Some((buy_sym.clone(), true)));
            };
            let sell = move |_| {
                notional.set(format!("{max_order}"));
                pending.set(Some((sell_sym.clone(), false)));
            };
            let toggle = move |_| {
                if expanded() == Some(i) { expanded.set(None); } else { expanded.set(Some(i)); }
            };
            let is_open = expanded() == Some(i);
            let v_err = v.err.clone();
            let day_cls = if v.day.starts_with('-') { "neg" } else { "pos" };
            let conf_w = format!("{:.0}%", v.conf_frac * 100.0);
            let loading = v_err.as_deref() == Some("loading...");
            rsx! {
                tr { key: "{i}",
                    td {
                        span { class: "strategy-name", "{v.symbol}" }
                        if v.stale {
                            span { class: "tag", style: "margin-left: 6px;", "cached" }
                        }
                    }
                    if loading {
                        td { colspan: "8",
                            div { class: "row", style: "gap: 18px; max-width: 560px;",
                                span { class: "skeleton skeleton-lg", style: "width: 84px;" }
                                span { class: "skeleton", style: "width: 120px;" }
                                span { class: "skeleton", style: "width: 90px;" }
                                span { class: "skeleton", style: "flex: 1;" }
                            }
                        }
                    } else if let Some(e) = v_err {
                        td { colspan: "8", class: "neg", "{e}" }
                    } else {
                        td { {ActionBadge(v.action)} }
                        td {
                            div { class: "conf-cell",
                                span { "{v.conf}" }
                                span { class: "conf-bar", style: "width: {conf_w}" }
                            }
                        }
                        td { "{v.price}" }
                        td { class: day_cls, "{v.day}" }
                        td { "{v.ret20}" }
                        td {
                            if let Some(seed) = v.model {
                                span { class: "tag", "seed {seed}" }
                            } else {
                                span { class: "dim", "-" }
                            }
                        }
                        td {
                            button { class: "ghost", onclick: toggle, if is_open { "hide" } else { "why" } }
                        }
                        td {
                            div { class: "row",
                                button { class: "ghost", onclick: buy, "Buy" }
                                button { class: "ghost", onclick: sell, "Sell" }
                            }
                        }
                    }
                }
            }
        })
        .collect();

    let why_card: Option<Element> = expanded()
        .and_then(|i| row_views.get(i))
        .filter(|v| v.err.is_none())
        .map(|v| {
            rsx! {
                div { class: "card",
                    div { class: "card-title", "{v.why_title}" }
                    div { class: "why-box",
                        for reason in &v.reasons {
                            div { class: "muted-note", "• {reason}" }
                        }
                    }
                    div { class: "row", style: "margin-top: 10px; gap: 18px; flex-wrap: wrap; font-size: 12px; color: #8FA3B8;",
                        span { "EMA20 {v.ema20}" }
                        span { "EMA50 {v.ema50}" }
                        span { "RSI {v.rsi}" }
                        span { "MACD hist {v.macd}" }
                        span { "5d {v.ret5}" }
                        span { "20d {v.ret20}" }
                    }
                }
            }
        });

    rsx! {
        div { class: "page-head",
            div {
                PageTitle { page: Page::Signals, title: "Signals" }
                div { class: "page-sub", "type any tickers - stocks and crypto - for a Buy / Sell / Hold verdict with its reasoning" }
            }
            div { class: "head-actions",
                button { class: "ghost", onclick: load_watchlist, "Load watchlist" }
                button { onclick: analyze, disabled: busy(), if busy() { "Analyzing..." } else { "Analyze" } }
            }
        }

        div { class: "card",
            div { class: "row",
                div { class: "field", style: "flex: 1; min-width: 280px;",
                    label { "Tickers" }
                    input {
                        r#type: "text",
                        value: "{input}",
                        placeholder: "AAPL, MSFT, BTC/USD",
                        oninput: move |e| input.set(e.value()),
                    }
                }
                div { class: "field",
                    label { " " }
                    button { class: "ghost", onclick: save_watchlist, "Save as watchlist" }
                }
            }
            if !note().is_empty() {
                div { class: "muted-note", style: "margin-top: 10px;", "{note}" }
            }
            if !config.watchlist.is_empty() {
                div { class: "muted-note", style: "margin-top: 8px;",
                    "Saved watchlist: {config.watchlist.join(\", \")}"
                }
            }
        }

        if !row_views.is_empty() {
            div { class: "card",
                div { class: "card-title", "Verdicts" }
                table {
                    thead {
                        tr {
                            th { "Ticker" }
                            th { "Verdict" }
                            th { "Confidence" }
                            th { "Price" }
                            th { "Day" }
                            th { "20d" }
                            th { "Model" }
                            th { "Why" }
                            th { "Trade" }
                        }
                    }
                    tbody { for e in table_rows { {e} } }
                }
            }
        }

        if let Some(wc) = why_card {
            {wc}
        }

        if let Some((sym, is_buy)) = pending() {
            div { class: "card confirm-box",
                div { class: "card-title",
                    {if is_buy { format!("Buy {sym}") } else { format!("Sell {sym}") }}
                }
                div { class: "row",
                    div { class: "field",
                        label { "Order size (USD)" }
                        input { r#type: "text", value: "{notional}", oninput: move |e| notional.set(e.value()) }
                    }
                    button { onclick: confirm, "Confirm paper order" }
                    button { class: "ghost", onclick: move |_| pending.set(None), "Cancel" }
                }
                div { class: "muted-note", style: "margin-top: 8px;",
                    "Market order on your Alpaca PAPER account; nothing here can reach live money."
                }
            }
        }
    }
}

/// Renders a pass-history RFC3339 stamp as a local "YYYY-MM-DD HH:MM" line;
/// falls back to the raw string when parsing fails.
fn fmt_pass_time(raw: &str) -> String {
    match chrono::DateTime::parse_from_rfc3339(raw) {
        Ok(t) => t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M:%S").to_string(),
        Err(_) => raw.to_string(),
    }
}

#[component]
fn MarketClock() -> Element {
    let mut clock = use_signal(|| None::<paperrl_alpaca::ClockView>);
    use_effect(move || {
        spawn(async move {
            let result = async {
                let client = alpaca_client().await?;
                client.clock().await.map_err(|e| e.to_string())
            }
            .await;
            if let Ok(c) = result {
                clock.set(Some(c));
            }
        });
    });
    match clock() {
        None => rsx! { span { class: "dim", "market clock unavailable" } },
        Some(c) => rsx! {
            span {
                class: if c.is_open { "tag badge-buy live" } else { "tag" },
                {if c.is_open {
                    "market open".to_string()
                } else {
                    format!("market closed · opens {}", fmt_date(c.next_open))
                }}
            }
        },
    }
}

#[component]
fn PortfolioPage() -> Element {
    let ctx = use_context::<AppCtx>();
    let mut busy = use_signal(|| false);
    let mut note = use_signal(String::new);
    let mut account = use_signal(|| None::<paperrl_alpaca::AccountSummary>);
    let mut positions = use_signal(Vec::<paperrl_alpaca::PositionView>::new);
    let mut equity_curve = use_signal(Vec::<paperrl_alpaca::EquityPoint>::new);
    let mut fills = use_signal(Vec::<paperrl_alpaca::FillView>::new);
    let mut period = use_signal(|| "1M".to_string());

    let mut reload = use_signal(|| 0u32);
    let bump_reload = move |_| reload.with_mut(|r| *r += 1);

    // Load on mount, when the period selector changes, and when refresh is hit.
    use_effect(move || {
        let want_period = period();
        let _ = reload();
        busy.set(true);
        spawn(async move {
            let result = async {
                let client = alpaca_client().await?;
                let acct = client.account().await.map_err(|e| e.to_string())?;
                let pos = client.positions().await.map_err(|e| e.to_string())?;
                let curve = client
                    .portfolio_history(&want_period)
                    .await
                    .map_err(|e| e.to_string())?;
                let fl = client.fills(20).await.map_err(|e| e.to_string())?;
                Ok::<_, String>((acct, pos, curve, fl))
            }
            .await;
            match result {
                Ok((a, p, c, f)) => {
                    account.set(Some(a));
                    positions.set(p);
                    equity_curve.set(c);
                    fills.set(f);
                    note.set(String::new());
                }
                Err(e) => note.set(format!("Error: {e}")),
            }
            busy.set(false);
        });
    });

    let fx = ctx.fx.cloned();
    let base = ctx.config.cloned().base_currency;
    let base_for_money = base.clone();
    let money = move |v: f64| fmt_money_display(v, &base_for_money, fx);
    let total_mv: f64 = positions().iter().map(|p| p.market_value).sum();
    let crypto_mv: f64 = positions()
        .iter()
        .filter(|p| p.asset_class == "crypto")
        .map(|p| p.market_value)
        .sum();
    let equity_mv = total_mv - crypto_mv;
    let unrealized: f64 = positions().iter().map(|p| p.unrealized_pl).sum();
    let equity_val: f64 = account()
        .as_ref()
        .and_then(|a| a.equity.as_deref().and_then(|v| v.parse().ok()))
        .unwrap_or(0.0);
    let fx_note = base != "USD" && fx.is_none();

    // Equity curve polyline plus a running-max drawdown line.
    let curve = equity_curve();
    let mut poly = String::new();
    let mut area = String::new();
    let mut dd_path = String::new();
    let mut min_e = 0.0;
    let mut max_e = 0.0;
    if curve.len() >= 2 {
        min_e = curve.iter().map(|p| p.equity).fold(f64::INFINITY, f64::min);
        max_e = curve.iter().map(|p| p.equity).fold(f64::NEG_INFINITY, f64::max);
        let span = (max_e - min_e).max(1e-9);
        let n = curve.len();
        poly = curve
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let x = i as f64 / (n - 1) as f64 * 880.0;
                let y = 200.0 - (p.equity - min_e) / span * 180.0;
                format!("{x:.1},{y:.1}")
            })
            .collect::<Vec<_>>()
            .join(" ");
        let mut running = f64::NEG_INFINITY;
        for (i, p) in curve.iter().enumerate() {
            running = running.max(p.equity);
            let x = i as f64 / (n - 1) as f64 * 880.0;
            let y = 200.0 - (running - min_e) / span * 180.0;
            dd_path.push_str(&if dd_path.is_empty() {
                format!("M {x:.1} {y:.1}")
            } else {
                format!(" L {x:.1} {y:.1}")
            });
        }
        let first_pt = curve.first().expect("checked non-empty");
        let last_pt = curve.last().expect("checked non-empty");
        let x0 = 0.0;
        let x1 = 880.0;
        let y0 = 200.0 - (first_pt.equity - min_e) / span * 180.0;
        let y1 = 200.0 - (last_pt.equity - min_e) / span * 180.0;
        area = format!("M {x0:.1} 200 L {x0:.1} {y0:.1} {poly} L {x1:.1} {y1:.1} L {x1:.1} 200 Z");
    }

    rsx! {
        div { class: "page-head",
            div {
                PageTitle { page: Page::Portfolio, title: "Portfolio" }
                div { class: "page-sub", "Alpaca paper holdings, P&L, and the equity curve. Displayed in {base}." }
            }
            div { class: "head-actions",
                button { onclick: bump_reload, disabled: busy(), if busy() { "Refreshing..." } else { "Refresh" } }
            }
        }

        if !note().is_empty() {
            div { class: "error-box", "{note}" }
        }
        if fx_note {
            div { class: "muted-note", style: "margin-bottom: 12px;",
                "FX unavailable right now, showing US dollars instead of {base}."
            }
        }

        div { class: "grid-4",
            Stat { label: "Equity".to_string(), value: money(equity_val), sub: "paper account".to_string(), class: String::new() }
            Stat { label: "Market value".to_string(), value: money(total_mv), sub: "open positions".to_string(), class: String::new() }
            Stat { label: "Unrealized P&L".to_string(),
                value: money(unrealized),
                sub: "all positions".to_string(),
                class: if unrealized >= 0.0 { "pos".to_string() } else { "neg".to_string() } }
            Stat { label: "Positions".to_string(), value: format!("{}", positions().len()), sub: "stocks and crypto".to_string(), class: String::new() }
        }

        div { class: "card",
            div { class: "card-title", "Holdings" }
            if positions().is_empty() {
                div { class: "muted-note",
                    "No open positions. Place a paper order in Signals, or run the bot in Automation."
                }
            } else {
                table {
                    thead { tr {
                        th { "Symbol" } th { "Class" } th { "Qty" } th { "Avg entry" }
                        th { "Price" } th { "Value" } th { "P&L" } th { "P&L %" } th { "Weight" }
                    } }
                    tbody {
                        for p in positions() {
                            tr { key: "{p.symbol}",
                                td { span { class: "strategy-name", "{p.symbol}" } }
                                td { span { class: "tag", "{p.asset_class}" } }
                                td { {format!("{:.4}", p.qty)} }
                                td { "{money(p.avg_entry_price)}" }
                                td { "{money(p.current_price)}" }
                                td { "{money(p.market_value)}" }
                                td { class: if p.unrealized_pl >= 0.0 { "pos" } else { "neg" }, "{money(p.unrealized_pl)}" }
                                td { class: if p.unrealized_plpc >= 0.0 { "pos" } else { "neg" }, {format!("{:+.2}%", p.unrealized_plpc * 100.0)} }
                                td { {format!("{:.1}%", if total_mv > 0.0 { p.market_value / total_mv * 100.0 } else { 0.0 })} }
                            }
                        }
                    }
                }
                div { class: "muted-note", style: "margin-top: 12px;",
                    "Currency exposure: equities {money(equity_mv)} · crypto {money(crypto_mv)}"
                }
            }
        }

        div { class: "card",
            div { class: "row", style: "justify-content: space-between; margin-bottom: 12px;",
                div { class: "card-title", style: "margin-bottom: 0;", "Equity curve" }
                div { class: "row",
                    for p in ["1W", "1M", "3M", "1A"] {
                        button {
                            class: if *period.read() == p { "" } else { "ghost" },
                            onclick: move |_| period.set(p.to_string()),
                            "{p}"
                        }
                    }
                }
            }
            if curve.len() < 2 {
                div { class: "muted-note", "Not enough history yet for a curve." }
            } else {
                svg { view_box: "0 0 880 210", width: "100%", height: "210", style: "display: block;",
                    defs {
                        linearGradient {
                            id: "eqgrad",
                            x1: "0", y1: "0", x2: "0", y2: "1",
                            stop { offset: "0%", "stop-color": "#08B98B", "stop-opacity": "0.22" }
                            stop { offset: "100%", "stop-color": "#08B98B", "stop-opacity": "0" }
                        }
                    }
                    path { d: "{area}", fill: "url(#eqgrad)" }
                    path { d: "{dd_path}", fill: "none", stroke: "#F2685C", stroke_width: "1", stroke_dasharray: "3 4", opacity: "0.7" }
                    polyline { class: "draw-line", path_length: "1", points: "{poly}", fill: "none", stroke: "#08B98B", stroke_width: "2", stroke_linejoin: "round" }
                    text { x: "4", y: "14", "{money(max_e)}" }
                    text { x: "4", y: "200", "{money(min_e)}" }
                }
                div { class: "muted-note", style: "margin-top: 8px;",
                    "Green: account equity. Red dashes: the running peak, so gaps show drawdown."
                }
            }
        }

        div { class: "card",
            div { class: "card-title", "Recent fills" }
            if fills().is_empty() {
                div { class: "muted-note", "No fills yet." }
            } else {
                table {
                    thead { tr { th { "Time" } th { "Symbol" } th { "Side" } th { "Qty" } th { "Price" } } }
                    tbody {
                        for f in fills() {
                            tr { key: "{f.id}",
                                td { "{fmt_date(f.time)}" }
                                td { span { class: "strategy-name", "{f.symbol}" } }
                                td { class: if f.side == "buy" { "pos" } else { "neg" }, "{f.side}" }
                                td { {format!("{:.4}", f.qty)} }
                                td { "{money(f.price)}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn CandleChart(bars: Vec<paperrl_core::Bar>) -> Element {
    let mut hovered = use_signal(|| None::<usize>);
    if bars.is_empty() {
        return rsx! { div { class: "muted-note", "No bars to chart." } };
    }
    let width = 900.0_f64;
    let height = 250.0_f64;
    let geom = chart_geometry(&bars, width, height);
    let last = bars.last().cloned().unwrap_or(bars[0]);
    let hover_bar = hovered().and_then(|i| bars.get(i).cloned());
    let first_date = bars.first().map(|b| fmt_date(b.timestamp)).unwrap_or_default();
    let last_date = bars.last().map(|b| fmt_date(b.timestamp)).unwrap_or_default();

    rsx! {
        div { class: "chart-wrap",
            div { class: "chart-head",
                div { class: "chart-legend",
                    if let Some(b) = hover_bar {
                        span { "{fmt_date(b.timestamp)}" }
                        span { "O {fmt_price(b.open)}" }
                        span { "H {fmt_price(b.high)}" }
                        span { "L {fmt_price(b.low)}" }
                        span { class: if b.close >= b.open { "pos" } else { "neg" }, "C {fmt_price(b.close)}" }
                    } else {
                        span { "hover a candle for OHLC" }
                    }
                }
                div { class: "chart-legend",
                    span { "{fmt_price(geom.min)} – {fmt_price(geom.max)}" }
                }
            }
            svg {
                view_box: "0 0 900 250",
                width: "100%",
                height: "250",
                style: "display: block;",
                line {
                    x1: "0", y1: "{geom.price_bottom:.1}", x2: "900", y2: "{geom.price_bottom:.1}",
                    stroke: "#22334A", stroke_width: "1", stroke_dasharray: "3 4"
                }
                line {
                    x1: "0", y1: "{geom.volume_top:.1}", x2: "900", y2: "{geom.volume_top:.1}",
                    stroke: "#22334A", stroke_width: "1", stroke_dasharray: "3 4"
                }
                for (i, c) in geom.candles.iter().enumerate() {
                    g {
                        class: "candle",
                        style: "animation-delay: {i * 6}ms",
                        onmouseenter: move |_| hovered.set(Some(i)),
                        line {
                            x1: "{c.center:.2}", y1: "{c.wick_top:.2}",
                            x2: "{c.center:.2}", y2: "{c.wick_bottom:.2}",
                            stroke: if c.up { "#34d399" } else { "#f87171" },
                            stroke_width: "1"
                        }
                        rect {
                            x: "{c.x:.2}", y: "{c.body_top:.2}",
                            width: "{c.width:.2}",
                            height: "{(c.body_bottom - c.body_top).max(1.0):.2}",
                            fill: if c.up { "#34d399" } else { "#f87171" },
                            rx: "0.5"
                        }
                        rect {
                            x: "{c.x:.2}", y: "{c.volume_top:.2}",
                            width: "{c.width:.2}", height: "{c.volume_height:.2}",
                            fill: if c.up { "#1d4a35" } else { "#3f1d28" }
                        }
                    }
                }
                text { x: "4", y: "{(geom.price_top + 10.0):.1}", "{fmt_price(geom.max)}" }
                text { x: "4", y: "{(geom.price_bottom - 4.0):.1}", "{fmt_price(geom.min)}" }
                text { x: "4", y: "{height + 0.0:.1}", "{first_date}" }
                text { x: "810", y: "{height + 0.0:.1}", "{last_date}" }
            }
        }
        div { class: "muted-note", style: "margin-top: 8px;",
            "Last close {fmt_price(last.close)}, volume {last.volume:.0}. Candles colored by close versus open."
        }
    }
}

struct RowView {
    name: String,
    ret: String,
    ret_pos: bool,
    sharpe: String,
    dd: String,
    win: String,
    exposure: String,
    trades: String,
    fees: String,
}

#[component]
fn ComparisonTable(rows: Vec<BacktestRow>) -> Element {
    let views: Vec<RowView> = rows
        .iter()
        .map(|r| RowView {
            name: r.name.clone(),
            ret: fmt_pct(r.net_return),
            ret_pos: r.net_return >= 0.0,
            sharpe: format!("{:.2}", r.sharpe),
            dd: fmt_pct(-r.max_drawdown),
            win: format!("{:.1}%", r.win_rate * 100.0),
            exposure: format!("{:.1}%", r.exposure * 100.0),
            trades: r.trades.to_string(),
            fees: fmt_money_cents(r.fees),
        })
        .collect();
    rsx! {
        table {
            thead {
                tr {
                    th { "Strategy" }
                    th { "Net return" }
                    th { "Sharpe" }
                    th { "Max drawdown" }
                    th { "Win rate" }
                    th { "Exposure" }
                    th { "Trades" }
                    th { "Fees" }
                }
            }
            tbody {
                for v in views {
                    tr {
                        td { span { class: "strategy-name", "{v.name}" } }
                        td { class: if v.ret_pos { "pos" } else { "neg" }, "{v.ret}" }
                        td { "{v.sharpe}" }
                        td { class: "neg", "{v.dd}" }
                        td { "{v.win}" }
                        td { "{v.exposure}" }
                        td { "{v.trades}" }
                        td { "{v.fees}" }
                    }
                }
            }
        }
    }
}

#[component]
fn MarketsPage(datasets: Signal<Vec<Dataset>>, sel: Signal<usize>) -> Element {
    let mut datasets = datasets;
    let mut sel = sel;
    let ctx = use_context::<AppCtx>();
    let mut symbols = use_signal(String::new);
    let mut timeframe = use_signal(|| "1day".to_string());
    let mut years = use_signal(|| "2".to_string());
    let mut busy = use_signal(|| false);
    let mut note = use_signal(String::new);
    let mut report = use_signal(|| None::<SyncReport>);
    let list: Vec<Dataset> = datasets.read().clone();
    let ds = selected_dataset(&datasets, &sel);
    let bars = use_memo(move || bars_for(&datasets, &sel));
    let chart_bars: Vec<paperrl_core::Bar> = bars().iter().rev().take(120).rev().cloned().collect();

    // Prefill from the saved watchlist so one click refreshes the usual names.
    let mut prefilled = use_signal(|| false);
    use_effect(move || {
        if prefilled() {
            return;
        }
        prefilled.set(true);
        let watchlist = ctx.config.cloned().watchlist;
        if symbols().trim().is_empty() && !watchlist.is_empty() {
            symbols.set(watchlist.join(", "));
        }
    });

    let sync = move |_| {
        let tags = parse_watchlist_input(&symbols());
        if tags.is_empty() {
            note.set("Type at least one ticker, e.g. \"AAPL, MSFT, BTC/USD\".".into());
            return;
        }
        let tf = match timeframe().as_str() {
            "1hour" => paperrl_core::Timeframe::Hour1,
            "15min" => paperrl_core::Timeframe::Min15,
            "5min" => paperrl_core::Timeframe::Min5,
            "1min" => paperrl_core::Timeframe::Min1,
            _ => paperrl_core::Timeframe::Day,
        };
        let span: u32 = years().trim().parse::<u32>().unwrap_or(2).clamp(1, 20);
        busy.set(true);
        note.set(format!("Fetching {} bars for {}...", timeframe(), tags.join(", ")));
        spawn(async move {
            match alpaca_client().await {
                Ok(client) => {
                    let outcome = paperrl_trade::sync_history(
                        &client,
                        &home().join("datasets"),
                        &tags,
                        tf,
                        span,
                    )
                    .await;
                    datasets.set(list_datasets(&home().join("datasets")));
                    note.set(outcome.summary());
                    report.set(Some(outcome));
                }
                Err(e) => note.set(format!("Error: {e}")),
            }
            busy.set(false);
        });
    };

    let run_report = report();
    rsx! {
        div { class: "page-head",
            div {
                PageTitle { page: Page::Markets, title: "Markets" }
                div { class: "page-sub", "{list.len()} local datasets cached from Alpaca" }
            }
        }

        div { class: "card",
            div { class: "card-title", "Fetch market data" }
            div { class: "row",
                div { class: "field", style: "flex: 1; min-width: 260px;",
                    label { "Tickers" }
                    input {
                        r#type: "text",
                        value: "{symbols}",
                        placeholder: "AAPL, MSFT, BTC/USD",
                        oninput: move |e| symbols.set(e.value()),
                    }
                }
                div { class: "field",
                    label { "Timeframe" }
                    select {
                        value: "{timeframe}",
                        onchange: move |e| timeframe.set(e.value()),
                        for (slug, label) in [
                            ("1day", "1 day"),
                            ("1hour", "1 hour"),
                            ("15min", "15 min"),
                            ("5min", "5 min"),
                            ("1min", "1 min"),
                        ] {
                            option { value: "{slug}", selected: *timeframe.read() == slug, "{label}" }
                        }
                    }
                }
                div { class: "field", style: "width: 84px;",
                    label { "Years" }
                    input { r#type: "text", value: "{years}", oninput: move |e| years.set(e.value()) }
                }
                div { class: "field",
                    label { " " }
                    button { onclick: sync, disabled: busy(),
                        if busy() { "Fetching..." } else { "Fetch market data" }
                    }
                }
                div { class: "field",
                    label { " " }
                    button {
                        class: "ghost",
                        onclick: move |_| symbols.set(ctx.config.cloned().watchlist.join(", ")),
                        "Use watchlist"
                    }
                }
            }
            if !note().is_empty() {
                div { class: "muted-note", style: "margin-top: 10px;", "{note}" }
            }
            if let Some(r) = run_report {
                for (symbol, error) in &r.failed {
                    div { class: "muted-note neg", style: "margin-top: 6px;",
                        "{symbol}: {short_error(error, 160)}"
                    }
                }
                if !r.synced.is_empty() {
                    div { style: "margin-top: 10px;",
                        table {
                            thead { tr {
                                th { "Symbol" } th { "Fetched" } th { "New bars" } th { "Stored" }
                            } }
                            tbody {
                                for s in &r.synced {
                                    tr { key: "{s.symbol}",
                                        td { span { class: "strategy-name", "{s.symbol}" } }
                                        td { "{s.fetched}" }
                                        td { "{s.added}" }
                                        td { "{s.stored}" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        div { class: "card",
            if list.is_empty() {
                div { class: "muted-note",
                    "Nothing cached yet. Type tickers above and press Fetch market data; with keys saved, a launch also syncs the watchlist for you."
                }
            } else {
                table { class: "clickable",
                    thead {
                        tr {
                            th { "Symbol" }
                            th { "Timeframe" }
                            th { "Bars" }
                            th { "First" }
                            th { "Last" }
                        }
                    }
                    tbody {
                        for (i, d) in list.iter().enumerate() {
                            tr {
                                class: if i == *sel.read() { "selected" } else { "" },
                                onclick: move |_| sel.set(i),
                                td { span { class: "strategy-name", "{d.symbol}" } }
                                td { span { class: "tag", "{d.timeframe}" } }
                                td { "{d.bars}" }
                                td { "{fmt_date(d.first)}" }
                                td { "{fmt_date(d.last)}" }
                            }
                        }
                    }
                }
            }
        }
        if let Some(d) = ds {
            div { class: "card",
                div { class: "card-title", "{d.symbol} · {d.timeframe}" }
                CandleChart { bars: chart_bars }
            }
        }
    }
}

/// Parses and validates Lab inputs; the simulator asserts on nonsense
/// numbers, so bad values surface as messages instead.
fn sim_params_from(cash: String, fee: String, slip: String) -> Result<SimParams, String> {
    let cash: f64 = cash.trim().parse().map_err(|_| "cash must be a number".to_string())?;
    let fee_bps: f64 = fee.trim().parse().map_err(|_| "fee bps must be a number".to_string())?;
    let slippage_bps: f64 = slip.trim().parse().map_err(|_| "slippage bps must be a number".to_string())?;
    if !cash.is_finite() || cash <= 0.0 {
        return Err("cash must be a positive number".into());
    }
    if !fee_bps.is_finite() || fee_bps < 0.0 || fee_bps > 1000.0 {
        return Err("fee bps must be between 0 and 1000".into());
    }
    if !slippage_bps.is_finite() || slippage_bps < 0.0 || slippage_bps > 1000.0 {
        return Err("slippage bps must be between 0 and 1000".into());
    }
    Ok(SimParams { cash, fee_bps, slippage_bps })
}

#[component]
fn LabPage(datasets: Signal<Vec<Dataset>>, sel: Signal<usize>) -> Element {
    let mut cash = use_signal(|| "100000".to_string());
    let mut fee = use_signal(|| "1".to_string());
    let mut slip = use_signal(|| "5".to_string());
    let mut months = use_signal(|| "6".to_string());
    let mut rows = use_signal(Vec::<BacktestRow>::new);
    let mut wf = use_signal(Vec::<(String, f64, f64, f64, usize, usize)>::new);
    let mut message = use_signal(String::new);

    let run = move |_| {
        let Some(ds) = selected_dataset(&datasets, &sel) else {
            message.set("Select a dataset in Markets first.".into());
            return;
        };
        let params = match sim_params_from(cash(), fee(), slip()) {
            Ok(p) => p,
            Err(e) => {
                message.set(format!("Error: {e}"));
                return;
            }
        };
        let bars = bars_for(&datasets, &sel);
        let model = ppo_model_for(&ds.symbol);
        rows.set(run_comparison(&bars, tf_of(&ds), params, model.as_deref()));
        message.set(format!("Ran {} strategies on {} bars.", rows().len(), bars.len()));
    };

    let run_wf = move |_| {
        let Some(ds) = selected_dataset(&datasets, &sel) else {
            message.set("Select a dataset in Markets first.".into());
            return;
        };
        let params = match sim_params_from(cash(), fee(), slip()) {
            Ok(p) => p,
            Err(e) => {
                message.set(format!("Error: {e}"));
                return;
            }
        };
        let m: u32 = months().parse().unwrap_or(6).max(1);
        let bars = bars_for(&datasets, &sel);
        let results = run_walk_forward(&bars, tf_of(&ds), params, m);
        wf.set(
            results
                .into_iter()
                .map(|(name, res)| {
                    let full = res.windows.iter().filter(|w| !w.partial).count();
                    (name, res.mean_return, res.return_stdev, res.worst_return, res.positive_windows, full)
                })
                .collect(),
        );
        message.set(format!("{m}-month walk-forward complete."));
    };

    rsx! {
        div { class: "page-head",
            div {
                PageTitle { page: Page::Lab, title: "Strategy Lab" }
                div { class: "page-sub", "same simulator, costs, and walk-forward engine as the CLI" }
            }
            div { class: "head-actions", SymbolSelector { datasets, sel } }
        }

        div { class: "card",
            div { class: "row",
                div { class: "field",
                    label { "Cash" }
                    input { r#type: "text", value: "{cash}", oninput: move |e| cash.set(e.value()) }
                }
                div { class: "field",
                    label { "Fee bps" }
                    input { r#type: "text", value: "{fee}", oninput: move |e| fee.set(e.value()) }
                }
                div { class: "field",
                    label { "Slippage bps" }
                    input { r#type: "text", value: "{slip}", oninput: move |e| slip.set(e.value()) }
                }
                div { class: "field",
                    label { "Walk-forward months" }
                    input { r#type: "text", value: "{months}", oninput: move |e| months.set(e.value()) }
                }
                div { class: "field",
                    label { " " }
                    button { onclick: run, "Run comparison" }
                }
                div { class: "field",
                    label { " " }
                    button { class: "ghost", onclick: run_wf, "Walk-forward" }
                }
            }
            if !message().is_empty() {
                div { class: "muted-note", style: "margin-top: 10px;", "{message}" }
            }
        }

        if !rows().is_empty() {
            div { class: "card",
                div { class: "card-title", "Full-period comparison" }
                ComparisonTable { rows: rows() }
            }
        }

        if !wf().is_empty() {
            div { class: "card",
                div { class: "card-title", "Walk-forward aggregates (full windows only)" }
                {
                    let views: Vec<(String, String, bool, String, String, String, String)> = wf()
                        .iter()
                        .map(|(name, mean, sd, worst, pos, full)| {
                            (
                                name.clone(),
                                fmt_pct(*mean),
                                *mean >= 0.0,
                                fmt_pct(*sd),
                                fmt_pct(*worst),
                                format!("{pos}/{full}"),
                                full.to_string(),
                            )
                        })
                        .collect();
                    rsx! {
                        table {
                            thead {
                                tr {
                                    th { "Strategy" }
                                    th { "Mean window" }
                                    th { "Window stdev" }
                                    th { "Worst window" }
                                    th { "Positive" }
                                    th { "Full windows" }
                                }
                            }
                            tbody {
                                for (name, mean, pos_style, sd, worst, pos, full) in views {
                                    tr {
                                        td { span { class: "strategy-name", "{name}" } }
                                        td { class: if pos_style { "pos" } else { "neg" }, "{mean}" }
                                        td { "{sd}" }
                                        td { class: "neg", "{worst}" }
                                        td { "{pos}" }
                                        td { "{full}" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
struct AutomationView {
    plan: Vec<(String, String)>,
    executed: Vec<String>,
    errors: Vec<String>,
    universe: usize,
}

fn start_pass(
    execute: bool,
    cfg: AppConfig,
    view: Signal<Option<AutomationView>>,
    busy: Signal<bool>,
    note: Signal<String>,
) {
    let mut busy = busy;
    let mut note = note;
    let mut view = view;
    if busy() {
        return;
    }
    let wcfg = match watchlist_config(&cfg, execute) {
        Ok(c) => c,
        Err(e) => {
            note.set(format!("Error: {e}"));
            return;
        }
    };
    busy.set(true);
    note.set(String::new());
    spawn(async move {
        let result = async {
            let client = alpaca_client().await?;
            let has_equity = wcfg
                .universe
                .as_ref()
                .is_some_and(|u| u.iter().any(|s| s.asset_class() == paperrl_core::AssetClass::UsEquity));
            if execute && has_equity {
                let clock = client.clock().await.map_err(|e| e.to_string())?;
                if !clock.is_open {
                    return Err("the equity market is closed; equities would only queue - use the bot (crypto still trades) or wait for the open".into());
                }
            }
            paperrl_trade::run_once(&client, &wcfg)
                .await
                .map_err(|e| e.to_string())
        }
        .await;
        match result {
            Ok(report) => {
                view.set(Some(AutomationView {
                    plan: report.plan.clone(),
                    executed: report.executed.clone(),
                    errors: report.errors.clone(),
                    universe: report.universe,
                }));
                note.set(if execute {
                    format!("paper pass submitted: {} orders", report.executed.len())
                } else {
                    "dry run complete; nothing submitted".to_string()
                });
            }
            Err(e) => note.set(format!("error: {e}")),
        }
        busy.set(false);
    });
}

#[component]
fn AutomationPage() -> Element {
    let mut ctx = use_context::<AppCtx>();
    let view = use_signal(|| None::<AutomationView>);
    let busy = use_signal(|| false);
    let mut note = use_signal(String::new);
    let mut interval_secs = use_signal(|| format!("{}", ctx.config.cloned().interval_secs));
    let mut history = use_signal(Vec::<paperrl_trade::PassEntry>::new);
    let bot_running = ctx.bot.cloned().is_some();

    use_effect(move || {
        let mut h = paperrl_trade::read_pass_history(home());
        h.reverse();
        h.truncate(50);
        history.set(h);
    });

    let start_bot = move |_| {
        let cfg = ctx.config.cloned();
        let wcfg = match watchlist_config(&cfg, true) {
            Ok(c) => c,
            Err(e) => {
                note.set(format!("Error: {e}"));
                return;
            }
        };
        let secs: u64 = interval_secs().trim().parse().unwrap_or(cfg.interval_secs);
        let secs = secs.max(60);
        // remember the interval
        let mut cfg2 = cfg.clone();
        cfg2.interval_secs = secs;
        ctx.config.set(cfg2.clone());
        let _ = cfg2.save(home());
        let interval = std::time::Duration::from_secs(secs);
        spawn(async move {
            match alpaca_client().await {
                Ok(client) => {
                    let handle = paperrl_trade::spawn_bot(client, wcfg, home(), interval);
                    ctx.bot.set(Some(handle));
                    note.set("bot started; first pass runs immediately".into());
                }
                Err(e) => note.set(format!("Error: {e}")),
            }
        });
    };

    let stop_bot = move |_| {
        spawn(async move {
            if let Some(h) = ctx.bot.cloned() {
                note.set("stopping; waiting for the current pass...".into());
                h.stop().await;
                h.join().await;
                ctx.bot.set(None);
                note.set("bot stopped; its open paper orders were cancelled".into());
            }
        });
    };

    let state_chip = match ctx.bot.cloned().map(|h| h.state()) {
        Some(paperrl_trade::BotState::Running) => rsx! { span { class: "badge badge-buy live", "running" } },
        Some(paperrl_trade::BotState::Stopping) => rsx! { span { class: "badge badge-hold live", "stopping" } },
        _ => rsx! { span { class: "badge badge-hold", "idle" } },
    };

    rsx! {
        div { class: "page-head",
            div {
                PageTitle { page: Page::Automation, title: "Automation" }
                div { class: "page-sub",
                    "the bot trades your saved watchlist: equities wait for market hours, crypto passes any time"
                }
            }
            div { class: "head-actions", {state_chip} }
        }

        div { class: "card",
            div { class: "row",
                div { class: "field",
                    label { "Interval (seconds)" }
                    input { r#type: "text", value: "{interval_secs}", oninput: move |e| interval_secs.set(e.value()) }
                }
                if !bot_running {
                    button { onclick: start_bot, "Start bot" }
                } else {
                    button { class: "ghost", onclick: stop_bot, "Stop bot" }
                }
                button { class: "ghost", disabled: busy(), onclick: move |_| start_pass(false, ctx.config.cloned(), view, busy, note),
                    if busy() { "Planning..." } else { "Plan watchlist pass (dry run)" } }
                button { class: "ghost", disabled: busy() || bot_running, onclick: move |_| start_pass(true, ctx.config.cloned(), view, busy, note),
                    "Execute watchlist pass" }
                button { class: "ghost", onclick: move |_| {
                    let mut h = paperrl_trade::read_pass_history(home());
                    h.reverse();
                    h.truncate(50);
                    history.set(h);
                }, "Refresh" }
            }
            div { class: "muted-note", style: "margin-top: 10px;",
                {format!("Orders go to your Alpaca PAPER account only. The risk gate caps positions, per-order size, and cash; orders in flight block duplicate buys. Stop cancels open orders this app created. Every pass appends to {}/logs/passes.jsonl.", home().display())}
            }
            if !note().is_empty() {
                div { class: "muted-note", style: "margin-top: 8px;", "{note}" }
            }
        }

        if !history().is_empty() {
            div { class: "card",
                div { class: "card-title", "Pass history (newest first)" }
                table {
                    thead { tr { th { "Time" } th { "Mode" } th { "Planned" } th { "Executed" } th { "Skipped" } th { "Errors" } } }
                    tbody {
                        for e in history() {
                            tr { key: "{e.time}",
                                td { {fmt_pass_time(&e.time)} }
                                td { span { class: "tag", "{e.mode}" } }
                                td { "{e.planned}" }
                                td { "{e.executed}" }
                                td { {e.skipped.clone().unwrap_or_else(|| "-".into())} }
                                td { class: if e.errors.is_empty() { "" } else { "neg" }, "{e.errors.len()}" }
                            }
                        }
                    }
                }
            }
        }

        if let Some(v) = view.read().clone() {
            div { class: "card",
                div { class: "card-title", {format!("Last pass: {} watchlist symbols screened", v.universe)} }
                div { class: "card-title", "Planned orders" }
                if v.plan.is_empty() {
                    div { class: "muted-note", "No orders: nothing qualifies or everything is already positioned or pending." }
                } else {
                    table {
                        thead { tr { th { "Order" } th { "Size" } } }
                        tbody {
                            for (side, detail) in v.plan {
                                tr {
                                    td { span { class: "strategy-name", "{side}" } }
                                    td { "{detail}" }
                                }
                            }
                        }
                    }
                }
                if !v.executed.is_empty() {
                    div { class: "card-title", style: "margin-top: 14px;", "Submitted" }
                    for e in v.executed {
                        div { class: "muted-note", "{e}" }
                    }
                }
                for e in v.errors {
                    div { class: "error-box", "{e}" }
                }
            }
        }
    }
}

#[component]
fn ExperimentsPage(datasets: Signal<Vec<Dataset>>) -> Element {
    let mut models = use_signal(|| list_models(&home().join("models")));
    let mut chosen = use_signal(|| 0usize);
    let mut rows = use_signal(Vec::<BacktestRow>::new);
    let mut pick = use_signal(|| 0usize);
    let mut iters = use_signal(|| "300".to_string());
    let mut seed = use_signal(|| "42".to_string());
    let mut window = use_signal(|| "10".to_string());
    let mut training = use_signal(|| false);
    let mut train_note = use_signal(String::new);

    let dataset_list: Vec<Dataset> = datasets.read().clone();
    let list: Vec<ModelEntry> = models.read().clone();
    let detail: Option<ModelEntry> = list.get(*chosen.read()).cloned();
    let train_datasets = dataset_list.clone();

    // Training runs in-process on the local bars, the same call the CLI makes.
    let train = move |_| {
        if training() {
            return;
        }
        let Some(ds) = train_datasets.get(pick()) else {
            train_note.set("Fetch market data in Markets first; training reads cached bars.".into());
            return;
        };
        let bars = match load_bars(&home().join("datasets"), &ds.symbol, &ds.timeframe) {
            Ok(b) => b,
            Err(e) => {
                train_note.set(format!("Could not read bars: {e}"));
                return;
            }
        };
        let iters_n: usize = iters().trim().parse().unwrap_or(300).max(1);
        let seed_n: u64 = seed().trim().parse().unwrap_or(42);
        let window_n: usize = window().trim().parse().unwrap_or(10).max(1);
        if bars.len() <= window_n + 1 {
            train_note.set(format!(
                "Need more than {} bars to train; {} has {}.",
                window_n + 1,
                ds.symbol,
                bars.len()
            ));
            return;
        }
        let symbol = match paperrl_core::Symbol::parse(&ds.symbol) {
            Ok(s) => s,
            Err(_) => {
                train_note.set("that dataset is not a valid ticker".into());
                return;
            }
        };
        let slug = symbol.file_slug();
        let symbol_name = symbol.to_string();
        let tf_slug = ds.timeframe.clone();
        let path = home().join("models").join(format!("ppo-{slug}-{seed_n}.bpk"));
        training.set(true);
        train_note.set(format!(
            "Training {symbol_name} {tf_slug}: {iters_n} iterations, seed {seed_n}. Runs on a background thread, so the rest of the app stays usable."
        ));
        spawn(async move {
            let outcome = tokio::task::spawn_blocking(move || {
                let env_cfg = paperrl_rl::EnvConfig { window: window_n, ..Default::default() };
                let ppo_cfg = paperrl_rl::PpoConfig { iters: iters_n, seed: seed_n, ..Default::default() };
                let trained = paperrl_rl::train_ppo(&bars, env_cfg, ppo_cfg, &symbol_name, &tf_slug);
                std::fs::create_dir_all(home().join("models"))?;
                trained.save(&path)?;
                Ok::<_, paperrl_core::Error>(path)
            })
            .await;
            match outcome {
                Ok(Ok(saved)) => {
                    models.set(list_models(&home().join("models")));
                    train_note.set(format!("checkpoint saved: {}", saved.display()));
                }
                Ok(Err(e)) => train_note.set(format!("Training failed: {e}")),
                Err(e) => train_note.set(format!("Training task stopped: {e}")),
            }
            training.set(false);
        });
    };

    rsx! {
        div { class: "page-head",
            div {
                PageTitle { page: Page::Experiments, title: "Experiments" }
                div { class: "page-sub", "{list.len()} PPO checkpoints in models/ with reproducibility metadata" }
            }
        }

        div { class: "card",
            div { class: "card-title", "Train a checkpoint" }
            if dataset_list.is_empty() {
                div { class: "muted-note",
                    "No cached bars yet. Open Markets, fetch market data, then train here: no terminal command needed."
                }
            } else {
                div { class: "row",
                    div { class: "field", style: "min-width: 230px;",
                        label { "Dataset" }
                        select {
                            value: "{pick}",
                            onchange: move |e| {
                                if let Ok(i) = e.value().parse::<usize>() { pick.set(i); }
                            },
                            for (i, d) in dataset_list.iter().enumerate() {
                                option {
                                    value: "{i}",
                                    selected: i == *pick.read(),
                                    "{d.symbol} · {d.timeframe} · {d.bars} bars"
                                }
                            }
                        }
                    }
                    div { class: "field", style: "width: 92px;",
                        label { "Iterations" }
                        input { r#type: "text", value: "{iters}", oninput: move |e| iters.set(e.value()) }
                    }
                    div { class: "field", style: "width: 74px;",
                        label { "Seed" }
                        input { r#type: "text", value: "{seed}", oninput: move |e| seed.set(e.value()) }
                    }
                    div { class: "field", style: "width: 76px;",
                        label { "Window" }
                        input { r#type: "text", value: "{window}", oninput: move |e| window.set(e.value()) }
                    }
                    div { class: "field",
                        label { " " }
                        button { onclick: train, disabled: training(),
                            if training() { "Training..." } else { "Train checkpoint" }
                        }
                    }
                }
                if !train_note().is_empty() {
                    div { class: "muted-note", style: "margin-top: 10px;", "{train_note}" }
                }
                div { class: "muted-note", style: "margin-top: 8px;",
                    "Writes models/ppo-SYMBOL-SEED.bpk plus its JSON metadata. Strategy Lab, Signals and Automation pick the newest checkpoint up automatically."
                }
            }
        }

        div { class: "card",
            div { class: "row", style: "justify-content: space-between; margin-bottom: 12px;",
                div { class: "card-title", style: "margin-bottom: 0;", "Checkpoints" }
                button {
                    class: "ghost",
                    onclick: move |_| models.set(list_models(&home().join("models"))),
                    "Refresh"
                }
            }
            if list.is_empty() {
                div { class: "muted-note",
                    "No checkpoints yet. Train one above; it will appear here with its seed, iterations and training curve."
                }
            } else {
                table { class: "clickable",
                    thead {
                        tr {
                            th { "Symbol" }
                            th { "Timeframe" }
                            th { "Seed" }
                            th { "Iters" }
                            th { "Window" }
                            th { "Final reward" }
                        }
                    }
                    tbody {
                        for (i, m) in list.iter().enumerate() {
                            tr {
                                class: if i == *chosen.read() { "selected" } else { "" },
                                onclick: move |_| { chosen.set(i); rows.set(Vec::new()); },
                                td { span { class: "strategy-name", "{m.meta.symbol}" } }
                                td { span { class: "tag", "{m.meta.timeframe}" } }
                                td { "{m.meta.config.seed}" }
                                td { "{m.meta.config.iters}" }
                                td { "{m.meta.window}" }
                                td {
                                    if let Some(s) = m.meta.stats.last() {
                                        {format!("{:+.4}", s.mean_reward)}
                                    } else {
                                        "-"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(m) = detail {
            div { class: "card",
                div { class: "card-title", "Training curve: {m.meta.symbol} · seed {m.meta.config.seed}" }
                {
                    let rewards: Vec<f64> = m.meta.stats.iter().map(|s| s.mean_reward as f64).collect();
                    let pts = sparkline_points(&rewards, 880.0, 120.0);
                    rsx! {
                        svg { view_box: "0 0 880 120", width: "100%", height: "120",
                            polyline {
                                class: "draw-line",
                                path_length: "1",
                                points: "{pts}",
                                fill: "none",
                                stroke: "#08B98B",
                                stroke_width: "2"
                            }
                        }
                    }
                }
                {
                    let note = format!(
                        "Mean step reward per iteration. Metadata: {} iters, window {}, dd {:.2}, turnover {:.4}, vol {:.3}.",
                        m.meta.config.iters,
                        m.meta.window,
                        m.meta.env.dd_penalty,
                        m.meta.env.turnover_penalty,
                        m.meta.env.vol_penalty
                    );
                    rsx! {
                        div { class: "muted-note", style: "margin-top: 6px;", "{note}" }
                    }
                }
                div { class: "row", style: "margin-top: 12px;",
                    button {
                        onclick: move |_| {
                            let Ok(bars) = load_bars(&home().join("datasets"), &m.meta.symbol, &m.meta.timeframe) else {
                                return;
                            };
                            let tf = tf_of(&Dataset {
                                symbol: m.meta.symbol.clone(),
                                timeframe: m.meta.timeframe.clone(),
                                bars: bars.len(),
                                first: bars.first().map(|b| b.timestamp).unwrap_or_default(),
                                last: bars.last().map(|b| b.timestamp).unwrap_or_default(),
                            });
                            rows.set(run_comparison(&bars, tf, SimParams::default(), Some(&m.model_path)));
                        },
                        "Backtest this checkpoint"
                    }
                }
            }
            if !rows().is_empty() {
                div { class: "card",
                    div { class: "card-title", "Model versus baselines" }
                    ComparisonTable { rows: rows() }
                }
            }
        }
    }
}

#[component]
fn SettingsPage(datasets: Signal<Vec<Dataset>>) -> Element {
    let mut ctx = use_context::<AppCtx>();
    let mut currency = use_signal(|| ctx.config.cloned().base_currency);
    let mut interval_secs = use_signal(|| format!("{}", ctx.config.cloned().interval_secs));
    let mut max_positions = use_signal(|| format!("{}", ctx.config.cloned().risk.max_positions));
    let mut position_pct = use_signal(|| format!("{:.0}", ctx.config.cloned().risk.position_pct * 100.0));
    let mut max_order = use_signal(|| format!("{}", ctx.config.cloned().risk.max_order_notional as i64));
    let mut note = use_signal(String::new);
    let mut key_id = use_signal(String::new);
    let mut secret = use_signal(String::new);
    let mut cred_note = use_signal(String::new);
    let mut testing = use_signal(|| false);

    let credentials_source = paperrl_trade::credential_source(home());
    let saved_key_hint = paperrl_trade::load_credentials(home())
        .map(|c| paperrl_trade::mask(&c.key_id));
    let key_placeholder = saved_key_hint
        .clone()
        .map(|m| format!("saved key {m}"))
        .unwrap_or_else(|| "PK...".to_string());

    let save_keys = move |_| {
        match paperrl_trade::save_credentials(home(), &key_id(), &secret()) {
            Ok(path) => {
                key_id.set(String::new());
                secret.set(String::new());
                cred_note.set(format!("keys saved to {}", path.display()));
                // The keys are usable now: run the same pass a launch would.
                run_startup(ctx, datasets);
            }
            Err(e) => cred_note.set(format!("Error: {e}")),
        }
    };

    let clear_keys = move |_| match paperrl_trade::clear_credentials(home()) {
        Ok(true) => {
            cred_note.set("saved keys removed; environment variables apply again".into());
            ctx.boot.set(BootStatus::failed(
                "No Alpaca paper keys now: save a key pair here to fetch data and place paper orders.",
            ));
        }
        Ok(false) => cred_note.set("no saved keys to remove".into()),
        Err(e) => cred_note.set(format!("Could not remove: {e}")),
    };

    let test_keys = move |_| {
        testing.set(true);
        cred_note.set("Testing the paper account connection...".into());
        spawn(async move {
            let result = async {
                let client = alpaca_client().await?;
                let acct = client.account().await.map_err(|e| e.to_string())?;
                Ok::<String, String>(format!(
                    "connected: paper account {} · equity {}",
                    acct.status,
                    acct.equity.unwrap_or_else(|| "-".into())
                ))
            }
            .await;
            match result {
                Ok(text) => cred_note.set(text),
                Err(e) => cred_note.set(format!("Error: {e}")),
            }
            testing.set(false);
        });
    };

    let save = move |_| {
        let risk = paperrl_trade::RiskLimits {
            max_positions: max_positions().trim().parse().unwrap_or(5),
            position_pct: (position_pct().trim().parse::<f64>().unwrap_or(20.0) / 100.0).clamp(0.01, 1.0),
            max_order_notional: max_order().trim().parse().unwrap_or(5_000.0),
            min_order_notional: 50.0,
        };
        let mut cfg = ctx.config.cloned();
        cfg.base_currency = currency();
        cfg.interval_secs = interval_secs().trim().parse::<u64>().unwrap_or(3600).max(60);
        cfg.risk = risk;
        ctx.config.set(cfg.clone());
        match cfg.save(home()) {
            Ok(_) => note.set("settings saved".into()),
            Err(e) => note.set(format!("Could not save: {e}")),
        }
    };

    rsx! {
        div { class: "page-head",
            div {
                PageTitle { page: Page::Settings, title: "Settings" }
                div { class: "page-sub", "currency, risk limits, paths, and credentials" }
            }
        }
        div { class: "card",
            div { class: "card-title", "Display and bot" }
            div { class: "row",
                div { class: "field",
                    label { "Base currency" }
                    select {
                        value: "{currency}",
                        onchange: move |e| currency.set(e.value()),
                        for (code, sym) in paperrl_core::CURRENCIES {
                            option { value: "{code}", selected: *currency.read() == *code, "{code} ({sym})" }
                        }
                    }
                }
                div { class: "field",
                    label { "Bot interval (seconds)" }
                    input { r#type: "text", value: "{interval_secs}", oninput: move |e| interval_secs.set(e.value()) }
                }
            }
            if !note().is_empty() {
                div { class: "muted-note", style: "margin-top: 10px;", "{note}" }
            }
        }
        div { class: "card",
            div { class: "card-title", "Risk limits" }
            div { class: "row",
                div { class: "field",
                    label { "Max positions" }
                    input { r#type: "text", value: "{max_positions}", oninput: move |e| max_positions.set(e.value()) }
                }
                div { class: "field",
                    label { "Position size (% of equity)" }
                    input { r#type: "text", value: "{position_pct}", oninput: move |e| position_pct.set(e.value()) }
                }
                div { class: "field",
                    label { "Max order (USD)" }
                    input { r#type: "text", value: "{max_order}", oninput: move |e| max_order.set(e.value()) }
                }
                div { class: "field",
                    label { " " }
                    button { onclick: save, "Save settings" }
                }
            }
            div { class: "muted-note", style: "margin-top: 10px;",
                "The bot and the confirm dialog read these when they start a pass; changes apply to the next pass."
            }
        }
        div { class: "card",
            div { class: "card-title", "Paths" }
            div { class: "muted-note",
                {format!("Data directory: {}. Datasets in datasets/, checkpoints in models/, config.json and logs/ alongside (PAPERRL_HOME overrides; a repo checkout wins; otherwise ~/paperrl).", home().display())}
            }
        }
        div { class: "card",
            div { class: "card-title", "Alpaca paper keys" }
            match credentials_source {
                CredentialSource::Saved => rsx! {
                    div { class: "row",
                        span { class: "tag", "saved" }
                        span { class: "dim", "key id {saved_key_hint.clone().unwrap_or_default()}" }
                    }
                    div { class: "muted-note", style: "margin-top: 8px;",
                        {format!(
                            "Stored in {}. Used when APCA_API_KEY_ID and APCA_API_SECRET are not set; the environment variables take precedence.",
                            paperrl_trade::credentials_path(home()).display()
                        )}
                    }
                },
                CredentialSource::Environment => rsx! {
                    div { span { class: "tag", "environment" } }
                    div { class: "muted-note", style: "margin-top: 8px;",
                        "APCA_API_KEY_ID and APCA_API_SECRET are set and take precedence over saved keys. Keys saved below are the fallback when the variables are absent."
                    }
                },
                CredentialSource::Missing => rsx! {
                    div { class: "error-box",
                        "No keys yet. Paste your Alpaca paper key id and secret below; market data, signals, the portfolio and paper orders need them. Backtests and the Strategy Lab work offline without keys."
                    }
                },
            }
            div { class: "row", style: "margin-top: 12px;",
                div { class: "field", style: "flex: 1; min-width: 220px;",
                    label { "Key ID" }
                    input {
                        r#type: "text",
                        value: "{key_id}",
                        placeholder: "{key_placeholder}",
                        oninput: move |e| key_id.set(e.value()),
                    }
                }
                div { class: "field", style: "flex: 1; min-width: 220px;",
                    label { "Secret" }
                    input {
                        r#type: "password",
                        value: "{secret}",
                        placeholder: "paper secret",
                        oninput: move |e| secret.set(e.value()),
                    }
                }
            }
            div { class: "row", style: "margin-top: 10px;",
                button { onclick: save_keys, "Save keys" }
                button {
                    class: "ghost",
                    disabled: testing(),
                    onclick: test_keys,
                    if testing() { "Testing..." } else { "Test connection" }
                }
                button { class: "ghost", onclick: clear_keys, "Remove saved keys" }
            }
            if !cred_note().is_empty() {
                div { class: "muted-note", style: "margin-top: 10px;", "{cred_note}" }
            }
            div { class: "muted-note", style: "margin-top: 8px;",
                "Keys stay on this machine in the app data folder, owner-only where the OS supports it. The trading client is hardcoded to Alpaca's paper endpoint."
            }
        }
        div { class: "card",
            div { class: "card-title", "Safety model" }
            div { class: "muted-note",
                "The trading client is hardcoded to the Alpaca paper endpoint, so live-money orders are unreachable from this build. Stop cancels only orders this app placed (client order ids starting with paperrl-)."
            }
        }
    }
}
