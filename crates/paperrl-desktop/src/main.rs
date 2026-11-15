//! PaperRL desktop dashboard: a local research desk over the same engine the
//! CLI uses (Alpaca data cache, simulator, PPO checkpoints).

use std::path::{Path, PathBuf};

use dioxus::prelude::*;
use paperrl_desktop::{
    chart_geometry, fmt_date, fmt_money_cents, fmt_pct, fmt_price, list_datasets,
    list_models, load_bars, run_comparison, run_walk_forward, sparkline_points, BacktestRow,
    Dataset, ModelEntry, SimParams,
};

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

fn main() {
    // PAPERRL_START_PAGE lets screenshots and demos open a specific page.
    // Retry once with a clear message: on Windows, WebView2 cannot create
    // its composition controller while the session is locked or the desktop
    // is not rendering (LogonUI active), and it fails with an opaque panic.
    for attempt in 1..=2 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dioxus::LaunchBuilder::desktop()
                .with_cfg(
                    dioxus::desktop::Config::new()
                        .with_icon(app_icon())
                        .with_window(
                            dioxus::desktop::WindowBuilder::new()
                                .with_title("PaperRL")
                                .with_inner_size(dioxus::desktop::LogicalSize::new(
                                    1280.0, 800.0,
                                )),
                        ),
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
        _ => Page::Dashboard,
    }
}

const DATASETS_ROOT: &str = "datasets";
const MODELS_ROOT: &str = "models";

#[derive(Clone, Copy, PartialEq)]
enum Page {
    Dashboard,
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
        svg {
            width: "32",
            height: "32",
            view_box: "0 0 64 64",
            rect { width: "64", height: "64", rx: "16", fill: "#101C2C" }
            path { d: "M18 12h19l11 11v29H18z", fill: "#EAF2F7" }
            path { d: "M37 12v12h11", fill: "#C8D9E5" }
            path {
                d: "M24 42l8-9 7 4 8-11",
                fill: "none",
                stroke: "#08B98B",
                stroke_width: "3.2",
                stroke_linecap: "round",
                stroke_linejoin: "round"
            }
            circle { cx: "24", cy: "42", r: "2.5", fill: "#08B98B" }
            circle { cx: "32", cy: "33", r: "2.5", fill: "#08B98B" }
            circle { cx: "39", cy: "37", r: "2.5", fill: "#08B98B" }
            circle { cx: "47", cy: "26", r: "3", fill: "#F5B94C" }
        }
    }
}

#[allow(non_snake_case)]
fn App() -> Element {
    let mut page = use_signal(initial_page);
    let datasets = use_signal(|| list_datasets(Path::new(DATASETS_ROOT)));
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

    let body = match *page.read() {
        Page::Dashboard => rsx! { DashboardPage { datasets, sel } },
        Page::Markets => rsx! { MarketsPage { datasets, sel } },
        Page::Lab => rsx! { LabPage { datasets, sel } },
        Page::Automation => rsx! { AutomationPage {} },
        Page::Experiments => rsx! { ExperimentsPage {} },
        Page::Settings => rsx! { SettingsPage {} },
    };

    rsx! {
        style { dangerous_inner_html: paperrl_desktop::theme::CSS.to_string() }
        div { class: "app",
            aside { class: "sidebar",
                div { class: "brand",
                    Logo {}
                    div {
                        div { class: "brand-name", "PaperRL" }
                        div { class: "brand-sub", "local research desk" }
                    }
                }
                for (p, icon, label) in [
                    (Page::Dashboard, "▦", "Dashboard"),
                    (Page::Markets, "≋", "Markets"),
                    (Page::Lab, "⚗", "Strategy Lab"),
                    (Page::Automation, "⇄", "Automation"),
                    (Page::Experiments, "◇", "Experiments"),
                    (Page::Settings, "⚙", "Settings"),
                ] {
                    div {
                        class: if *page.read() == p { "nav-item active" } else { "nav-item" },
                        onclick: move |_| page.set(p),
                        span { class: "nav-icon", "{icon}" }
                        span { "{label}" }
                    }
                }
                div { class: "sidebar-foot",
                    div { "Paper trading only" }
                    span { class: "safety-pill", "no live orders" }
                }
            }
            main { class: "content", {body} }
        }
    }
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
            span { class: "dim", "no datasets yet; run `paperrl data sync` first" }
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
        Some(ds) => load_bars(Path::new(DATASETS_ROOT), &ds.symbol, &ds.timeframe).unwrap_or_default(),
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
    let p = Path::new(MODELS_ROOT).join(format!("ppo-{symbol}-42.bpk"));
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
                div { class: "page-title", "Dashboard" }
                div { class: "page-sub",
                    if let Some(ds) = ds.as_ref() {
                        "{ds.symbol} · {ds.timeframe} · {ds.bars} bars · {fmt_date(ds.first)} to {fmt_date(ds.last)}"
                    } else {
                        "local-first RL trading research"
                    }
                }
            }
            div { class: "head-actions", SymbolSelector { datasets, sel } }
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
fn Stat(label: String, value: String, sub: String, class: String) -> Element {
    rsx! {
        div { class: "stat",
            div { class: "stat-label", "{label}" }
            div { class: "stat-value {class}", "{value}" }
            div { class: "stat-sub", "{sub}" }
        }
    }
}

#[component]
fn AccountCard() -> Element {
    let mut account = use_signal(|| None::<Result<String, String>>);
    use_effect(move || {
        spawn(async move {
            let result = async {
                let creds = paperrl_alpaca::Credentials::from_env().map_err(|e| e.to_string())?;
                let client =
                    paperrl_alpaca::AlpacaClient::new(&creds).map_err(|e| e.to_string())?;
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
                    let equity = parts.get(1).copied().unwrap_or("-");
                    let cash = parts.get(2).copied().unwrap_or("-");
                    let bp = parts.get(3).copied().unwrap_or("-");
                    rsx! {
                        div { class: "row", style: "justify-content: space-between;",
                            span { class: "dim", "status" }
                            span { class: "tag", "{parts.first().copied().unwrap_or(\"-\")}" }
                        }
                        div { class: "grid-4", style: "grid-template-columns: repeat(3, 1fr); margin-top: 12px; margin-bottom: 0;",
                            Stat { label: "Equity".to_string(), value: equity.to_string(), sub: "paper".to_string(), class: String::new() }
                            Stat { label: "Cash".to_string(), value: cash.to_string(), sub: "available".to_string(), class: String::new() }
                            Stat { label: "Buying power".to_string(), value: bp.to_string(), sub: "margin".to_string(), class: String::new() }
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
    let mut sel = sel;
    let list = datasets.read();
    let ds = selected_dataset(&datasets, &sel);
    let bars = use_memo(move || bars_for(&datasets, &sel));
    let chart_bars: Vec<paperrl_core::Bar> = bars().iter().rev().take(120).rev().cloned().collect();

    rsx! {
        div { class: "page-head",
            div {
                div { class: "page-title", "Markets" }
                div { class: "page-sub", "{list.len()} local datasets cached from Alpaca" }
            }
        }
        div { class: "card",
            if list.is_empty() {
                div { class: "muted-note",
                    "Nothing cached yet. From a terminal: `paperrl data sync --symbols AAPL,MSFT,NVDA --years 3`"
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
        let params = SimParams {
            cash: cash().parse().unwrap_or(100_000.0),
            fee_bps: fee().parse().unwrap_or(1.0),
            slippage_bps: slip().parse().unwrap_or(5.0),
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
        let params = SimParams {
            cash: cash().parse().unwrap_or(100_000.0),
            fee_bps: fee().parse().unwrap_or(1.0),
            slippage_bps: slip().parse().unwrap_or(5.0),
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
                div { class: "page-title", "Strategy Lab" }
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
    gainers: Vec<(String, String, String, String)>,
    losers: Vec<(String, String, String, String)>,
    plan: Vec<(String, String)>,
    executed: Vec<String>,
    errors: Vec<String>,
    universe: usize,
}

fn start_pass(
    execute: bool,
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
    busy.set(true);
        note.set(String::new());
        spawn(async move {
            let result = async {
                let creds =
                    paperrl_alpaca::Credentials::from_env().map_err(|e| e.to_string())?;
                let client =
                    paperrl_alpaca::AlpacaClient::new(&creds).map_err(|e| e.to_string())?;
                let cfg = paperrl_trade::AutoConfig {
                    top: 100,
                    long_count: 3,
                    limits: paperrl_trade::RiskLimits::default(),
                    datasets: std::path::PathBuf::from("datasets"),
                    model_dir: std::path::PathBuf::from("models"),
                    filter: paperrl_trade::ScreenFilter::default(),
                    execute,
                    stale_days: 5,
                    universe: None,
                };
                paperrl_trade::run_once(&client, &cfg)
                    .await
                    .map_err(|e| e.to_string())
            }
            .await;
            match result {
                Ok(report) => {
                    let row = |c: &paperrl_trade::Candidate| {
                        (
                            c.symbol.clone(),
                            fmt_pct(c.ret20),
                            fmt_pct(c.ret5),
                            fmt_pct(c.score),
                        )
                    };
                    view.set(Some(AutomationView {
                        gainers: report.gainers.iter().take(10).map(row).collect(),
                        losers: report.losers.iter().rev().take(10).map(row).collect(),
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
    let view = use_signal(|| None::<AutomationView>);
    let busy = use_signal(|| false);
    let note = use_signal(String::new);

    rsx! {
        div { class: "page-head",
            div {
                div { class: "page-title", "Automation" }
                div { class: "page-sub",
                    "momentum screen over the top 100 most active symbols; sells losers, buys leaders through the risk gate"
                }
            }
            div { class: "head-actions",
                button {
                    class: "ghost",
                    disabled: busy(),
                    onclick: move |_| start_pass(false, view, busy, note),
                    if busy() { "Scanning..." } else { "Scan (dry run)" }
                }
                button {
                    disabled: busy(),
                    onclick: move |_| start_pass(true, view, busy, note),
                    "Execute paper pass"
                }
            }
        }
        div { class: "card",
            div { class: "muted-note",
                "Orders go to your Alpaca PAPER account only. The risk gate caps positions, per-order size, and cash; orders in flight block duplicate buys. Every submission is appended to logs/orders.csv."
            }
            if !note().is_empty() {
                div { class: "muted-note", style: "margin-top: 8px;", "{note}" }
            }
        }
        if let Some(v) = view.read().clone() {
            div { class: "card",
                div { class: "card-title", "Universe: {v.universe} symbols screened" }
                div { class: "split",
                    div {
                        div { class: "card-title", "Gaining momentum" }
                        table {
                            thead { tr { th { "Symbol" } th { "20d" } th { "5d" } th { "Score" } } }
                            tbody {
                                for (sym, r20, r5, score) in v.gainers {
                                    tr {
                                        td { span { class: "strategy-name", "{sym}" } }
                                        td { class: "pos", "{r20}" }
                                        td { class: "pos", "{r5}" }
                                        td { "{score}" }
                                    }
                                }
                            }
                        }
                    }
                    div {
                        div { class: "card-title", "Losing momentum" }
                        table {
                            thead { tr { th { "Symbol" } th { "20d" } th { "5d" } th { "Score" } } }
                            tbody {
                                for (sym, r20, r5, score) in v.losers {
                                    tr {
                                        td { span { class: "strategy-name", "{sym}" } }
                                        td { class: "neg", "{r20}" }
                                        td { "{r5}" }
                                        td { "{score}" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            div { class: "card",
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
fn ExperimentsPage() -> Element {
    let models = use_signal(|| list_models(Path::new(MODELS_ROOT)));
    let mut chosen = use_signal(|| 0usize);
    let mut rows = use_signal(Vec::<BacktestRow>::new);
    let list = models.read();

    let detail: Option<ModelEntry> = list.get(*chosen.read()).cloned();
    rsx! {
        div { class: "page-head",
            div {
                div { class: "page-title", "Experiments" }
                div { class: "page-sub", "{list.len()} PPO checkpoints in models/ with reproducibility metadata" }
            }
        }
        div { class: "card",
            if list.is_empty() {
                div { class: "muted-note",
                    "No checkpoints yet. Try: `paperrl train --symbol AAPL --iters 300 --seed 42`"
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
                            let Ok(bars) = load_bars(Path::new(DATASETS_ROOT), &m.meta.symbol, &m.meta.timeframe) else {
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
fn SettingsPage() -> Element {
    let has_keys = std::env::var("APCA_API_KEY_ID").is_ok() && std::env::var("APCA_API_SECRET").is_ok();
    rsx! {
        div { class: "page-head",
            div {
                div { class: "page-title", "Settings" }
                div { class: "page-sub", "paths and environment" }
            }
        }
        div { class: "card",
            div { class: "card-title", "Paths" }
            div { class: "muted-note",
                "Datasets are read from {DATASETS_ROOT}/ and checkpoints from {MODELS_ROOT}/ relative to the working directory. Launch from the repository root, or override with a shortcut that sets the working directory."
            }
        }
        div { class: "card",
            div { class: "card-title", "Alpaca credentials" }
            if has_keys {
                div { span { class: "tag", "configured" } }
                div { class: "muted-note", style: "margin-top: 8px;",
                    "APCA_API_KEY_ID and APCA_API_SECRET are set. The dashboard reads the paper account; order placement stays disabled until the risk engine ships."
                }
            } else {
                div { class: "error-box", "APCA_API_KEY_ID / APCA_API_SECRET are not set. Data sync and account reads need them." }
            }
        }
        div { class: "card",
            div { class: "card-title", "Safety model" }
            div { class: "muted-note",
                "The trading client is hardcoded to the Alpaca paper endpoint, so live-money orders are unreachable from this build. Backtests and training run fully offline against cached Parquet data."
            }
        }
    }
}
