//! App stylesheet: one constant so the app ships a single inline style block
//! with no asset pipeline. Design language: layered near-black navy surfaces
//! with hairline borders (Linear), one emerald accent reserved for positive
//! states (fintech price semantics), gold for warnings, tabular numerals,
//! and short purposeful motion (Vercel-style tokens). Glass blur appears on
//! exactly one surface per page (the sticky page header).

pub const CSS: &str = r#"
/* ---------- tokens ---------- */
:root {
  --bg: #0B1220;
  --surface-1: #0F1828;
  --surface-2: #141F33;
  --surface-3: #1B2B42;
  --border: rgba(255, 255, 255, 0.07);
  --border-strong: rgba(255, 255, 255, 0.14);
  --text: #EAF2F7;
  --text-2: #9FB2C4;
  --text-3: #64748B;
  --accent: #08B98B;
  --accent-bright: #10D69E;
  --accent-dim: rgba(8, 185, 139, 0.14);
  --gold: #F5B94C;
  --red: #F2685C;
  --ease: cubic-bezier(0.2, 0, 0, 1);
  --ease-swift: cubic-bezier(0.175, 0.885, 0.32, 1.1);
  --dur-fast: 140ms;
  --dur: 200ms;
  --dur-slow: 320ms;
  --radius: 14px;
  --radius-sm: 9px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, #main { height: 100%; }
body {
  font-family: "Segoe UI Variable Display", "Segoe UI", Inter, system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  letter-spacing: -0.01em;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

/* atmosphere: two faint color washes plus a hairline grid, kept away from data */
.app {
  background:
    radial-gradient(900px 500px at 88% -8%, rgba(8, 185, 139, 0.07), transparent 60%),
    radial-gradient(700px 420px at -6% 104%, rgba(245, 185, 76, 0.05), transparent 55%),
    linear-gradient(rgba(255, 255, 255, 0.016) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.016) 1px, transparent 1px),
    var(--bg);
  background-size: auto, auto, 32px 32px, 32px 32px, auto;
}
.app { display: grid; grid-template-columns: 232px minmax(0, 1fr); height: 100vh; overflow: hidden; }

::selection { background: rgba(8, 185, 139, 0.35); }
:focus-visible { outline: 2px solid rgba(8, 185, 139, 0.55); outline-offset: 2px; border-radius: 4px; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.08); border-radius: 999px;
  border: 3px solid transparent; background-clip: padding-box;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.16); background-clip: padding-box; border: 3px solid transparent; }

/* ---------- motion ---------- */
@keyframes rise { from { opacity: 0; transform: translateY(9px); } to { opacity: 1; transform: none; } }
@keyframes pop { from { opacity: 0; transform: translateY(5px) scale(0.98); } to { opacity: 1; transform: none; } }
@keyframes overlay-in { from { opacity: 0; transform: scale(0.96) translateY(5px); } to { opacity: 1; transform: none; } }
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
@keyframes draw { to { stroke-dashoffset: 0; } }
@keyframes candle-in { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: none; } }
@keyframes pulse-ring {
  0% { transform: scale(0.9); opacity: 0.7; }
  70% { transform: scale(2.1); opacity: 0; }
  100% { transform: scale(2.1); opacity: 0; }
}

.page > * { animation: rise var(--dur-slow) var(--ease) backwards; }
.page > *:nth-child(1) { animation-delay: 0ms; }
.page > *:nth-child(2) { animation-delay: 40ms; }
.page > *:nth-child(3) { animation-delay: 80ms; }
.page > *:nth-child(4) { animation-delay: 120ms; }
.page > *:nth-child(5) { animation-delay: 160ms; }
.page > *:nth-child(6) { animation-delay: 200ms; }
.page > *:nth-child(7) { animation-delay: 240ms; }
.page > *:nth-child(n+8) { animation-delay: 280ms; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* ---------- sidebar ---------- */
.sidebar {
  background: linear-gradient(180deg, #0C1524 0%, #0A1220 100%);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  padding: 16px 12px;
  gap: 3px;
}
.brand { display: flex; align-items: center; gap: 10px; padding: 6px 10px 16px; }
.brand-name { font-weight: 650; font-size: 15px; letter-spacing: -0.02em; }
.brand-sub { font-size: 10.5px; color: var(--text-3); letter-spacing: 0.02em; }
.nav-item {
  position: relative;
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px; border-radius: 10px; cursor: pointer;
  color: var(--text-2); font-weight: 500; border: 1px solid transparent;
  transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease),
    transform var(--dur-fast) var(--ease);
}
.nav-item:hover { background: rgba(255, 255, 255, 0.04); color: var(--text); transform: translateX(2px); }
.nav-item.active {
  background: var(--accent-dim); color: var(--text); border-color: rgba(8, 185, 139, 0.25);
}
.nav-item.active::before {
  content: ""; position: absolute; left: 0; top: 8px; bottom: 8px; width: 3px;
  border-radius: 999px; background: var(--accent);
  box-shadow: 0 0 12px rgba(8, 185, 139, 0.8);
}
.nav-item:active { transform: translateX(2px) scale(0.985); }
.nav-icon { width: 18px; text-align: center; opacity: 0.9; }
.sidebar-foot { margin-top: auto; padding: 10px; font-size: 11px; color: var(--text-3); line-height: 1.5; }
.safety-pill {
  display: inline-block; margin-top: 6px; padding: 2px 9px; border-radius: 999px;
  background: var(--accent-dim); color: var(--accent); border: 1px solid rgba(8, 185, 139, 0.3);
  font-size: 10.5px;
}

/* ---------- content shell ---------- */
.content { overflow-y: auto; overflow-x: hidden; min-width: 0; padding: 0 26px 40px; }
.page { padding-top: 16px; }

/* sticky glass page header: the one blurred surface */
.page-head {
  position: sticky; top: 0; z-index: 10;
  display: flex; align-items: baseline; justify-content: space-between;
  margin: 0 -26px 16px; padding: 14px 26px 12px;
  gap: 16px; flex-wrap: wrap;
  background: rgba(11, 18, 32, 0.72);
  -webkit-backdrop-filter: blur(14px);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--border);
}
@supports not (backdrop-filter: blur(1px)) { .page-head { background: rgba(11, 18, 32, 0.97); } }
.page-title { font-size: 21px; font-weight: 650; letter-spacing: -0.02em; }
.page-sub { color: var(--text-2); font-size: 12.5px; margin-top: 3px; }
.head-actions { display: flex; gap: 8px; align-items: center; }

/* ---------- cards ---------- */
.card {
  position: relative;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.022), rgba(255, 255, 255, 0) 42%), var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px; margin-bottom: 16px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.28), 0 10px 28px rgba(0, 0, 0, 0.16);
  transition: border-color var(--dur) var(--ease), transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}
.card:hover { border-color: var(--border-strong); }
.card-title { font-size: 12.5px; font-weight: 600; color: #C8D9E5; margin-bottom: 12px; letter-spacing: 0.02em; }

.grid-4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
.stat, .card { min-width: 0; }
.stat {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.022), rgba(255, 255, 255, 0) 42%), var(--surface-2);
  border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
  transition: border-color var(--dur) var(--ease), transform var(--dur) var(--ease);
}
.stat:hover { border-color: var(--border-strong); transform: translateY(-2px); }
.stat-label {
  font-size: 10.5px; color: var(--text-3); text-transform: uppercase;
  letter-spacing: 0.09em; font-weight: 600;
}
.stat-value { font-size: 22px; font-weight: 650; margin-top: 6px; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; animation: pop var(--dur-slow) var(--ease-swift) backwards; animation-delay: 120ms; }
.stat-sub { font-size: 11.5px; color: var(--text-3); margin-top: 4px; }
.pos { color: var(--accent); }
.neg { color: var(--red); }
.dim { color: var(--text-3); }

/* ---------- tables ---------- */
table { width: 100%; border-collapse: separate; border-spacing: 0; font-variant-numeric: tabular-nums; }
th {
  text-align: right; font-size: 10.5px; color: var(--text-3); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.09em;
  padding: 8px 10px; border-bottom: 1px solid var(--border);
}
th:first-child, td:first-child { text-align: left; }
td { padding: 9px 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.045); font-size: 13px; text-align: right; }
tbody tr { transition: background var(--dur-fast) var(--ease); }
tbody tr:hover td { background: rgba(255, 255, 255, 0.03); }
tbody tr:hover td:first-child { box-shadow: inset 2px 0 0 var(--accent); }
tr.selected td { background: rgba(8, 185, 139, 0.07) !important; }
.clickable tbody tr { cursor: pointer; }
.strategy-name { font-weight: 600; color: var(--text); }
.tag {
  display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 10.5px;
  background: rgba(255, 255, 255, 0.05); color: #C8D9E5; border: 1px solid var(--border);
  font-variant-numeric: tabular-nums;
}

/* ---------- badges and live pulse ---------- */
.badge {
  position: relative;
  display: inline-flex; align-items: center; gap: 7px;
  padding: 3px 11px; border-radius: 999px;
  font-size: 11px; font-weight: 650; letter-spacing: 0.03em;
  font-variant-numeric: tabular-nums;
}
.badge-buy { background: var(--accent-dim); color: var(--accent); border: 1px solid rgba(8, 185, 139, 0.3); }
.badge-sell { background: rgba(242, 104, 92, 0.14); color: var(--red); border: 1px solid rgba(242, 104, 92, 0.32); }
.badge-hold { background: rgba(245, 185, 76, 0.13); color: var(--gold); border: 1px solid rgba(245, 185, 76, 0.3); }
.badge.live::after {
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: currentColor; flex: none;
}
.badge.live::before {
  content: ""; position: absolute; left: 11px; top: 50%; width: 7px; height: 7px;
  margin-top: -3.5px; border-radius: 50%;
  border: 1.5px solid currentColor;
  animation: pulse-ring 1.8s var(--ease) infinite;
}

/* ---------- charts ---------- */
.chart-wrap { position: relative; }
.chart-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 12px; flex-wrap: wrap; }
.chart-legend { display: flex; gap: 14px; font-size: 12px; color: var(--text-2); font-variant-numeric: tabular-nums; }
svg text { font-size: 10px; fill: var(--text-3); font-variant-numeric: tabular-nums; }
.candle { animation: candle-in 240ms var(--ease) backwards; }
.draw-line {
  stroke-dasharray: 1; stroke-dashoffset: 1;
  animation: draw 800ms var(--ease) 150ms forwards;
}

/* ---------- controls ---------- */
select, input[type="text"], input[type="number"] {
  background: var(--surface-1); color: var(--text); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: var(--radius-sm);
  padding: 8px 11px; font-size: 13px; outline: none;
  transition: border-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}
select:hover, input:hover { border-color: var(--border-strong); }
select:focus, input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(8, 185, 139, 0.14); }
::placeholder { color: var(--text-3); }

button {
  position: relative;
  background: linear-gradient(180deg, #0ED49E, #08B98B); color: #06251F;
  border: none; border-radius: var(--radius-sm);
  padding: 8px 15px; font-size: 13px; font-weight: 600; cursor: pointer;
  letter-spacing: -0.01em;
  transition: transform var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease),
    filter var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
}
button:hover { filter: brightness(1.07); box-shadow: 0 6px 18px rgba(8, 185, 139, 0.28); transform: translateY(-1px); }
button:active { transform: translateY(0) scale(0.97); box-shadow: 0 2px 8px rgba(8, 185, 139, 0.2); }
button.ghost {
  background: rgba(255, 255, 255, 0.045); color: #C8D9E5;
  border: 1px solid rgba(255, 255, 255, 0.11);
  box-shadow: none;
}
button.ghost:hover {
  background: rgba(255, 255, 255, 0.08); color: var(--text);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3); transform: translateY(-1px);
}
button.ghost:active { transform: scale(0.97); }
button:disabled { opacity: 0.45; cursor: not-allowed; transform: none !important; box-shadow: none !important; filter: none !important; }

.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 10.5px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; }
.muted-note { font-size: 12px; color: var(--text-2); line-height: 1.65; }
.error-box {
  background: rgba(242, 104, 92, 0.09); border: 1px solid rgba(242, 104, 92, 0.32); color: #F5A3A8;
  border-radius: 10px; padding: 10px 14px; font-size: 12.5px; margin-bottom: 16px;
  animation: rise var(--dur) var(--ease) backwards;
}

/* ---------- skeletons ---------- */
.skeleton {
  height: 12px; border-radius: 6px;
  background: linear-gradient(90deg, rgba(255, 255, 255, 0.04) 25%, rgba(255, 255, 255, 0.11) 50%, rgba(255, 255, 255, 0.04) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s ease-in-out infinite;
}
.skeleton-lg { height: 16px; }

/* ---------- split and overlays ---------- */
.split { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 16px; }

/* ---------- signals badges and reasoning ---------- */
.why-box {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; display: flex; flex-direction: column; gap: 7px;
  animation: rise var(--dur) var(--ease) backwards;
}
.confirm-box {
  border-color: rgba(8, 185, 139, 0.35); background: linear-gradient(180deg, rgba(8, 185, 139, 0.05), transparent 50%), var(--surface-2);
  animation: overlay-in 240ms var(--ease-swift) backwards;
}


/* confidence cell: value plus a micro bar */
.conf-cell { display: inline-flex; flex-direction: column; align-items: flex-end; gap: 4px; min-width: 52px; }
.conf-bar {
  height: 3px; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent-bright));
  box-shadow: 0 0 8px rgba(8, 185, 139, 0.5);
  animation: bar-in 600ms var(--ease-swift) 200ms backwards;
}
@keyframes bar-in { from { width: 0 !important; opacity: 0; } }


/* compact stats fit inside narrower cards without clipping money strings */
.stat.compact { padding: 12px 12px; }
.stat.compact .stat-value { font-size: 16.5px; }

@media (max-width: 1100px) { .split { grid-template-columns: 1fr; } .grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
"#;
