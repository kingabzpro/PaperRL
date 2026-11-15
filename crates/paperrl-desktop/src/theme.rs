//! Dashboard stylesheet. Kept as one constant so the app ships a single
//! inline style block with no asset pipeline.

pub const CSS: &str = r#"
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, #main { height: 100%; }
body {
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  background: #101C2C;
  color: #EAF2F7;
  font-size: 14px;
}
.app { display: grid; grid-template-columns: 228px minmax(0, 1fr); height: 100vh; overflow: hidden; }

/* Sidebar */
.sidebar {
  background: #0C1725;
  border-right: 1px solid #22334A;
  display: flex; flex-direction: column;
  padding: 18px 12px;
  gap: 4px;
}
.brand { display: flex; align-items: center; gap: 10px; padding: 4px 10px 18px; }
.brand-logo {
  width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;
  filter: drop-shadow(0 2px 6px rgba(8, 185, 139, 0.25));
}
.brand-name { font-weight: 650; font-size: 15px; letter-spacing: 0.2px; }
.brand-sub { font-size: 10.5px; color: #8FA3B8; }
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px; border-radius: 8px; cursor: pointer;
  color: #9FB2C4; font-weight: 500; border: 1px solid transparent;
}
.nav-item:hover { background: #17283E; color: #EAF2F7; }
.nav-item.active {
  background: #0E2A26; color: #EAF2F7; border-color: #1B5A4C;
}
.nav-icon { width: 18px; text-align: center; opacity: 0.9; }
.sidebar-foot { margin-top: auto; padding: 10px; font-size: 11px; color: #7C8FA5; line-height: 1.5; }
.safety-pill {
  display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 999px;
  background: #0E2A26; color: #08B98B; border: 1px solid #1B5A4C; font-size: 10.5px;
}

/* Content */
.content { overflow-y: auto; overflow-x: hidden; min-width: 0; padding: 22px 26px 40px; }
.page-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 18px; gap: 16px; flex-wrap: wrap; }
.page-title { font-size: 20px; font-weight: 650; }
.page-sub { color: #8FA3B8; font-size: 12.5px; margin-top: 3px; }
.head-actions { display: flex; gap: 8px; align-items: center; }

.card {
  background: #16243A; border: 1px solid #22334A; border-radius: 12px;
  padding: 16px; margin-bottom: 16px;
}
.card-title { font-size: 13px; font-weight: 600; color: #C8D9E5; margin-bottom: 12px; letter-spacing: 0.2px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
.stat, .card { min-width: 0; }
.stat {
  background: #16243A; border: 1px solid #22334A; border-radius: 12px; padding: 14px 16px;
}
.stat-label { font-size: 11.5px; color: #8FA3B8; text-transform: uppercase; letter-spacing: 0.6px; }
.stat-value { font-size: 22px; font-weight: 650; margin-top: 6px; }
.stat-sub { font-size: 11.5px; color: #8FA3B8; margin-top: 4px; }
.pos { color: #08B98B; }
.neg { color: #F2685C; }
.dim { color: #8FA3B8; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th {
  text-align: right; font-size: 11px; color: #8FA3B8; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.5px;
  padding: 8px 10px; border-bottom: 1px solid #22334A;
}
th:first-child, td:first-child { text-align: left; }
td { padding: 8px 10px; border-bottom: 1px solid #1B2B42; font-size: 13px; text-align: right; }
tr:hover td { background: #182A40; }
tr.selected td { background: #123029; }
tbody tr { cursor: default; }
.clickable tbody tr { cursor: pointer; }
.strategy-name { font-weight: 600; color: #EAF2F7; }
.tag {
  display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 10.5px;
  background: #182B42; color: #C8D9E5; border: 1px solid #2A3F5C;
}

/* Chart */
.chart-wrap { position: relative; }
.chart-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 12px; flex-wrap: wrap; }
.chart-legend { display: flex; gap: 14px; font-size: 12px; color: #9FB2C4; font-variant-numeric: tabular-nums; }
svg text { font-size: 10px; fill: #7C8FA5; }

/* Controls */
select, input[type="text"], input[type="number"] {
  background: #0C1725; color: #EAF2F7; border: 1px solid #24314f; border-radius: 8px;
  padding: 7px 10px; font-size: 13px; outline: none;
}
select:focus, input:focus { border-color: #08B98B; }
button {
  background: #08B98B; color: #06251F; border: none; border-radius: 8px;
  padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer;
}
button:hover { background: #07A77D; }
button.ghost { background: #17283E; color: #C8D9E5; border: 1px solid #2A3F5C; }
button.ghost:hover { background: #1E3350; }
button:disabled { opacity: 0.45; cursor: not-allowed; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 11px; color: #8FA3B8; text-transform: uppercase; letter-spacing: 0.5px; }
.muted-note { font-size: 12px; color: #8FA3B8; line-height: 1.6; }
.error-box {
  background: #2E1A20; border: 1px solid #6B2A34; color: #F5A3A8;
  border-radius: 10px; padding: 10px 14px; font-size: 12.5px; margin-bottom: 16px;
}
.split { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 16px; }

/* Signals badges and reasoning */
.badge {
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 650; letter-spacing: 0.3px;
}
.badge-buy { background: #0E2A26; color: #08B98B; border: 1px solid #1B5A4C; }
.badge-sell { background: #2E1A20; color: #F2685C; border: 1px solid #6B2A34; }
.badge-hold { background: #2E2A1A; color: #F5B94C; border: 1px solid #6B552A; }
.why-box {
  background: #0F1C2E; border: 1px solid #22334A; border-radius: 10px;
  padding: 12px 14px; display: flex; flex-direction: column; gap: 6px;
}
.confirm-box { border-color: #1B5A4C; background: #132A24; }

@media (max-width: 1100px) { .split { grid-template-columns: 1fr; } .grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
"#;
