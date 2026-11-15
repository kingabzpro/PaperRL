//! Dashboard stylesheet. Kept as one constant so the app ships a single
//! inline style block with no asset pipeline.

pub const CSS: &str = r#"
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, #main { height: 100%; }
body {
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  background: #0b1220;
  color: #dbe4f3;
  font-size: 14px;
}
.app { display: grid; grid-template-columns: 228px minmax(0, 1fr); height: 100vh; overflow: hidden; }

/* Sidebar */
.sidebar {
  background: #0e1626;
  border-right: 1px solid #1e2a44;
  display: flex; flex-direction: column;
  padding: 18px 12px;
  gap: 4px;
}
.brand { display: flex; align-items: center; gap: 10px; padding: 4px 10px 18px; }
.brand-logo {
  width: 32px; height: 32px; border-radius: 9px;
  background: linear-gradient(135deg, #4f8cff, #7c5cff);
  display: flex; align-items: center; justify-content: center;
  font-size: 17px; line-height: 1;
  box-shadow: 0 2px 8px rgba(79, 140, 255, 0.35);
}
.brand-name { font-weight: 650; font-size: 15px; letter-spacing: 0.2px; }
.brand-sub { font-size: 10.5px; color: #6b7a99; }
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px; border-radius: 8px; cursor: pointer;
  color: #93a1bd; font-weight: 500; border: 1px solid transparent;
}
.nav-item:hover { background: #15203a; color: #dbe4f3; }
.nav-item.active {
  background: #17264a; color: #eaf0fb; border-color: #2a3f6e;
}
.nav-icon { width: 18px; text-align: center; opacity: 0.9; }
.sidebar-foot { margin-top: auto; padding: 10px; font-size: 11px; color: #5d6b88; line-height: 1.5; }
.safety-pill {
  display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 999px;
  background: #122a1e; color: #34d399; border: 1px solid #1d4a35; font-size: 10.5px;
}

/* Content */
.content { overflow-y: auto; overflow-x: hidden; min-width: 0; padding: 22px 26px 40px; }
.page-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 18px; gap: 16px; flex-wrap: wrap; }
.page-title { font-size: 20px; font-weight: 650; }
.page-sub { color: #6b7a99; font-size: 12.5px; margin-top: 3px; }
.head-actions { display: flex; gap: 8px; align-items: center; }

.card {
  background: #111a2c; border: 1px solid #1e2a44; border-radius: 12px;
  padding: 16px; margin-bottom: 16px;
}
.card-title { font-size: 13px; font-weight: 600; color: #aab6d0; margin-bottom: 12px; letter-spacing: 0.2px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
.stat, .card { min-width: 0; }
.stat {
  background: #111a2c; border: 1px solid #1e2a44; border-radius: 12px; padding: 14px 16px;
}
.stat-label { font-size: 11.5px; color: #6b7a99; text-transform: uppercase; letter-spacing: 0.6px; }
.stat-value { font-size: 22px; font-weight: 650; margin-top: 6px; }
.stat-sub { font-size: 11.5px; color: #6b7a99; margin-top: 4px; }
.pos { color: #34d399; }
.neg { color: #f87171; }
.dim { color: #6b7a99; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th {
  text-align: right; font-size: 11px; color: #6b7a99; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.5px;
  padding: 8px 10px; border-bottom: 1px solid #1e2a44;
}
th:first-child, td:first-child { text-align: left; }
td { padding: 8px 10px; border-bottom: 1px solid #16203a; font-size: 13px; text-align: right; }
tr:hover td { background: #14203a; }
tr.selected td { background: #17264a; }
tbody tr { cursor: default; }
.clickable tbody tr { cursor: pointer; }
.strategy-name { font-weight: 600; color: #eaf0fb; }
.tag {
  display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 10.5px;
  background: #17264a; color: #8fb0ff; border: 1px solid #2a3f6e;
}

/* Chart */
.chart-wrap { position: relative; }
.chart-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 12px; flex-wrap: wrap; }
.chart-legend { display: flex; gap: 14px; font-size: 12px; color: #93a1bd; font-variant-numeric: tabular-nums; }
svg text { font-size: 10px; fill: #5d6b88; }

/* Controls */
select, input[type="text"], input[type="number"] {
  background: #0e1626; color: #dbe4f3; border: 1px solid #24314f; border-radius: 8px;
  padding: 7px 10px; font-size: 13px; outline: none;
}
select:focus, input:focus { border-color: #4f8cff; }
button {
  background: #4f8cff; color: #fff; border: none; border-radius: 8px;
  padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer;
}
button:hover { background: #3f7bef; }
button.ghost { background: #17264a; color: #b9c8e8; border: 1px solid #2a3f6e; }
button.ghost:hover { background: #1d3057; }
button:disabled { opacity: 0.45; cursor: not-allowed; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 11px; color: #6b7a99; text-transform: uppercase; letter-spacing: 0.5px; }
.muted-note { font-size: 12px; color: #6b7a99; line-height: 1.6; }
.error-box {
  background: #2a1620; border: 1px solid #5c2537; color: #fda4af;
  border-radius: 10px; padding: 10px 14px; font-size: 12.5px; margin-bottom: 16px;
}
.split { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 16px; }
@media (max-width: 1100px) { .split { grid-template-columns: 1fr; } .grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
"#;
