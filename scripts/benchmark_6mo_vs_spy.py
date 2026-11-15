"""Trailing 6-month benchmark: the 27-stock universe with every strategy and
the per-stock PPO checkpoints, plus a SPY buy-and-hold baseline over the same
window. Writes experiments/benchmark-6mo-vs-spy.csv and prints aggregates."""
import csv
import os
import re
import subprocess

ROOT = r"C:\Repository\GitHub\PaperRL"
EXE = os.path.join(ROOT, r"target\release\paperrl.exe")
OUT = os.path.join(ROOT, r"experiments\benchmark-6mo-vs-spy.csv")
START, END = "2026-05-14", "2026-11-13"
STRATEGIES = "buyhold,ema,momentum,meanrev,voltarget,ppo"

symbols = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "JPM",
    "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "ABBV", "KO",
    "PEP", "CSCO", "ORCL", "AMD", "NFLX", "INTC", "DIS",
]

# strategy, net ret %, sharpe, max dd %, win %, exposure %, trades
ROW = re.compile(
    r"^(buyhold|ema|momentum|meanrev|voltarget|ppo)\s+"
    r"(-?[\d.]+)%\s+(-?[\d.]+)\s+(-?[\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)\s"
)


def run_backtest(symbol, with_ppo):
    strategies = STRATEGIES if with_ppo else STRATEGIES.replace(",ppo", "")
    args = [EXE, "backtest", "--symbol", symbol, "--strategies", strategies,
            "--start", START, "--end", END]
    if with_ppo:
        args += ["--model", os.path.join(ROOT, "models", f"ppo-{symbol}-42.bpk")]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ROOT).stdout
    rows = {}
    for line in out.splitlines():
        m = ROW.match(line.strip())
        if m:
            name, ret, _sharpe, dd, _win, exp, trades = m.groups()
            rows[name] = (float(ret), float(dd), int(trades), float(exp))
    return rows


def main():
    results = {}
    missing_ppo = []
    for sym in symbols:
        rows = run_backtest(sym, True)
        if not rows:
            rows = run_backtest(sym, False)
        if "ppo" not in rows:
            missing_ppo.append(sym)
        results[sym] = rows
    if missing_ppo:
        print("no ppo row for:", ", ".join(missing_ppo))

    spy = run_backtest("SPY", False).get("buyhold")

    names = ["buyhold", "ema", "momentum", "meanrev", "voltarget", "ppo"]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "bh_ret", "ema_ret", "mom_ret", "mr_ret", "vt_ret",
                    "ppo_ret", "bh_dd", "ppo_dd", "bh_exp", "ema_exp", "mom_exp",
                    "mr_exp", "vt_exp", "ppo_exp", "ppo_trades"])
        for sym in symbols:
            r = results[sym]
            get = lambda n, i: r[n][i] if n in r else ""
            w.writerow([sym, get("buyhold", 0), get("ema", 0), get("momentum", 0),
                        get("meanrev", 0), get("voltarget", 0), get("ppo", 0),
                        get("buyhold", 1), get("ppo", 1),
                        get("buyhold", 3), get("ema", 3), get("momentum", 3),
                        get("meanrev", 3), get("voltarget", 3), get("ppo", 3),
                        get("ppo", 2)])
        if spy:
            w.writerow(["SPY", spy[0], "", "", "", "", "", spy[1], "", spy[2], "", ""])

    print(f"\nTrailing window {START} .. {END}, 27 stocks + SPY")
    print(f"{'strategy':<12} {'mean ret':>9} {'median':>9} {'mean dd':>9} {'mean exp':>9}")
    for n in names:
        vals = sorted(results[s][n][0] for s in symbols if n in results[s])
        dds = [results[s][n][1] for s in symbols if n in results[s]]
        exps = [results[s][n][3] for s in symbols if n in results[s]]
        if vals:
            print(f"{n:<12} {sum(vals)/len(vals):>8.2f}% {vals[len(vals)//2]:>8.2f}% "
                  f"{sum(dds)/len(dds):>8.2f}% {sum(exps)/len(exps):>8.1f}%")
    if spy:
        print(f"{'SPY bh':<12} {spy[0]:>8.2f}% {'':>9} {spy[1]:>8.2f}%")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
