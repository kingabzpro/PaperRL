"""Fetch daily OHLCV bars from Yahoo Finance into CSV files.

Writes <out>/<SYMBOL>.csv with an ISO-8601 UTC timestamp column, using
split/dividend-adjusted prices (the same treatment Alpaca applies to its
adjusted bars), so a long history is internally consistent.

Usage:
    python scripts/fetch_yahoo.py --range 10y --out datasets/csv SPY QQQ AAPL
"""

import argparse
import csv
import json
import os
import sys
import urllib.request

CHART = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    "?range={rng}&interval=1d&events=div%2Csplit"
)


def fetch(symbol: str, rng: str) -> dict:
    url = CHART.format(sym=symbol, rng=rng)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def rows(payload: dict) -> list[tuple]:
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    out = []
    for i, t in enumerate(ts):
        o, h, low, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, low, c) or c in (0, None):
            continue
        # Scale raw prices by the adjustment factor so the series is smooth
        # across splits and dividends.
        k = (adj[i] / c) if adj and adj[i] is not None else 1.0
        out.append(
            (
                t,
                o * k,
                h * k,
                low * k,
                c * k,
                float(v or 0),
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", default="10y")
    ap.add_argument("--out", default="datasets/csv")
    ap.add_argument("symbols", nargs="+")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for sym in a.symbols:
        try:
            data = rows(fetch(sym, a.range))
        except Exception as e:  # noqa: BLE001 - report and continue per symbol
            print(f"{sym}: FETCH FAILED: {e}", file=sys.stderr)
            continue
        if not data:
            print(f"{sym}: no rows", file=sys.stderr)
            continue
        path = os.path.join(a.out, f"{sym.upper()}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for t, o, h, low, c, v in data:
                w.writerow([t, f"{o:.6f}", f"{h:.6f}", f"{low:.6f}", f"{c:.6f}", f"{v:.0f}"])
        from datetime import datetime, timezone

        first = datetime.fromtimestamp(data[0][0], timezone.utc).date()
        last = datetime.fromtimestamp(data[-1][0], timezone.utc).date()
        print(f"{sym}: {len(data)} bars -> {path} ({first} .. {last})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
