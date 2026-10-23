"""python -m volfc run --index NASDAQCOM   (or --index SP500 --window 1000 --oos-start 2020-11-01)"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import models
from .data import INDICES, ROOT, load
from .evaluate import Z99, diebold_mariano, kupiec, mincer_zarnowitz, mse, perf, qlike, vol_target
from .plots import plot_all

BENCH = "GARCH(1,1)"


def forecasts(df: pd.DataFrame, start: int, window: int, use_lstm: bool) -> pd.DataFrame:
    r, vix = df["r"], df["vix"]
    out = {}
    steps = [
        ("EWMA (RiskMetrics)", lambda: models.ewma(r)),
        ("GARCH(1,1)", lambda: models.garch(r, start, window)),
        ("GJR-GARCH(1,1)-t", lambda: models.garch(r, start, window, o=1, dist="t")),
        ("HAR", lambda: models.har(r, start, window=window)),
        ("HAR-X (VIX)", lambda: models.har(r, start, vix, window=window)),
        ("VIX, rescaled", lambda: models.vix_implied(r, vix, start, window)),
    ]
    if use_lstm:
        from .lstm import lstm
        steps.append(("LSTM (QLIKE loss)", lambda: lstm(r, vix, start)))
    for name, fn in steps:
        t0 = time.time()
        out[name] = fn()
        print(f"  {name:22s} {time.time() - t0:6.1f}s")
    return pd.DataFrame(out)


def evaluate(df: pd.DataFrame, F: pd.DataFrame) -> dict:
    y = (df["r"] ** 2).to_numpy()
    ok = F.notna().all(axis=1).to_numpy()
    yy = y[ok]
    base_q = qlike(yy, F[BENCH].to_numpy()[ok])
    base_m = mse(yy, F[BENCH].to_numpy()[ok])
    stat_rows = {}
    for name in F:
        h = F[name].to_numpy()[ok]
        q, m = qlike(yy, h), mse(yy, h)
        dm, p = diebold_mariano(q, base_q) if name != BENCH else (0.0, 1.0)
        exceed = df["r"].to_numpy()[ok] < Z99 * np.sqrt(h)
        rate, kp = kupiec(exceed)
        stat_rows[name] = {
            "qlike": float(q.mean()),
            "qlike_vs_garch_pct": float((q.mean() - base_q.mean()) / abs(base_q.mean()) * 100),
            "dm_stat": dm, "dm_p": p,
            "mse_vs_garch_pct": float((m.mean() / base_m.mean() - 1) * 100),
            **{f"mz_{k}": v for k, v in mincer_zarnowitz(yy, h).items()},
            "var99_exceed_rate": rate, "kupiec_p": kp,
        }
    idx = df.index[ok]
    strat = {"Buy and hold": perf(pd.Series(np.expm1(df["r"].to_numpy()[ok] / 100), index=idx))}
    curves = {"Buy and hold": np.expm1(df["r"][ok] / 100)}
    for name in F:
        vt = vol_target(df["r"][ok], F[name][ok])
        strat[name] = perf(vt["ret"])
        strat[name]["turnover"] = float(vt["w"].diff().abs().mean() * 252)
        curves[name] = vt["ret"]
    return {"stats": stat_rows, "strategy": strat, "curves": pd.DataFrame(curves, index=idx),
            "sample": [str(idx[0].date()), str(idx[-1].date()), int(ok.sum())]}


def write_markdown(index: str, res: dict, path) -> None:
    s, st = res["stats"], res["strategy"]
    lines = [f"Out-of-sample: {res['sample'][0]} to {res['sample'][1]} ({res['sample'][2]:,} trading days).", "",
             "| Model | QLIKE vs GARCH | DM stat (p) | MSE vs GARCH | Mincer-Zarnowitz R² | 99% VaR breaches (Kupiec p) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, r in sorted(s.items(), key=lambda kv: kv[1]["qlike"]):
        dm = "benchmark" if name == BENCH else f"{r['dm_stat']:.2f} ({r['dm_p']:.3f})"
        lines.append(f"| {name} | {r['qlike_vs_garch_pct']:+.2f}% | {dm} | {r['mse_vs_garch_pct']:+.1f}% | "
                     f"{r['mz_r2']:.3f} | {r['var99_exceed_rate']:.2%} ({r['kupiec_p']:.3f}) |")
    lines += ["", "Volatility targeting at 15% a year (weight capped at 2x, 5 bp per unit traded):", "",
              "| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Vol of rolling vol | Turnover/yr |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in st.items():
        to = f"{r['turnover']:.1f}x" if "turnover" in r else "-"
        lines.append(f"| {name} | {r['cagr']:.1%} | {r['vol']:.1%} | {r['sharpe']:.2f} | {r['max_drawdown']:.1%} | "
                     f"{r['vol_of_vol']:.1%} | {to} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="volfc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--index", default="NASDAQCOM", choices=list(INDICES))
    run.add_argument("--start", default="1990-01-02")
    run.add_argument("--oos-start", default="1998-01-01")
    run.add_argument("--window", type=int, default=2000)
    run.add_argument("--no-lstm", action="store_true")
    rep = sub.add_parser("report", help="rebuild tables and figures from saved forecasts")
    rep.add_argument("--index", default="NASDAQCOM", choices=list(INDICES))
    rep.add_argument("--start", default="1990-01-02")
    args = ap.parse_args(argv)

    df = load(args.index, args.start)
    out = ROOT / "reports" / args.index
    if args.cmd == "report":
        F = pd.read_csv(out / "forecasts.csv", index_col=0, parse_dates=True).reindex(df.index)
    else:
        start = max(int(np.searchsorted(df.index, pd.Timestamp(args.oos_start))), args.window)
        print(f"{INDICES[args.index]}: {len(df):,} days, out-of-sample from {df.index[start].date()}")
        F = forecasts(df, start, args.window, not args.no_lstm)
        out.mkdir(parents=True, exist_ok=True)
        F.dropna(how="all").to_csv(out / "forecasts.csv", float_format="%.6g")
    res = evaluate(df, F)
    json.dump({k: res[k] for k in ("stats", "strategy", "sample")}, open(out / "results.json", "w"), indent=2)
    write_markdown(args.index, res, out / "results.md")
    plot_all(df, F, res, INDICES[args.index], out)


if __name__ == "__main__":
    main()
