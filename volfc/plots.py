"""Figures for the README."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
COLORS = {
    "realised": "#9aa5b1", "GARCH(1,1)": "#2f6fb3", "HAR-X (VIX)": "#d9822b", "LSTM (QLIKE loss)": "#2a9d6f",
    "Buy and hold": "#9aa5b1",
}


def _style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_all(df: pd.DataFrame, F: pd.DataFrame, res: dict, title: str, out) -> None:
    ok = F.notna().all(axis=1)
    ann = lambda h: np.sqrt(h * 252)  # noqa: E731
    realised = ann((df["r"] ** 2).rolling(21).mean().shift(-20))  # vol over the next 21 days

    # 1. forecasts against what happened
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), dpi=160, gridspec_kw={"height_ratios": [1.4, 1]})
    for ax, (lo, hi) in zip(axes, [(df.index[ok][0], df.index[ok][-1]),
                                    (pd.Timestamp("2020-01-15"), pd.Timestamp("2020-06-30"))]):
        sl = ok & (df.index >= lo) & (df.index <= hi)
        ax.plot(df.index[sl], realised[sl], color=COLORS["realised"], lw=1.6, label="realised, next 21 days")
        for name in ("GARCH(1,1)", "HAR-X (VIX)", "LSTM (QLIKE loss)"):
            if name in F:
                ax.plot(df.index[sl], ann(F[name][sl]), color=COLORS[name], lw=0.9, label=name)
        ax.set_ylabel("annualised vol, %", color=MUTED, fontsize=8)
        _style(ax)
    axes[0].set_title(f"{title}: one-day-ahead volatility forecasts", color=INK, fontsize=10, loc="left")
    axes[1].set_title("Zoom: March 2020", color=MUTED, fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "forecasts.png")
    plt.close(fig)

    # 2. loss relative to GARCH
    stats = res["stats"]
    names = sorted(stats, key=lambda n: stats[n]["qlike_vs_garch_pct"])
    vals = [stats[n]["qlike_vs_garch_pct"] for n in names]
    sig = [stats[n]["dm_p"] < 0.05 and n != "GARCH(1,1)" for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * len(names)), dpi=160)
    ax.barh(names, vals, color=["#2a9d6f" if v < 0 and s else "#c23b3b" if v > 0 and s else "#cbd2d9"
                                for v, s in zip(vals, sig)], height=0.6)
    ax.axvline(0, color=INK, lw=0.8)
    for i, (v, n) in enumerate(zip(vals, names)):
        p = stats[n]["dm_p"]
        label = "benchmark" if n == "GARCH(1,1)" else f"{v:+.2f}%  (DM p={p:.3f})"
        ax.text(v + (0.05 if v >= 0 else -0.05), i, label, va="center", ha="left" if v >= 0 else "right",
                fontsize=7, color=INK)
    ax.invert_yaxis()
    span = max(abs(min(vals)), abs(max(vals))) * 1.9 + 0.5
    ax.set_xlim(-span, span)
    ax.set_xlabel("QLIKE loss vs GARCH(1,1), % (lower is better; coloured = significant at 5%)",
                  color=MUTED, fontsize=8)
    ax.set_title(f"{title}: forecast accuracy, out of sample", color=INK, fontsize=10, loc="left")
    _style(ax, grid="x")
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(out / "qlike.png")
    plt.close(fig)

    # 3. volatility targeting
    curves = res["curves"]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    for name in ("Buy and hold", "GARCH(1,1)", "HAR-X (VIX)", "LSTM (QLIKE loss)"):
        if name in curves:
            w = (1 + curves[name]).cumprod()
            ax.plot(w.index, w, color=COLORS[name], lw=1.0 if name != "Buy and hold" else 1.4,
                    label=f"{name}  (Sharpe {res['strategy'][name]['sharpe']:.2f}, "
                          f"max DD {res['strategy'][name]['max_drawdown']:.0%})")
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1, log scale", color=MUTED, fontsize=8)
    ax.set_title(f"{title}: targeting 15% volatility with each forecast", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "vol_targeting.png")
    plt.close(fig)
