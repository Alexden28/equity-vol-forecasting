"""Statistical and economic evaluation of variance forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

Z99 = stats.norm.ppf(0.01)


def qlike(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    """QLIKE loss, robust to the noise in squared returns as a variance proxy (Patton, 2011)."""
    return np.log(h) + y / h


def mse(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    return (y - h) ** 2


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """DM statistic for E[loss_a - loss_b] = 0 with a Newey-West variance; negative = a better."""
    d = loss_a - loss_b
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def mincer_zarnowitz(y: np.ndarray, h: np.ndarray) -> dict:
    """Regress realised on forecast variance: unbiased forecasts have a=0, b=1."""
    X = np.column_stack([np.ones_like(h), h])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    return {"a": float(beta[0]), "b": float(beta[1]), "r2": float(r2)}


def kupiec(exceed: np.ndarray, p: float = 0.01) -> tuple[float, float]:
    """Kupiec proportion-of-failures test; returns (exceedance rate, p-value)."""
    n, x = len(exceed), int(exceed.sum())
    rate = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - rate) - x * np.log(rate))
    return rate, float(stats.chi2.sf(lr, 1))


def vol_target(r: pd.Series, h: pd.Series, target: float = 15.0, max_lev: float = 2.0,
               cost_bp: float = 5.0) -> pd.DataFrame:
    """Daily volatility targeting: weight = target / forecast vol, capped, with trading costs.

    `h` at date t is known at the close of t-1, so the weight is set before r_t happens.
    Cash earns nothing and leverage costs nothing (a simplification, same for every model).
    """
    sigma = np.sqrt(h * 252)
    w = (target / sigma).clip(upper=max_lev)
    simple = np.expm1(r / 100)
    cost = cost_bp / 1e4 * w.diff().abs().fillna(0)
    ret = w * simple - cost
    return pd.DataFrame({"w": w, "ret": ret})


def perf(ret: pd.Series) -> dict:
    wealth = (1 + ret).cumprod()
    years = len(ret) / 252
    dd = wealth / wealth.cummax() - 1
    vol = ret.std() * np.sqrt(252)
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    rolling = ret.rolling(63).std() * np.sqrt(252)
    return {
        "cagr": float(cagr), "vol": float(vol), "sharpe": float(ret.mean() * 252 / vol),
        "max_drawdown": float(dd.min()), "vol_of_vol": float(rolling.std()),
    }
