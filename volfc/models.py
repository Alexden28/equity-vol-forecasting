"""One-day-ahead variance forecasts.

Every function returns a Series indexed like the returns, where the value at date t is the
forecast of the variance of r_t made with information up to the close of t-1. Returns are
log returns in percent, so variances are in %^2 per day.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model


def ewma(r: pd.Series, lam: float = 0.94, warmup: int = 250) -> pd.Series:
    """RiskMetrics exponentially weighted moving average."""
    x = r.to_numpy()
    h = np.full(len(x), np.nan)
    h[warmup] = np.var(x[:warmup])
    for t in range(warmup + 1, len(x)):
        h[t] = lam * h[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(h, index=r.index)


def garch(r: pd.Series, start: int, window: int = 2000, refit: int = 21, o: int = 0,
          dist: str = "normal") -> pd.Series:
    """GARCH(1,1) (o=1: GJR-GARCH) on a rolling window, refitted every `refit` days.

    Between refits the parameters stay fixed and the variance recursion keeps running on
    new data, which is how a desk would run it.
    """
    out = pd.Series(np.nan, index=r.index)
    for T in range(start, len(r), refit):
        train = r.iloc[T - window:T]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(train, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist).fit(disp="off")
        stop = min(T + refit, len(r))
        seg = r.iloc[T - window:stop]
        fixed = arch_model(seg, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist).fix(res.params)
        fc = fixed.forecast(horizon=1, start=window - 1, reindex=False).variance["h.1"].to_numpy()
        # forecast made at origin T-1+k is for day T+k
        n = stop - T
        out.iloc[T:stop] = fc[:n]
    return out


def har_features(r: pd.Series, vix: pd.Series | None = None) -> pd.DataFrame:
    """HAR regressors known at the close of day t (daily, weekly and monthly squared returns)."""
    r2 = r ** 2
    X = pd.DataFrame({"d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean()}, index=r.index)
    if vix is not None:
        X["vix"] = vix ** 2 / 252  # implied daily variance, %^2
    return X


LOG_EPS = 1e-2  # %^2; keeps log(r^2) finite on flat days


def har(r: pd.Series, start: int, vix: pd.Series | None = None, window: int = 2000, refit: int = 21) -> pd.Series:
    """Log-HAR(-X) by rolling OLS; regressors at t-1 predict log r_t^2.

    A HAR in levels can forecast near-zero or negative variances, which QLIKE punishes
    without limit; in logs every forecast is positive. Exponentiating a log forecast
    underestimates the mean, so forecasts are rescaled by Duan's smearing factor.
    """
    X = np.log(har_features(r, vix) + LOG_EPS).shift(1)  # information up to t-1
    X.insert(0, "const", 1.0)
    y2 = r ** 2
    y = np.log(y2 + LOG_EPS)
    out = pd.Series(np.nan, index=r.index)
    for T in range(start, len(r), refit):
        tr = slice(max(0, T - window), T)
        Xt, yt = X.iloc[tr], y.iloc[tr]
        ok = Xt.notna().all(axis=1)
        beta, *_ = np.linalg.lstsq(Xt[ok].to_numpy(), yt[ok].to_numpy(), rcond=None)
        fitted = np.exp(Xt[ok].to_numpy() @ beta)
        smear = float(y2.iloc[tr][ok].mean() / fitted.mean())
        stop = min(T + refit, len(r))
        out.iloc[T:stop] = smear * np.exp(X.iloc[T:stop].to_numpy() @ beta)
    return out


def vix_implied(r: pd.Series, vix: pd.Series, start: int, window: int = 2000, refit: int = 21) -> pd.Series:
    """Yesterday's VIX as a variance forecast, rescaled by a rolling no-intercept regression
    (the VIX is the S&P 500's 30-day implied vol and carries a variance risk premium)."""
    iv = (vix ** 2 / 252).shift(1)
    y = r ** 2
    out = pd.Series(np.nan, index=r.index)
    for T in range(start, len(r), refit):
        tr = slice(max(1, T - window), T)
        b = float((iv.iloc[tr] * y.iloc[tr]).sum() / (iv.iloc[tr] ** 2).sum())
        stop = min(T + refit, len(r))
        out.iloc[T:stop] = b * iv.iloc[T:stop]
    return out
