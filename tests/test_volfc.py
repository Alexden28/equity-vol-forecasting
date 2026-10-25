import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from volfc import models
from volfc.evaluate import diebold_mariano, kupiec, qlike, vol_target


def simulate_garch(n=3000, omega=0.02, alpha=0.08, beta=0.90, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    h = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * r[t] ** 2 + beta * h
    return pd.Series(r, index=pd.bdate_range("2000-01-03", periods=n))


@pytest.fixture(scope="module")
def r():
    return simulate_garch()


def test_garch_recovers_persistence(r):
    res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
    assert res.params["alpha[1]"] + res.params["beta[1]"] == pytest.approx(0.98, abs=0.02)


@pytest.mark.parametrize("make", [
    lambda r: models.ewma(r),
    lambda r: models.garch(r, start=1500, window=1000, refit=50),
    lambda r: models.har(r, start=1500, window=1000, refit=50),
])
def test_no_look_ahead(r, make):
    """Changing returns from day k on must not change any forecast up to and including day k."""
    k = 1700
    base = make(r)
    shocked = r.copy()
    shocked.iloc[k:] *= 3
    after = make(shocked)
    pd.testing.assert_series_equal(base.iloc[:k + 1], after.iloc[:k + 1])
    assert not np.allclose(base.iloc[k + 1:].dropna(), after.iloc[k + 1:].dropna())


def test_garch_forecasts_track_true_variance(r):
    f = models.garch(r, start=1500, window=1000, refit=50).dropna()
    # rebuild the true conditional variance of the simulation
    h = np.empty(len(r))
    h[0] = 0.02 / 0.02
    for t in range(1, len(r)):
        h[t] = 0.02 + 0.08 * r.iloc[t - 1] ** 2 + 0.90 * h[t - 1]
    true = pd.Series(h, index=r.index).loc[f.index]
    assert np.corrcoef(f, true)[0, 1] > 0.95


def test_qlike_is_minimised_by_the_true_variance():
    rng = np.random.default_rng(1)
    y = 2.0 * rng.standard_normal(200_000) ** 2  # true variance 2
    losses = {h: qlike(y, np.full_like(y, h)).mean() for h in (1.0, 1.5, 2.0, 2.5, 3.0)}
    assert min(losses, key=losses.get) == 2.0


def test_diebold_mariano():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(1000) ** 2
    assert diebold_mariano(a, a) == (0.0, 1.0)
    stat, p = diebold_mariano(a, a + 0.5)
    assert stat < 0 and p < 0.01


def test_kupiec():
    rng = np.random.default_rng(3)
    _, p_ok = kupiec(rng.random(5000) < 0.01)
    _, p_bad = kupiec(rng.random(5000) < 0.03)
    assert p_ok > 0.05 and p_bad < 0.001


def test_vol_target_caps_leverage():
    r = pd.Series([0.1, -0.2, 0.3])
    h = pd.Series([1e-6, 1.0, 4.0])
    w = vol_target(r, h, target=15, max_lev=2)["w"]
    assert w.iloc[0] == 2.0 and w.max() <= 2.0


def test_fhs_var_fixes_fat_tails():
    """With Student-t(4) shocks the normal 99% VaR is breached ~1.5% of the time; FHS gets close
    to 1%. The residual ~0.2 pp comes from estimating a 1% quantile from 500 days (5 tail points)."""
    from volfc.evaluate import Z99, fhs_var
    rng = np.random.default_rng(4)
    shocks = rng.standard_t(4, 20_000) / np.sqrt(2.0)  # unit variance
    r = pd.Series(shocks)
    h = pd.Series(np.ones(len(r)))
    normal_rate = (r < Z99).mean()
    var = fhs_var(r, h)
    ok = var.notna()
    fhs_rate = (r[ok] < var[ok]).mean()
    assert normal_rate > 0.014
    assert fhs_rate < normal_rate and abs(fhs_rate - 0.01) < 0.003
