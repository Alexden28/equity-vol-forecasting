Out-of-sample: 1998-01-02 to 2026-10-22 (7,246 trading days).

| Model | QLIKE vs GARCH | DM stat (p) | MSE vs GARCH | Mincer-Zarnowitz R² | 99% VaR breaches, normal (Kupiec p) | 99% VaR breaches, FHS (Kupiec p) |
| --- | --- | --- | --- | --- | --- | --- |
| HAR-X (VIX) | -1.98% | -3.49 (0.000) | +11.7% | 0.169 | 1.71% (0.000) | 1.19% (0.128) |
| GJR-GARCH(1,1)-t | -1.34% | -2.09 (0.037) | -3.3% | 0.217 | 1.79% (0.000) | 1.27% (0.028) |
| GARCH(1,1) | +0.00% | benchmark | +0.0% | 0.188 | 1.81% (0.000) | 1.19% (0.128) |
| LSTM (QLIKE loss) | +0.15% | 0.18 (0.855) | +3.5% | 0.175 | 2.48% (0.000) | 1.11% (0.343) |
| EWMA (RiskMetrics) | +1.25% | 2.55 (0.011) | +2.2% | 0.171 | 1.93% (0.000) | 1.24% (0.049) |
| VIX, rescaled | +1.61% | 1.83 (0.067) | +7.8% | 0.140 | 1.48% (0.000) | 1.16% (0.196) |
| HAR | +1.72% | 3.32 (0.001) | +2.6% | 0.166 | 1.71% (0.000) | 1.19% (0.128) |

FHS = filtered historical simulation: the 1% quantile of the previous 500 standardised returns instead of the normal one (6,996 days, after a 250-day warm-up).

Volatility targeting at 15% a year (weight capped at 2x, 5 bp per unit traded):

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Vol of rolling vol | Turnover/yr |
| --- | --- | --- | --- | --- | --- | --- |
| Buy and hold | 10.3% | 25.1% | 0.51 | -77.9% | 11.6% | - |
| EWMA (RiskMetrics) | 8.8% | 15.6% | 0.62 | -46.5% | 1.9% | 6.7x |
| GARCH(1,1) | 8.6% | 15.3% | 0.62 | -48.2% | 1.9% | 10.3x |
| GJR-GARCH(1,1)-t | 8.4% | 15.3% | 0.60 | -48.2% | 2.0% | 11.8x |
| HAR | 8.3% | 14.9% | 0.61 | -48.1% | 2.3% | 12.4x |
| HAR-X (VIX) | 7.9% | 15.0% | 0.58 | -50.7% | 2.6% | 15.5x |
| VIX, rescaled | 7.0% | 14.1% | 0.55 | -54.3% | 3.5% | 8.9x |
| LSTM (QLIKE loss) | 7.9% | 16.3% | 0.55 | -59.5% | 3.3% | 15.0x |
