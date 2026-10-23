Out-of-sample: 2020-11-02 to 2026-10-21 (1,499 trading days).

| Model | QLIKE vs GARCH | DM stat (p) | MSE vs GARCH | Mincer-Zarnowitz R² | 99% VaR breaches (Kupiec p) |
| --- | --- | --- | --- | --- | --- |
| GJR-GARCH(1,1)-t | -5.52% | -4.11 (0.000) | -5.6% | 0.112 | 1.73% (0.010) |
| HAR-X (VIX) | -4.95% | -2.05 (0.041) | -12.8% | 0.174 | 1.87% (0.003) |
| LSTM (QLIKE loss) | -1.32% | -0.53 (0.595) | -9.2% | 0.140 | 1.67% (0.018) |
| GARCH(1,1) | +0.00% | benchmark | +0.0% | 0.064 | 1.80% (0.005) |
| EWMA (RiskMetrics) | +2.88% | 1.82 (0.069) | -0.8% | 0.062 | 1.87% (0.003) |
| HAR | +3.21% | 2.45 (0.014) | -0.6% | 0.065 | 1.40% (0.141) |
| VIX, rescaled | +16.53% | 4.64 (0.000) | +5.9% | 0.088 | 0.60% (0.093) |

Volatility targeting at 15% a year (weight capped at 2x, 5 bp per unit traded):

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Vol of rolling vol | Turnover/yr |
| --- | --- | --- | --- | --- | --- | --- |
| Buy and hold | 15.6% | 16.4% | 0.97 | -25.4% | 5.8% | - |
| EWMA (RiskMetrics) | 13.0% | 15.5% | 0.86 | -20.3% | 1.9% | 8.6x |
| GARCH(1,1) | 11.4% | 14.9% | 0.80 | -20.2% | 1.7% | 18.0x |
| GJR-GARCH(1,1)-t | 11.0% | 14.8% | 0.78 | -19.7% | 1.4% | 18.9x |
| HAR | 11.6% | 13.9% | 0.86 | -19.0% | 1.7% | 17.9x |
| HAR-X (VIX) | 11.9% | 14.5% | 0.85 | -23.2% | 2.6% | 26.8x |
| VIX, rescaled | 9.4% | 11.1% | 0.86 | -15.9% | 2.8% | 10.2x |
| LSTM (QLIKE loss) | 10.8% | 14.3% | 0.79 | -22.2% | 2.5% | 15.1x |
