# Pairs Trading Backtest

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quantitative Finance](https://img.shields.io/badge/Domain-Statistical%20Arbitrage-brightgreen.svg)](#)

An institutional-grade **Statistical Arbitrage & Pairs Trading Backtesting Lab** implementing **Engle-Granger Cointegration**, dynamic hedge ratio estimation (OLS and Kalman Filters), rolling $Z$-score signal thresholding, and risk-adjusted performance attribution.

---

## 📈 Quantitative Methodology

### 1. Cointegration & Stationarity Verification
Unlike short-term correlation, cointegration verifies whether a stationary linear combination exists between two non-stationary price series $Y_t$ and $X_t$:
$$Y_t = \alpha + \beta X_t + \epsilon_t$$
where the residual spread $\epsilon_t \sim I(0)$ is tested for stationarity using the **Augmented Dickey-Fuller (ADF)** test.

### 2. Rolling Z-Score Normalized Spread
$$\text{Spread}_t = Y_t - \beta X_t$$
$$Z_t = \frac{\text{Spread}_t - \mu_{\text{roll}}(t, W)}{\sigma_{\text{roll}}(t, W)}$$

* **Short Spread Signal ($Z_t > +2.0\sigma$):** Short Asset $Y$, Long $\beta$ units of Asset $X$.
* **Long Spread Signal ($Z_t < -2.0\sigma$):** Long Asset $Y$, Short $\beta$ units of Asset $X$.
* **Exit Signal ($|Z_t| \le 0.0\sigma$):** Close both legs when the spread reverts to historical mean.

---

## 📊 Backtest Performance Summary

```
ADF t-statistic       : -4.8912
Cointegration p-value : 0.000342 (< 0.05 => Stationary Spread)
Hedge Ratio (Beta)    : 1.2504
Cointegration Verified: PASSED (Statistically Significant)

--- Strategy Performance ---
Total Cumulative Return   : 38.45%
Annualized Return         : 9.24%
Annualized Volatility     : 5.38%
Sharpe Ratio              : 1.72
Maximum Drawdown          : -7.62%
Win Rate (Active Days)    : 64.2%
```

---

## 🛠 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Harshithb23me1023/sigma-arbitrage-backtester.git
cd sigma-arbitrage-backtester

# Install dependencies
pip install -r requirements.txt

# Run backtester
python main.py
```

---

## 👤 Author
**Janavath Venkata Harshith Naik**  
B.Tech in Mechanical Engineering, IIT Jodhpur  
GitHub: [@Harshithb23me1023](https://github.com/Harshithb23me1023)
