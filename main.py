import numpy as np
import pandas as pd
from engine.cointegration import CointegrationAnalyzer
from engine.strategy import PairsTradingBacktester
from engine.metrics import PerformanceAnalytics

def generate_synthetic_cointegrated_pair(n_days=1000, seed=42):
    """
    Generates synthetic cointegrated daily asset prices (Asset Y and Asset X)
    with a stationary mean-reverting Ornstein-Uhlenbeck spread.
    """
    np.random.seed(seed)
    # Common random walk trend (market factor)
    trend = np.cumsum(np.random.normal(0.0003, 0.012, n_days))
    
    # Asset X follows trend + noise
    x = 100.0 * np.exp(trend + np.random.normal(0, 0.005, n_days))
    
    # Stationary OU process for spread
    spread = np.zeros(n_days)
    theta = 0.15 # Mean-reversion speed
    sigma_spread = 0.8
    for t in range(1, n_days):
        spread[t] = spread[t-1] - theta * spread[t-1] + np.random.normal(0, sigma_spread)
        
    true_beta = 1.25
    y = true_beta * x + spread + 15.0

    dates = pd.date_range(end='2024-01-01', periods=n_days, freq='B')
    return pd.Series(y, index=dates, name='Asset_Y'), pd.Series(x, index=dates, name='Asset_X')

def main():
    print("=" * 65)
    print("   SIGMA ARBITRAGE BACKTESTER - Statistical Pairs Trading Lab   ")
    print("=" * 65)

    print("\n[1] Generating Cointegrated Asset Series (1,000 Trading Days)...")
    y, x = generate_synthetic_cointegrated_pair()
    print(f"Loaded series length: {len(y)} trading days.")

    # 1. Cointegration Testing
    print("\n--- [2] Engle-Granger Cointegration & Stationarity Analysis ---")
    score, p_val, is_coint = CointegrationAnalyzer.test_cointegration(y, x)
    ols_beta = CointegrationAnalyzer.calculate_ols_hedge_ratio(y, x)

    print(f"ADF t-statistic       : {score:.4f}")
    print(f"Cointegration p-value : {p_val:.6f} (< 0.05 => Stationary Spread)")
    print(f"Hedge Ratio (Beta)    : {ols_beta:.4f}")
    print(f"Cointegration Verified: {'PASSED (Statistically Significant)' if is_coint else 'FAILED'}")

    # 2. Backtest Execution
    print("\n--- [3] Running Backtest Simulation ---")
    backtester = PairsTradingBacktester(
        y=y, x=x, window=30, entry_threshold=2.0, exit_threshold=0.0, slippage_bps=2.0
    )
    results = backtester.run_backtest(beta=ols_beta)

    # 3. Performance Metrics
    print("\n--- [4] Quantitative Strategy Performance Summary ---")
    metrics = PerformanceAnalytics.calculate_metrics(results)
    for k, v in metrics.items():
        print(f"  {k:<25} : {v}")

    print("\nSigma Arbitrage Backtest Completed Successfully.\n")

if __name__ == "__main__":
    main()
