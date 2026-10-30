import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from typing import Tuple

class CointegrationAnalyzer:
    """
    Analyzes cointegration and estimates dynamic hedge ratios for pairs trading.
    """
    @staticmethod
    def test_cointegration(y: pd.Series, x: pd.Series) -> Tuple[float, float, bool]:
        """
        Engle-Granger Two-Step Cointegration Test.
        Returns: (t-statistic, p-value, is_cointegrated_at_5pct)
        """
        score, p_value, _ = coint(y, x)
        is_cointegrated = p_value < 0.05
        return float(score), float(p_value), is_cointegrated

    @staticmethod
    def calculate_ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
        """
        Calculates static hedge ratio Beta via Ordinary Least Squares:
        y = alpha + beta * x + epsilon
        """
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[1]
        return float(beta)

    @staticmethod
    def calculate_kalman_hedge_ratio(y: pd.Series, x: pd.Series) -> pd.Series:
        """
        Estimates time-varying hedge ratio using a 1D Kalman Filter.
        """
        n = len(y)
        betas = np.zeros(n)
        P = 1.0       # Estimation error variance
        R = 0.001     # Observation noise covariance
        Q = 0.0001    # Process noise covariance
        beta = 1.0    # Initial hedge ratio

        for t in range(n):
            # Time update
            P = P + Q
            # Measurement update
            xt = x.iloc[t]
            yt = y.iloc[t]
            y_pred = beta * xt
            error = yt - y_pred
            S = xt * P * xt + R
            K = (P * xt) / S
            beta = beta + K * error
            P = (1.0 - K * xt) * P
            betas[t] = beta

        return pd.Series(betas, index=y.index)
