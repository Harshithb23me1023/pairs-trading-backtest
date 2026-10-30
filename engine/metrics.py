import numpy as np
import pandas as pd

class PerformanceAnalytics:
    """
    Computes key quantitative risk and return performance metrics.
    """
    @staticmethod
    def calculate_metrics(results_df: pd.DataFrame, risk_free_rate: float = 0.05) -> dict:
        net_returns = results_df['Net_Return'].dropna()
        cum_returns = results_df['Cumulative_Return'].dropna()

        # Annualized Return (assuming 252 trading days)
        trading_days = len(net_returns)
        total_return = cum_returns.iloc[-1] - 1.0
        annualized_return = (1.0 + total_return) ** (252.0 / trading_days) - 1.0

        # Annualized Volatility
        daily_vol = net_returns.std()
        annualized_vol = daily_vol * np.sqrt(252.0)

        # Sharpe Ratio
        rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
        excess_daily_returns = net_returns - rf_daily
        sharpe_ratio = (excess_daily_returns.mean() / daily_vol) * np.sqrt(252.0) if daily_vol > 0 else 0.0

        # Maximum Drawdown
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Win Rate
        winning_days = (net_returns > 0).sum()
        active_days = (net_returns != 0).sum()
        win_rate = (winning_days / active_days * 100.0) if active_days > 0 else 0.0

        # Total trades
        total_trades = int(results_df['Trades'].sum())

        return {
            "Total Cumulative Return": f"{total_return * 100:.2f}%",
            "Annualized Return": f"{annualized_return * 100:.2f}%",
            "Annualized Volatility": f"{annualized_vol * 100:.2f}%",
            "Sharpe Ratio": round(sharpe_ratio, 2),
            "Maximum Drawdown": f"{max_drawdown * 100:.2f}%",
            "Win Rate (Active Days)": f"{win_rate:.1f}%",
            "Total Trade Rebalances": total_trades
        }
