import numpy as np
import pandas as pd

class PairsTradingBacktester:
    """
    Backtests a mean-reverting statistical arbitrage strategy with rolling Z-score signals.
    """
    def __init__(self, y: pd.Series, x: pd.Series, window: int = 30, 
                 entry_threshold: float = 2.0, exit_threshold: float = 0.0,
                 slippage_bps: float = 2.0):
        self.y = y
        self.x = x
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.slippage = slippage_bps / 10000.0

    def run_backtest(self, beta: float) -> pd.DataFrame:
        df = pd.DataFrame({'Asset_Y': self.y, 'Asset_X': self.x})
        df['Spread'] = df['Asset_Y'] - beta * df['Asset_X']

        # Rolling Mean and Std
        df['Rolling_Mean'] = df['Spread'].rolling(window=self.window).mean()
        df['Rolling_Std'] = df['Spread'].rolling(window=self.window).std()
        df['Z_Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']

        # Signal Generation
        # Long Spread: buy Y, short X when Z < -entry_threshold
        # Short Spread: short Y, buy X when Z > +entry_threshold
        df['Position'] = 0

        current_pos = 0
        positions = []

        for z in df['Z_Score']:
            if np.isnan(z):
                positions.append(0)
                continue

            if z > self.entry_threshold:
                current_pos = -1 # Short spread
            elif z < -self.entry_threshold:
                current_pos = 1  # Long spread
            elif abs(z) <= self.exit_threshold:
                current_pos = 0  # Reverted to mean: close position

            positions.append(current_pos)

        df['Position'] = positions

        # Calculate returns
        df['Ret_Y'] = df['Asset_Y'].pct_change().fillna(0)
        df['Ret_X'] = df['Asset_X'].pct_change().fillna(0)

        # Spread return based on position held previous day
        df['Pos_Lagged'] = df['Position'].shift(1).fillna(0)
        df['Gross_Return'] = df['Pos_Lagged'] * (df['Ret_Y'] - beta * df['Ret_X'])

        # Transaction cost deduction on position changes
        df['Trades'] = df['Position'].diff().abs().fillna(0)
        df['Net_Return'] = df['Gross_Return'] - (df['Trades'] * self.slippage)
        df['Cumulative_Return'] = (1.0 + df['Net_Return']).cumprod()

        return df
