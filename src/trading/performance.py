"""
Performance Tracker - Tracks and calculates trading performance metrics
"""
import pandas as pd
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks portfolio performance metrics over time"""
    
    def __init__(self):
        """Initialize PerformanceTracker"""
        self.performance_history: List[Dict] = []
        
    def record_snapshot(
        self,
        timestamp: datetime,
        portfolio_value: float,
        cash: float,
        positions_value: float,
        positions: Dict[str, int],
        current_prices: Dict[str, float]
    ):
        """
        Record a portfolio snapshot
        
        Args:
            timestamp: Current timestamp
            portfolio_value: Total portfolio value
            cash: Current cash balance
            positions_value: Value of all positions
            positions: Current positions {symbol: quantity}
            current_prices: Current prices {symbol: price}
        """
        snapshot = {
            "timestamp": timestamp,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "positions_value": positions_value,
            "positions": positions.copy(),
            "prices": current_prices.copy()
        }
        self.performance_history.append(snapshot)
    
    def get_performance_df(self) -> pd.DataFrame:
        """Get performance history as DataFrame"""
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        return df
    
    def calculate_metrics(self, initial_capital: float) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            initial_capital: Initial portfolio value
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.performance_history:
            return {}
        
        df = self.get_performance_df()
        
        # Current value
        current_value = df['portfolio_value'].iloc[-1]
        
        # Total return
        total_return = current_value - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0
        
        # Max drawdown
        df['peak'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['peak'] - df['portfolio_value']) / df['peak'] * 100
        max_drawdown = df['drawdown'].max()
        
        # Volatility (standard deviation of returns)
        df['returns'] = df['portfolio_value'].pct_change()
        volatility = df['returns'].std() * 100  # As percentage
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        avg_return = df['returns'].mean()
        sharpe_ratio = (avg_return / df['returns'].std()) if df['returns'].std() > 0 else 0
        
        # Win rate (percentage of positive return periods)
        positive_periods = (df['returns'] > 0).sum()
        total_periods = len(df['returns'].dropna())
        win_rate = (positive_periods / total_periods * 100) if total_periods > 0 else 0
        
        return {
            "current_value": current_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe_ratio,
            "win_rate_pct": win_rate,
            "num_snapshots": len(self.performance_history)
        }
    
    def get_recent_performance(self, periods: int = 10) -> List[Dict]:
        """Get recent performance snapshots"""
        return self.performance_history[-periods:] if self.performance_history else []
    
    def clear(self):
        """Clear performance history"""
        self.performance_history.clear()
        logger.info("Performance history cleared")
