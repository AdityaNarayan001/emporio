"""
Data Loader - Fetches historical stock data for simulation
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads historical stock data from Yahoo Finance"""
    
    def __init__(self, symbol: str, cache_dir: str = "data"):
        """
        Initialize DataLoader
        
        Args:
            symbol: Stock symbol (e.g., "IRCTC.NS" for NSE)
            cache_dir: Directory to cache downloaded data
        """
        self.symbol = symbol
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def fetch_historical_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period_days: int = 365,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            start_date: Start date (YYYY-MM-DD) or None for auto-calculation
            end_date: End date (YYYY-MM-DD) or None for today
            period_days: Days to fetch if start_date is None
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        logger.info(f"Loading historical data for {self.symbol}")
        
        # First, try to load from cached CSV files
        # Look for both exact match and sample data
        import glob
        
        # Try exact match first
        cache_pattern = os.path.join(self.cache_dir, f"{self.symbol.replace('.', '_')}*.csv")
        cached_files = glob.glob(cache_pattern)
        
        # If no exact match, try sample data
        if not cached_files:
            sample_pattern = os.path.join(self.cache_dir, "*sample*.csv")
            cached_files = glob.glob(sample_pattern)
            if cached_files:
                logger.info("No exact match found, using sample data for simulation")
        
        if cached_files:
            # Use the most recent cache file
            cache_file = max(cached_files, key=os.path.getctime)
            logger.info(f"Loading data from cache: {cache_file}")
            try:
                df = pd.read_csv(cache_file)
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                
                # Ensure timezone-aware
                if df['Datetime'].dt.tz is None:
                    df['Datetime'] = df['Datetime'].dt.tz_localize('Asia/Kolkata')
                
                logger.info(
                    f"Successfully loaded {len(df)} data points from cache "
                    f"({df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]})"
                )
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}, will try downloading...")
        
        # If no cache, try downloading
        logger.info("No cached data found, attempting download...")
        return self._download_data(start_date, end_date, period_days, interval)
    
    def _download_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period_days: int = 365,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            start_date: Start date (YYYY-MM-DD) or None for auto-calculation
            end_date: End date (YYYY-MM-DD) or None for today
            period_days: Days to fetch if start_date is None
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        logger.info(f"Fetching historical data for {self.symbol}")
        
        # Calculate dates if not provided
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)
            
        if start_date is None:
            start_date = end_date - timedelta(days=period_days)
        else:
            start_date = pd.to_datetime(start_date)
        
        # Yahoo Finance has limitations on intraday data
        # For intervals < 1d, max period is 60 days
        interval_limits = {
            "1m": 7,    # 1 minute data: max 7 days
            "5m": 60,   # 5 minute data: max 60 days
            "15m": 60,  # 15 minute data: max 60 days
            "30m": 60,  # 30 minute data: max 60 days
            "1h": 730,  # 1 hour data: max 730 days
        }
        
        if interval in interval_limits:
            max_days = interval_limits[interval]
            actual_days = (end_date - start_date).days
            if actual_days > max_days:
                logger.warning(
                    f"Requested {actual_days} days of {interval} data, "
                    f"but maximum is {max_days} days. Adjusting..."
                )
                start_date = end_date - timedelta(days=max_days)
        
        try:
            # Download data from Yahoo Finance
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,  # Adjust for splits and dividends
                actions=False
            )
            
            if df.empty:
                logger.error(f"No data found for {self.symbol}")
                logger.error("This might be due to:")
                logger.error("  1. Yahoo Finance rate limiting (429 Too Many Requests)")
                logger.error("  2. Internet connectivity issues")
                logger.error("  3. Invalid stock symbol")
                logger.error("\nSuggestion: Run 'python3 download_data.py' to create sample data")
                raise ValueError(f"No data found for {self.symbol}")
            
            # Clean and prepare data
            df.index.name = 'Datetime'
            df.reset_index(inplace=True)
            
            # Ensure datetime is timezone-aware
            if df['Datetime'].dt.tz is None:
                df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize('Asia/Kolkata')
            
            # Rename columns to standard format
            df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Remove any NaN values
            df.dropna(inplace=True)
            
            logger.info(
                f"Successfully loaded {len(df)} data points from "
                f"{df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}"
            )
            
            # Cache the data
            cache_file = self._get_cache_filename(start_date, end_date, interval)
            df.to_csv(cache_file, index=False)
            logger.info(f"Data cached to {cache_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def _get_cache_filename(self, start_date, end_date, interval) -> str:
        """Generate cache filename"""
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return os.path.join(
            self.cache_dir,
            f"{self.symbol}_{start_str}_{end_str}_{interval}.csv"
        )
    
    def load_from_cache(self, cache_file: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache file
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        if os.path.exists(cache_file):
            logger.info(f"Loading data from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            return df
        return None
    
    def get_stock_info(self) -> dict:
        """
        Get basic stock information
        
        Returns:
            Dictionary with stock info
        """
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            return {
                "name": info.get("longName", self.symbol),
                "currency": info.get("currency", "INR"),
                "exchange": info.get("exchange", "NSE"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
            }
        except Exception as e:
            logger.warning(f"Could not fetch stock info: {str(e)}")
            return {
                "name": self.symbol,
                "currency": "INR",
                "exchange": "NSE",
                "sector": "Unknown",
                "industry": "Unknown",
            }
