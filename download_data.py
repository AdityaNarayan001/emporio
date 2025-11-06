"""
Data Downloader - Pre-download historical stock data
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_irctc_data():
    """Download IRCTC historical data and save to CSV"""
    
    symbol = "IRCTC.NS"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Try different approaches
    logger.info(f"Downloading data for {symbol}...")
    
    # Approach 1: Try daily data for 1 year using period parameter
    try:
        logger.info(f"Fetching 1 year of daily data using period='1y'")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            period="1y",  # Get 1 year of data
            interval="1d",
            auto_adjust=True,
            actions=False
        )
        
        if df.empty:
            raise ValueError("No data returned")
        
        # Clean data
        df.reset_index(inplace=True)
        df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Save to CSV
        output_file = os.path.join(data_dir, f"{symbol.replace('.', '_')}_daily_1year.csv")
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Successfully downloaded {len(df)} days of data")
        logger.info(f"📁 Saved to: {output_file}")
        logger.info(f"📅 Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        
        # Display sample
        print("\n" + "="*60)
        print("Sample data (first 5 rows):")
        print("="*60)
        print(df.head())
        print("\n" + "="*60)
        print("Sample data (last 5 rows):")
        print("="*60)
        print(df.tail())
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error downloading data: {str(e)}")
        
        # Approach 2: Try alternative symbols
        alternative_symbols = ["IRCTC.BO"]  # Bombay Stock Exchange
        
        for alt_symbol in alternative_symbols:
            try:
                logger.info(f"Trying alternative symbol: {alt_symbol}")
                ticker = yf.Ticker(alt_symbol)
                df = ticker.history(
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    actions=False
                )
                
                if not df.empty:
                    df.reset_index(inplace=True)
                    df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                    
                    output_file = os.path.join(data_dir, f"{alt_symbol.replace('.', '_')}_daily_1year.csv")
                    df.to_csv(output_file, index=False)
                    
                    logger.info(f"✅ Successfully downloaded from {alt_symbol}")
                    logger.info(f"📁 Saved to: {output_file}")
                    return output_file
                    
            except Exception as e2:
                logger.error(f"Failed with {alt_symbol}: {str(e2)}")
                continue
        
        # Approach 3: Create sample data as fallback
        logger.warning("⚠️  Could not download real data. Creating sample data for testing...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = create_sample_data(start_date, end_date)
        output_file = os.path.join(data_dir, "IRCTC_sample_data.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"📁 Sample data saved to: {output_file}")
        return output_file


def create_sample_data(start_date, end_date):
    """Create sample IRCTC-like data for testing"""
    import numpy as np
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic stock price movements
    # IRCTC typically trades in 800-1000 range
    initial_price = 850
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Create OHLC data
    df = pd.DataFrame({
        'Datetime': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(prices))),
        'High': prices * (1 + np.random.uniform(0.01, 0.03, len(prices))),
        'Low': prices * (1 - np.random.uniform(0.01, 0.03, len(prices))),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, len(prices))
    })
    
    logger.info(f"Created sample data with {len(df)} days")
    return df


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📥 IRCTC Data Downloader")
    print("="*60 + "\n")
    
    output_file = download_irctc_data()
    
    print("\n" + "="*60)
    print("✅ Download complete!")
    print(f"📁 Data file: {output_file}")
    print("="*60)
    print("\nYou can now run the simulator with: python3 main.py")
    print("="*60 + "\n")
