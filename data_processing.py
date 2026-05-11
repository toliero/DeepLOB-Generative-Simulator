import yfinance as yf
import pandas as pd
import numpy as np

def get_1m_data_and_calc_volatility(tickers=["SPY", "QQQ"]):
    """
    Downloads 1-minute interval data, cleans it, and calculates realized volatility.
    
    Note: Yahoo Finance strictly limits 1-minute data downloads to the last 7 days.
    To retrieve 30 days of 1-minute data, a paid API (e.g., Polygon.io, Alpaca) is typically required.
    This script will fetch the maximum available 1-minute data (7 days) using yfinance.
    """
    print("Note: yfinance limits 1-minute data to the last 7 days. Fetching max available (7 days)...")
    
    # Download data
    data = yf.download(tickers, period="7d", interval="1m", group_by="ticker")
    
    results = {}
    
    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        
        # Handle different dataframe structures depending on the number of tickers
        df = data[ticker].copy() if len(tickers) > 1 else data.copy()
        
        # 1. Clean the data for missing timestamps
        # Remove completely empty rows (common outside of regular market hours)
        df.dropna(how='all', inplace=True)
        
        # Forward fill to handle any remaining intermediate missing values
        df.ffill(inplace=True)
        
        # 2. Calculate Realized Volatility
        # Calculate logarithmic returns
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Drop the first row which will have a NaN log return
        df.dropna(subset=['Log_Return'], inplace=True)
        
        # Period Realized Volatility: Square root of the sum of squared logarithmic returns
        realized_vol = np.sqrt((df['Log_Return'] ** 2).sum())
        
        # Annualized Realized Volatility:
        # Assuming ~390 trading minutes per day and 252 trading days per year
        annualized_vol = df['Log_Return'].std() * np.sqrt(252 * 390)
        
        results[ticker] = {
            'realized_volatility': realized_vol,
            'annualized_volatility': annualized_vol,
            'data': df
        }
        
        print(f"Total 1m data points: {len(df)}")
        print(f"Period Realized Volatility: {realized_vol:.4f}")
        print(f"Annualized Volatility:      {annualized_vol:.4f}")
        
    return results

if __name__ == "__main__":
    get_1m_data_and_calc_volatility(["SPY", "QQQ"])
