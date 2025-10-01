"""
Cryptocurrency Historical Data Downloader (Alternative)
Downloads 6 years of historical data for BTC, ETH, and SOL using multiple sources
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import yfinance as yf

class CryptoDataDownloader:
    def __init__(self):
        # Yahoo Finance symbols for crypto
        self.yahoo_symbols = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD', 
            'SOL': 'SOL-USD'
        }
        
        # Alternative API endpoints
        self.binance_base = "https://api.binance.com/api/v3"
        
    def download_from_yahoo(self, symbol, crypto_symbol):
        """
        Download data from Yahoo Finance
        """
        try:
            print(f"📊 Downloading {symbol} data from Yahoo Finance...")
            
            # Calculate date range (6 years)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=6*365)
            
            # Download data
            ticker = yf.Ticker(crypto_symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                print(f"❌ No data available for {symbol}")
                return None
            
            # Clean and format the data
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            
            # Rename columns to match our format
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open_usd',
                'High': 'high_usd', 
                'Low': 'low_usd',
                'Close': 'close_usd',
                'Volume': 'volume'
            })
            
            # Select relevant columns
            df = df[['date', 'open_usd', 'high_usd', 'low_usd', 'close_usd', 'volume']]
            
            print(f"✅ Successfully downloaded {len(df)} days of data for {symbol}")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading {symbol} from Yahoo Finance: {e}")
            return None
    
    def get_binance_data(self, symbol):
        """
        Get recent data from Binance API (free, but limited history)
        """
        try:
            # Binance symbols
            binance_symbols = {
                'BTC': 'BTCUSDT',
                'ETH': 'ETHUSDT',
                'SOL': 'SOLUSDT'
            }
            
            if symbol not in binance_symbols:
                return None
                
            binance_symbol = binance_symbols[symbol]
            
            print(f"📊 Getting recent data for {symbol} from Binance...")
            
            # Get kline data (limited to 1000 records ≈ 2.7 years for daily)
            url = f"{self.binance_base}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': '1d',
                'limit': 1000  # Maximum allowed
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df_data = []
            for kline in data:
                timestamp = int(kline[0])
                date = datetime.fromtimestamp(timestamp / 1000).date()
                
                df_data.append({
                    'date': date,
                    'open_usd': float(kline[1]),
                    'high_usd': float(kline[2]),
                    'low_usd': float(kline[3]),
                    'close_usd': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(df_data)
            print(f"✅ Got {len(df)} days from Binance for {symbol}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting Binance data for {symbol}: {e}")
            return None
    
    def download_all_crypto_data(self):
        """
        Download historical data for all cryptocurrencies using multiple sources
        """
        print("🚀 Starting cryptocurrency data download...")
        print("🎯 Attempting multiple data sources for best coverage")
        print("=" * 60)
        
        # Create data directory if it doesn't exist
        data_dir = "crypto_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"📁 Created directory: {data_dir}")
        
        successful_downloads = 0
        failed_downloads = 0
        
        for symbol in self.yahoo_symbols.keys():
            print(f"\n💰 Processing {symbol}...")
            
            df = None
            
            # Try Yahoo Finance first (best for long-term historical data)
            if symbol in self.yahoo_symbols:
                df = self.download_from_yahoo(symbol, self.yahoo_symbols[symbol])
            
            # If Yahoo fails, try Binance (limited history but reliable)
            if df is None:
                print(f"🔄 Trying Binance as fallback for {symbol}...")
                df = self.get_binance_data(symbol)
            
            if df is not None:
                # Save to CSV
                filename = f"{data_dir}/{symbol}_6year_data.csv"
                df.to_csv(filename, index=False)
                
                # Calculate statistics
                latest_price = df['close_usd'].iloc[-1]
                oldest_price = df['close_usd'].iloc[0]
                total_return = ((latest_price - oldest_price) / oldest_price) * 100
                days_of_data = len(df)
                
                print(f"💾 Saved to: {filename}")
                print(f"📅 Days of data: {days_of_data}")
                print(f"📈 First price: ${oldest_price:,.2f}")
                print(f"📈 Latest price: ${latest_price:,.2f}")
                print(f"📈 Total return: {total_return:+.1f}%")
                
                successful_downloads += 1
            else:
                print(f"❌ Failed to get data for {symbol}")
                failed_downloads += 1
            
            # Add delay between requests
            if symbol != list(self.yahoo_symbols.keys())[-1]:
                print("⏱️  Waiting 1 second...")
                time.sleep(1)
        
        print("\n" + "=" * 60)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"✅ Successful downloads: {successful_downloads}")
        print(f"❌ Failed downloads: {failed_downloads}")
        
        if successful_downloads > 0:
            print(f"\n📁 Data files saved in: {os.path.abspath(data_dir)}/")
            print("🎉 Download completed!")
            
            # Show file information
            print("\n📄 Generated files:")
            for symbol in self.yahoo_symbols.keys():
                filename = f"{data_dir}/{symbol}_6year_data.csv"
                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    df_temp = pd.read_csv(filename)
                    print(f"   • {filename} ({file_size:,} bytes, {len(df_temp):,} records)")
        
        return successful_downloads > 0

def main():
    """
    Main function to run the crypto data downloader
    """
    downloader = CryptoDataDownloader()
    
    print("💰 Cryptocurrency Historical Data Downloader v2")
    print("📅 Downloading up to 6 years of daily data for BTC, ETH, and SOL")
    print("🌐 Data sources: Yahoo Finance, Binance API")
    print()
    
    success = downloader.download_all_crypto_data()
    
    if success:
        print("\n🎯 Next steps:")
        print("   • Check the 'crypto_data' folder for your CSV files")
        print("   • Use pandas to load and analyze the data:")
        print("     df = pd.read_csv('crypto_data/BTC_6year_data.csv')")
        print("   • Each file contains: date, open, high, low, close, volume")
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()