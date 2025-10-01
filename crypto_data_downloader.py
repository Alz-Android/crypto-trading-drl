"""
Cryptocurrency Historical Data Downloader
Downloads 6 years of historical data for BTC, ETH, and SOL
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json

class CryptoDataDownloader:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.crypto_ids = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'SOL': 'solana'
        }
        
    def get_historical_data(self, crypto_id, days=2190):  # 6 years ≈ 2190 days
        """
        Fetch historical data from CoinGecko API
        """
        url = f"{self.base_url}/coins/{crypto_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        print(f"Fetching data for {crypto_id}...")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract prices, market caps, and volumes
            prices = data['prices']
            market_caps = data['market_caps'] 
            volumes = data['total_volumes']
            
            # Convert to DataFrame
            df_data = []
            for i in range(len(prices)):
                timestamp = prices[i][0]
                date = datetime.fromtimestamp(timestamp / 1000)
                
                df_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'timestamp': timestamp,
                    'price_usd': prices[i][1],
                    'market_cap_usd': market_caps[i][1] if i < len(market_caps) else None,
                    'volume_24h_usd': volumes[i][1] if i < len(volumes) else None
                })
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            print(f"✅ Successfully fetched {len(df)} days of data for {crypto_id}")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data for {crypto_id}: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error for {crypto_id}: {e}")
            return None
    
    def download_all_crypto_data(self):
        """
        Download historical data for all cryptocurrencies
        """
        print("🚀 Starting cryptocurrency data download...")
        print("=" * 60)
        
        # Create data directory if it doesn't exist
        data_dir = "crypto_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"📁 Created directory: {data_dir}")
        
        successful_downloads = 0
        failed_downloads = 0
        
        for symbol, crypto_id in self.crypto_ids.items():
            print(f"\n📊 Processing {symbol} ({crypto_id})...")
            
            # Fetch data
            df = self.get_historical_data(crypto_id)
            
            if df is not None:
                # Save to CSV
                filename = f"{data_dir}/{symbol}_6year_data.csv"
                df.to_csv(filename, index=False)
                
                # Calculate some statistics
                latest_price = df['price_usd'].iloc[-1]
                price_6y_ago = df['price_usd'].iloc[0]
                total_return = ((latest_price - price_6y_ago) / price_6y_ago) * 100
                
                print(f"💾 Saved to: {filename}")
                print(f"📈 Price 6 years ago: ${price_6y_ago:,.2f}")
                print(f"📈 Latest price: ${latest_price:,.2f}")
                print(f"📈 6-year return: {total_return:+.1f}%")
                
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Add delay to be respectful to the API
            if symbol != list(self.crypto_ids.keys())[-1]:  # Don't delay after the last one
                print("⏱️  Waiting 2 seconds before next request...")
                time.sleep(2)
        
        print("\n" + "=" * 60)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"✅ Successful downloads: {successful_downloads}")
        print(f"❌ Failed downloads: {failed_downloads}")
        
        if successful_downloads > 0:
            print(f"\n📁 Data files saved in: {os.path.abspath(data_dir)}/")
            print("🎉 Download completed successfully!")
            
            # Show file information
            print("\n📄 Generated files:")
            for symbol in self.crypto_ids.keys():
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
    
    print("💰 Cryptocurrency Historical Data Downloader")
    print("📅 Downloading 6 years of daily data for BTC, ETH, and SOL")
    print("🌐 Data source: CoinGecko API")
    print()
    
    success = downloader.download_all_crypto_data()
    
    if success:
        print("\n🎯 Next steps:")
        print("   • Check the 'crypto_data' folder for your CSV files")
        print("   • Use pandas to load and analyze the data:")
        print("     df = pd.read_csv('crypto_data/BTC_6year_data.csv')")
        print("   • Each file contains: date, price, market cap, and volume data")
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()