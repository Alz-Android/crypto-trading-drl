#!/usr/bin/env python3
"""
Simple test script for the ETH data fetcher
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import CryptoCompareDataFetcher

def test_eth_data_fetcher():
    """Test the data fetcher functionality for Ethereum"""
    print("Testing CryptoCompare Data Fetcher for Ethereum...")

    # Initialize fetcher
    fetcher = CryptoCompareDataFetcher()

    # Test basic fetch for ETH
    print("Fetching recent ETH/USD data...")
    eth_data = fetcher.get_historical_hourly('ETH', 'USD', limit=10)

    if eth_data is not None:
        print("✅ Successfully fetched ETH data!")
        print(f"Data shape: {eth_data.shape}")
        print("Sample ETH data:")
        print(eth_data.head())
        print(f"Columns: {list(eth_data.columns)}")
        print(f"Price range: ${eth_data['close'].min():.2f} - ${eth_data['close'].max():.2f}")

        # Save test data
        fetcher.save_data(eth_data, 'eth_usd_test')
        print("✅ ETH data saved to data/eth_usd_test.csv")
        
        # Test data loading
        loaded_data = fetcher.load_data('eth_usd_test')
        if loaded_data is not None:
            print("✅ ETH data loaded successfully from file")
        else:
            print("❌ Failed to load ETH data from file")
            
    else:
        print("❌ Failed to fetch ETH data")

def test_eth_multi_day_fetch():
    """Test fetching multiple days of ETH data"""
    print("\nTesting multi-day ETH data fetch...")
    
    fetcher = CryptoCompareDataFetcher()
    
    # Fetch 7 days of ETH data
    print("Fetching 7 days of ETH data...")
    eth_data = fetcher.get_multiple_days('ETH', 'USD', days=7)
    
    if eth_data is not None:
        print(f"✅ Successfully fetched {len(eth_data)} hours of ETH data")
        print(f"Date range: {eth_data.index[0]} to {eth_data.index[-1]}")
        print(f"ETH price range: ${eth_data['close'].min():.2f} - ${eth_data['close'].max():.2f}")
        
        # Save multi-day data
        fetcher.save_data(eth_data, 'eth_usd_7day_test')
        print("✅ 7-day ETH data saved")
    else:
        print("❌ Failed to fetch multi-day ETH data")

if __name__ == "__main__":
    test_eth_data_fetcher()
    test_eth_multi_day_fetch()