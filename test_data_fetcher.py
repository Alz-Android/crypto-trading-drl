#!/usr/bin/env python3
"""
Simple test script for the data fetcher
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import CryptoCompareDataFetcher

def test_data_fetcher():
    """Test the data fetcher functionality"""
    print("Testing CryptoCompare Data Fetcher...")

    # Initialize fetcher
    fetcher = CryptoCompareDataFetcher()

    # Test basic fetch
    print("Fetching recent BTC/USD data...")
    btc_data = fetcher.get_historical_hourly('BTC', 'USD', limit=10)

    if btc_data is not None:
        print("✓ Successfully fetched data!")
        print(f"Data shape: {btc_data.shape}")
        print("Sample data:")
        print(btc_data.head())
        print(f"Columns: {list(btc_data.columns)}")

        # Save test data
        fetcher.save_data(btc_data, 'btc_usd_test')
        print("✓ Data saved to data/btc_usd_test.csv")
    else:
        print("✗ Failed to fetch data")

if __name__ == "__main__":
    test_data_fetcher()
