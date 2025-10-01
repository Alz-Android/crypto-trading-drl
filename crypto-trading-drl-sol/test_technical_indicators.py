#!/usr/bin/env python3
"""
Test script for technical indicators
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from technical_indicators import TechnicalIndicators
from data_fetcher import CryptoCompareDataFetcher

def test_technical_indicators():
    """Test the technical indicators functionality"""
    print("Testing Technical Indicators...")

    # Load test data
    fetcher = CryptoCompareDataFetcher()
    df = fetcher.load_data('btc_usd_test')

    if df is None or df.empty:
        print("No test data found. Fetching new data...")
        df = fetcher.get_historical_hourly('BTC', 'USD', limit=50)
        if df is None:
            print("Failed to fetch data")
            return

    print(f"Original data shape: {df.shape}")
    print("Original columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())

    # Initialize technical indicators
    ti = TechnicalIndicators()

    # Add technical indicators
    df_with_indicators = ti.add_technical_indicators(df)

    print(f"\nData with indicators shape: {df_with_indicators.shape}")
    print("New columns:", list(df_with_indicators.columns))

    # Show sample data with indicators
    print("\nSample data with indicators:")
    print(df_with_indicators[['close', 'rsi', 'atr', 'obv']].tail(10))

    # Test individual indicators
    print("\nTesting individual indicators:")

    # Test RSI
    rsi = ti.calculate_rsi(df['close'])
    print(f"RSI range: {rsi.min():.2f} - {rsi.max():.2f}")

    # Test ATR
    atr = ti.calculate_atr(df['high'], df['low'], df['close'])
    print(f"ATR range: {atr.min():.2f} - {atr.max():.2f}")

    # Test OBV
    obv = ti.calculate_obv(df['close'], df['volume_from'])
    print(f"OBV range: {obv.min():.0f} - {obv.max():.0f}")

    # Test feature preparation
    features = ti.prepare_features_for_model(df_with_indicators, lookback_window=10)
    if features is not None:
        print(f"\nPrepared features shape: {features.shape}")
        print("Features ready for model input!")
        print(f"Sample feature window shape: {features[0].shape}")
    else:
        print("\nNot enough data for feature preparation")

    # Save data with indicators
    fetcher.save_data(df_with_indicators, 'btc_usd_with_indicators')
    print("\n✓ Data with indicators saved!")

if __name__ == "__main__":
    test_technical_indicators()
