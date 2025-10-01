"""
Quick test of technical indicators
"""
import pandas as pd
import sys
import os

sys.path.append('src')
from src.technical_indicators import TechnicalIndicators

# Load minimal data
print("Loading SOL data...")
df = pd.read_csv('../crypto_data/SOL_6year_data.csv')

# Rename columns
column_mapping = {
    'date': 'timestamp',
    'open_usd': 'open',
    'high_usd': 'high', 
    'low_usd': 'low',
    'close_usd': 'close',
    'volume': 'volume'
}
df = df.rename(columns=column_mapping)

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()

# Take recent data
df = df.tail(180)  # Last 180 days

print(f"Original columns: {list(df.columns)}")
print(f"Volume exists: {'volume' in df.columns}")

# Add technical indicators
ti = TechnicalIndicators()
df = ti.add_technical_indicators(df)
df = df.dropna()

print(f"After indicators: {list(df.columns)}")
print(f"Required features present:")
features = ['close', 'rsi', 'atr', 'obv']
for feature in features:
    present = feature in df.columns
    print(f"  {feature}: {'✅' if present else '❌'}")

print(f"\nData shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Test observation extraction
print(f"\nTesting observation extraction...")
lookback = 30
start_idx = lookback
end_idx = start_idx + lookback

try:
    obs_data = df[features].iloc[start_idx:end_idx].values
    print(f"✅ Observation shape: {obs_data.shape}")
    print("✅ Technical indicators working correctly!")
except Exception as e:
    print(f"❌ Error: {e}")