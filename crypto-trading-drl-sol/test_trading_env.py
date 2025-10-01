#!/usr/bin/env python3
"""
Test script for the trading environment
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from trading_env import CryptoTradingEnv
from data_fetcher import CryptoCompareDataFetcher
from technical_indicators import TechnicalIndicators

def test_trading_env():
    """Test the trading environment functionality"""
    print("Testing Crypto Trading Environment...")

    # Load data with indicators
    fetcher = CryptoCompareDataFetcher()
    df = fetcher.load_data('btc_usd_with_indicators')

    if df is None or len(df) < 110:  # Need at least 110 points for 100 lookback + 10 steps
        print("Not enough data. Fetching more...")
        df = fetcher.get_historical_hourly('BTC', 'USD', limit=150)
        if df is not None:
            ti = TechnicalIndicators()
            df = ti.add_technical_indicators(df)
            fetcher.save_data(df, 'btc_usd_with_indicators')
        else:
            print("Failed to fetch data")
            return

    if df is None or len(df) < 110:
        print("Still not enough data for testing")
        return

    print(f"Data shape: {df.shape}")
    print("Data columns:", list(df.columns))
    print(f"Data date range: {df.index.min()} to {df.index.max()}")

    # Create environment
    env = CryptoTradingEnv(df, initial_balance=10000, lookback_window=100)

    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test a few steps with different actions
    total_reward = 0
    actions_taken = []

    print("\nTesting environment steps:")
    for i in range(10):
        # Choose action (0=Hold, 1=Buy, 2=Sell)
        action = np.random.randint(0, 3)
        actions_taken.append(action)

        # Take step
        obs, reward, done, info = env.step(action)

        total_reward += reward

        print(f"Step {i+1}:")
        print(f"  Action: {['Hold', 'Buy', 'Sell'][action]}")
        print(f"  Reward: ${reward:.2f}")
        print(f"  Net Worth: ${info['net_worth']:.2f}")
        print(f"  Balance: ${info['balance']:.2f}")
        print(f"  Crypto Held: {info['crypto_held']:.6f}")
        print(f"  Current Price: ${info['current_price']:.2f}")

        if done:
            print("Episode finished!")
            break

    print(f"\nTotal reward: ${total_reward:.2f}")
    print(f"Actions taken: {[ ['Hold', 'Buy', 'Sell'][a] for a in actions_taken ]}")

    # Test environment reset
    print("\nTesting environment reset...")
    obs2 = env.reset()
    print(f"Reset observation shape: {obs2.shape}")
    print("Environment reset successful!")

    print("\nTrading environment test completed successfully! ✅")

if __name__ == "__main__":
    test_trading_env()
