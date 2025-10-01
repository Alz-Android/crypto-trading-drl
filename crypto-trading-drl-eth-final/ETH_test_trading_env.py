#!/usr/bin/env python3
"""
Test script for the ETH trading environment
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trading_env import CryptoTradingEnv
from src.data_fetcher import CryptoCompareDataFetcher
from src.technical_indicators import TechnicalIndicators

def test_eth_trading_env():
    """Test the trading environment functionality for Ethereum"""
    print("Testing ETH Crypto Trading Environment...")

    # Load data with indicators
    fetcher = CryptoCompareDataFetcher()
    df = fetcher.load_data('eth_usd_with_indicators')

    if df is None or len(df) < 110:  # Need at least 110 points for 100 lookback + 10 steps
        print("Not enough ETH data. Fetching more...")
        df = fetcher.get_historical_hourly('ETH', 'USD', limit=150)
        if df is not None:
            ti = TechnicalIndicators()
            df = ti.add_technical_indicators(df)
            fetcher.save_data(df, 'eth_usd_with_indicators')
        else:
            print("Failed to fetch ETH data")
            return

    if df is None or len(df) < 110:
        print("Still not enough ETH data for testing")
        return

    print(f"ETH data shape: {df.shape}")
    print("ETH data columns:", list(df.columns))
    print(f"ETH data date range: {df.index.min()} to {df.index.max()}")
    print(f"ETH price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

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

    print("\nTesting ETH environment steps:")
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
        print(f"  ETH Held: {info['crypto_held']:.6f}")
        print(f"  Current ETH Price: ${info['current_price']:.2f}")

        if done:
            print("Episode finished!")
            break

    print(f"\nTotal reward: ${total_reward:.2f}")
    print(f"Actions taken: {[ ['Hold', 'Buy', 'Sell'][a] for a in actions_taken ]}")

    # Test environment reset
    print("\nTesting ETH environment reset...")
    obs2 = env.reset()
    print(f"Reset observation shape: {obs2.shape}")
    print("ETH environment reset successful!")

    # Test trading logic
    print("\nTesting ETH trading logic...")
    env.reset()
    
    # Test buying
    initial_balance = env.balance
    current_price = env.df.iloc[env.current_step]['close']
    obs, reward, done, info = env.step(1)  # Buy action
    print(f"After buy: Balance=${info['balance']:.2f}, ETH Held={info['crypto_held']:.6f}")
    
    # Test selling
    if info['crypto_held'] > 0:
        obs, reward, done, info = env.step(2)  # Sell action
        print(f"After sell: Balance=${info['balance']:.2f}, ETH Held={info['crypto_held']:.6f}")

    print("\nETH Trading environment test completed successfully! ✅")

def test_eth_technical_indicators():
    """Test technical indicators with ETH data"""
    print("\nTesting technical indicators with ETH data...")
    
    fetcher = CryptoCompareDataFetcher()
    ti = TechnicalIndicators()
    
    # Fetch fresh ETH data
    eth_data = fetcher.get_historical_hourly('ETH', 'USD', limit=50)
    
    if eth_data is not None:
        print(f"Original ETH data: {eth_data.shape}")
        
        # Add indicators
        eth_with_indicators = ti.add_technical_indicators(eth_data)
        print(f"ETH data with indicators: {eth_with_indicators.shape}")
        print("ETH indicators columns:", [col for col in eth_with_indicators.columns if col in ['rsi', 'atr', 'obv']])
        
        # Show sample values
        print("\nSample ETH indicators:")
        print(eth_with_indicators[['close', 'rsi', 'atr', 'obv']].tail())
        
        print("✅ ETH technical indicators test passed!")
    else:
        print("❌ Failed to fetch ETH data for indicators test")

if __name__ == "__main__":
    test_eth_trading_env()
    test_eth_technical_indicators()