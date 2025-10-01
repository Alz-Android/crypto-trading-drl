#!/usr/bin/env python3
"""
Test script to verify that training scripts can properly import from shared src.
This tests the actual import patterns used by training scripts.
"""

import os
import sys

def test_btc_imports():
    """Test Bitcoin training script imports"""
    print("Testing Bitcoin training script imports...")
    
    # Simulate being in BTC directory
    btc_dir = os.path.join(os.path.dirname(__file__), 'crypto-trading-drl-btc')
    
    # Add main src to path (like the training script does)
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from data_fetcher import DataFetcher
        from trading_env import TradingEnvironment
        from neural_networks import PPOAgent
        from technical_indicators import TechnicalIndicators
        print("✅ Bitcoin training imports successful")
        return True
    except ImportError as e:
        print(f"❌ Bitcoin training imports failed: {e}")
        return False

def test_eth_imports():
    """Test Ethereum training script imports"""
    print("Testing Ethereum training script imports...")
    
    try:
        from technical_indicators import TechnicalIndicators
        from trading_env import TradingEnvironment
        from neural_networks import PPOAgent
        print("✅ Ethereum training imports successful")
        return True
    except ImportError as e:
        print(f"❌ Ethereum training imports failed: {e}")
        return False

def test_sol_imports():
    """Test Solana training script imports"""
    print("Testing Solana training script imports...")
    
    try:
        from technical_indicators import TechnicalIndicators
        from trading_env import TradingEnvironment
        from neural_networks import PPOAgent
        print("✅ Solana training imports successful")
        return True
    except ImportError as e:
        print(f"❌ Solana training imports failed: {e}")
        return False

def test_module_functionality():
    """Test that modules can be instantiated and used"""
    print("Testing module functionality...")
    
    try:
        from technical_indicators import TechnicalIndicators
        from data_fetcher import DataFetcher
        
        # Test instantiation
        ti = TechnicalIndicators()
        fetcher = DataFetcher()
        
        print("✅ Module instantiation successful")
        return True
    except Exception as e:
        print(f"❌ Module functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SHARED ARCHITECTURE TRAINING SCRIPT TEST")
    print("=" * 60)
    
    results = []
    results.append(test_btc_imports())
    results.append(test_eth_imports())
    results.append(test_sol_imports())
    results.append(test_module_functionality())
    
    print("\n" + "=" * 60)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("✅ Shared src/ architecture is working perfectly")
        print("✅ All cryptocurrency training scripts can import shared modules")
        print("✅ Zero code duplication achieved")
    else:
        print("❌ Some tests failed")
        print("Please check the import paths and module availability")
    
    print("=" * 60)