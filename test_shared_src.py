#!/usr/bin/env python3
"""
Test script to verify that the shared src directory structure works correctly.
This script tests importing from the main src directory.
"""

import os
import sys

# Add main src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all main modules can be imported from shared src."""
    print("Testing shared src directory imports...")
    
    try:
        from data_fetcher import DataFetcher
        print("✅ data_fetcher.DataFetcher imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_fetcher: {e}")
        
    try:
        from technical_indicators import TechnicalIndicators
        print("✅ technical_indicators.TechnicalIndicators imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import technical_indicators: {e}")
        
    try:
        from trading_env import TradingEnvironment
        print("✅ trading_env.TradingEnvironment imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import trading_env: {e}")
        
    try:
        from neural_networks import PPOAgent
        print("✅ neural_networks.PPOAgent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import neural_networks: {e}")

def test_basic_functionality():
    """Test basic functionality of imported modules."""
    print("\nTesting basic functionality...")
    
    try:
        from technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        print("✅ TechnicalIndicators instance created successfully")
    except Exception as e:
        print(f"❌ Failed to create TechnicalIndicators: {e}")
        
    try:
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
        print("✅ DataFetcher instance created successfully")
    except Exception as e:
        print(f"❌ Failed to create DataFetcher: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("SHARED SRC DIRECTORY TEST")
    print("=" * 60)
    
    # Check if src directory exists
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if os.path.exists(src_path):
        print(f"✅ Shared src directory found at: {src_path}")
        
        # List files in src directory
        files = os.listdir(src_path)
        print(f"📁 Files in src/: {files}")
    else:
        print(f"❌ Shared src directory not found at: {src_path}")
        sys.exit(1)
    
    test_imports()
    test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)