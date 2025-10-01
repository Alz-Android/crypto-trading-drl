"""
Quick test script to verify all ETH components work before training
"""

import os
import sys
import pandas as pd
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing Dependencies...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import gym
        print("✅ OpenAI Gym imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_access():
    """Test if we can access the ETH data"""
    print("\n🔍 Testing Data Access...")
    
    data_file = 'data/ETH_6year_data.csv'
    
    if not os.path.exists(data_file):
        print(f"⚠️ Data file not found: {data_file}")
        print("   Please ensure ETH_6year_data.csv is in the data/ directory")
        print("   The file should contain 6 years of ETH historical data")
        print("   Required columns: date, open_usd, high_usd, low_usd, close_usd, volume")
        return False
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ ETH data loaded successfully: {len(df)} rows")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['date', 'open_usd', 'high_usd', 'low_usd', 'close_usd', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print(f"✅ Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"✅ Price range: ${df['close_usd'].min():.2f} - ${df['close_usd'].max():.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading ETH data: {e}")
        return False

def test_components():
    """Test if our custom components work"""
    print("\n🔍 Testing ETH Trading Components...")
    
    try:
        from src.technical_indicators import TechnicalIndicators
        print("✅ TechnicalIndicators imported successfully")
        
        from src.trading_env import CryptoTradingEnv
        print("✅ CryptoTradingEnv imported successfully")
        
        from src.neural_networks import PPOAgent
        print("✅ PPOAgent imported successfully")
        
        from src.data_fetcher import CryptoCompareDataFetcher
        print("✅ CryptoCompareDataFetcher imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Component import error: {e}")
        return False

def test_component_functionality():
    """Test basic component functionality"""
    print("\n🔍 Testing Component Functionality...")
    
    try:
        # Test Technical Indicators
        from src.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        print("✅ TechnicalIndicators initialized")
        
        # Test PPO Agent
        from src.neural_networks import PPOAgent
        agent = PPOAgent(input_shape=(100, 4), action_dim=3)
        print("✅ PPOAgent initialized")
        
        # Test Data Fetcher
        from src.data_fetcher import CryptoCompareDataFetcher
        fetcher = CryptoCompareDataFetcher()
        print("✅ CryptoCompareDataFetcher initialized")
        
        return True
    except Exception as e:
        print(f"❌ Component functionality error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\n🔍 Testing GPU Availability...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print("🚀 Training will use GPU acceleration")
        return True
    else:
        print("⚠️ CUDA not available - will use CPU")
        print("⏰ Training will be slower on CPU")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\n🔍 Testing Directory Structure...")
    
    required_dirs = ['src', 'data', 'full_training_results']
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}/")
        else:
            print(f"❌ Directory missing: {dir_name}/")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 Pre-Training System Check - Ethereum DRL Bot")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_dependencies()
    all_passed &= test_directories()
    all_passed &= test_components()
    all_passed &= test_component_functionality()
    data_available = test_data_access()
    test_gpu_availability()  # GPU is optional
    
    print("\n" + "=" * 60)
    if all_passed and data_available:
        print("✅ All tests passed! Ready to start ETH training!")
        print("\nNext steps:")
        print("1. Run: python train_ethereum_agent.py")
        print("2. Wait for training to complete (4-8 hours)")
        print("3. Run: python backtest_ethereum_agent.py")
        print("\n🎯 Training Configuration:")
        print("   - Cryptocurrency: Ethereum (ETH)")
        print("   - Episodes: 1000 (full training)")
        print("   - Data: 6 years historical")
        print("   - Initial balance: $10,000")
    elif all_passed and not data_available:
        print("⚠️ System ready but missing ETH data!")
        print("\nTo proceed:")
        print("1. Add ETH_6year_data.csv to the data/ directory")
        print("2. Ensure the file has required columns")
        print("3. Re-run this test")
        print("4. Then start training")
    else:
        print("❌ Some tests failed. Please fix the issues before training.")
    
    return all_passed and data_available

if __name__ == "__main__":
    main()