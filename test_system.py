"""
Quick test script to verify all components work before training
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
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_access():
    """Test if we can access the SOL data"""
    print("\n🔍 Testing Data Access...")
    
    data_file = '../crypto_data/SOL_6year_data.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Data loaded successfully: {len(df)} rows")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_components():
    """Test if our custom components work"""
    print("\n🔍 Testing Components...")
    
    try:
        from src.technical_indicators import TechnicalIndicators
        print("✅ TechnicalIndicators imported successfully")
        
        from src.trading_env import CryptoTradingEnv
        print("✅ CryptoTradingEnv imported successfully")
        
        from src.neural_networks import PPOAgent
        print("✅ PPOAgent imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Component import error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\n🔍 Testing GPU Availability...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        return True
    else:
        print("⚠️ CUDA not available - will use CPU")
        return False

def main():
    """Run all tests"""
    print("🧪 Pre-Training System Check")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_dependencies()
    all_passed &= test_data_access()
    all_passed &= test_components()
    test_gpu_availability()  # GPU is optional
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! Ready to start training!")
        print("\nNext steps:")
        print("1. Run: python train_solana_agent.py")
        print("2. Wait for training to complete")
        print("3. Run: python backtest_solana_agent.py")
    else:
        print("❌ Some tests failed. Please fix the issues before training.")
    
    return all_passed

if __name__ == "__main__":
    main()