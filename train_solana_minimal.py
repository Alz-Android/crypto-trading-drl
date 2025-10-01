"""
MINIMAL training configuration for ultra-fast testing (5-10 minutes).
Perfect for testing the system works before running longer training.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.technical_indicators import TechnicalIndicators
from src.trading_env import CryptoTradingEnv
from src.neural_networks import PPOAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalTrainingConfig:
    """Minimal training configuration - completes in 5-10 minutes"""
    # Data parameters
    CRYPTO_SYMBOL = 'SOL'
    DATA_FILE = '../crypto_data/SOL_6year_data.csv'
    LOOKBACK_WINDOW = 30  # Minimal lookback
    
    # Training parameters - ULTRA MINIMAL
    TOTAL_EPISODES = 50   # Very few episodes
    EPISODES_PER_UPDATE = 5
    MAX_EPISODE_LENGTH = 100  # Very short episodes
    
    # Environment parameters
    INITIAL_BALANCE = 10000
    
    # Neural network parameters
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # PPO parameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    
    # Model saving
    MODEL_SAVE_FREQ = 10
    RESULTS_DIR = 'minimal_training_results'
    
    # Use only 6 months of recent data
    USE_RECENT_DATA_ONLY = True
    RECENT_DATA_DAYS = 180

# Import from the original training script
from train_solana_agent import TrainingManager

class MinimalTrainingManager(TrainingManager):
    """Minimal training manager for ultra-fast testing"""
    
    def setup_data(self):
        """Load minimal dataset for ultra-fast training"""
        logger.info(f"Loading MINIMAL {self.config.CRYPTO_SYMBOL} data...")
        
        if not os.path.exists(self.config.DATA_FILE):
            raise FileNotFoundError(f"Data file not found: {self.config.DATA_FILE}")
        
        # Read and process data
        self.df = pd.read_csv(self.config.DATA_FILE)
        
        column_mapping = {
            'date': 'timestamp',
            'open_usd': 'open',
            'high_usd': 'high', 
            'low_usd': 'low',
            'close_usd': 'close',
            'volume': 'volume'
        }
        self.df = self.df.rename(columns=column_mapping)
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.set_index('timestamp')
        self.df = self.df.sort_index()
        
        # Use only very recent data for minimal training
        self.df = self.df.tail(180)  # Last 180 days (6 months)
        logger.info(f"Using minimal dataset: {len(self.df)} days")
        
        # Add technical indicators
        ti = TechnicalIndicators()
        self.df = ti.add_technical_indicators(self.df)
        self.df = self.df.dropna()
        
        logger.info(f"Data after indicators: {len(self.df)} days")
        logger.info(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
        
        # Split data
        split_idx = int(0.8 * len(self.df))
        self.train_df = self.df[:split_idx].copy()
        self.val_df = self.df[split_idx:].copy()
        
        logger.info(f"Training: {len(self.train_df)} days, Validation: {len(self.val_df)} days")


def main():
    """Main minimal training function"""
    print("🚀 MINIMAL Solana Trading DRL Training")
    print("⚡ Ultra-fast test - Expected time: 5-10 minutes")
    print("🔬 Perfect for testing before full training")
    print("=" * 60)
    
    config = MinimalTrainingConfig()
    trainer = MinimalTrainingManager(config)
    
    print(f"Minimal Training Configuration:")
    print(f"  Episodes: {config.TOTAL_EPISODES} (ultra minimal)")
    print(f"  Episode length: {config.MAX_EPISODE_LENGTH} (very short)")
    print(f"  Lookback: {config.LOOKBACK_WINDOW} (minimal)")
    print(f"  Data: Last {config.RECENT_DATA_DAYS} days only")
    print(f"  Device: {config.DEVICE}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ Minimal training completed!")
        print(f"⏱️ Training time: {duration}")
        print(f"📁 Results: {config.RESULTS_DIR}/")
        
        if trainer.metrics['episode_rewards']:
            avg_reward = np.mean(trainer.metrics['episode_rewards'][-10:])
            avg_return = np.mean(trainer.metrics['total_returns'][-10:])
            print(f"📊 Final avg reward: ${avg_reward:.2f}")
            print(f"📈 Final avg return: {avg_return:.2f}%")
            
        print(f"\n🚀 System working! Next steps:")
        print(f"   Fast training: python train_solana_fast.py (15-30 min)")
        print(f"   Full training: python train_solana_agent.py (2-4 hours)")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")
        trainer.save_model('interrupted_minimal')
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        logger.error(f"Training failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()