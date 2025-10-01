"""
Fast training configuration for quick testing and development.
Significantly reduced parameters for faster training on CPU.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.technical_indicators import TechnicalIndicators
from src.trading_env import CryptoTradingEnv
from src.neural_networks import PPOAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastTrainingConfig:
    """Fast training configuration for CPU - completes in 15-30 minutes"""
    # Data parameters
    CRYPTO_SYMBOL = 'SOL'  # Solana
    DATA_FILE = '../crypto_data/SOL_6year_data.csv'
    LOOKBACK_WINDOW = 50  # Reduced from 100 - less memory, faster
    
    # Training parameters - SIGNIFICANTLY REDUCED
    TOTAL_EPISODES = 200  # Reduced from 1000 - 5x faster
    EPISODES_PER_UPDATE = 5  # Reduced from 10 - 2x faster updates
    MAX_EPISODE_LENGTH = 300  # Reduced from 1000 - 3x faster episodes
    
    # Environment parameters
    INITIAL_BALANCE = 10000
    
    # Neural network parameters
    LEARNING_RATE = 1e-3  # Slightly higher for faster convergence
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # PPO parameters
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # GAE parameter
    
    # Model saving
    MODEL_SAVE_FREQ = 25  # Save every 25 episodes (was 50)
    RESULTS_DIR = 'fast_training_results'
    
    # Data sampling for faster training
    USE_RECENT_DATA_ONLY = True  # Use only recent 2 years instead of full 6 years
    RECENT_DATA_YEARS = 2

# Import the TrainingManager from the original script but with our fast config
from train_solana_agent import TrainingManager

class FastTrainingManager(TrainingManager):
    """Fast training manager with optimizations for speed"""
    
    def setup_data(self):
        """Load and prepare training data with fast training optimizations"""
        logger.info(f"Loading {self.config.CRYPTO_SYMBOL} data from {self.config.DATA_FILE}...")
        
        # Load the existing CSV data
        if not os.path.exists(self.config.DATA_FILE):
            raise FileNotFoundError(f"Data file not found: {self.config.DATA_FILE}")
        
        # Read CSV with proper column names
        self.df = pd.read_csv(self.config.DATA_FILE)
        
        # Rename columns to match expected format
        column_mapping = {
            'date': 'timestamp',
            'open_usd': 'open',
            'high_usd': 'high', 
            'low_usd': 'low',
            'close_usd': 'close',
            'volume': 'volume'
        }
        self.df = self.df.rename(columns=column_mapping)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.set_index('timestamp')
        self.df = self.df.sort_index()
        
        logger.info(f"Original data loaded: {len(self.df)} days")
        
        # FAST TRAINING OPTIMIZATION: Use only recent data
        if hasattr(self.config, 'USE_RECENT_DATA_ONLY') and getattr(self.config, 'USE_RECENT_DATA_ONLY', False):
            # Use only last N years of data
            recent_data_years = getattr(self.config, 'RECENT_DATA_YEARS', 2)
            recent_data_days = recent_data_years * 365
            self.df = self.df.tail(recent_data_days)
            logger.info(f"Using recent {recent_data_years} years: {len(self.df)} days")
        
        logger.info(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
        
        # Add technical indicators
        logger.info("Adding technical indicators...")
        ti = TechnicalIndicators()
        self.df = ti.add_technical_indicators(self.df)
        
        # Remove rows with NaN values (from technical indicators)
        self.df = self.df.dropna()
        
        logger.info(f"Data after adding indicators: {len(self.df)} days")
        
        # Split data for training/validation
        # Use first 80% for training, last 20% for validation
        split_idx = int(0.8 * len(self.df))
        self.train_df = self.df[:split_idx].copy()
        self.val_df = self.df[split_idx:].copy()
        
        logger.info(f"Training data: {len(self.train_df)} days ({self.train_df.index[0]} to {self.train_df.index[-1]})")
        logger.info(f"Validation data: {len(self.val_df)} days ({self.val_df.index[0]} to {self.val_df.index[-1]})")
        
        # Display data statistics
        logger.info(f"Price range - Min: ${self.df['close'].min():.2f}, Max: ${self.df['close'].max():.2f}")


def main():
    """Main fast training function"""
    print("🚀 Starting FAST Solana Trading DRL Training")
    print("⚡ Optimized for CPU - Expected time: 15-30 minutes")
    print("=" * 70)
    
    # Initialize fast training
    config = FastTrainingConfig()
    trainer = FastTrainingManager(config)
    
    print(f"Fast Training Configuration:")
    print(f"  Cryptocurrency: {config.CRYPTO_SYMBOL}")
    print(f"  Data file: {config.DATA_FILE}")
    print(f"  Recent data only: {config.USE_RECENT_DATA_ONLY} ({config.RECENT_DATA_YEARS} years)")
    print(f"  Device: {config.DEVICE}")
    print(f"  Total episodes: {config.TOTAL_EPISODES} (reduced from 1000)")
    print(f"  Episodes per update: {config.EPISODES_PER_UPDATE} (reduced from 10)")
    print(f"  Max episode length: {config.MAX_EPISODE_LENGTH} (reduced from 1000)")
    print(f"  Lookback window: {config.LOOKBACK_WINDOW} (reduced from 100)")
    print(f"  Initial balance: ${config.INITIAL_BALANCE:,}")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # Start training
        trainer.train()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print(f"\n✅ Fast training completed successfully!")
        print(f"⏱️ Training time: {training_duration}")
        print(f"Results saved in: {config.RESULTS_DIR}/")
        
        # Show final statistics
        if trainer.metrics['episode_rewards']:
            final_avg_reward = np.mean(trainer.metrics['episode_rewards'][-25:])  # Last 25 episodes
            final_avg_return = np.mean(trainer.metrics['total_returns'][-25:])
            print(f"Final average reward (last 25 episodes): ${final_avg_reward:.2f}")
            print(f"Final average return (last 25 episodes): {final_avg_return:.2f}%")
            
        print(f"\n🔄 To run full training with more episodes:")
        print(f"   python train_solana_agent.py")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        trainer.save_model('interrupted_fast_model')
        trainer.save_metrics()
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        logger.error(f"Training failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()