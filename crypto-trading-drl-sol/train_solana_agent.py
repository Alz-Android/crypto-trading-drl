"""
Training script for the cryptocurrency trading DRL agent using existing 6-year data.
Implements PPO training loop with Solana data from crypto_data folder.
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

# Add main src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from technical_indicators import TechnicalIndicators
from trading_env import CryptoTradingEnv
from neural_networks import PPOAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Training configuration parameters"""
    # Data parameters
    CRYPTO_SYMBOL = 'SOL'  # Solana
    DATA_FILE = 'data/SOL_6year_data.csv'
    LOOKBACK_WINDOW = 100
    
    # Training parameters
    TOTAL_EPISODES = 1000
    EPISODES_PER_UPDATE = 10
    MAX_EPISODE_LENGTH = 1000
    
    # Environment parameters
    INITIAL_BALANCE = 10000
    
    # Neural network parameters
    LEARNING_RATE = 3e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # PPO parameters
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # GAE parameter
    
    # Model saving
    MODEL_SAVE_FREQ = 50  # Save every N episodes
    RESULTS_DIR = 'full_training_results'

class TrainingManager:
    """Manages the training process for the DRL agent"""
    
    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.setup_directories()
        self.setup_data()
        self.setup_agent()
        self.setup_metrics()
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.config.RESULTS_DIR, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.config.RESULTS_DIR, 'plots'), exist_ok=True)
        
    def setup_data(self):
        """Load and prepare training data from existing CSV file"""
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
        
        # Sort by timestamp
        self.df = self.df.sort_index()
        
        logger.info(f"Original data loaded: {len(self.df)} days")
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
        
    def setup_agent(self):
        """Initialize the PPO agent"""
        input_shape = (self.config.LOOKBACK_WINDOW, 4)  # (timesteps, features)
        action_dim = 3  # Buy, Hold, Sell
        
        self.agent = PPOAgent(
            input_shape=input_shape,
            action_dim=action_dim,
            lr=self.config.LEARNING_RATE,
            device=self.config.DEVICE
        )
        
        logger.info(f"Agent initialized on device: {self.config.DEVICE}")
        
    def setup_metrics(self):
        """Initialize training metrics tracking"""
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'net_worths': [],
            'total_returns': [],
            'validation_rewards': [],
            'best_validation_reward': -np.inf
        }
        
    def calculate_advantages(self, rewards, values, dones):
        """Calculate advantages using GAE"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                next_value = values[t + 1] if t + 1 < len(values) else 0
                delta = rewards[t] + self.config.GAMMA * next_value - values[t]
                last_gae = delta + self.config.GAMMA * self.config.GAE_LAMBDA * last_gae
            
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
            
        return advantages, returns
        
    def collect_trajectories(self, env, num_episodes):
        """Collect trajectories for training"""
        trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            initial_net_worth = env.net_worth
            
            episode_states = []
            episode_actions = []
            episode_rewards_list = []
            episode_log_probs = []
            episode_values = []
            episode_dones = []
            
            for step in range(self.config.MAX_EPISODE_LENGTH):
                # Get action from agent
                action, log_prob, action_probs = self.agent.get_action(state)
                
                # Get value estimate
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                value = self.agent.critic(state_tensor).item()
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                episode_states.append(state.copy())
                episode_actions.append(action)
                episode_rewards_list.append(reward)
                episode_log_probs.append(log_prob)
                episode_values.append(value)
                episode_dones.append(done)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # Calculate total return percentage
            final_net_worth = env.net_worth
            total_return = (final_net_worth / initial_net_worth - 1) * 100
            
            # Add episode data to trajectories
            trajectories['states'].extend(episode_states)
            trajectories['actions'].extend(episode_actions)
            trajectories['rewards'].extend(episode_rewards_list)
            trajectories['log_probs'].extend(episode_log_probs)
            trajectories['values'].extend(episode_values)
            trajectories['dones'].extend(episode_dones)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_returns.append(total_return)
            
            logger.debug(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Return = {total_return:.2f}%, Length = {episode_length}")
        
        # Calculate advantages and returns
        advantages, returns = self.calculate_advantages(
            np.array(trajectories['rewards']),
            np.array(trajectories['values']),
            np.array(trajectories['dones'])
        )
        
        trajectories['advantages'] = advantages.tolist()
        trajectories['returns'] = returns.tolist()
        
        return trajectories, episode_rewards, episode_lengths, episode_returns
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Create training environment
        train_env = CryptoTradingEnv(
            self.train_df,
            initial_balance=self.config.INITIAL_BALANCE,
            lookback_window=self.config.LOOKBACK_WINDOW
        )
        
        # Create validation environment
        val_env = CryptoTradingEnv(
            self.val_df,
            initial_balance=self.config.INITIAL_BALANCE,
            lookback_window=self.config.LOOKBACK_WINDOW
        )
        
        total_episodes = 0
        
        # Training loop
        while total_episodes < self.config.TOTAL_EPISODES:
            # Collect trajectories
            trajectories, episode_rewards, episode_lengths, episode_returns = self.collect_trajectories(
                train_env, self.config.EPISODES_PER_UPDATE
            )
            
            # Update agent
            self.agent.update(trajectories)
            
            # Update metrics
            self.metrics['episode_rewards'].extend(episode_rewards)
            self.metrics['episode_lengths'].extend(episode_lengths)
            self.metrics['total_returns'].extend(episode_returns)
            
            total_episodes += len(episode_rewards)
            
            # Logging
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_return = np.mean(episode_returns)
            
            logger.info(f"Episodes {total_episodes-len(episode_rewards)+1}-{total_episodes}:")
            logger.info(f"  Average Reward: ${avg_reward:.2f}")
            logger.info(f"  Average Return: {avg_return:.2f}%")
            logger.info(f"  Average Length: {avg_length:.1f}")
            
            # Validation every 5 updates
            if total_episodes % (self.config.EPISODES_PER_UPDATE * 5) == 0:
                val_reward, val_return = self.evaluate(val_env)
                self.metrics['validation_rewards'].append(val_reward)
                
                logger.info(f"  Validation Reward: ${val_reward:.2f}")
                logger.info(f"  Validation Return: {val_return:.2f}%")
                
                # Save best model
                if val_reward > self.metrics['best_validation_reward']:
                    self.metrics['best_validation_reward'] = val_reward
                    self.save_model('best_model')
                    logger.info("  🏆 New best model saved!")
            
            # Periodic model saving
            if total_episodes % self.config.MODEL_SAVE_FREQ == 0:
                self.save_model(f'model_episode_{total_episodes}')
                self.plot_training_progress()
        
        logger.info("Training completed!")
        self.save_model('final_model')
        self.plot_training_progress()
        self.save_metrics()
        
    def evaluate(self, env, num_episodes=5):
        """Evaluate the agent"""
        total_rewards = []
        total_returns = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            initial_net_worth = env.net_worth
            
            for step in range(self.config.MAX_EPISODE_LENGTH):
                action, _, _ = self.agent.get_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            final_net_worth = env.net_worth
            total_return = (final_net_worth / initial_net_worth - 1) * 100
            
            total_rewards.append(episode_reward)
            total_returns.append(total_return)
        
        return np.mean(total_rewards), np.mean(total_returns)
    
    def save_model(self, name):
        """Save the trained model"""
        model_path = os.path.join(self.config.RESULTS_DIR, 'models', f'{name}.pt')
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'config': self.config.__dict__,
            'metrics': self.metrics
        }, model_path)
        logger.info(f"Model saved: {model_path}")
    
    def load_model(self, name):
        """Load a trained model"""
        model_path = os.path.join(self.config.RESULTS_DIR, 'models', f'{name}.pt')
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']
        
        logger.info(f"Model loaded: {model_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if self.metrics['episode_rewards']:
            axes[0, 0].plot(self.metrics['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward ($)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode returns
        if self.metrics['total_returns']:
            axes[0, 1].plot(self.metrics['total_returns'])
            axes[0, 1].set_title('Episode Returns')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Moving average rewards
        if len(self.metrics['episode_rewards']) >= 50:
            moving_avg_rewards = pd.Series(self.metrics['episode_rewards']).rolling(window=50).mean()
            
            axes[1, 0].plot(moving_avg_rewards, label='Rewards')
            axes[1, 0].set_title('Moving Average (50 episodes)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Moving Avg Reward ($)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Validation rewards
            if self.metrics['validation_rewards']:
                val_x = np.arange(len(self.metrics['validation_rewards'])) * self.config.EPISODES_PER_UPDATE * 5
                axes[1, 1].plot(val_x, self.metrics['validation_rewards'], 'ro-')
                axes[1, 1].set_title('Validation Performance')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Validation Reward ($)')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.RESULTS_DIR, 'plots', 'training_progress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training progress plot saved: {plot_path}")
    
    def save_metrics(self):
        """Save training metrics"""
        metrics_path = os.path.join(self.config.RESULTS_DIR, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")


def main():
    """Main training function"""
    print("🚀 Starting Solana Trading DRL Training")
    print("=" * 60)
    
    # Initialize training
    config = TrainingConfig()
    trainer = TrainingManager(config)
    
    print(f"Training Configuration:")
    print(f"  Cryptocurrency: {config.CRYPTO_SYMBOL}")
    print(f"  Data file: {config.DATA_FILE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Total episodes: {config.TOTAL_EPISODES}")
    print(f"  Episodes per update: {config.EPISODES_PER_UPDATE}")
    print(f"  Max episode length: {config.MAX_EPISODE_LENGTH}")
    print(f"  Initial balance: ${config.INITIAL_BALANCE:,}")
    print("=" * 60)
    
    try:
        # Start training
        trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"Results saved in: {config.RESULTS_DIR}/")
        
        # Show final statistics
        if trainer.metrics['episode_rewards']:
            final_avg_reward = np.mean(trainer.metrics['episode_rewards'][-50:])  # Last 50 episodes
            final_avg_return = np.mean(trainer.metrics['total_returns'][-50:])
            print(f"Final average reward (last 50 episodes): ${final_avg_reward:.2f}")
            print(f"Final average return (last 50 episodes): {final_avg_return:.2f}%")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        trainer.save_model('interrupted_model')
        trainer.save_metrics()
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        logger.error(f"Training failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()