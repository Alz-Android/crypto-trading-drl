"""
Backtesting script for evaluating the full training DRL agent (1000 episodes, 6-year data).
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.technical_indicators import TechnicalIndicators
from src.trading_env import CryptoTradingEnv
from src.neural_networks import PPOAgent

class FullTrainingBacktestRunner:
    """Runs backtesting for the full training DRL agent using complete 6-year dataset"""
    
    def __init__(self, model_name='best_model', results_dir='full_training_results'):
        self.model_name = model_name
        self.results_dir = results_dir
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, self.results_dir, 'models', f'{self.model_name}.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        # If config is empty (common issue), provide default full training config
        if not config:
            print("⚠️ Config is empty, using default full training configuration")
            config = {
                'LOOKBACK_WINDOW': 100,
                'INITIAL_BALANCE': 10000,
                'LEARNING_RATE': 3e-4,
                'CRYPTO_SYMBOL': 'ETH',
                'TOTAL_EPISODES': 1000,
                'MAX_EPISODE_LENGTH': 1000
            }
        
        # Initialize agent
        self.agent = PPOAgent(
            input_shape=(config['LOOKBACK_WINDOW'], 4),
            action_dim=3,
            lr=config['LEARNING_RATE'],
            device='cpu'
        )
        
        # Load weights
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        self.config = config
        print(f"✅ Full training model loaded: {model_path}")
        
    def load_test_data(self, symbol='ETH', test_period='validation'):
        """Load test data from existing CSV files"""
        # Get the directory where this script is located and navigate to data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, 'data', f'{symbol}_6year_data.csv')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Read CSV with proper column names
        df = pd.read_csv(data_file)
        
        # Rename columns to match expected format
        column_mapping = {
            'date': 'timestamp',
            'open_usd': 'open',
            'high_usd': 'high', 
            'low_usd': 'low',
            'close_usd': 'close',
            'volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        # Add technical indicators
        ti = TechnicalIndicators()
        df = ti.add_technical_indicators(df)
        df = df.dropna()
        
        # Select test period based on full 6-year dataset
        if test_period == 'validation':
            # Use last 20% of data (same as training validation split)
            split_idx = int(0.8 * len(df))
            test_df = df[split_idx:].copy()
        elif test_period == 'recent':
            # Use last 1 year for full model testing
            test_df = df.last('365D').copy()
        elif test_period == 'full':
            # Use full dataset
            test_df = df.copy()
        elif test_period == 'holdout':
            # Use most recent 6 months not seen during training
            test_df = df.last('180D').copy()
        else:
            # Use last N days
            try:
                days = int(test_period)
                test_df = df.tail(days).copy()
            except:
                raise ValueError(f"Invalid test_period: {test_period}")
        
        print(f"Full training test data loaded: {len(test_df)} days ({test_df.index[0]} to {test_df.index[-1]})")
        return test_df
        
    def run_comprehensive_backtest(self, symbol='ETH'):
        """Run comprehensive backtest on multiple periods"""
        print(f"🧪 Running Comprehensive Backtest for Full Training Model")
        print(f"🔍 Symbol: {symbol} | Model: Full Training (1000 episodes, 6-year data)")
        print("=" * 80)
        
        test_periods = ['validation', 'recent', 'holdout']
        all_results = {}
        
        for period in test_periods:
            print(f"\n📊 Testing on {period} period...")
            try:
                # Load test data
                df = self.load_test_data(symbol, period)
                
                if len(df) < self.config['LOOKBACK_WINDOW'] + 50:
                    print(f"⚠️ Skipping {period}: insufficient data ({len(df)} days)")
                    continue
                
                # Create environment
                env = CryptoTradingEnv(
                    df,
                    initial_balance=self.config['INITIAL_BALANCE'],
                    lookback_window=self.config['LOOKBACK_WINDOW']
                )
                
                # Run backtest
                results = self._run_single_backtest(env, symbol, df)
                
                # Compare with buy-and-hold
                buy_hold_results = self._run_buy_hold_strategy(df, self.config['INITIAL_BALANCE'], self.config['LOOKBACK_WINDOW'])
                
                # Generate report
                self._generate_period_report(results, buy_hold_results, symbol, period, df)
                
                # Store results
                all_results[period] = {
                    'drl_results': results,
                    'buy_hold_results': buy_hold_results,
                    'df': df
                }
                
            except Exception as e:
                print(f"❌ Error testing {period} period: {str(e)}")
                continue
        
        # Generate comprehensive summary report
        self._generate_comprehensive_report(all_results, symbol)
        
        return all_results
    
    def _run_single_backtest(self, env, symbol, df):
        """Run a single backtest episode"""
        state = env.reset()
        results = {
            'actions': [],
            'rewards': [],
            'net_worths': [],
            'balances': [],
            'crypto_held': [],
            'prices': [],
            'timestamps': []
        }
        
        total_reward = 0
        step_count = 0
        
        # Get timestamps for plotting
        start_idx = self.config['LOOKBACK_WINDOW']
        timestamps = df.index[start_idx:].tolist()
        
        while True:
            # Get action from trained agent
            action, _, _ = self.agent.get_action(state, deterministic=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Record results
            if step_count < len(timestamps):
                results['timestamps'].append(timestamps[step_count])
            results['actions'].append(action)
            results['rewards'].append(reward)
            results['net_worths'].append(info['net_worth'])
            results['balances'].append(info['balance'])
            results['crypto_held'].append(info['crypto_held'])
            results['prices'].append(info['current_price'])
            
            total_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        results['total_reward'] = total_reward
        results['final_net_worth'] = results['net_worths'][-1]
        results['total_return'] = (results['final_net_worth'] / self.config['INITIAL_BALANCE'] - 1) * 100
        
        return results
        
    def _run_buy_hold_strategy(self, df, initial_balance, lookback_window):
        """Run buy-and-hold strategy for comparison"""
        start_price = df['close'].iloc[lookback_window]
        end_price = df['close'].iloc[-1]
        
        # Buy at start, hold until end
        crypto_bought = initial_balance / start_price
        final_value = crypto_bought * end_price
        
        return {
            'initial_balance': initial_balance,
            'final_value': final_value,
            'total_return': (final_value / initial_balance - 1) * 100,
            'start_price': start_price,
            'end_price': end_price
        }
        
    def _generate_period_report(self, drl_results, buy_hold_results, symbol, test_period, df):
        """Generate report for a specific test period"""
        print(f"\n{'='*60}")
        print(f"FULL TRAINING BACKTEST - {symbol} ({test_period.upper()} PERIOD)")
        print(f"{'='*60}")
        
        # DRL Strategy Results
        print(f"\n🤖 DRL Strategy (Full Training):")
        print(f"  Initial Balance: ${self.config['INITIAL_BALANCE']:,.2f}")
        print(f"  Final Net Worth: ${drl_results['final_net_worth']:,.2f}")
        print(f"  Total Return: {drl_results['total_return']:.2f}%")
        print(f"  Total Reward: ${drl_results['total_reward']:,.2f}")
        
        # Buy-and-Hold Results
        print(f"\n📈 Buy-and-Hold Strategy:")
        print(f"  Initial Balance: ${buy_hold_results['initial_balance']:,.2f}")
        print(f"  Final Value: ${buy_hold_results['final_value']:,.2f}")
        print(f"  Total Return: {buy_hold_results['total_return']:.2f}%")
        
        # Comparison
        outperformance = drl_results['total_return'] - buy_hold_results['total_return']
        print(f"\n🏆 Performance Comparison:")
        print(f"  DRL vs Buy-Hold: {outperformance:+.2f}%")
        
        if outperformance > 0:
            print(f"  ✅ DRL strategy outperformed buy-and-hold!")
        else:
            print(f"  ❌ DRL strategy underperformed buy-and-hold")
        
        # Trading Statistics
        actions = np.array(drl_results['actions'])
        buy_actions = np.sum(actions == 1)
        hold_actions = np.sum(actions == 0)
        sell_actions = np.sum(actions == 2)
        total_actions = len(actions)
        
        print(f"\n📊 Trading Statistics:")
        print(f"  Total Trading Days: {total_actions}")
        print(f"  Buy Actions: {buy_actions} ({buy_actions/total_actions*100:.1f}%)")
        print(f"  Hold Actions: {hold_actions} ({hold_actions/total_actions*100:.1f}%)")
        print(f"  Sell Actions: {sell_actions} ({sell_actions/total_actions*100:.1f}%)")
        
        # Price statistics
        start_price = drl_results['prices'][0]
        end_price = drl_results['prices'][-1]
        max_price = max(drl_results['prices'])
        min_price = min(drl_results['prices'])
        
        print(f"\n💰 Price Statistics:")
        print(f"  Start Price: ${start_price:.2f}")
        print(f"  End Price: ${end_price:.2f}")
        print(f"  Max Price: ${max_price:.2f}")
        print(f"  Min Price: ${min_price:.2f}")
        print(f"  Price Change: {(end_price/start_price - 1)*100:+.2f}%")
        
        # Generate plots for this period
        self._plot_period_results(drl_results, buy_hold_results, symbol, test_period)
        
    def _plot_period_results(self, drl_results, buy_hold_results, symbol, test_period):
        """Plot backtest results for a specific period"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Net worth over time
        axes[0].plot(drl_results['net_worths'], label='DRL Strategy (Full Training)', linewidth=2, color='blue')
        
        # Buy-and-hold line
        initial_balance = self.config['INITIAL_BALANCE']
        start_price = buy_hold_results['start_price']
        crypto_held = initial_balance / start_price
        buy_hold_values = [price * crypto_held for price in drl_results['prices']]
        axes[0].plot(buy_hold_values, label='Buy-and-Hold', linewidth=2, alpha=0.7, color='green')
        
        axes[0].set_title(f'{symbol} Full Training Performance ({test_period.title()} Period)')
        axes[0].set_ylabel('Net Worth ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Price and actions
        axes[1].plot(drl_results['prices'], label='Price', color='black', alpha=0.7)
        
        # Mark buy/sell actions
        for i, action in enumerate(drl_results['actions']):
            if action == 1:  # Buy
                axes[1].scatter(i, drl_results['prices'][i], color='green', s=30, alpha=0.8, marker='^')
            elif action == 2:  # Sell
                axes[1].scatter(i, drl_results['prices'][i], color='red', s=30, alpha=0.8, marker='v')
        
        axes[1].set_title(f'{symbol} Price and Trading Actions (Full Training)')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend(['Price', 'Buy', 'Sell'])
        axes[1].grid(True, alpha=0.3)
        
        # Cumulative returns
        drl_returns = [(nw / self.config['INITIAL_BALANCE'] - 1) * 100 for nw in drl_results['net_worths']]
        buy_hold_returns = [(val / initial_balance - 1) * 100 for val in buy_hold_values]
        
        axes[2].plot(drl_returns, label='DRL Strategy (Full Training)', linewidth=2, color='blue')
        axes[2].plot(buy_hold_returns, label='Buy-and-Hold', linewidth=2, alpha=0.7, color='green')
        axes[2].set_title('Cumulative Returns Comparison (Full Training)')
        axes[2].set_xlabel('Trading Days')
        axes[2].set_ylabel('Return (%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(script_dir, self.results_dir, 'backtest_plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{symbol}_{test_period}_full_training_backtest.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot instead of showing it
        
        print(f"📊 Plot saved: {plot_path}")
        
    def _generate_comprehensive_report(self, all_results, symbol):
        """Generate comprehensive summary report"""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE FULL TRAINING REPORT - {symbol}")
        print(f"{'='*80}")
        
        print(f"\n🧠 Model Configuration:")
        print(f"  Training Episodes: {self.config['TOTAL_EPISODES']}")
        print(f"  Episode Length: {self.config['MAX_EPISODE_LENGTH']} steps")
        print(f"  Lookback Window: {self.config['LOOKBACK_WINDOW']} days")
        print(f"  Learning Rate: {self.config['LEARNING_RATE']}")
        print(f"  Dataset: Complete 6-year historical data")
        
        print(f"\n📊 Performance Summary Across All Test Periods:")
        print(f"{'Period':<15} {'DRL Return':<12} {'Buy-Hold':<12} {'Outperformance':<15} {'Status'}")
        print(f"{'-'*65}")
        
        for period, results in all_results.items():
            drl_return = results['drl_results']['total_return']
            bh_return = results['buy_hold_results']['total_return']
            outperf = drl_return - bh_return
            status = "✅ WIN" if outperf > 0 else "❌ LOSS"
            
            print(f"{period.title():<15} {drl_return:>10.2f}% {bh_return:>10.2f}% {outperf:>+13.2f}% {status}")
        
        # Calculate average performance
        if all_results:
            avg_drl = np.mean([r['drl_results']['total_return'] for r in all_results.values()])
            avg_bh = np.mean([r['buy_hold_results']['total_return'] for r in all_results.values()])
            avg_outperf = avg_drl - avg_bh
            
            print(f"{'-'*65}")
            print(f"{'AVERAGE':<15} {avg_drl:>10.2f}% {avg_bh:>10.2f}% {avg_outperf:>+13.2f}% {'✅ WIN' if avg_outperf > 0 else '❌ LOSS'}")
        
    def _compare_with_fast_training(self):
        """Compare full training results with fast training results"""
        fast_results_path = os.path.join(os.path.dirname(__file__), 'fast_training_results', 'training_metrics.json')
        
        if os.path.exists(fast_results_path):
            try:
                with open(fast_results_path, 'r') as f:
                    fast_metrics = json.load(f)
                
                print(f"\n🔄 Comparison with Fast Training (200 episodes):")
                print(f"  Fast Training Final Rewards: {fast_metrics['episode_rewards'][-10:]}")
                print(f"  Fast Training showed high variance with potential for improvement")
                print(f"  Full Training (1000 episodes) should show more stable and improved performance")
                
            except Exception as e:
                print(f"\n⚠️ Could not load fast training results for comparison: {e}")


def main():
    """Main backtesting function for full training model"""
    print("🧪 Running Full Training Backtest Analysis")
    print("=" * 60)
    
    try:
        # Initialize backtester
        backtester = FullTrainingBacktestRunner()
        
        # Run comprehensive backtest
        print("Running comprehensive backtest on multiple periods...")
        all_results = backtester.run_comprehensive_backtest('ETH')
        
        print("\n✅ Full training backtest completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Full training backtest failed: {str(e)}")


if __name__ == "__main__":
    main()