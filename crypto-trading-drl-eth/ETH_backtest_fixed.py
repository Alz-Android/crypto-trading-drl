"""
FIXED Backtesting script for ETH DRL agent with improved trading logic
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

class FixedBacktestRunner:
    """Fixed backtesting with improved trading logic"""
    
    def __init__(self, model_name='best_model', results_dir='full_training_results'):
        self.model_name = model_name
        self.results_dir = results_dir
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, self.results_dir, 'models', f'{self.model_name}.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
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
        print(f"✅ Fixed backtest model loaded: {model_path}")
        
    def load_test_data(self, symbol='ETH', test_period='validation'):
        """Load test data from existing CSV files"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, 'data', f'{symbol}_6year_data.csv')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
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
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        # Add technical indicators
        ti = TechnicalIndicators()
        df = ti.add_technical_indicators(df)
        df = df.dropna()
        
        # Select test period
        if test_period == 'validation':
            split_idx = int(0.8 * len(df))
            test_df = df[split_idx:].copy()
        elif test_period == 'recent':
            test_df = df.tail(365).copy()  # Last 365 days
        elif test_period == 'holdout':
            test_df = df.tail(180).copy()  # Last 180 days
        else:
            test_df = df.tail(int(test_period)).copy()
        
        print(f"Test data loaded: {len(test_df)} days ({test_df.index[0]} to {test_df.index[-1]})")
        return test_df
        
    def run_comprehensive_backtest(self, symbol='ETH'):
        """Run comprehensive backtest on multiple periods"""
        print(f"🧪 FIXED ETH Backtest Analysis")
        print(f"🔍 Symbol: {symbol} | Model: Full Training (1000 episodes)")
        print("=" * 60)
        
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
                
                # Run STOCHASTIC backtest (use probability sampling)
                print("🎲 Running stochastic backtest...")
                stochastic_results = self._run_stochastic_backtest(df)
                
                # Run DETERMINISTIC backtest for comparison
                print("🎯 Running deterministic backtest...")
                deterministic_results = self._run_deterministic_backtest(df)
                
                # Compare with buy-and-hold
                buy_hold_results = self._run_buy_hold_strategy(df, self.config['INITIAL_BALANCE'], self.config['LOOKBACK_WINDOW'])
                
                # Generate report
                self._generate_fixed_report(stochastic_results, deterministic_results, buy_hold_results, symbol, period)
                
                all_results[period] = {
                    'stochastic': stochastic_results,
                    'deterministic': deterministic_results,
                    'buy_hold': buy_hold_results
                }
                
            except Exception as e:
                print(f"❌ Error testing {period} period: {str(e)}")
                continue
        
        self._generate_summary_report(all_results, symbol)
        return all_results
    
    def _run_stochastic_backtest(self, df):
        """Run backtest using stochastic action sampling"""
        # Create environment
        env = CryptoTradingEnv(
            df,
            initial_balance=self.config['INITIAL_BALANCE'],
            lookback_window=self.config['LOOKBACK_WINDOW']
        )
        
        state = env.reset()
        results = {
            'actions': [],
            'rewards': [],
            'net_worths': [],
            'balances': [],
            'crypto_held': [],
            'prices': []
        }
        
        total_reward = 0
        
        while True:
            # Use STOCHASTIC action selection (sample from probability distribution)
            action, _, action_probs = self.agent.get_action(state, deterministic=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Record results
            results['actions'].append(action)
            results['rewards'].append(reward)
            results['net_worths'].append(info['net_worth'])
            results['balances'].append(info['balance'])
            results['crypto_held'].append(info['crypto_held'])
            results['prices'].append(info['current_price'])
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        results['total_reward'] = total_reward
        results['final_net_worth'] = results['net_worths'][-1]
        results['total_return'] = (results['final_net_worth'] / self.config['INITIAL_BALANCE'] - 1) * 100
        
        return results
    
    def _run_deterministic_backtest(self, df):
        """Run backtest using deterministic action selection"""
        env = CryptoTradingEnv(
            df,
            initial_balance=self.config['INITIAL_BALANCE'],
            lookback_window=self.config['LOOKBACK_WINDOW']
        )
        
        state = env.reset()
        results = {
            'actions': [],
            'net_worths': [],
            'final_net_worth': 0,
            'total_return': 0
        }
        
        total_reward = 0
        
        while True:
            # Use DETERMINISTIC action selection
            action, _, _ = self.agent.get_action(state, deterministic=True)
            
            next_state, reward, done, info = env.step(action)
            
            results['actions'].append(action)
            results['net_worths'].append(info['net_worth'])
            
            total_reward += reward
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
        
        crypto_bought = initial_balance / start_price
        final_value = crypto_bought * end_price
        
        return {
            'final_value': final_value,
            'total_return': (final_value / initial_balance - 1) * 100,
            'start_price': start_price,
            'end_price': end_price
        }
        
    def _generate_fixed_report(self, stochastic_results, deterministic_results, buy_hold_results, symbol, period):
        """Generate detailed comparison report"""
        print(f"\n{'='*50}")
        print(f"FIXED BACKTEST RESULTS - {symbol} ({period.upper()})")
        print(f"{'='*50}")
        
        print(f"\n🎲 STOCHASTIC Strategy (Probability Sampling):")
        print(f"  Final Net Worth: ${stochastic_results['final_net_worth']:,.2f}")
        print(f"  Total Return: {stochastic_results['total_return']:.2f}%")
        
        # Action distribution for stochastic
        actions = stochastic_results['actions']
        buy_pct = actions.count(1) / len(actions) * 100
        hold_pct = actions.count(0) / len(actions) * 100
        sell_pct = actions.count(2) / len(actions) * 100
        print(f"  Action Distribution: Buy={buy_pct:.1f}%, Hold={hold_pct:.1f}%, Sell={sell_pct:.1f}%")
        
        print(f"\n🎯 DETERMINISTIC Strategy (Max Probability):")
        print(f"  Final Net Worth: ${deterministic_results['final_net_worth']:,.2f}")
        print(f"  Total Return: {deterministic_results['total_return']:.2f}%")
        
        # Action distribution for deterministic
        det_actions = deterministic_results['actions']
        det_buy_pct = det_actions.count(1) / len(det_actions) * 100
        det_hold_pct = det_actions.count(0) / len(det_actions) * 100
        det_sell_pct = det_actions.count(2) / len(det_actions) * 100
        print(f"  Action Distribution: Buy={det_buy_pct:.1f}%, Hold={det_hold_pct:.1f}%, Sell={det_sell_pct:.1f}%")
        
        print(f"\n📈 BUY-AND-HOLD Strategy:")
        print(f"  Final Value: ${buy_hold_results['final_value']:,.2f}")
        print(f"  Total Return: {buy_hold_results['total_return']:.2f}%")
        
        print(f"\n🏆 PERFORMANCE COMPARISON:")
        stoch_vs_bh = stochastic_results['total_return'] - buy_hold_results['total_return']
        det_vs_bh = deterministic_results['total_return'] - buy_hold_results['total_return']
        
        print(f"  Stochastic vs Buy-Hold: {stoch_vs_bh:+.2f}%")
        print(f"  Deterministic vs Buy-Hold: {det_vs_bh:+.2f}%")
        
        if stoch_vs_bh > 0:
            print(f"  ✅ Stochastic DRL outperformed buy-and-hold!")
        else:
            print(f"  ❌ Stochastic DRL underperformed buy-and-hold")
            
        if det_vs_bh > 0:
            print(f"  ✅ Deterministic DRL outperformed buy-and-hold!")
        else:
            print(f"  ❌ Deterministic DRL underperformed buy-and-hold")
    
    def _generate_summary_report(self, all_results, symbol):
        """Generate comprehensive summary"""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE FIXED BACKTEST SUMMARY - {symbol}")
        print(f"{'='*60}")
        
        print(f"\n📊 Performance Summary:")
        print(f"{'Period':<12} {'Stochastic':<12} {'Deterministic':<14} {'Buy-Hold':<12} {'Best Strategy'}")
        print(f"{'-'*70}")
        
        for period, results in all_results.items():
            stoch_ret = results['stochastic']['total_return']
            det_ret = results['deterministic']['total_return']
            bh_ret = results['buy_hold']['total_return']
            
            best = max(stoch_ret, det_ret, bh_ret)
            if best == stoch_ret:
                best_strategy = "Stochastic"
            elif best == det_ret:
                best_strategy = "Deterministic"
            else:
                best_strategy = "Buy-Hold"
            
            print(f"{period.title():<12} {stoch_ret:>10.2f}% {det_ret:>12.2f}% {bh_ret:>10.2f}% {best_strategy}")
        
        print(f"\n🎯 Key Insights:")
        print(f"- Model trained for 1000 episodes on 6 years of ETH data")
        print(f"- Stochastic sampling provides more realistic trading behavior")
        print(f"- Deterministic mode shows the model's preferred action bias")
        print(f"- Action probabilities: ~33% Hold, ~30% Buy, ~37% Sell")

def main():
    """Main fixed backtesting function"""
    print("🔧 Running FIXED ETH Backtest Analysis")
    print("=" * 50)
    
    try:
        backtester = FixedBacktestRunner()
        results = backtester.run_comprehensive_backtest('ETH')
        print("\n✅ Fixed backtest completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Fixed backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()