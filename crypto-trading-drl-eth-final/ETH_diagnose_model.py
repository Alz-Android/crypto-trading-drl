import torch
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.technical_indicators import TechnicalIndicators
from src.trading_env import CryptoTradingEnv
from src.neural_networks import PPOAgent

def diagnose_model_behavior():
    """Diagnose why the ETH model is only taking sell actions"""
    print("🔍 DIAGNOSING ETH MODEL BEHAVIOR")
    print("=" * 50)
    
    # Load the trained model
    model_path = 'full_training_results/models/best_model.pt'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    config = {
        'LOOKBACK_WINDOW': 100,
        'INITIAL_BALANCE': 10000,
        'LEARNING_RATE': 3e-4,
        'CRYPTO_SYMBOL': 'ETH',
        'TOTAL_EPISODES': 1000,
        'MAX_EPISODE_LENGTH': 1000
    }
    
    agent = PPOAgent(
        input_shape=(config['LOOKBACK_WINDOW'], 4),
        action_dim=3,
        lr=config['LEARNING_RATE'],
        device='cpu'
    )
    
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    print("✅ Model loaded successfully")
    
    # Load and prepare test data
    df = pd.read_csv('data/ETH_6year_data.csv')
    
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
    
    print(f"✅ Data loaded: {len(df)} days")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    
    # Take a recent sample for analysis
    test_df = df.tail(200).copy()
    print(f"Testing on last 200 days: {test_df.index[0]} to {test_df.index[-1]}")
    
    # Create environment
    env = CryptoTradingEnv(
        test_df,
        initial_balance=config['INITIAL_BALANCE'],
        lookback_window=config['LOOKBACK_WINDOW']
    )
    
    # Test the model's action probabilities
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"State sample (first 5 timesteps, all features):")
    print(state[:5])
    
    # Check if state has any NaN or infinite values
    if np.any(np.isnan(state)):
        print("⚠️ WARNING: State contains NaN values!")
        nan_count = np.sum(np.isnan(state))
        print(f"NaN count: {nan_count}")
    
    if np.any(np.isinf(state)):
        print("⚠️ WARNING: State contains infinite values!")
        inf_count = np.sum(np.isinf(state))
        print(f"Infinite count: {inf_count}")
    
    # Test action selection multiple times
    print("\n🎯 Testing Action Selection:")
    actions_tested = []
    action_probs_tested = []
    
    for i in range(10):
        action, log_prob, action_probs = agent.get_action(state, deterministic=False)
        actions_tested.append(action)
        action_probs_tested.append(action_probs)
        
        print(f"Test {i+1}: Action={action} ({['Hold', 'Buy', 'Sell'][action]}), Probs={[f'{p:.3f}' for p in action_probs]}")
    
    print(f"\nAction distribution: {np.bincount(actions_tested, minlength=3)}")
    print(f"Action percentages: Hold={actions_tested.count(0)/10*100:.1f}%, Buy={actions_tested.count(1)/10*100:.1f}%, Sell={actions_tested.count(2)/10*100:.1f}%")
    
    # Test deterministic actions
    print("\n🎯 Testing Deterministic Actions:")
    det_actions = []
    for i in range(5):
        action, _, action_probs = agent.get_action(state, deterministic=True)
        det_actions.append(action)
        print(f"Deterministic Test {i+1}: Action={action} ({['Hold', 'Buy', 'Sell'][action]}), Probs={[f'{p:.3f}' for p in action_probs]}")
    
    # Analyze the network outputs directly
    print("\n🧠 Analyzing Network Outputs:")
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    with torch.no_grad():
        # Get raw actor output
        raw_output = agent.actor(state_tensor)
        print(f"Raw actor output: {raw_output.numpy()}")
        
        # Get critic value
        critic_value = agent.critic(state_tensor)
        print(f"Critic value: {critic_value.item():.3f}")
    
    # Check data statistics
    print("\n📊 Data Statistics:")
    feature_cols = ['close', 'rsi', 'atr', 'obv']
    for col in feature_cols:
        if col in test_df.columns:
            data_col = test_df[col]
            print(f"{col}: min={data_col.min():.3f}, max={data_col.max():.3f}, mean={data_col.mean():.3f}, std={data_col.std():.3f}")
    
    # Test with different states
    print("\n🔄 Testing Different States:")
    for step in range(5):
        if step > 0:
            action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            print(f"Step {step}: Action={action}, Price=${info['current_price']:.2f}, Balance=${info['balance']:.2f}, ETH={info['crypto_held']:.6f}")
        else:
            print(f"Step {step}: Initial state, Price=${env.df.iloc[env.current_step]['close']:.2f}")

if __name__ == "__main__":
    diagnose_model_behavior()