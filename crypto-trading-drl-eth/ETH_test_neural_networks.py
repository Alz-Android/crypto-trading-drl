#!/usr/bin/env python3
"""
Test script for the ETH neural networks
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.neural_networks import CNNLSTMActor, CNNLSTMCritic, PPOAgent

def test_eth_neural_networks():
    """Test the neural networks for ETH trading"""
    print("Testing CNN-LSTM Neural Networks for ETH...")

    # Create a test state (100 timesteps, 4 features)
    test_state = np.random.randn(100, 4).astype(np.float32)
    print(f"Test state shape: {test_state.shape}")

    # Test Actor Network
    print("\n🧠 Testing Actor Network...")
    actor = CNNLSTMActor(input_shape=(100, 4), action_dim=3)
    
    # Forward pass
    state_tensor = torch.FloatTensor(test_state).unsqueeze(0)  # Add batch dimension
    actor_output = actor(state_tensor)
    
    print(f"Actor input shape: {state_tensor.shape}")
    print(f"Actor output shape: {actor_output.shape}")
    print(f"Actor output (probabilities): {actor_output.detach().numpy()}")
    print(f"Probabilities sum to: {actor_output.sum().item():.6f}")
    
    # Test action selection
    action, log_prob, probs = actor.get_action(test_state)
    print(f"Selected action: {action} ({['Hold', 'Buy', 'Sell'][int(action)]})")
    print(f"Action probabilities: {probs}")
    print(f"Log probability: {log_prob}")

    # Test Critic Network
    print("\n🎯 Testing Critic Network...")
    critic = CNNLSTMCritic(input_shape=(100, 4))
    
    critic_output = critic(state_tensor)
    print(f"Critic input shape: {state_tensor.shape}")
    print(f"Critic output shape: {critic_output.shape}")
    print(f"Critic output (value): {critic_output.item():.4f}")

    # Test PPO Agent
    print("\n🤖 Testing PPO Agent...")
    agent = PPOAgent(input_shape=(100, 4), action_dim=3, lr=3e-4, device='cpu')
    
    # Test action selection
    action, log_prob, probs = agent.get_action(test_state)
    print(f"PPO Agent selected action: {action} ({['Hold', 'Buy', 'Sell'][int(action)]})")
    print(f"Action probabilities: {probs}")
    print(f"Log probability: {log_prob}")
    
    # Test with deterministic action
    det_action, det_log_prob, det_probs = agent.get_action(test_state, deterministic=True)
    print(f"Deterministic action: {det_action} ({['Hold', 'Buy', 'Sell'][int(det_action)]})")

    print("\n✅ Neural networks test completed successfully!")

def test_network_training():
    """Test network training functionality"""
    print("\n🏋️ Testing Network Training Functionality...")
    
    # Create PPO agent
    agent = PPOAgent(input_shape=(100, 4), action_dim=3, lr=3e-4, device='cpu')
    
    # Create dummy trajectory data
    batch_size = 32
    trajectory_length = 100
    
    dummy_trajectories = {
        'states': np.random.randn(batch_size * trajectory_length, 100, 4).astype(np.float32).tolist(),
        'actions': np.random.randint(0, 3, batch_size * trajectory_length).tolist(),
        'rewards': np.random.randn(batch_size * trajectory_length).tolist(),
        'log_probs': np.random.randn(batch_size * trajectory_length).tolist(),
        'advantages': np.random.randn(batch_size * trajectory_length).tolist(),
        'returns': np.random.randn(batch_size * trajectory_length).tolist()
    }
    
    print(f"Dummy trajectory length: {len(dummy_trajectories['states'])}")
    
    try:
        # Test update
        agent.update(dummy_trajectories)
        print("✅ PPO update completed successfully!")
    except Exception as e:
        print(f"❌ PPO update failed: {e}")

def test_gpu_compatibility():
    """Test GPU compatibility if available"""
    print("\n🚀 Testing GPU Compatibility...")
    
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        
        try:
            # Test networks on GPU
            device = 'cuda'
            agent = PPOAgent(input_shape=(100, 4), action_dim=3, lr=3e-4, device=device)
            
            # Test with GPU tensor
            test_state = np.random.randn(100, 4).astype(np.float32)
            action, log_prob, probs = agent.get_action(test_state)
            
            print(f"✅ GPU training ready!")
            print(f"GPU action: {action} ({['Hold', 'Buy', 'Sell'][int(action)]})")
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
    else:
        print("⚠️ No GPU available - will use CPU")

def test_model_save_load():
    """Test model saving and loading"""
    print("\n💾 Testing Model Save/Load...")
    
    # Create and train a simple agent
    agent = PPOAgent(input_shape=(100, 4), action_dim=3, lr=3e-4, device='cpu')
    
    # Get initial action
    test_state = np.random.randn(100, 4).astype(np.float32)
    initial_action, _, _ = agent.get_action(test_state)
    
    # Save model state
    model_state = {
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'config': {'test': True}
    }
    
    # Create new agent and load state
    new_agent = PPOAgent(input_shape=(100, 4), action_dim=3, lr=3e-4, device='cpu')
    new_agent.actor.load_state_dict(model_state['actor_state_dict'])
    new_agent.critic.load_state_dict(model_state['critic_state_dict'])
    
    # Test if actions are the same
    loaded_action, _, _ = new_agent.get_action(test_state, deterministic=True)
    
    print(f"Initial action: {initial_action}")
    print(f"Loaded action: {loaded_action}")
    print("✅ Model save/load test completed!")

if __name__ == "__main__":
    test_eth_neural_networks()
    test_network_training()
    test_gpu_compatibility()
    test_model_save_load()