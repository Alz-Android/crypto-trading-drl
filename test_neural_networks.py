#!/usr/bin/env python3
"""
Test script for neural networks
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from neural_networks import CNNLSTMActor, CNNLSTMCritic, PPOAgent

def test_neural_networks():
    """Test the neural networks functionality"""
    print("Testing CNN-LSTM Neural Networks...")

    # Create a simple test state (100 timesteps, 4 features)
    test_state = np.random.randn(100, 4).astype(np.float32)
    print(f"Test state shape: {test_state.shape}")

    # Initialize networks
    actor = CNNLSTMActor()
    critic = CNNLSTMCritic()

    # Test forward pass
    import torch
    state_tensor = torch.FloatTensor(test_state).unsqueeze(0)

    actor_output = actor(state_tensor)
    critic_output = critic(state_tensor)

    print(f"Actor output shape: {actor_output.shape}")
    print(f"Actor output (probabilities): {actor_output.detach().numpy()}")
    print(f"Critic output shape: {critic_output.shape}")
    print(f"Critic output (value): {critic_output.item():.4f}")

    # Test PPO agent
    agent = PPOAgent()
    action, log_prob, probs = agent.get_action(test_state)

    print(f"\nPPO Agent test:")
    print(f"Selected action: {action}")
    print(f"Action probabilities: {probs}")
    print(f"Log probability: {log_prob}")

    # Test deterministic action
    det_action, det_log_prob, det_probs = agent.get_action(test_state, deterministic=True)
    print(f"\nDeterministic action: {det_action}")
    print(f"Deterministic probabilities: {det_probs}")

    print("\nNeural networks test completed successfully! ✅")

if __name__ == "__main__":
    test_neural_networks()
