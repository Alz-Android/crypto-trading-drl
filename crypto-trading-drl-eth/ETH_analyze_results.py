import json

# Load training metrics
with open('full_training_results/training_metrics.json', 'r') as f:
    data = json.load(f)

print("🎉 ETH FULL TRAINING COMPLETED! 🎉")
print("=" * 50)
print(f"Total episodes completed: {len(data['episode_rewards'])}")
print(f"Final 10 episode rewards: {[round(r, 2) for r in data['episode_rewards'][-10:]]}")
print(f"Average final 50 rewards: ${sum(data['episode_rewards'][-50:])/50:,.2f}")
print(f"Average final 100 rewards: ${sum(data['episode_rewards'][-100:])/100:,.2f}")
print(f"Best individual episode reward: ${max(data['episode_rewards']):,.2f}")
print(f"Worst individual episode reward: ${min(data['episode_rewards']):,.2f}")
print(f"Best validation reward: ${data['best_validation_reward']:,.2f}")

# Training progress metrics
episode_returns = data.get('total_returns', [])
if episode_returns:
    print(f"Final 10 episode returns (%): {[round(r, 2) for r in episode_returns[-10:]]}")
    print(f"Average final 50 returns: {sum(episode_returns[-50:])/50:.2f}%")

print("\nTraining Status: ✅ COMPLETE")
print("Models available:")
print("- best_model.pt (recommended for backtesting)")
print("- final_model.pt (from last episode)")
print("- Periodic checkpoints every 50 episodes")