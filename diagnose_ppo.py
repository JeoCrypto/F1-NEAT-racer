#!/usr/bin/env python3
"""Diagnose PPO model behavior."""

import numpy as np
from stable_baselines3 import PPO
from racing_env import RacingEnv
import time

# Load model
print("Loading PPO model...")
model = PPO.load("ppo_simple_rewards.zip")

# Create environment
env = RacingEnv(render_mode=None, max_steps=500)

print("\nRunning diagnostic episode...")
print("=" * 60)

obs, info = env.reset()
done = False
step = 0
positions = []
actions_taken = []
rewards_received = []

while not done and step < 500:
    step += 1

    # Get action from model
    action, _states = model.predict(obs, deterministic=True)
    actions_taken.append(action.copy())

    # Record position
    positions.append((env.car.x, env.car.y))

    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    rewards_received.append(reward)
    done = terminated or truncated

    if step % 100 == 0:
        print(f"Step {step}: pos=({env.car.x:.1f}, {env.car.y:.1f}), "
              f"speed={env.car.speed:.2f}, reward={reward:.2f}")

print("\n" + "=" * 60)
print("DIAGNOSTIC RESULTS")
print("=" * 60)

# Analyze actions
actions_array = np.array(actions_taken)
print(f"\nSteps taken: {step}")
print(f"Final position: ({env.car.x:.1f}, {env.car.y:.1f})")
print(f"Starting position: (375, 410)")

# Movement analysis
if len(positions) > 1:
    start_pos = positions[0]
    end_pos = positions[-1]
    total_dist = sum(
        np.sqrt((positions[i+1][0] - positions[i][0])**2 +
                (positions[i+1][1] - positions[i][1])**2)
        for i in range(len(positions) - 1)
    )
    net_dist = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)

    print(f"\nMovement:")
    print(f"  Total distance: {total_dist:.1f}px")
    print(f"  Net distance: {net_dist:.1f}px")
    print(f"  Efficiency: {(net_dist/total_dist*100):.1f}%")

# Action analysis
print(f"\nActions taken:")
print(f"  Steering mean: {actions_array[:, 0].mean():.3f}")
print(f"  Steering std: {actions_array[:, 0].std():.3f}")
print(f"  Acceleration mean: {actions_array[:, 1].mean():.3f}")
print(f"  Acceleration std: {actions_array[:, 1].std():.3f}")

# Check if actions are constant
steering_variance = actions_array[:, 0].var()
accel_variance = actions_array[:, 1].var()

if steering_variance < 0.01 and accel_variance < 0.01:
    print("\n⚠️  PROBLEM: Actions are nearly constant!")
    print("   The model is not exploring - it's outputting the same action every step")
    print(f"   Typical action: steering={actions_array[0, 0]:.3f}, accel={actions_array[0, 1]:.3f}")

# Reward analysis
print(f"\nRewards:")
print(f"  Total reward: {sum(rewards_received):.2f}")
print(f"  Mean reward per step: {np.mean(rewards_received):.3f}")

# Checkpoint progress
print(f"\nCheckpoints passed: {info['checkpoint']}")
print(f"Distance to next checkpoint: {info['distance_to_next']:.1f}px")

env.close()
