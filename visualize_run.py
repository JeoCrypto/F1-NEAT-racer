#!/usr/bin/env python3
"""Visualize a single run with detailed output."""

import numpy as np
from stable_baselines3 import PPO
from racing_env import RacingEnv
import time

# Load model
print("Loading PPO model...")
model = PPO.load("ppo_ultra_extended.zip")

# Create environment with rendering
env = RacingEnv(render_mode="human", max_steps=1000)

print("\nRunning visualization...")
print("=" * 60)

obs, info = env.reset()
done = False
step = 0

while not done and step < 200:
    step += 1

    # Get action from model
    action, _states = model.predict(obs, deterministic=True)

    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Render
    env.render()
    time.sleep(0.03)  # Slow down for visibility

    # Print info every 10 steps
    if step % 10 == 0:
        print(f"Step {step}: pos=({env.car.x:.1f}, {env.car.y:.1f}), "
              f"CP={info['checkpoint']}, speed={env.car.speed:.2f}, "
              f"angle={env.car.angle:.1f}, reward={reward:.2f}")

print("\n" + "=" * 60)
print(f"Run ended at step {step}")
print(f"Final checkpoint: {info['checkpoint']}/44")
print(f"Final position: ({env.car.x:.1f}, {env.car.y:.1f})")
print(f"Reason: {'Off-track' if terminated else 'Max steps' if truncated else 'Completed'}")

input("\nPress Enter to close...")
env.close()
