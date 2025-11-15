#!/usr/bin/env python3
"""
Play/test a trained PPO model for Karting Racing.
"""

import argparse
import time
from stable_baselines3 import PPO
from racing_env import RacingEnv


def play(model_path: str, num_episodes: int = 3):
    """Play the racing game with a trained PPO model."""

    print("=" * 60)
    print("PPO Model Testing - Karting Racing")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print()

    # Load the trained model
    print("Loading model...")
    model = PPO.load(model_path)

    # Create environment with rendering
    env = RacingEnv(render_mode="human", max_steps=2000)

    print("Starting playback...\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_checkpoint = 0
        start_time = time.time()

        print(f"Episode {episode + 1}/{num_episodes}")

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            max_checkpoint = max(max_checkpoint, info["checkpoint"])

            # Render
            env.render()

        elapsed = time.time() - start_time

        print(f"  Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Max checkpoint: {max_checkpoint}/{len(env.checkpoints) - 1}")
        print(f"  Time: {elapsed:.2f}s")

        # Check if completed a lap
        if max_checkpoint == 0 and steps > 100:
            print("  🏁 COMPLETED A LAP!")

        print()

    env.close()
    print("Playback complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained PPO model")
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_racing.zip",
        help="Path to trained model (default: ppo_racing.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to play (default: 3)"
    )

    args = parser.parse_args()

    play(model_path=args.model, num_episodes=args.episodes)
