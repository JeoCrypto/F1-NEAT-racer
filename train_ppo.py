#!/usr/bin/env python3
"""
Train Karting Racing AI using PPO (Proximal Policy Optimization).

This uses stable-baselines3, which is much better at handling exploration/exploitation
than NEAT for continuous control tasks.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from racing_env import RacingEnv


def make_env():
    """Create and wrap the environment."""
    env = RacingEnv(render_mode=None, max_steps=1000)
    env = Monitor(env)  # Wrap with Monitor for logging
    return env


def train(total_timesteps: int, save_path: str = "ppo_racing", visualize: bool = False):
    """Train the PPO agent."""

    print("=" * 60)
    print("PPO Training for Karting Racing")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Save path: {save_path}")
    print()

    # Create environment
    if visualize:
        # Single environment with rendering
        env = RacingEnv(render_mode="human", max_steps=1000)
        env = Monitor(env)
    else:
        # Vectorized environment for faster training
        env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel environments

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # Number of steps to collect before update
        batch_size=64,
        n_epochs=10,  # Number of epochs for optimization
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter
        clip_range=0.2,  # PPO clipping parameter
        ent_coef=0.01,  # Entropy coefficient (encourages exploration)
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        tensorboard_log=None,  # Disabled tensorboard (not installed)
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path="./checkpoints/",
        name_prefix=save_path
    )

    print("\nStarting training...")
    print("TIP: Monitor training progress with:")
    print("  tensorboard --logdir ./ppo_racing_tensorboard/")
    print()

    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False  # Disabled (requires tqdm)
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save final model
    print(f"\nSaving final model to {save_path}.zip")
    model.save(save_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nTo test the trained model, run:")
    print(f"  python3 play_ppo.py --model {save_path}.zip")
    print()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Karting Racing AI with PPO")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train (default: 100,000)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="ppo_racing",
        help="Model save path (default: ppo_racing)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization during training (slower)"
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_path=args.save,
        visualize=args.visualize
    )
