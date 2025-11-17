"""
Vision-Based PPO Training for Karting Simulator

Uses CNN policy with hybrid observations (vision + telemetry).
This script demonstrates proper GPU utilization with convolutional networks.

Training Features:
- Transfer learning support (pre-trained CNN backbones)
- Frame stacking for temporal awareness
- Telemetry integration for physics understanding
- TensorBoard logging with video recordings
- Checkpoint management
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from typing import Dict
import argparse
from pathlib import Path
import os
from racing_env_vision import RacingEnvVision


class HybridCNNExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for hybrid observations (vision + telemetry).
    
    Architecture:
    - Vision branch: CNN (inspired by Nature DQN)
    - Telemetry branch: MLP
    - Fusion: Concatenate and process with shared layers
    
    This can use transfer learning by loading pre-trained CNN weights.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        # Total features from both branches
        super().__init__(observation_space, features_dim)
        
        # Vision branch (CNN)
        vision_shape = observation_space["vision"].shape
        n_input_channels = vision_shape[-1]  # Channels (3 for RGB or stacked frames)
        
        # Modified Nature DQN CNN for 84x84 input
        # Adjusted kernel sizes to work with smaller images
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),  # 84 -> 42
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 42 -> 21
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 21 -> 11
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, vision_shape[0], vision_shape[1])
            cnn_output_dim = self.cnn(sample_input).shape[1]
        
        # Telemetry branch (MLP) - if telemetry is included
        if "telemetry" in observation_space.spaces:
            telemetry_dim = observation_space["telemetry"].shape[0]
            self.telemetry_mlp = nn.Sequential(
                nn.Linear(telemetry_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            fusion_input_dim = cnn_output_dim + 64
        else:
            self.telemetry_mlp = None
            fusion_input_dim = cnn_output_dim
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process vision
        # Note: CNN expects (batch, channels, height, width)
        # Gym returns (batch, height, width, channels), so we permute
        vision = observations["vision"].float() / 255.0  # Normalize to [0, 1]
        vision = vision.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        cnn_features = self.cnn(vision)
        
        # Process telemetry if available
        if self.telemetry_mlp is not None and "telemetry" in observations:
            telemetry_features = self.telemetry_mlp(observations["telemetry"])
            combined = torch.cat([cnn_features, telemetry_features], dim=1)
        else:
            combined = cnn_features
        
        # Fusion
        return self.fusion(combined)


def make_env(track_name: str = "circuit.png", render_mode: str = None, max_steps: int = 1000):
    """Create a single environment instance."""
    def _init():
        env = RacingEnvVision(
            render_mode=render_mode,
            max_steps=max_steps,
            track_name=track_name,
            camera_height=84,
            camera_width=84,
            use_grayscale=False,  # Use RGB for richer features
            include_telemetry=True,  # Hybrid model
            telemetry_history=1,  # Can increase for frame stacking
        )
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with Vision (CNN)")
    parser.add_argument("--track-name", type=str, default="circuit.png", help="Track image name")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--vector-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint frequency")
    parser.add_argument("--save-name", type=str, default="ppo_vision", help="Model save name")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU (for testing)")
    parser.add_argument("--features-dim", type=int, default=512, help="Feature extractor output dim")
    
    args = parser.parse_args()
    
    # Setup device
    device = "cpu" if args.use_cpu else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create vectorized environments
    if args.vector_envs > 1:
        env = SubprocVecEnv([make_env(args.track_name) for _ in range(args.vector_envs)])
    else:
        env = DummyVecEnv([make_env(args.track_name)])
    
    # Policy kwargs for custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=HybridCNNExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=[256, 256],  # Shared MLP after feature extraction
    )
    
    # Create or load model
    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming from {args.resume_from}")
        model = PPO.load(
            args.resume_from,
            env=env,
            device=device,
            custom_objects={"policy_kwargs": policy_kwargs}
        )
    else:
        print("Creating new model with CnnPolicy")
        model = PPO(
            "MultiInputPolicy",  # For Dict observation space
            env,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard_vision/",
            device=device,
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path="./checkpoints_vision/",
        name_prefix=args.save_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    callbacks = [checkpoint_callback]
    
    # Train
    print(f"Starting training for {args.timesteps:,} timesteps on {device.upper()}...")
    print(f"Vector envs: {args.vector_envs}, Batch size: {args.batch_size}")
    print(f"This WILL use GPU efficiently with CNN policy!" if device == "cuda" else "")
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_path = f"{args.save_name}_final.zip"
    model.save(final_path)
    print(f"\nTraining complete! Model saved to {final_path}")
    
    env.close()


if __name__ == "__main__":
    main()
