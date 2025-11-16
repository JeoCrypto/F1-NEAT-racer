"""
Gymnasium Environment for Karting Racing with PPO.

This environment wraps the racing game logic to work with stable-baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from PIL import Image
from car import Car
from typing import Tuple, Optional
import os
from pathlib import Path


TRACKS_JSON_CANDIDATES = [
    os.environ.get("TRACKS_JSON"),
    "/root/tracks/checkpoints.json",  # remote volume path
    str(Path(__file__).resolve().parents[1] / "tracks" / "checkpoints.json"),  # repo local
    str(Path.cwd() / "tracks" / "checkpoints.json"),
]


class RacingEnv(gym.Env):
    """Custom Gymnasium Environment for Karting Racing.

    Parameters:
        render_mode: Optional render mode ("human" or None).
        max_steps: Episode step cap.
        track_name: Filename of track PNG (default 'circuit.png').
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 1000, track_name: str = "circuit.png"):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        # Resolve checkpoints metadata JSON
        import json
        checkpoints_path = None
        for cand in TRACKS_JSON_CANDIDATES:
            if cand and Path(cand).exists():
                checkpoints_path = Path(cand)
                break
        if checkpoints_path is None:
            raise FileNotFoundError("No checkpoints.json found in expected locations. Set TRACKS_JSON or create tracks/checkpoints.json.")
        with open(checkpoints_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        env_override = os.environ.get("TRACK_IMAGE")
        chosen = env_override if env_override else track_name
        if chosen not in meta:
            raise KeyError(f"Track '{chosen}' missing in checkpoints.json. Add entry before use.")
        track_meta = meta[chosen]
        # Resolve track image path (prefer remote volume /root/tracks)
        candidate_paths = [
            Path("/root/tracks") / chosen,
            Path(__file__).resolve().parents[1] / chosen,
            Path.cwd() / chosen,
        ]
        chosen_img = None
        for p in candidate_paths:
            if p.exists():
                chosen_img = p
                break
        if chosen_img is None:
            raise FileNotFoundError(f"Track image file '{chosen}' not found in volume or repo.")
        self.circuit = Image.open(chosen_img)
        self.bg_array = np.array(self.circuit)
        self.track_name = chosen
        self.checkpoints = track_meta["checkpoints"]
        self.starting_position = tuple(track_meta.get("start_position", [375, 410]))
        self.starting_angle = float(track_meta.get("start_angle", 240.8))


        # Starting position already set from metadata above

        # Define action space: [steering, acceleration]
        # Steering: -1 to 1, Acceleration: -1 to 1
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Define observation space: [6 vision rays, speed, angle, dx to checkpoint, dy to checkpoint]
        # Total: 10 values
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(10,),
            dtype=np.float32
        )

        # Initialize pygame for rendering
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Karting Racing - PPO Training")
            self.clock = pygame.time.Clock()
            self.background_image = pygame.image.fromstring(
                self.circuit.tobytes(), self.circuit.size, self.circuit.mode
            )
        else:
            self.screen = None
            self.background_image = None

        # State variables
        self.car = None
        self.current_checkpoint = 0
        self.steps = 0
        self.last_distance = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.car = Car(self.starting_position, angle=self.starting_angle)
        self.current_checkpoint = 0
        self.steps = 0
        self.last_distance = self._get_checkpoint_distance()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        self.steps += 1

        # Convert normalized actions to game actions
        steering = float(action[0]) * 2.0  # -2 to 2
        acceleration = float(action[1]) * 5.0  # -5 to 5

        # Update car
        self.car.update(steering, acceleration)

        # SIMPLIFIED REWARD: Only checkpoint progress!
        reward = 0.0
        terminated = False
        truncated = False

        # Off-track penalty (strong enough to prevent shortcuts)
        if self.car.check_off_track(self.bg_array):
            reward = -50.0  # Must stay on track!
            terminated = True

        # HUGE reward for passing checkpoint!
        next_cp = (self.current_checkpoint + 1) % len(self.checkpoints)
        if self.car.get_collide_checkpoint(self.checkpoints[next_cp], threshold=30.0):
            reward = 100.0  # Big reward!
            self.current_checkpoint = next_cp
            self.last_distance = None

            # Lap completion bonus
            if self.current_checkpoint == 0 and self.steps > 10:
                reward += 500.0
                terminated = True

        # ONLY reward: Moving closer to checkpoint (no other rewards!)
        current_distance = self._get_checkpoint_distance()
        if self.last_distance is not None and reward < 50:  # Only if didn't just pass checkpoint
            distance_improvement = self.last_distance - current_distance
            reward += distance_improvement * 10.0  # Very strong distance signal!

        self.last_distance = current_distance

        # Speed control: Penalize going too fast when walls are close
        # This teaches the car to brake for corners
        vision = self.car.vision(self.bg_array)
        min_wall_distance = min(vision)  # Closest wall
        if min_wall_distance < 40 and abs(self.car.speed) > 3.5:
            # Going too fast near walls = dangerous!
            reward -= (abs(self.car.speed) - 3.5) * 3.0

        # Check max steps
        if self.steps >= self.max_steps:
            truncated = True

        observation = self._get_observation()
        info = {
            "checkpoint": self.current_checkpoint,
            "distance_to_next": current_distance,
            "steps": self.steps
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.screen is not None:
            self.screen.blit(self.background_image, (0, 0))

            # Draw checkpoints
            next_cp = (self.current_checkpoint + 1) % len(self.checkpoints)
            for idx, cp in enumerate(self.checkpoints):
                color = (0, 255, 255) if idx == next_cp else (0, 255, 0)
                pygame.draw.line(self.screen, color, cp[0], cp[1],
                               3 if idx == next_cp else 2)

            # Draw car
            self.car.draw(self.screen)
            self.car.vision(self.bg_array, screen=self.screen)

            pygame.display.flip()
            self.clock.tick(60)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()

    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        # Get vision data (6 values, distances to walls)
        vision = self.car.vision(self.bg_array)
        vision_normalized = np.array(vision, dtype=np.float32) / 200.0  # Normalize to 0-1

        # Get speed (normalize to 0-1, assuming max speed ~10)
        speed_normalized = np.clip(abs(self.car.speed) / 10.0, 0, 1)

        # Get angle (normalize to 0-1)
        angle_normalized = (self.car.angle % 360) / 360.0

        # Get position relative to next checkpoint
        next_cp = (self.current_checkpoint + 1) % len(self.checkpoints)
        dx, dy = self.car.pos_relative_to_next_cp(self.checkpoints[next_cp])
        # Normalize to -1 to 1 range (assuming max distance ~400 pixels)
        dx_normalized = np.clip(dx / 400.0, -1, 1)
        dy_normalized = np.clip(dy / 400.0, -1, 1)

        # Combine all observations
        observation = np.concatenate([
            vision_normalized,
            [speed_normalized],
            [angle_normalized],
            [dx_normalized],
            [dy_normalized]
        ])

        return observation.astype(np.float32)

    def _get_checkpoint_distance(self) -> float:
        """Get distance to next checkpoint."""
        next_cp = (self.current_checkpoint + 1) % len(self.checkpoints)
        cp_mid_x = (self.checkpoints[next_cp][0][0] + self.checkpoints[next_cp][1][0]) / 2
        cp_mid_y = (self.checkpoints[next_cp][0][1] + self.checkpoints[next_cp][1][1]) / 2
        return np.sqrt((self.car.x - cp_mid_x)**2 + (self.car.y - cp_mid_y)**2)
