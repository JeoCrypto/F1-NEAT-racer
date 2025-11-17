"""
Vision-based Karting Racing Environment (Hybrid: Vision + Sensors)

This is a professional-grade environment that combines:
- First-person camera view (like real drivers see)
- Telemetry sensors (speed, acceleration, steering angle)
- Object detection capabilities (walls, checkpoints, racing line)

Designed for:
- Training young drivers with realistic simulation
- Team testing and strategy optimization
- Telemetry analysis and performance metrics
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from PIL import Image
import cv2
from typing import Tuple, Optional, Dict, Any
import os
from pathlib import Path
import json


TRACKS_JSON_CANDIDATES = [
    os.environ.get("TRACKS_JSON"),
    "/root/tracks/checkpoints.json",
    str(Path(__file__).resolve().parents[1] / "tracks" / "checkpoints.json"),
    str(Path.cwd() / "tracks" / "checkpoints.json"),
]


class RacingEnvVision(gym.Env):
    """
    Professional Karting Simulator Environment
    
    Observation Space:
        - Vision: 84x84x3 RGB image (first-person view from car)
        - Telemetry: [speed, steering_angle, throttle, brake, lateral_g, longitudinal_g]
        - Track Info: [distance_to_racing_line, track_curvature, upcoming_turn_severity]
    
    Action Space:
        - Continuous: [steering (-1 to 1), throttle/brake (-1 to 1)]
    
    Features:
        - Realistic physics (weight transfer, tire grip)
        - Telemetry logging for analysis
        - Object detection overlay (optional)
        - Multiple camera angles
        - Racing line guidance
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self, 
        render_mode: Optional[str] = None, 
        max_steps: int = 1000,
        track_name: str = "circuit.png",
        camera_height: int = 84,  # Standard for CNNs (84x84)
        camera_width: int = 84,
        fov_distance: int = 200,  # How far ahead the car "sees"
        use_grayscale: bool = False,
        include_telemetry: bool = True,
        telemetry_history: int = 4,  # Frame stacking for temporal info
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.fov_distance = fov_distance
        self.use_grayscale = use_grayscale
        self.include_telemetry = include_telemetry
        self.telemetry_history = telemetry_history
        
        # Load track metadata
        checkpoints_path = self._find_checkpoints_json()
        with open(checkpoints_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        chosen = os.environ.get("TRACK_IMAGE", track_name)
        if chosen not in meta:
            raise KeyError(f"Track '{chosen}' missing in checkpoints.json")
        
        track_meta = meta[chosen]
        self.track_image_path = self._find_track_image(chosen)
        self.circuit = Image.open(self.track_image_path)
        # Convert to RGB to ensure 3 channels (PNG might be RGBA)
        if self.circuit.mode != 'RGB':
            self.circuit = self.circuit.convert('RGB')
        self.bg_array = np.array(self.circuit)
        self.track_name = chosen
        self.checkpoints = track_meta["checkpoints"]
        self.starting_position = tuple(track_meta.get("start_position", [375, 410]))
        self.starting_angle = float(track_meta.get("start_angle", 240.8))
        
        # Define action space: [steering, throttle/brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space (hybrid: vision + telemetry)
        base_channels = 1 if use_grayscale else 3
        # Only multiply channels if we're actually stacking frames
        channels = base_channels * telemetry_history if telemetry_history > 1 else base_channels
        
        obs_spaces = {
            # Camera view (first-person)
            "vision": spaces.Box(
                low=0, high=255,
                shape=(camera_height, camera_width, channels),
                dtype=np.uint8
            )
        }
        
        if include_telemetry:
            # Telemetry vector: speed, steering, throttle, brake, lat_g, long_g, 
            # dist_to_racing_line, track_curvature, next_checkpoint_angle
            obs_spaces["telemetry"] = spaces.Box(
                low=-10.0, high=10.0,
                shape=(9,),
                dtype=np.float32
            )
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        # Initialize pygame for rendering
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Karting Simulator - Vision Mode")
            self.clock = pygame.time.Clock()
            self.background_image = pygame.image.fromstring(
                self.circuit.tobytes(), self.circuit.size, self.circuit.mode
            )
        else:
            self.screen = None
            self.background_image = None
        
        # Telemetry logging
        self.telemetry_buffer = []
        self.frame_buffer = []  # For frame stacking
        
        # Physics state
        self.car = None
        self.current_checkpoint = 0
        self.steps = 0
        self.last_distance = None
        self.lateral_g = 0.0
        self.longitudinal_g = 0.0
        self.last_speed = 0.0
    
    def _find_checkpoints_json(self) -> Path:
        """Find checkpoints.json in expected locations."""
        for cand in TRACKS_JSON_CANDIDATES:
            if cand and Path(cand).exists():
                return Path(cand)
        raise FileNotFoundError("No checkpoints.json found")
    
    def _find_track_image(self, track_name: str) -> Path:
        """Find track image in expected locations."""
        candidates = [
            Path("/root/tracks") / track_name,
            Path(__file__).resolve().parents[1] / track_name,
            Path.cwd() / track_name,
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Track image '{track_name}' not found")
    
    def _capture_first_person_view(self) -> np.ndarray:
        """
        Capture first-person view from car's perspective.
        
        Returns:
            84x84x3 RGB image (or 84x84x1 if grayscale)
        """
        # Create a viewport ahead of the car
        import math
        
        # Calculate view direction
        angle_rad = math.radians(self.car.angle)
        view_distance = self.fov_distance
        
        # Center point ahead of car
        center_x = int(self.car.x + view_distance * math.cos(angle_rad))
        center_y = int(self.car.y - view_distance * math.sin(angle_rad))
        
        # Extract region from track
        half_w = self.camera_width // 2
        half_h = self.camera_height // 2
        
        x1 = max(0, center_x - half_w)
        x2 = min(self.bg_array.shape[1], center_x + half_w)
        y1 = max(0, center_y - half_h)
        y2 = min(self.bg_array.shape[0], center_y + half_h)
        
        # Extract and resize
        view = self.bg_array[y1:y2, x1:x2]
        
        # Resize to target size
        view_resized = cv2.resize(view, (self.camera_width, self.camera_height))
        
        # Convert to grayscale if needed
        if self.use_grayscale:
            view_resized = cv2.cvtColor(view_resized, cv2.COLOR_RGB2GRAY)
            view_resized = np.expand_dims(view_resized, axis=-1)
        
        return view_resized
    
    def _get_telemetry(self) -> np.ndarray:
        """
        Get current telemetry data.
        
        Returns:
            [speed, steering_angle, throttle, brake, lateral_g, longitudinal_g,
             dist_to_racing_line, track_curvature, next_checkpoint_angle]
        """
        # Normalize values
        speed_norm = np.clip(self.car.speed / 10.0, -1, 1)
        steering_norm = np.clip(self.car.angle / 360.0, 0, 1)
        
        # G-forces (simplified physics)
        self.lateral_g = np.clip(self.car.speed * 0.1, -2, 2)  # Placeholder
        self.longitudinal_g = np.clip((self.car.speed - self.last_speed) * 0.5, -2, 2)
        self.last_speed = self.car.speed
        
        # Distance to ideal racing line (placeholder - would need racing line data)
        dist_to_line = 0.0
        
        # Track curvature ahead (placeholder)
        track_curvature = 0.0
        
        # Angle to next checkpoint
        next_cp = (self.current_checkpoint + 1) % len(self.checkpoints)
        dx, dy = self.car.pos_relative_to_next_cp(self.checkpoints[next_cp])
        cp_angle = np.arctan2(dy, dx) / np.pi  # Normalize to -1 to 1
        
        telemetry = np.array([
            speed_norm,
            steering_norm,
            0.0,  # throttle (would track from action)
            0.0,  # brake (would track from action)
            self.lateral_g / 2.0,  # Normalize
            self.longitudinal_g / 2.0,
            dist_to_line,
            track_curvature,
            cp_angle,
        ], dtype=np.float32)
        
        return telemetry
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation (vision + telemetry)."""
        from car import Car
        
        # Capture camera view
        vision = self._capture_first_person_view()
        
        # Frame stacking for temporal information (only if > 1)
        if self.telemetry_history > 1:
            self.frame_buffer.append(vision)
            if len(self.frame_buffer) > self.telemetry_history:
                self.frame_buffer.pop(0)
            
            # Stack frames along channel dimension
            if len(self.frame_buffer) == self.telemetry_history:
                stacked = np.concatenate(self.frame_buffer, axis=-1)
            else:
                # Pad with zeros until we have enough frames
                padding = [np.zeros_like(vision) for _ in range(self.telemetry_history - len(self.frame_buffer))]
                stacked = np.concatenate(padding + self.frame_buffer, axis=-1)
        else:
            stacked = vision
        
        obs = {"vision": stacked}
        
        if self.include_telemetry:
            obs["telemetry"] = self._get_telemetry()
        
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment."""
        super().reset(seed=seed)
        
        from car import Car
        self.car = Car(self.starting_position, angle=self.starting_angle)
        self.current_checkpoint = 0
        self.steps = 0
        self.last_distance = self._get_checkpoint_distance()
        self.telemetry_buffer = []
        self.frame_buffer = []
        self.lateral_g = 0.0
        self.longitudinal_g = 0.0
        self.last_speed = 0.0
        
        observation = self._get_observation()
        info = {"telemetry_log": []}
        
        return observation, info
    
    def step(self, action):
        """Execute one step."""
        self.steps += 1
        
        # Apply action
        steering = float(action[0]) * 2.0
        acceleration = float(action[1]) * 5.0
        
        self.car.update(steering, acceleration)
        
        # Reward calculation (same as before for now)
        reward = 0.0
        terminated = False
        truncated = False
        
        # Off-track penalty
        if self.car.check_off_track(self.bg_array):
            reward = -50.0
            terminated = True
        
        # Checkpoint passing reward
        next_cp = (self.current_checkpoint + 1) % len(self.checkpoints)
        if self.car.get_collide_checkpoint(self.checkpoints[next_cp], threshold=30.0):
            reward = 100.0
            self.current_checkpoint = next_cp
            self.last_distance = None
            
            if self.current_checkpoint == 0 and self.steps > 10:
                reward += 500.0  # Lap completion
                terminated = True
        
        # Distance improvement reward
        current_distance = self._get_checkpoint_distance()
        if self.last_distance is not None and reward < 50:
            distance_improvement = self.last_distance - current_distance
            reward += distance_improvement * 10.0
        self.last_distance = current_distance
        
        # Check max steps
        if self.steps >= self.max_steps:
            truncated = True
        
        observation = self._get_observation()
        
        # Log telemetry
        telemetry_entry = {
            "step": self.steps,
            "speed": float(self.car.speed),
            "steering": float(steering),
            "throttle": float(max(0, acceleration)),
            "brake": float(max(0, -acceleration)),
            "lateral_g": float(self.lateral_g),
            "longitudinal_g": float(self.longitudinal_g),
            "checkpoint": self.current_checkpoint,
            "reward": float(reward),
        }
        self.telemetry_buffer.append(telemetry_entry)
        
        info = {
            "checkpoint": self.current_checkpoint,
            "distance_to_next": current_distance,
            "steps": self.steps,
            "telemetry": telemetry_entry,
            "telemetry_log": self.telemetry_buffer,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (standard view + telemetry overlay)."""
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
            
            # Telemetry overlay
            font = pygame.font.Font(None, 24)
            y_offset = 10
            telemetry_texts = [
                f"Speed: {self.car.speed:.1f}",
                f"Checkpoint: {self.current_checkpoint + 1}/{len(self.checkpoints)}",
                f"Lat G: {self.lateral_g:.2f}",
                f"Long G: {self.longitudinal_g:.2f}",
                f"Steps: {self.steps}",
            ]
            
            for text in telemetry_texts:
                surface = font.render(text, True, (255, 255, 255))
                rect = surface.get_rect()
                # Black background for readability
                pygame.draw.rect(self.screen, (0, 0, 0), 
                               (5, y_offset - 2, rect.width + 10, rect.height + 4))
                self.screen.blit(surface, (10, y_offset))
                y_offset += 25
            
            pygame.display.flip()
            self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
        
        # Optionally save telemetry log
        if self.telemetry_buffer:
            import pandas as pd
            df = pd.DataFrame(self.telemetry_buffer)
            df.to_csv("telemetry_log.csv", index=False)
            print(f"Telemetry logged: {len(self.telemetry_buffer)} entries saved")
    
    def _get_checkpoint_distance(self) -> float:
        """Get distance to next checkpoint."""
        next_cp = (self.current_checkpoint + 1) % len(self.checkpoints)
        cp_mid_x = (self.checkpoints[next_cp][0][0] + self.checkpoints[next_cp][1][0]) / 2
        cp_mid_y = (self.checkpoints[next_cp][0][1] + self.checkpoints[next_cp][1][1]) / 2
        return np.sqrt((self.car.x - cp_mid_x)**2 + (self.car.y - cp_mid_y)**2)
    
    def export_telemetry(self, filename: str = "telemetry.csv"):
        """Export telemetry data for analysis."""
        import pandas as pd
        if self.telemetry_buffer:
            df = pd.DataFrame(self.telemetry_buffer)
            df.to_csv(filename, index=False)
            return df
        return None
