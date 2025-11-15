import math
from typing import Optional, Sequence
import numpy as np
import pygame


class Car:
    def __init__(
        self,
        pos: Sequence[float],
        width: int = 20,
        height: int = 10,
        color: tuple[int, int, int] = (255, 0, 0),
        angle: float = 90.0,
        max_speed: float = 5.0,
        acceleration: float = 0.2,
        friction: float = 0.05,
    ) -> None:
        self.starting_pos = tuple(pos)
        self.x, self.y = self.starting_pos
        self.width = width
        self.height = height
        self.color = color
        self.angle = angle
        self.speed = 0.0
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.friction = friction

        self.base_image: pygame.Surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.base_image.fill(self.color)
        self.image: pygame.Surface = self.base_image
        self.rect: pygame.Rect = self.image.get_rect(center=(self.x, self.y))

    def reset(
        self,
        last_checkpoint_midpoint: Sequence[float],
        heading_target: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Reset the car onto the provided checkpoint midpoint and optionally orient it
        toward the next checkpoint (heading_target).  If no heading target is given,
        the previous heading is preserved.
        """
        self.x, self.y = last_checkpoint_midpoint
        dx: float
        dy: float
        if heading_target is not None:
            dx = heading_target[0] - self.x
            dy = heading_target[1] - self.y
        else:
            # Preserve current orientation if no heading information is available.
            dx = math.cos(math.radians(90.0 - self.angle))
            dy = math.sin(math.radians(90.0 - self.angle))

        if dx == 0 and dy == 0:
            # Default to "pointing up" if we still lack a direction vector.
            dx, dy = 0.0, -1.0

        self.angle = (90.0 - math.degrees(math.atan2(dy, dx)))
        self.speed = 0.0
        self.image = pygame.transform.rotate(self.base_image, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self.image, self.rect)
        pygame.draw.circle(
            self.base_image,
            (0, 255, 0),
            (self.width, self.height // 2),
            5,
        )
        pygame.draw.rect(screen, (0, 0, 255), self.rect, 2)

    def check_off_track(
        self, bg_array: np.ndarray, target_color: tuple[int, int, int] = (255, 255, 255)
    ) -> bool:
        """Check if any part of the car's bounding box is on the off-track color (white)."""
        height, width = bg_array.shape[:2]
        for x in range(max(0, self.rect.left), min(width, self.rect.right)):
            for y in range(max(0, self.rect.top), min(height, self.rect.bottom)):
                if tuple(bg_array[y, x, :3]) == target_color:
                    return True
        return False

    def vision(
        self,
        bg_array: np.ndarray,
        target_color: tuple[int, int, int] = (255, 255, 255),
        max_distance: int = 200,
        screen: Optional[pygame.Surface] = None,
    ) -> list[int]:
        # Vision in 6 directions relative to current angle
        vision_angles = [-90, -45, 0, 45, 90, 180]
        vision_data: list[int] = []
        for v_angle in vision_angles:
            dist = 0
            total_angle = self.angle + v_angle
            rad = math.radians(total_angle)
            end_x, end_y = int(self.x), int(self.y)
            while dist < max_distance:
                end_x = int(self.x + dist * math.cos(rad))
                end_y = int(self.y - dist * math.sin(rad))
                # Boundary protection
                if (0 <= end_x < bg_array.shape[1]) and (0 <= end_y < bg_array.shape[0]):
                    if tuple(bg_array[end_y, end_x, :3]) == target_color:
                        break
                else:
                    break
                dist += 1
            if screen is not None:
                pygame.draw.line(
                    screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 1
                )
            vision_data.append(dist)
        return vision_data

    def update(self, steering: float, accel_input: float) -> None:
        # --- Update speed with acceleration and friction ---
        if accel_input != 0:
            self.speed += accel_input * self.acceleration
        else:
            # Apply friction when no input
            if self.speed > 0:
                self.speed -= self.friction
            elif self.speed < 0:
                self.speed += self.friction

        # Clamp speed
        self.speed = max(-self.max_speed, min(self.speed, self.max_speed))

        # --- Update rotation ---
        speed_factor = self.speed / self.max_speed if self.max_speed else 0
        self.angle += steering * speed_factor * 5  # rotate faster with speed

        # --- Update position ---
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)

        # --- Rotate image and update rect ---
        self.image = pygame.transform.rotate(self.base_image, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def get_collide_checkpoint(
        self,
        checkpoint: tuple[tuple[int, int], tuple[int, int]],
        threshold: float = 20.0,
    ) -> bool:
        mid_point = (
            (checkpoint[0][0] + checkpoint[1][0]) // 2,
            (checkpoint[0][1] + checkpoint[1][1]) // 2,
        )
        return math.dist(self.rect.center, mid_point) < threshold

    def dist_to_checkpoint(
        self,
        checkpoint: tuple[tuple[int, int], tuple[int, int]],
        screen: pygame.Surface,
    ) -> float:
        mid_cp = (
            (checkpoint[0][0] + checkpoint[1][0]) // 2,
            (checkpoint[0][1] + checkpoint[1][1]) // 2,
        )
        pygame.draw.line(screen, (255, 0, 255), (self.x, self.y), mid_cp, 1)
        return math.hypot(self.x - mid_cp[0], self.y - mid_cp[1])

    def pos_relative_to_next_cp(
        self, checkpoint: tuple[tuple[int, int], tuple[int, int]]
    ) -> tuple[float, float]:
        mid_cp = (
            (checkpoint[0][0] + checkpoint[1][0]) // 2,
            (checkpoint[0][1] + checkpoint[1][1]) // 2,
        )
        return mid_cp[0] - self.x, mid_cp[1] - self.y