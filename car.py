import pygame
import math
from PIL import Image
import numpy as np

class Car:
    def __init__(self, pos, width=20, height=10, color=(255,0,0), angle = 90, max_speed=5, acceleration=0.2, friction=0.05):
        self.starting_pos = pos
        self.x, self.y = pos
        self.width = width
        self.height = height
        self.color = color
        self.angle = angle
        self.speed = 0
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.friction = friction

        self.base_image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.base_image.fill(color)
        self.image = self.base_image
        self.rect = self.image.get_rect(center=(self.x, self.y))
    
    def reset(self, last_checkpoint_midpoint):
        self.x, self.y = last_checkpoint_midpoint
        self.angle = (90-math.atan2(self.y - last_checkpoint_midpoint[1], self.x - last_checkpoint_midpoint[0])) * 180 / math.pi
        self.speed = 0
        self.image = pygame.transform.rotate(self.base_image, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def draw(self, screen):
        screen.blit(self.image, self.rect)
        pygame.draw.circle(self.base_image, (0, 255, 0), (self.width, self.height//2), 5)
        pygame.draw.rect(screen, (0,0,255), self.rect, 2)

    def check_off_track(self, bg_array, target_color1=(255,255,255)):
        # Get the color of the pixel at the car's position
        for x in range(self.rect.left, self.rect.right):
            for y in range(self.rect.top, self.rect.bottom):
                if tuple(bg_array[y,x,:3]) == target_color1:
                    return True
        return False

    def vision(self, bg_array, target_color=(255, 255, 255), max_distance=200, screen = None):
        # Example vision lines at -45, 0, +45 degrees relative to car's angle
        vision_angles = [-90,-45, 0, 45,90,180]
        vision_data = []
        max_distance = max_distance  # Maximum distance to check
        for v_angle in vision_angles:
            dist = 0
            total_angle = self.angle + v_angle
            rad = math.radians(total_angle)
            while dist < max_distance:
                end_x = int(self.x + dist * math.cos(rad))
                end_y = int(self.y - dist * math.sin(rad))
                if tuple(bg_array[end_y, end_x, :3]) == target_color:
                    break
                dist += 1
            if screen:
                pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 1)
            vision_data.append(dist)
        return vision_data

    def update(self, steering, accel_input):
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
        self.angle += steering * (self.speed / self.max_speed) * 5  # rotate faster with speed

        # --- Update position ---
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)

        # --- Rotate image and update rect ---
        self.image = pygame.transform.rotate(self.base_image, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))
    
    def get_collide_checkpoint(self, checkpoint, theshold=20):
        return math.dist(self.rect.center, ((checkpoint[0][0]+checkpoint[1][0])//2, (checkpoint[0][1]+checkpoint[1][1])//2)) < theshold

    def dist_to_checkpoint(self, checkpoint, screen):
        mid_cp = ((checkpoint[0][0]+checkpoint[1][0])//2, (checkpoint[0][1]+checkpoint[1][1])//2)
        pygame.draw.line(screen, (255,0,255), (self.x, self.y), mid_cp, 1)
        return math.hypot(self.x - mid_cp[0], self.y - mid_cp[1])
    
    def pos_relative_to_next_cp(self, checkpoint):
        mid_cp = ((checkpoint[0][0]+checkpoint[1][0])//2, (checkpoint[0][1]+checkpoint[1][1])//2)
        return (mid_cp[0]-self.x, mid_cp[1]-self.y)