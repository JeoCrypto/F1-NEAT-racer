#!/usr/bin/env python3
"""
Interactive checkpoint placement helper for F1-NEAT-racer.
Click on the track to place checkpoint lines.
"""

import pygame
import sys
from PIL import Image
import numpy as np

# Initialize pygame
pygame.init()

# Load the circuit image
try:
    pil_image = Image.open("circuit.png")
    print(f"Loaded circuit.png - Size: {pil_image.size}")
except FileNotFoundError:
    print("Error: circuit.png not found!")
    sys.exit(1)

WIDTH, HEIGHT = pil_image.size
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkpoint Placement Helper")

background_image = pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode)
bg_array = np.array(pil_image)

# Checkpoint data
checkpoints = []
current_point = None

# Try to initialize font, but continue if it fails
try:
    font = pygame.font.SysFont("Arial", 14)
    FONT_AVAILABLE = True
except (ImportError, NotImplementedError):
    font = None
    FONT_AVAILABLE = False
    print("Note: Font rendering disabled (SDL_ttf not available)")

# Instructions
print("\n" + "="*60)
print("CHECKPOINT PLACEMENT HELPER")
print("="*60)
print("CONTROLS:")
print("  - LEFT CLICK: Place checkpoint points (2 clicks per checkpoint)")
print("  - RIGHT CLICK: Remove last checkpoint")
print("  - S: Save checkpoints to checkpoints.txt")
print("  - G: Click to set starting grid position")
print("  - ESC: Quit without saving")
print("="*60)
print("\nPlace checkpoints by clicking two points to form a line.")
print("Checkpoints should cross the track perpendicular to racing direction.\n")

mode = "checkpoint"  # or "grid"
grid_position = None
grid_angle = 0.0

running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("\nExiting without saving...")
                running = False

            elif event.key == pygame.K_s:
                # Save checkpoints
                if checkpoints:
                    with open("checkpoints.txt", "w") as f:
                        f.write("checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [\n")
                        for i, cp in enumerate(checkpoints):
                            comment = "  # start/finish" if i == 0 else ""
                            f.write(f"    {cp},{comment}\n")
                        f.write("]\n\n")
                        if grid_position:
                            f.write(f"STARTING_POSITION = {grid_position}\n")
                            f.write(f"STARTING_ANGLE = {grid_angle}\n")
                    print(f"\n✓ Saved {len(checkpoints)} checkpoints to checkpoints.txt")
                else:
                    print("\n✗ No checkpoints to save!")

            elif event.key == pygame.K_g:
                mode = "grid"
                print("\nGrid mode: Click to set starting position")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()

            # Check pixel color at click position
            x, y = pos
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                pixel_color = tuple(bg_array[y, x, :3])

                if event.button == 1:  # Left click
                    if mode == "checkpoint":
                        if current_point is None:
                            current_point = pos
                            print(f"Checkpoint {len(checkpoints)} - Point 1: {pos}")
                        else:
                            checkpoints.append((current_point, pos))
                            print(f"Checkpoint {len(checkpoints)} - Point 2: {pos} -> COMPLETE")
                            current_point = None
                    elif mode == "grid":
                        grid_position = pos
                        print(f"Grid position set: {pos}")
                        mode = "checkpoint"

                elif event.button == 3:  # Right click
                    if checkpoints:
                        removed = checkpoints.pop()
                        print(f"Removed checkpoint: {removed}")
                        current_point = None

    # Draw
    screen.blit(background_image, (0, 0))

    # Draw completed checkpoints
    for i, cp in enumerate(checkpoints):
        pygame.draw.line(screen, (0, 255, 0), cp[0], cp[1], 3)
        pygame.draw.circle(screen, (0, 255, 0), cp[0], 5)
        pygame.draw.circle(screen, (0, 255, 0), cp[1], 5)

        # Draw checkpoint number (if font available)
        if FONT_AVAILABLE:
            mid_x = (cp[0][0] + cp[1][0]) // 2
            mid_y = (cp[0][1] + cp[1][1]) // 2
            text = font.render(str(i), True, (255, 255, 0))
            screen.blit(text, (mid_x + 10, mid_y - 10))

    # Draw current point being placed
    if current_point:
        pygame.draw.circle(screen, (255, 0, 0), current_point, 5)
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.line(screen, (255, 0, 0), current_point, mouse_pos, 2)

    # Draw grid position
    if grid_position:
        pygame.draw.circle(screen, (255, 0, 255), grid_position, 8, 2)
        pygame.draw.circle(screen, (255, 0, 255), grid_position, 2)

    # Draw info text (if font available)
    if FONT_AVAILABLE:
        info_text = f"Checkpoints: {len(checkpoints)} | Mode: {mode.upper()}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        pygame.draw.rect(screen, (0, 0, 0), (5, 5, text_surface.get_width() + 10, 25))
        screen.blit(text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("\nDone!")
