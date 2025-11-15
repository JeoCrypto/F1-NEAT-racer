#!/usr/bin/env python3
"""
Manually test if a car can even reach CP0 from the starting position with simple controls.
"""

import pygame
import numpy as np
from PIL import Image
from car import Car

# Starting position and angle
STARTING_POSITION = (404, 399)
STARTING_ANGLE = -60.9

# Load circuit
circuit = Image.open("circuit.png")
bg_array = np.array(circuit)

# Checkpoints
checkpoints = [
    ((350, 421), (354, 435)),  # 0
    ((245, 473), (250, 491)),  # 1
]

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Manual Drive Test")
clock = pygame.time.Clock()

# Load background
background_image = pygame.image.fromstring(circuit.tobytes(), circuit.size, circuit.mode)

# Create car
car = Car(STARTING_POSITION, angle=STARTING_ANGLE)

print("Manual Drive Test")
print("=" * 60)
print("Controls:")
print("  Arrow Keys: Steer left/right")
print("  Up Arrow: Accelerate")
print("  Down Arrow: Brake")
print("  Q: Quit")
print()
print(f"Starting position: {STARTING_POSITION}")
print(f"Starting angle: {STARTING_ANGLE:.1f}°")
print(f"Distance to CP0: {np.hypot(car.x - 352, car.y - 428):.1f}px")
print()
print("Try to drive through CP0 (the cyan line)")
print("=" * 60)

curr_cp = 0
frames = 0
running = True

while running:
    frames += 1

    # Handle input
    keys = pygame.key.get_pressed()
    steering = 0.0
    acceleration = 0.0

    if keys[pygame.K_LEFT]:
        steering = -2.0
    if keys[pygame.K_RIGHT]:
        steering = 2.0
    if keys[pygame.K_UP]:
        acceleration = 5.0
    if keys[pygame.K_DOWN]:
        acceleration = -5.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    # Update car
    car.update(steering, acceleration)

    # Check off-track
    if car.check_off_track(bg_array):
        print(f"\n⚠️  Went off track at frame {frames}!")
        print(f"Position: ({car.x:.1f}, {car.y:.1f})")
        print(f"Speed: {car.speed:.2f}")
        running = False

    # Check checkpoint collision
    next_cp = (curr_cp + 1) % len(checkpoints)
    if car.get_collide_checkpoint(checkpoints[next_cp]):
        curr_cp = next_cp
        print(f"✓ Passed CP{curr_cp} at frame {frames}!")
        if curr_cp == 1:
            print("\n🎉 SUCCESS! You reached CP1!")
            print(f"Total frames: {frames}")
            running = False

    # Draw
    screen.blit(background_image, (0, 0))

    # Draw checkpoints
    for idx, cp in enumerate(checkpoints):
        color = (0, 255, 255) if idx == next_cp else (0, 255, 0)
        pygame.draw.line(screen, color, cp[0], cp[1], 5 if idx == next_cp else 2)

    car.draw(screen)
    car.vision(bg_array, screen=screen)
    car.dist_to_checkpoint(checkpoints[next_cp], screen)

    # Show info
    font = None
    try:
        font = pygame.font.Font(None, 24)
    except:
        pass

    if font:
        info_text = f"Frame: {frames} | Speed: {car.speed:.1f} | Angle: {car.angle:.1f}° | CP: {curr_cp}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

if curr_cp == 0:
    print("\n❌ Did not reach CP0")
    print("This suggests the starting position/angle might be problematic")
