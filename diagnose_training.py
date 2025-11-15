#!/usr/bin/env python3
"""Diagnose where the car is going off track during training."""

import pickle
import pygame
import numpy as np
from PIL import Image
from car import Car
import neat

# Load configuration
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-feedforward.txt"
)

# Load the winner
import sys
model_file = sys.argv[1] if len(sys.argv) > 1 else "winner.pkl"
with open(model_file, "rb") as f:
    winner = pickle.load(f)
print(f"Testing model: {model_file}\n")

# Checkpoints from main.py
checkpoints = [
    ((350, 421), (354, 435)),  # 0
    ((245, 473), (250, 491)),  # 1
    ((214, 524), (216, 544)),  # 2
    ((134, 522), (127, 534)),  # 3
    ((81, 481), (76, 494)),  # 4
    ((45, 401), (29, 395)),  # 5
    ((71, 382), (70, 402)),  # 6
    ((96, 416), (89, 430)),  # 7
    ((120, 450), (117, 462)),  # 8
    ((167, 453), (165, 465)),  # 9
    ((240, 418), (261, 408)),  # 10
    ((251, 357), (265, 356)),  # 11
    ((290, 333), (291, 348)),  # 12
    ((320, 345), (327, 361)),  # 13
    ((358, 345), (367, 353)),  # 14
    ((463, 309), (473, 315)),  # 15
    ((473, 280), (487, 278)),  # 16
    ((451, 265), (452, 277)),  # 17
    ((414, 288), (403, 303)),  # 18
    ((304, 293), (292, 305)),  # 19
    ((251, 292), (237, 304)),  # 20
    ((212, 332), (201, 340)),  # 21
    ((220, 376), (207, 385)),  # 22
    ((199, 405), (169, 387)),  # 23
    ((146, 388), (120, 380)),  # 24
    ((118, 340), (103, 324)),  # 25
    ((140, 270), (168, 265)),  # 26
    ((239, 214), (265, 213)),  # 27
    ((294, 183), (317, 187)),  # 28
    ((379, 141), (400, 137)),  # 29
    ((417, 113), (453, 110)),  # 30
    ((465, 89), (501, 91)),  # 31
    ((531, 94), (532, 116)),  # 32
    ((554, 132), (541, 151)),  # 33
    ((512, 164), (502, 169)),  # 34
    ((488, 146), (479, 157)),  # 35
    ((463, 162), (450, 162)),  # 36
    ((466, 198), (459, 208)),  # 37
    ((543, 206), (535, 219)),  # 38
    ((602, 218), (593, 229)),  # 39
    ((614, 255), (599, 265)),  # 40
    ((603, 293), (578, 299)),  # 41
    ((573, 315), (543, 319)),  # 42
    ((528, 341), (504, 339)),  # 43
    ((494, 360), (472, 357)),  # 44
]

STARTING_POSITION = (375, 410)  # Updated to match main.py
STARTING_ANGLE = -50

# Load circuit
circuit = Image.open("circuit.png")
bg_array = np.array(circuit)

def get_inputs(car, next_checkpoint_idx):
    """Get neural network inputs."""
    return [
        *car.vision(bg_array),
        car.speed,
        car.angle,
        *car.pos_relative_to_next_cp(checkpoints[next_checkpoint_idx]),
    ]

# Create network
net = neat.nn.FeedForwardNetwork.create(winner, config)
car = Car(STARTING_POSITION, angle=STARTING_ANGLE)

print("=" * 60)
print("DIAGNOSTIC RUN - Tracking car behavior")
print("=" * 60)

curr_cp = 0
checkpoints_passed = []
positions = []
max_frames = 1000

for frame in range(max_frames):
    next_cp = (curr_cp + 1) % len(checkpoints)
    inputs = get_inputs(car, next_cp)
    output = net.activate(inputs)

    steering = output[0] * 2
    acceleration = output[1] * 5

    # Store position
    positions.append((int(car.x), int(car.y)))

    # Update car
    car.update(steering, acceleration)

    # Check for checkpoint collision
    if car.get_collide_checkpoint(checkpoints[next_cp]):
        checkpoints_passed.append(next_cp)
        curr_cp = next_cp
        print(f"Frame {frame:4d}: Passed CP{curr_cp} at position ({car.x:.1f}, {car.y:.1f}), speed: {car.speed:.2f}")

    # Check if car went off track
    if frame > 5 and car.check_off_track(bg_array):
        print(f"\n{'='*60}")
        print(f"CAR WENT OFF TRACK!")
        print(f"{'='*60}")
        print(f"Frame: {frame}")
        print(f"Position: ({car.x:.1f}, {car.y:.1f})")
        print(f"Speed: {car.speed:.2f}")
        print(f"Angle: {car.angle:.1f}")
        print(f"Current checkpoint: {curr_cp}")
        print(f"Next checkpoint: {next_cp}")
        print(f"Total checkpoints passed: {len(checkpoints_passed)}")
        print(f"\nCheckpoints reached: {checkpoints_passed}")

        # Show last few steering decisions
        print(f"\nFinal network output:")
        print(f"  Steering: {steering:.3f}")
        print(f"  Acceleration: {acceleration:.3f}")

        # Show vision at crash point
        vision = car.vision(bg_array)
        print(f"\nVision distances at crash:")
        angles = [-90, -45, 0, 45, 90, 180]
        for i, (angle, dist) in enumerate(zip(angles, vision)):
            print(f"  {angle:>4d}°: {dist:>3d}px")

        break

if frame >= max_frames - 1:
    print(f"\nReached max frames ({max_frames})")
    print(f"Checkpoints passed: {len(checkpoints_passed)}")
    print(f"Final position: ({car.x:.1f}, {car.y:.1f})")
    print(f"Final speed: {car.speed:.2f}")

    # Check distance to checkpoints
    import math
    for i in range(min(5, len(checkpoints))):
        cp_mid_x = (checkpoints[i][0][0] + checkpoints[i][1][0]) / 2
        cp_mid_y = (checkpoints[i][0][1] + checkpoints[i][1][1]) / 2
        dist = math.sqrt((car.x - cp_mid_x)**2 + (car.y - cp_mid_y)**2)
        print(f"  Distance to CP{i}: {dist:.1f}px")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Frames survived: {frame}")
print(f"Checkpoints passed: {len(checkpoints_passed)}")
print(f"Furthest checkpoint: {curr_cp}")
print(f"Expected fitness: ~{len(checkpoints_passed) * 500 + frame * 0.5:.1f}")

# Analyze movement
if len(positions) > 10:
    start_pos = positions[0]
    end_pos = positions[-1]
    total_distance = sum(
        math.sqrt((positions[i+1][0] - positions[i][0])**2 + (positions[i+1][1] - positions[i][1])**2)
        for i in range(len(positions) - 1)
    )
    net_distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)

    print(f"\nMovement analysis:")
    print(f"  Start: {start_pos}")
    print(f"  End: {end_pos}")
    print(f"  Total distance traveled: {total_distance:.1f}px")
    print(f"  Net displacement: {net_distance:.1f}px")
    print(f"  Movement efficiency: {(net_distance/total_distance)*100:.1f}%")

    if total_distance < 100:
        print("\n⚠️  WARNING: Car barely moved! It's staying nearly stationary.")
