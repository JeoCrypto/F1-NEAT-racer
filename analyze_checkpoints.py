#!/usr/bin/env python3
"""Analyze checkpoint spacing and difficulty."""

import numpy as np
from PIL import Image

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

def checkpoint_midpoint(cp):
    """Get midpoint of checkpoint."""
    (x1, y1), (x2, y2) = cp
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Starting position
start_pos = (404, 399)

print("=" * 60)
print("CHECKPOINT SPACING ANALYSIS")
print("=" * 60)

# Distance from start to CP0
cp0_mid = checkpoint_midpoint(checkpoints[0])
dist_to_cp0 = distance(start_pos, cp0_mid)
print(f"\nStart position: {start_pos}")
print(f"CP0 midpoint: ({cp0_mid[0]:.1f}, {cp0_mid[1]:.1f})")
print(f"Distance to CP0: {dist_to_cp0:.1f} pixels")

# Analyze spacing between checkpoints
print(f"\n{'CP':<5} {'Distance':<10} {'Status'}")
print("-" * 40)

distances = []
for i in range(len(checkpoints)):
    next_i = (i + 1) % len(checkpoints)
    mid1 = checkpoint_midpoint(checkpoints[i])
    mid2 = checkpoint_midpoint(checkpoints[next_i])
    dist = distance(mid1, mid2)
    distances.append(dist)

    # Flag unusual distances
    status = ""
    if dist < 30:
        status = "⚠️  TOO CLOSE"
    elif dist > 120:
        status = "⚠️  TOO FAR"

    print(f"{i:>2} → {next_i:<2} {dist:>6.1f}px   {status}")

print(f"\n{'Statistic':<20} {'Value'}")
print("-" * 40)
print(f"{'Min distance:':<20} {min(distances):.1f} pixels")
print(f"{'Max distance:':<20} {max(distances):.1f} pixels")
print(f"{'Mean distance:':<20} {np.mean(distances):.1f} pixels")
print(f"{'Median distance:':<20} {np.median(distances):.1f} pixels")
print(f"{'Std deviation:':<20} {np.std(distances):.1f} pixels")

# Find problematic sections
print("\n" + "=" * 60)
print("POTENTIAL PROBLEMS")
print("=" * 60)

too_close = [(i, d) for i, d in enumerate(distances) if d < 30]
too_far = [(i, d) for i, d in enumerate(distances) if d > 120]

if too_close:
    print("\nCheckpoints too close together (< 30px):")
    for i, d in too_close:
        next_i = (i + 1) % len(checkpoints)
        print(f"  CP{i} → CP{next_i}: {d:.1f}px")
else:
    print("\n✓ No checkpoints too close")

if too_far:
    print("\nCheckpoints too far apart (> 120px):")
    for i, d in too_far:
        next_i = (i + 1) % len(checkpoints)
        print(f"  CP{i} → CP{next_i}: {d:.1f}px")
        print(f"    Consider adding intermediate checkpoint")
else:
    print("\n✓ No checkpoints too far")

# Check first few checkpoints (where car is getting stuck)
print("\n" + "=" * 60)
print("FIRST 5 CHECKPOINTS (Critical for learning)")
print("=" * 60)

for i in range(min(5, len(checkpoints))):
    cp = checkpoints[i]
    mid = checkpoint_midpoint(cp)
    width = distance(cp[0], cp[1])

    print(f"\nCP{i}:")
    print(f"  Endpoints: {cp[0]} → {cp[1]}")
    print(f"  Midpoint: ({mid[0]:.1f}, {mid[1]:.1f})")
    print(f"  Width: {width:.1f}px")

    if i > 0:
        prev_mid = checkpoint_midpoint(checkpoints[i-1])
        dist = distance(prev_mid, mid)
        print(f"  Distance from CP{i-1}: {dist:.1f}px")
