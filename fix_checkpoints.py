#!/usr/bin/env python3
"""
Automatically fix existing checkpoints to ensure both endpoints are on track.

This script:
1. Loads current checkpoints from main.py
2. For each checkpoint, checks if endpoints are on track (dark pixels)
3. If an endpoint is off-track (white), it searches nearby for a dark pixel
4. Saves the fixed checkpoints to checkpoints_fixed.txt
"""

import re
import numpy as np
from PIL import Image
from typing import List, Tuple

Point = Tuple[int, int]
Checkpoint = Tuple[Point, Point]


def load_track(filename: str = "circuit.png") -> np.ndarray:
    """Load track image."""
    img = Image.open(filename)
    return np.array(img)


def is_on_track(point: Point, track_arr: np.ndarray) -> bool:
    """Check if point is on dark track pixel."""
    x, y = point
    if 0 <= x < track_arr.shape[1] and 0 <= y < track_arr.shape[0]:
        pixel = tuple(track_arr[y, x, :3])
        return pixel[0] < 100  # Dark = track
    return False


def find_nearest_track_pixel(point: Point, track_arr: np.ndarray, max_dist: int = 10) -> Point:
    """
    Find the nearest dark track pixel within max_dist of the given point.
    Returns the original point if no track pixel is found nearby.
    """
    x, y = point

    # Check if already on track
    if is_on_track(point, track_arr):
        return point

    # Search in expanding square around point
    for dist in range(1, max_dist + 1):
        for dx in range(-dist, dist + 1):
            for dy in range(-dist, dist + 1):
                if abs(dx) == dist or abs(dy) == dist:  # Only check perimeter
                    new_point = (x + dx, y + dy)
                    if is_on_track(new_point, track_arr):
                        return new_point

    # No track pixel found nearby, return original
    return point


def load_checkpoints_from_main(filename: str = "main.py") -> List[Checkpoint]:
    """Load checkpoints from main.py."""
    with open(filename, 'r') as f:
        content = f.read()

    # Extract checkpoint tuples
    pattern = r'\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\)'
    matches = re.findall(pattern, content)

    checkpoints = []
    for match in matches:
        p1 = (int(match[0]), int(match[1]))
        p2 = (int(match[2]), int(match[3]))
        checkpoints.append((p1, p2))

    return checkpoints


def fix_checkpoint(cp: Checkpoint, track_arr: np.ndarray) -> Checkpoint:
    """Fix a checkpoint by moving endpoints to nearest track pixels."""
    p1, p2 = cp

    # Fix each endpoint if needed
    p1_fixed = find_nearest_track_pixel(p1, track_arr)
    p2_fixed = find_nearest_track_pixel(p2, track_arr)

    return (p1_fixed, p2_fixed)


def save_checkpoints(checkpoints: List[Checkpoint], filename: str = "checkpoints_fixed.txt"):
    """Save checkpoints to file."""
    with open(filename, 'w') as f:
        f.write("# Fixed checkpoints - both endpoints on track\n")
        f.write("# Copy into main.py\n\n")
        f.write("checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [\n")

        for i, cp in enumerate(checkpoints):
            f.write(f"    {cp},  # {i}\n")

        f.write("]\n")

    print(f"Saved {len(checkpoints)} checkpoints to {filename}")


def main():
    print("Loading track image...")
    track_arr = load_track()

    print("Loading checkpoints from main.py...")
    checkpoints = load_checkpoints_from_main()
    print(f"Loaded {len(checkpoints)} checkpoints")

    print("\nAnalyzing and fixing checkpoints...")
    fixed_checkpoints = []
    fixed_count = 0

    for i, cp in enumerate(checkpoints):
        p1, p2 = cp
        p1_on_track = is_on_track(p1, track_arr)
        p2_on_track = is_on_track(p2, track_arr)

        if p1_on_track and p2_on_track:
            # Already valid
            fixed_checkpoints.append(cp)
            print(f"CP {i:2d}: ✓ Valid")
        else:
            # Needs fixing
            fixed_cp = fix_checkpoint(cp, track_arr)
            fixed_checkpoints.append(fixed_cp)
            fixed_count += 1

            status = []
            if not p1_on_track:
                status.append(f"P1 moved {p1}→{fixed_cp[0]}")
            if not p2_on_track:
                status.append(f"P2 moved {p2}→{fixed_cp[1]}")

            print(f"CP {i:2d}: ✗ Fixed - {', '.join(status)}")

    print(f"\nSummary:")
    print(f"  Total checkpoints: {len(checkpoints)}")
    print(f"  Already valid: {len(checkpoints) - fixed_count}")
    print(f"  Fixed: {fixed_count}")

    # Save fixed checkpoints
    save_checkpoints(fixed_checkpoints)

    print(f"\nNext steps:")
    print(f"1. Review checkpoints_fixed.txt")
    print(f"2. Copy into main.py replacing old checkpoints")
    print(f"3. Test with: python3 main.py train --generations 5 --visualize")


if __name__ == "__main__":
    main()
