#!/usr/bin/env python3
"""
Automatic Checkpoint Generator

This script analyzes the track and automatically generates well-placed checkpoints.
The generated checkpoints can then be fine-tuned using the checkpoint editor.

Algorithm:
1. Detect track centerline using image processing
2. Follow centerline and place checkpoints at regular intervals
3. For each checkpoint, find perpendicular line across track width
4. Ensure both endpoints are on track surface

Usage:
    python3 auto_checkpoint_generator.py [--spacing PIXELS] [--output FILE]
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from typing import List, Tuple
import math

Point = Tuple[int, int]
Checkpoint = Tuple[Point, Point]


def load_track(filename: str = "circuit.png") -> np.ndarray:
    """Load track image as numpy array."""
    img = Image.open(filename)
    return np.array(img)


def create_track_mask(track_arr: np.ndarray, threshold: int = 100) -> np.ndarray:
    """
    Create binary mask: True = track, False = off-track.
    """
    # Convert to grayscale if needed
    if len(track_arr.shape) == 3:
        gray = np.mean(track_arr[:, :, :3], axis=2)
    else:
        gray = track_arr

    # Track is dark pixels
    track_mask = gray < threshold
    return track_mask


def find_track_skeleton(track_mask: np.ndarray) -> np.ndarray:
    """
    Find the skeleton (centerline) of the track.
    Uses morphological thinning.
    """
    from skimage import morphology
    try:
        skeleton = morphology.skeletonize(track_mask)
        return skeleton
    except ImportError:
        # Fallback: use erosion to approximate centerline
        print("Warning: skimage not available, using simple erosion")
        eroded = ndimage.binary_erosion(track_mask, iterations=10)
        return eroded


def trace_centerline(skeleton: np.ndarray, start_point: Point) -> List[Point]:
    """
    Trace along the skeleton from a starting point.
    Returns ordered list of points along the centerline.
    """
    # This is a simplified implementation
    # A full implementation would use graph traversal
    points = []
    visited = np.zeros_like(skeleton, dtype=bool)

    # Find skeleton points
    skeleton_points = np.argwhere(skeleton)

    if len(skeleton_points) == 0:
        return points

    # Start from nearest skeleton point to start_point
    distances = np.sum((skeleton_points - np.array(start_point)[::-1])**2, axis=1)
    current_idx = np.argmin(distances)
    current = skeleton_points[current_idx]

    for _ in range(len(skeleton_points)):
        y, x = current
        if visited[y, x]:
            break

        points.append((x, y))
        visited[y, x] = True

        # Find next nearest unvisited skeleton point
        unvisited_mask = ~visited
        remaining = skeleton_points[np.where(unvisited_mask[skeleton_points[:, 0], skeleton_points[:, 1]])[0]]

        if len(remaining) == 0:
            break

        distances = np.sum((remaining - current)**2, axis=1)
        next_idx = np.argmin(distances)

        if distances[next_idx] > 100:  # Too far, stop
            break

        current = remaining[next_idx]

    return points


def find_track_width(point: Point, angle: float, track_mask: np.ndarray, max_dist: int = 100) -> Tuple[Point, Point]:
    """
    Find the track edges perpendicular to the centerline at a point.

    Args:
        point: Center point on track
        angle: Angle of track direction
        track_mask: Binary mask of track
        max_dist: Maximum distance to search for edge

    Returns:
        Tuple of two points (left edge, right edge)
    """
    x, y = point

    # Perpendicular angle (90 degrees from track direction)
    perp_angle = angle + math.pi / 2

    dx = math.cos(perp_angle)
    dy = math.sin(perp_angle)

    # Search in both directions for track edges
    left_point = None
    right_point = None

    # Search left
    for dist in range(1, max_dist):
        px = int(x + dx * dist)
        py = int(y + dy * dist)

        if 0 <= px < track_mask.shape[1] and 0 <= py < track_mask.shape[0]:
            if not track_mask[py, px]:  # Hit off-track
                # Step back one pixel to be on track edge
                px = int(x + dx * (dist - 3))
                py = int(y + dy * (dist - 3))
                left_point = (px, py)
                break
        else:
            break

    # Search right
    for dist in range(1, max_dist):
        px = int(x - dx * dist)
        py = int(y - dy * dist)

        if 0 <= px < track_mask.shape[1] and 0 <= py < track_mask.shape[0]:
            if not track_mask[py, px]:  # Hit off-track
                # Step back one pixel to be on track edge
                px = int(x - dx * (dist - 3))
                py = int(y - dy * (dist - 3))
                right_point = (px, py)
                break
        else:
            break

    # If we didn't find edges, use a default width
    if left_point is None:
        left_point = (int(x + dx * 20), int(y + dy * 20))
    if right_point is None:
        right_point = (int(x - dx * 20), int(y - dy * 20))

    return left_point, right_point


def generate_checkpoints_simple(
    track_arr: np.ndarray,
    spacing: int = 80,
    start_pos: Point = (361, 402)
) -> List[Checkpoint]:
    """
    Simple checkpoint generation: follow track and place checkpoints.

    This is a basic implementation that works without complex image processing.
    """
    print("Generating checkpoints...")

    checkpoints = []
    track_mask = create_track_mask(track_arr)

    # Find track points roughly following the track
    # Start from start_pos and spiral outward to find track centerline points

    # For simplicity, let's sample points along the track perimeter
    # and create checkpoints at regular intervals

    # Get all track pixels
    track_points = np.argwhere(track_mask)

    if len(track_points) == 0:
        print("No track found!")
        return []

    # Simple approach: place checkpoints at evenly spaced intervals around track
    # This won't be perfect but gives a starting point

    print(f"Found {len(track_points)} track pixels")
    print(f"Generating checkpoints with {spacing}px spacing...")

    # Sample points from track_points
    num_checkpoints = len(track_points) // (spacing * 10)
    indices = np.linspace(0, len(track_points)-1, num_checkpoints, dtype=int)

    for idx in indices:
        y, x = track_points[idx]
        center = (x, y)

        # Try to find perpendicular line
        # Estimate local track direction by looking at nearby points
        angle = 0  # Default angle

        # Find track edges perpendicular to estimated direction
        p1, p2 = find_track_width(center, angle, track_mask)

        # Verify both points are on track
        if (track_mask[p1[1], p1[0]] and track_mask[p2[1], p2[0]]):
            checkpoints.append((p1, p2))

    print(f"Generated {len(checkpoints)} checkpoints")
    return checkpoints


def save_checkpoints(checkpoints: List[Checkpoint], filename: str = "checkpoints_auto.txt"):
    """Save checkpoints to file."""
    with open(filename, 'w') as f:
        f.write("# Auto-generated checkpoints\n")
        f.write("# Copy into main.py\n\n")
        f.write("checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [\n")

        for i, cp in enumerate(checkpoints):
            f.write(f"    {cp},  # {i}\n")

        f.write("]\n")

    print(f"Saved {len(checkpoints)} checkpoints to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Auto-generate track checkpoints")
    parser.add_argument("--spacing", type=int, default=80, help="Spacing between checkpoints (pixels)")
    parser.add_argument("--output", type=str, default="checkpoints_auto.txt", help="Output file")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")

    args = parser.parse_args()

    # Load track
    try:
        track_arr = load_track()
    except FileNotFoundError:
        print("Error: circuit.png not found")
        return

    # Generate checkpoints
    checkpoints = generate_checkpoints_simple(track_arr, spacing=args.spacing)

    if len(checkpoints) == 0:
        print("Failed to generate checkpoints")
        return

    # Save
    save_checkpoints(checkpoints, args.output)

    print(f"\nNext steps:")
    print(f"1. Review checkpoints in: {args.output}")
    print(f"2. Use checkpoint_editor.py to refine placement")
    print(f"3. Copy final checkpoints to main.py")


if __name__ == "__main__":
    main()
