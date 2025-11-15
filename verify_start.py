"""Quick script to verify starting position is on the track."""
import numpy as np
from PIL import Image

# Load circuit
circuit = Image.open("circuit.png")
bg_array = np.array(circuit)

# Check starting position (from main.py)
start_pos = (404, 399)
x, y = start_pos

print(f"Starting position: {start_pos}")
print(f"Image size: {circuit.size}")

# Check if position is within bounds
if 0 <= x < bg_array.shape[1] and 0 <= y < bg_array.shape[0]:
    pixel_color = tuple(bg_array[y, x, :3])
    print(f"Pixel color at start: RGB{pixel_color}")

    # Check surrounding area
    print("\nChecking 3x3 area around start:")
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            px, py = x + dx, y + dy
            if 0 <= px < bg_array.shape[1] and 0 <= py < bg_array.shape[0]:
                color = tuple(bg_array[py, px, :3])
                marker = "█" if color == (255, 255, 255) else "▓" if color[0] > 200 else "░"
                print(f"{marker}", end="")
            else:
                print("X", end="")
        print()

    print("\n█ = White (off-track)")
    print("▓ = Light color")
    print("░ = Dark color (track)")

    if pixel_color == (255, 255, 255):
        print("\n⚠️  WARNING: Starting position is on WHITE (off-track)!")
    else:
        print(f"\n✓ Starting position appears to be on track")
else:
    print("ERROR: Starting position is out of bounds!")

# Check first checkpoint (from main.py)
checkpoint_0 = ((350, 421), (354, 435))
mid_x = (checkpoint_0[0][0] + checkpoint_0[1][0]) // 2
mid_y = (checkpoint_0[0][1] + checkpoint_0[1][1]) // 2
print(f"\nFirst checkpoint midpoint: ({mid_x}, {mid_y})")
print(f"Distance from start to checkpoint 0: {np.hypot(x - mid_x, y - mid_y):.1f} pixels")
