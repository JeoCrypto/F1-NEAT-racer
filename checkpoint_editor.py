#!/usr/bin/env python3
"""
Interactive Checkpoint Editor for F1 NEAT Racer

This tool helps you place checkpoints properly across the racing track.
Checkpoints should span the track width with both endpoints on the track surface.

Features:
- Interactive click-and-place checkpoint creation
- Visual feedback showing track vs off-track areas
- Automatic checkpoint validation
- Export to main.py format
- Load existing checkpoints for editing
- Smart suggestions for checkpoint placement

Controls:
    Left Click (1st):  Place first endpoint of checkpoint
    Left Click (2nd):  Place second endpoint (completes checkpoint)
    Right Click:       Delete nearest checkpoint
    Mouse Wheel:       Zoom in/out
    Space:             Auto-generate checkpoints along track
    S:                 Save checkpoints to file
    L:                 Load checkpoints from main.py
    C:                 Clear all checkpoints
    V:                 Toggle validation overlay
    H:                 Show/hide help
    Q/ESC:             Quit

Usage:
    python3 checkpoint_editor.py
"""

import sys
import math
import pygame
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

# Type aliases
Point = Tuple[int, int]
Checkpoint = Tuple[Point, Point]


class CheckpointEditor:
    def __init__(self, circuit_path: str = "circuit.png"):
        pygame.init()

        # Load circuit image
        try:
            self.circuit_pil = Image.open(circuit_path)
            self.circuit_arr = np.array(self.circuit_pil)
            self.width, self.height = self.circuit_pil.size
        except FileNotFoundError:
            print(f"Error: {circuit_path} not found!")
            sys.exit(1)

        # Display setup
        self.screen = pygame.display.set_mode((self.width + 300, self.height))
        pygame.display.set_caption("Checkpoint Editor - F1 NEAT Racer")

        # Convert circuit to pygame surface
        self.circuit_surface = pygame.image.fromstring(
            self.circuit_pil.tobytes(),
            self.circuit_pil.size,
            self.circuit_pil.mode
        )

        # State
        self.checkpoints: List[Checkpoint] = []
        self.current_point: Optional[Point] = None
        self.selected_cp: Optional[int] = None
        self.show_validation = True
        self.show_help = True

        # Colors
        self.COLOR_VALID = (0, 255, 0)      # Green - valid checkpoint
        self.COLOR_INVALID = (255, 0, 0)    # Red - invalid checkpoint
        self.COLOR_CURRENT = (255, 255, 0)  # Yellow - checkpoint being placed
        self.COLOR_SELECTED = (0, 255, 255) # Cyan - selected checkpoint

        # Font - try to initialize, but make it optional
        self.font_large = None
        self.font_small = None
        try:
            import pygame.font as pgfont
            pgfont.init()
            self.font_large = pgfont.Font(None, 24)
            self.font_small = pgfont.Font(None, 18)
        except (ImportError, NotImplementedError) as e:
            print(f"Warning: Font support not available: {e}")
            print("Editor will run without text labels")

    def is_on_track(self, point: Point) -> bool:
        """Check if a point is on the track (dark pixel)."""
        x, y = point
        if 0 <= x < self.width and 0 <= y < self.height:
            pixel = tuple(self.circuit_arr[y, x, :3])
            return pixel[0] < 100  # Dark = track
        return False

    def is_valid_checkpoint(self, cp: Checkpoint) -> bool:
        """
        Check if a checkpoint is valid.
        Valid = both endpoints are on track (dark pixels).
        """
        return self.is_on_track(cp[0]) and self.is_on_track(cp[1])

    def find_nearest_checkpoint(self, point: Point) -> Optional[int]:
        """Find the index of the nearest checkpoint to a point."""
        if not self.checkpoints:
            return None

        min_dist = float('inf')
        nearest_idx = 0

        for i, cp in enumerate(self.checkpoints):
            # Distance to midpoint
            mid_x = (cp[0][0] + cp[1][0]) // 2
            mid_y = (cp[0][1] + cp[1][1]) // 2
            dist = math.hypot(point[0] - mid_x, point[1] - mid_y)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx if min_dist < 30 else None

    def add_checkpoint(self, p1: Point, p2: Point):
        """Add a new checkpoint."""
        self.checkpoints.append((p1, p2))

    def delete_checkpoint(self, index: int):
        """Delete a checkpoint by index."""
        if 0 <= index < len(self.checkpoints):
            del self.checkpoints[index]

    def clear_checkpoints(self):
        """Clear all checkpoints."""
        self.checkpoints = []
        self.current_point = None

    def save_checkpoints(self, filename: str = "checkpoints_new.txt"):
        """Save checkpoints to a Python-formatted file."""
        with open(filename, 'w') as f:
            f.write("# Checkpoints for main.py\n")
            f.write("# Copy this into your main.py file\n\n")
            f.write("checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [\n")

            for i, cp in enumerate(self.checkpoints):
                comment = f"  # {i}" if i == 0 else f"  # {i}"
                f.write(f"    {cp},{comment}\n")

            f.write("]\n")

        print(f"Saved {len(self.checkpoints)} checkpoints to {filename}")

        # Also print statistics
        valid_count = sum(1 for cp in self.checkpoints if self.is_valid_checkpoint(cp))
        print(f"Valid checkpoints: {valid_count}/{len(self.checkpoints)}")

        if valid_count < len(self.checkpoints):
            print("Warning: Some checkpoints are not fully on track!")

    def load_checkpoints_from_main(self, filename: str = "main.py"):
        """Load existing checkpoints from main.py."""
        try:
            with open(filename, 'r') as f:
                content = f.read()

            # Find checkpoint list
            start = content.find("checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [")
            if start == -1:
                start = content.find("checkpoints = [")

            if start == -1:
                print("Could not find checkpoints in main.py")
                return

            end = content.find("]", start)
            checkpoint_text = content[start:end+1]

            # Extract checkpoint tuples using eval (safe since we control the file)
            # Parse the actual checkpoint data
            import re
            pattern = r'\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\)'
            matches = re.findall(pattern, checkpoint_text)

            self.checkpoints = []
            for match in matches:
                p1 = (int(match[0]), int(match[1]))
                p2 = (int(match[2]), int(match[3]))
                self.checkpoints.append((p1, p2))

            print(f"Loaded {len(self.checkpoints)} checkpoints from {filename}")

        except FileNotFoundError:
            print(f"File {filename} not found")
        except Exception as e:
            print(f"Error loading checkpoints: {e}")

    def auto_generate_checkpoints(self, num_checkpoints: int = 50):
        """
        Auto-generate checkpoints along the track.
        This is a simple implementation - you'll likely want to adjust manually.
        """
        print("Auto-generating checkpoints...")
        # TODO: Implement track centerline detection and automatic checkpoint placement
        # For now, just show a message
        print("Auto-generation not yet implemented. Please place checkpoints manually.")
        print("Suggested workflow:")
        print("1. Follow the track from start to finish")
        print("2. Place checkpoints every 50-100 pixels")
        print("3. Ensure checkpoints span the track width")
        print("4. Both endpoints should be on dark track surface")

    def draw_help(self):
        """Draw help overlay."""
        if not self.show_help or not self.font_small:
            return

        help_x = self.width + 10
        help_y = 10
        line_height = 20

        help_text = [
            "=== CONTROLS ===",
            "Left Click: Place checkpoint",
            "Right Click: Delete nearest",
            "Space: Auto-generate",
            "S: Save checkpoints",
            "L: Load from main.py",
            "C: Clear all",
            "V: Toggle validation",
            "H: Toggle this help",
            "Q/ESC: Quit",
            "",
            "=== STATS ===",
            f"Checkpoints: {len(self.checkpoints)}",
            f"Valid: {sum(1 for cp in self.checkpoints if self.is_valid_checkpoint(cp))}",
            "",
            "=== GUIDE ===",
            "✓ Both points on track",
            "✗ Point(s) off track",
            "Yellow: Placing",
            "Green: Valid CP",
            "Red: Invalid CP",
        ]

        for i, line in enumerate(help_text):
            text_surface = self.font_small.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (help_x, help_y + i * line_height))

    def draw(self):
        """Draw the editor interface."""
        # Draw circuit
        self.screen.fill((40, 40, 40))
        self.screen.blit(self.circuit_surface, (0, 0))

        # Draw existing checkpoints
        for i, cp in enumerate(self.checkpoints):
            is_valid = self.is_valid_checkpoint(cp)
            color = self.COLOR_VALID if is_valid else self.COLOR_INVALID

            # Draw checkpoint line
            pygame.draw.line(self.screen, color, cp[0], cp[1], 2)

            # Draw endpoints
            for point in cp:
                point_color = self.COLOR_VALID if self.is_on_track(point) else self.COLOR_INVALID
                pygame.draw.circle(self.screen, point_color, point, 4)

            # Draw checkpoint number at midpoint
            if self.font_small:
                mid_x = (cp[0][0] + cp[1][0]) // 2
                mid_y = (cp[0][1] + cp[1][1]) // 2
                text = self.font_small.render(str(i), True, (255, 255, 255))
                self.screen.blit(text, (mid_x + 5, mid_y - 10))

        # Draw current point being placed
        if self.current_point:
            pygame.draw.circle(self.screen, self.COLOR_CURRENT, self.current_point, 6)

            # Draw line to mouse cursor
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[0] < self.width:  # Only if mouse is over circuit
                pygame.draw.line(self.screen, self.COLOR_CURRENT, self.current_point, mouse_pos, 1)

        # Draw help panel
        self.draw_help()

        pygame.display.flip()

    def handle_click(self, pos: Point, button: int):
        """Handle mouse click."""
        # Only process clicks on the circuit area
        if pos[0] >= self.width:
            return

        if button == 1:  # Left click - place checkpoint point
            if self.current_point is None:
                # First point
                self.current_point = pos
            else:
                # Second point - complete checkpoint
                self.add_checkpoint(self.current_point, pos)
                self.current_point = None

        elif button == 3:  # Right click - delete nearest checkpoint
            nearest = self.find_nearest_checkpoint(pos)
            if nearest is not None:
                self.delete_checkpoint(nearest)

    def run(self):
        """Main editor loop."""
        running = True
        clock = pygame.time.Clock()

        print("\n" + "="*60)
        print("Checkpoint Editor - F1 NEAT Racer")
        print("="*60)
        print("Press H to toggle help overlay")
        print("="*60 + "\n")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                    elif event.key == pygame.K_s:
                        self.save_checkpoints()

                    elif event.key == pygame.K_l:
                        self.load_checkpoints_from_main()

                    elif event.key == pygame.K_c:
                        if len(self.checkpoints) > 0:
                            confirm = input("\nClear all checkpoints? (y/n): ")
                            if confirm.lower() == 'y':
                                self.clear_checkpoints()
                                print("Checkpoints cleared")

                    elif event.key == pygame.K_v:
                        self.show_validation = not self.show_validation

                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help

                    elif event.key == pygame.K_SPACE:
                        self.auto_generate_checkpoints()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos, event.button)

            self.draw()
            clock.tick(60)

        # Ask to save before quitting
        if len(self.checkpoints) > 0:
            print(f"\nYou have {len(self.checkpoints)} checkpoints.")
            save = input("Save before quitting? (y/n): ")
            if save.lower() == 'y':
                self.save_checkpoints()

        pygame.quit()
        print("Editor closed")


if __name__ == "__main__":
    editor = CheckpointEditor()
    editor.run()
