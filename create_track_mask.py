#!/usr/bin/env python3
"""
Track Mask Generator

This tool creates a clean binary mask from the circuit image where:
- Black (0, 0, 0) = Valid track surface (where the car can drive)
- White (255, 255, 255) = Off-track/barriers (where the car should not go)

This removes all visual clutter (grid lines, text, corner numbers) and provides
a clean reference for the AI to learn track boundaries.

Usage:
    python create_track_mask.py [--interactive]

Options:
    --interactive    Launch interactive mode to manually paint track boundaries
"""

import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
import pygame


def create_mask_from_color_range(image_path: str, output_path: str = "track_mask.png") -> None:
    """
    Create a track mask by detecting track colors (darker grays/blacks) vs barriers.

    Track pixels (dark colors) -> Black in mask
    Barrier pixels (light colors, white) -> White in mask
    """
    print("Loading circuit image...")
    img = Image.open(image_path)
    arr = np.array(img)

    # Create mask (default to white = off-track)
    mask = np.ones((arr.shape[0], arr.shape[1], 3), dtype=np.uint8) * 255

    # Define track color range - typically dark grays and blacks on asphalt
    # We want to be permissive with track colors but exclude white markings

    # Convert to grayscale for easier thresholding
    if len(arr.shape) == 3:
        gray = np.mean(arr[:, :, :3], axis=2)
    else:
        gray = arr

    # Track is typically darker (asphalt, painted lines that aren't white)
    # Threshold: pixels darker than 180 are considered track
    # This excludes white (255) and light gray barriers
    track_pixels = gray < 180

    # Set track pixels to black
    mask[track_pixels] = [0, 0, 0]

    # Save mask
    mask_img = Image.fromarray(mask)
    mask_img.save(output_path)
    print(f"Track mask saved to {output_path}")

    # Print statistics
    track_percentage = (np.sum(track_pixels) / track_pixels.size) * 100
    print(f"Track coverage: {track_percentage:.1f}%")

    return mask_img


def create_mask_interactive(image_path: str, output_path: str = "track_mask.png") -> None:
    """
    Interactive mode: Display the circuit and let user paint the track boundaries.

    Controls:
    - Left Click + Drag: Mark as TRACK (black)
    - Right Click + Drag: Mark as BARRIER (white)
    - S: Save mask
    - R: Reset to auto-generated mask
    - Q/ESC: Quit
    """
    pygame.init()

    # Load circuit image
    circuit_img = Image.open(image_path)
    circuit_arr = np.array(circuit_img)

    # Create initial mask from color range
    print("Generating initial mask from color detection...")
    mask_img = create_mask_from_color_range(image_path, output_path)
    mask_arr = np.array(mask_img)

    # Setup pygame display
    width, height = circuit_img.size
    screen = pygame.display.set_mode((width * 2, height))
    pygame.display.set_caption("Track Mask Editor - Left: Circuit | Right: Mask")

    # Convert to pygame surfaces
    circuit_surface = pygame.image.fromstring(circuit_img.tobytes(), circuit_img.size, circuit_img.mode)

    brush_size = 10
    drawing = False
    draw_color = (0, 0, 0)  # Black = track

    print("\n=== Interactive Track Mask Editor ===")
    print("Left side: Original circuit")
    print("Right side: Track mask (Black=track, White=barrier)")
    print("\nControls:")
    print("  Left Click + Drag:  Paint TRACK (black)")
    print("  Right Click + Drag: Paint BARRIER (white)")
    print("  Mouse Wheel:        Change brush size")
    print("  S:                  Save mask")
    print("  R:                  Reset to auto-generated")
    print("  Q/ESC:              Quit")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # Save mask
                    Image.fromarray(mask_arr).save(output_path)
                    print(f"Mask saved to {output_path}")
                elif event.key == pygame.K_r:
                    # Reset mask
                    print("Resetting to auto-generated mask...")
                    mask_img = create_mask_from_color_range(image_path, output_path)
                    mask_arr = np.array(mask_img)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    drawing = True
                    draw_color = (0, 0, 0)  # Track
                elif event.button == 3:  # Right click
                    drawing = True
                    draw_color = (255, 255, 255)  # Barrier

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in [1, 3]:
                    drawing = False

            elif event.type == pygame.MOUSEWHEEL:
                brush_size = max(5, min(50, brush_size + event.y * 2))
                print(f"Brush size: {brush_size}")

        # Handle drawing
        if drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Only draw on right side (mask)
            if mouse_x >= width:
                # Convert to mask coordinates
                mask_x = mouse_x - width
                mask_y = mouse_y

                # Paint circular area
                for dx in range(-brush_size, brush_size + 1):
                    for dy in range(-brush_size, brush_size + 1):
                        if dx*dx + dy*dy <= brush_size*brush_size:
                            px = mask_x + dx
                            py = mask_y + dy
                            if 0 <= px < width and 0 <= py < height:
                                mask_arr[py, px] = draw_color

        # Render
        screen.fill((50, 50, 50))
        screen.blit(circuit_surface, (0, 0))

        # Convert mask to pygame surface and display
        mask_surface = pygame.surfarray.make_surface(mask_arr.swapaxes(0, 1))
        screen.blit(mask_surface, (width, 0))

        # Draw brush preview
        if pygame.mouse.get_focused():
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (255, 0, 0), (mouse_x, mouse_y), brush_size, 2)

        pygame.display.flip()

    pygame.quit()

    # Final save
    final_save = input("\nSave final mask before exiting? (y/n): ")
    if final_save.lower() == 'y':
        Image.fromarray(mask_arr).save(output_path)
        print(f"Final mask saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate track mask from circuit image")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive editor to manually refine the mask"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="circuit.png",
        help="Input circuit image (default: circuit.png)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="track_mask.png",
        help="Output mask image (default: track_mask.png)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=180,
        help="Brightness threshold for track detection (default: 180)"
    )

    args = parser.parse_args()

    try:
        if args.interactive:
            create_mask_interactive(args.input, args.output)
        else:
            create_mask_from_color_range(args.input, args.output)
            print("\nTip: Use --interactive to manually refine the mask if needed")
    except FileNotFoundError:
        print(f"Error: Could not find {args.input}")
        print("Make sure the circuit image exists in the current directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
