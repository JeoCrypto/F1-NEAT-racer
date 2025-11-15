"""
Utility script to (re)generate the `circuit.png` asset used by the game.

Run with:

    python tools/generate_track.py

Requirements: pillow
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter

WIDTH, HEIGHT = 800, 600
TRACK_PATH: list[tuple[int, int]] = [
    (140, 520),
    (110, 420),
    (120, 310),
    (210, 210),
    (350, 150),
    (470, 170),
    (600, 250),
    (660, 340),
    (660, 430),
    (620, 500),
    (520, 550),
    (360, 535),
    (240, 500),
    (160, 460),
    (140, 520),
]


def draw_track(draw: ImageDraw.ImageDraw) -> None:
    track_width = 90
    shadow_width = track_width + 16

    draw.line(
        TRACK_PATH,
        fill=(30, 30, 30),
        width=shadow_width,
        joint="curve",
    )
    draw.line(
        TRACK_PATH,
        fill=(62, 62, 62),
        width=track_width,
        joint="curve",
    )
    draw.line(
        TRACK_PATH,
        fill=(82, 82, 82),
        width=track_width - 18,
        joint="curve",
    )


def draw_curbs(draw: ImageDraw.ImageDraw, specs: Iterable[tuple[tuple[int, int], int]]) -> None:
    for (center_x, center_y), radius in specs:
        bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        draw.arc(bbox, 210, 310, fill=(220, 60, 60), width=12)
        draw.arc(
            (bbox[0] + 10, bbox[1] + 10, bbox[2] - 10, bbox[3] - 10),
            210,
            310,
            fill=(255, 210, 120),
            width=8,
        )


def draw_start_line(draw: ImageDraw.ImageDraw) -> None:
    base_line = [(150, 505), (200, 505)]
    draw.line(base_line, fill="white", width=8)
    for i in range(6):
        color = "black" if i % 2 == 0 else "white"
        x = 150 + i * 8
        draw.line([(x, 505), (x, 535)], fill=color, width=4)


def build_track_image() -> Image.Image:
    surface = Image.new("RGB", (WIDTH, HEIGHT), "white")
    painter = ImageDraw.Draw(surface)
    draw_track(painter)
    draw_curbs(
        painter,
        [
            ((140, 450), 80),
            ((230, 220), 85),
            ((520, 520), 70),
            ((650, 360), 85),
        ],
    )
    draw_start_line(painter)
    return surface.filter(ImageFilter.GaussianBlur(radius=0.6))


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "circuit.png"
    image = build_track_image()
    image.save(output_path)
    print(f"Saved track image to {output_path}")


if __name__ == "__main__":
    main()

