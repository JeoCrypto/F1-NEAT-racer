import pygame
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, QUIT
from car import Car
from PIL import Image
import numpy as np

WIDTH: int = 800
HEIGHT: int = 600
FPS: int = 60

pygame.init()
screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Karting Race")
clock: pygame.time.Clock = pygame.time.Clock()
running: bool = True

def find_midpoint(point1: tuple[int, int], point2: tuple[int, int]) -> tuple[int, int]:
    """Return the pixel-wise midpoint of two (x, y) points as integers."""
    return ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

def find_midpoint_line(checkpoint: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, int]:
    """Return the pixel-wise midpoint between two (x, y) coordinates forming a checkpoint line."""
    (x1, y1), (x2, y2) = checkpoint
    return ((x1 + x2) // 2, (y1 + y2) // 2)

checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((130, 530), (210, 510)),  # start/finish straight
    ((125, 470), (195, 420)),
    ((140, 360), (205, 300)),
    ((210, 270), (275, 215)),
    ((300, 190), (375, 155)),
    ((410, 165), (495, 170)),
    ((520, 210), (590, 260)),
    ((575, 275), (645, 335)),
    ((600, 330), (670, 410)),
    ((600, 420), (650, 500)),
    ((520, 500), (580, 560)),
    ((420, 520), (500, 550)),
    ((320, 520), (380, 545)),
    ((230, 500), (300, 540)),
    ((150, 485), (215, 525)),
]

test_car: Car = Car(find_midpoint(checkpoints[0][0], checkpoints[0][1]))
steering: int = 0
acceleration: int = 0

pil_image: Image.Image = Image.open("circuit.png")
background_image: pygame.Surface = pygame.image.fromstring(
    pil_image.tobytes(), pil_image.size, pil_image.mode
)
bg_array: np.ndarray = np.array(pil_image)

current_checkpoint: int = 0

while running:
    screen.blit(background_image, (0, 0))

    if test_car.check_off_track(bg_array):
        reset_midpoint = find_midpoint_line(checkpoints[current_checkpoint])
        next_cp_idx = (current_checkpoint + 1) % len(checkpoints)
        next_midpoint = find_midpoint_line(checkpoints[next_cp_idx])
        test_car.reset(reset_midpoint, heading_target=next_midpoint)
        print("Car reset due to off-track!")

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[K_LEFT]:
        steering = 2
    elif keys[K_RIGHT]:
        steering = -2
    else:
        steering = 0

    if keys[K_UP]:
        acceleration = 1
    elif keys[K_DOWN]:
        acceleration = -1
    else:
        acceleration = 0

    steering = max(-10, min(steering, 10))
    test_car.update(steering, acceleration)
    test_car.draw(screen)
    test_car.vision(bg_array, screen=screen)
    test_car.dist_to_checkpoint(checkpoints[(current_checkpoint + 1) % len(checkpoints)], screen)

    if test_car.get_collide_checkpoint(checkpoints[(current_checkpoint + 1) % len(checkpoints)]):
        current_checkpoint = (current_checkpoint + 1) % len(checkpoints)
        test_car.starting_pos = find_midpoint(
            checkpoints[current_checkpoint][0], checkpoints[current_checkpoint][1]
        )
        print(f"Checkpoint reached! Next: {current_checkpoint}")

    for cp in checkpoints:
        pygame.draw.line(screen, (0, 255, 0), cp[0], cp[1], 2)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()