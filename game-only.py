import pygame
from pygame.locals import *
from car import Car
from PIL import Image
import numpy as np

WIDTH, HEIGHT = 800,600
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 RACE")
running = True
clock = pygame.time.Clock()

def find_midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
def find_midpoint_line(cp):
    return ((cp[0][0] + cp[1][0]) // 2, (cp[0][1] + cp[1][1]) // 2)

checkpoints = [((9, 376), (93, 380)), ((117, 290), (189, 345)), ((112, 219), (195, 211)), ((185, 123), (219, 212)), ((255, 190), (313, 145)), ((356, 232), (299, 307)), ((594, 235), (659, 293)), ((557, 201), (632, 173)), ((581, 115), (629, 174)), ((677, 120), (629, 174)), ((727, 269), (792, 240)), ((729, 474), (788, 536)), ((645, 437), (675, 522)), ((600, 520), (670, 578)), ((327, 514), (324, 591))]

test_car = Car(find_midpoint(checkpoints[0][0], checkpoints[0][1]))
steering = 0
acceleration = 0


pilImage = Image.open("circuit.png")
background_image = pygame.image.fromstring(pilImage.tobytes(), pilImage.size, pilImage.mode)
bg_array = np.array(pilImage)

current_checkpoint = 0

pygame.init()
while running:
    screen.blit(background_image, (0,0))
    if test_car.check_off_track(bg_array):
        test_car.reset(find_midpoint_line(checkpoints[current_checkpoint]))
        print("Car reset due to off-track!")
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[K_LEFT]:
        steering  = 2
    elif keys[K_RIGHT]:
        steering  = -2
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
    test_car.dist_to_checkpoint(checkpoints[(current_checkpoint+1)%len(checkpoints)], screen)
    if test_car.get_collide_checkpoint(checkpoints[current_checkpoint+1 % len(checkpoints)]):
        current_checkpoint = (current_checkpoint + 1) % len(checkpoints)
        test_car.starting_pos = find_midpoint(checkpoints[current_checkpoint][0], checkpoints[current_checkpoint][1])
        print(f"Checkpoint reached! Next: {current_checkpoint}")

    for cp in checkpoints:
        pygame.draw.line(screen, (0,255,0), cp[0], cp[1], 2)
    
    
    pygame.display.flip()
    clock.tick(FPS)
pygame.quit()