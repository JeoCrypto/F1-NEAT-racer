import sys
import time
import pickle
import pygame
import pandas as pd
import numpy as np
from pygame.locals import QUIT
from car import Car
from PIL import Image
import neat

WIDTH: int = 800
HEIGHT: int = 600
FPS: int = 100

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 RACE")
clock = pygame.time.Clock()

def find_midpoint(point1: tuple[int, int], point2: tuple[int, int]) -> tuple[int, int]:
    """Return the pixel-wise midpoint of two (x, y) points as integers."""
    return ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

def find_midpoint_line(checkpoint: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, int]:
    """Return the pixel-wise midpoint between two (x, y) coordinates forming a checkpoint line."""
    (x1, y1), (x2, y2) = checkpoint
    return ((x1 + x2) // 2, (y1 + y2) // 2)

checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((9, 376), (93, 380)),
    ((117, 290), (189, 345)),
    ((112, 219), (195, 211)),
    ((185, 123), (219, 212)),
    ((255, 190), (313, 145)),
    ((356, 232), (299, 307)),
    ((594, 235), (659, 293)),
    ((557, 201), (632, 173)),
    ((581, 115), (629, 174)),
    ((677, 120), (629, 174)),
    ((727, 269), (792, 240)),
    ((729, 474), (788, 536)),
    ((645, 437), (675, 522)),
    ((600, 520), (670, 578)),
    ((327, 514), (324, 591)),
]

current_checkpoint: int = 0

pil_image = Image.open("circuit.png")
background_image = pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode)
bg_array = np.array(pil_image)

VISUALIZE: bool = False
FITNESS_DATA: list = []

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-feedforward.txt"
)

def get_inputs(car: Car) -> list[float]:
    """
    Get neural network inputs for the current car state.
    """
    return [
        *car.vision(bg_array),
        car.speed,
        car.angle,
        *car.pos_relative_to_next_cp(checkpoints[(current_checkpoint + 1) % len(checkpoints)]),
    ]

def eval_genomes(
    genomes: list[tuple[int, neat.DefaultGenome]],
    config: neat.Config
) -> None:
    """
    The evaluation function for each genome in the NEAT population.
    """
    global FITNESS_DATA  # For analytics tracking
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = Car(find_midpoint(checkpoints[0][0], checkpoints[0][1]))
        fitness = 0
        running = True
        curr_cp = 0
        max_frames = 1000
        frames = 0

        while running and frames < max_frames:
            frames += 1
            inputs = get_inputs(car)
            output = net.activate(inputs)
            steering: float = output[0] * 2
            acceleration: float = output[1] * 5
            car.update(steering, acceleration)

            if car.check_off_track(bg_array):
                fitness -= 1000
                running = False

            next_cp = (curr_cp + 1) % len(checkpoints)
            prev_cp = (curr_cp - 1) % len(checkpoints)
            # Passed next checkpoint
            if car.get_collide_checkpoint(checkpoints[next_cp]):
                curr_cp = next_cp
                fitness += 500
                frames = 0

            # Penalty for going back
            if car.get_collide_checkpoint(checkpoints[prev_cp]):
                curr_cp = prev_cp
                fitness -= 50

            # Small penalty to encourage faster completion
            fitness -= 1

            if VISUALIZE:
                car.draw(screen)
                car.vision(pil_image, screen=screen)
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                pygame.display.flip()
                clock.tick(FPS)

        genome.fitness = fitness
        FITNESS_DATA.append(fitness)

def train(
    generations: int,
    filename: str,
    analytics: bool = True
) -> None:
    """
    Train NEAT for a given number of generations and save the winning genome.
    """
    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))

    winner = p.run(eval_genomes, generations)
    if analytics:
        gen = list(range(len(stats.most_fit_genomes)))
        best = [c.fitness for c in stats.most_fit_genomes]
        avg = stats.get_fitness_mean()
        stdev = stats.get_fitness_stdev()
        df = pd.DataFrame({
            "generation": gen,
            "best_fitness": best,
            "avg_fitness": avg,
            "stdev": stdev
        })
        df.to_csv("fitness_log.csv", index=False)

    with open(filename, "wb") as f:
        pickle.dump(winner, f)

    print("Winner saved to winner.pkl")

def load_play(filename: str) -> None:
    """
    Load a trained genome and play one episode using NEAT neural network.
    """
    with open(filename, "rb") as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    car = Car(find_midpoint(checkpoints[0][0], checkpoints[0][1]))

    running = True
    start_time = time.time()

    global current_checkpoint
    while running:
        inputs = get_inputs(car)
        output = net.activate(inputs)
        car.update(output[0] * 2, output[1] * 5)

        screen.blit(background_image, (0, 0))
        if car.check_off_track(bg_array):
            running = False

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        car.draw(screen)
        car.vision(bg_array, screen=screen)
        next_cp_idx = (current_checkpoint + 1) % len(checkpoints)
        car.dist_to_checkpoint(checkpoints[next_cp_idx], screen)

        elapsed = time.time() - start_time
        font = pygame.font.SysFont("Arial", 18)
        time_surface = font.render(f"time: {elapsed:.2f}", True, (0, 0, 0))
        screen.blit(time_surface, (10, 10))

        for cp in checkpoints:
            pygame.draw.line(screen, (0, 255, 0), cp[0], cp[1], 2)

        pygame.display.flip()
        clock.tick(200)

    pygame.quit()
    print("Replay finished")
    duration = time.time() - start_time
    print(f"Time taken: {duration:.2f} seconds")

# Uncomment to train:
# train(100, "winner.pkl", analytics=True)
load_play("winner.pkl")