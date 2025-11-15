import sys
import time
import warnings
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
pygame.display.set_caption("Karting Race")
clock = pygame.time.Clock()


def _init_text_renderer(font_name: str = "Arial", font_size: int = 18) -> tuple[str | None, object | None]:
    """
    Initialize a text rendering backend. Prefer pygame.font, fall back to pygame.freetype.
    """
    try:
        import pygame.font as pg_font  # pylint: disable=import-outside-toplevel

        pg_font.init()
        font = pg_font.SysFont(font_name, font_size)
        return "font", font
    except (ImportError, NotImplementedError):
        pass

    try:
        import pygame.freetype as pg_freetype  # pylint: disable=import-outside-toplevel

        pg_freetype.init()
        font = pg_freetype.SysFont(font_name, font_size)
        return "freetype", font
    except (ImportError, NotImplementedError):
        warnings.warn(
            "Text overlay disabled: pygame font modules unavailable. "
            "Install SDL_ttf / freetype to enable HUD text.",
            RuntimeWarning,
        )
        return None, None


_TEXT_BACKEND, _TEXT_FONT = _init_text_renderer()


def draw_text(surface: pygame.Surface, text: str, position: tuple[int, int], color: tuple[int, int, int] = (0, 0, 0)) -> None:
    """
    Render text to the provided surface using whichever backend is available.
    """
    if _TEXT_BACKEND is None or _TEXT_FONT is None:
        return

    if _TEXT_BACKEND == "font":
        text_surface = _TEXT_FONT.render(text, True, color)  # type: ignore[call-arg]
        surface.blit(text_surface, position)
    else:
        _TEXT_FONT.render_to(surface, position, text, color)  # type: ignore[attr-defined]

def find_midpoint(point1: tuple[int, int], point2: tuple[int, int]) -> tuple[int, int]:
    """Return the pixel-wise midpoint of two (x, y) points as integers."""
    return ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

def find_midpoint_line(checkpoint: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, int]:
    """Return the pixel-wise midpoint between two (x, y) coordinates forming a checkpoint line."""
    (x1, y1), (x2, y2) = checkpoint
    return ((x1 + x2) // 2, (y1 + y2) // 2)

checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((350, 421), (354, 435)),  # 0 - start/finish
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

# Starting position - CLOSER to CP0 for easier initial learning
# Once AI learns to pass CP0, can gradually move this back
STARTING_POSITION = (375, 410)  # Much closer to CP0 (~30px away instead of 60px)
STARTING_ANGLE = -50  # Point more directly at checkpoint 0

try:
    pil_image = Image.open("circuit.png")
    background_image = pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode)
    bg_array = np.array(pil_image)
except FileNotFoundError:
    print("Error: circuit.png not found. Please ensure the file exists in the current directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading circuit image: {e}")
    sys.exit(1)

VISUALIZE: bool = False
FITNESS_DATA: list = []

try:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-feedforward.txt"
    )
except FileNotFoundError:
    print("Error: config-feedforward.txt not found. Please ensure the file exists in the current directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading NEAT configuration: {e}")
    sys.exit(1)

def get_inputs(car: Car, next_checkpoint_idx: int) -> list[float]:
    """
    Get neural network inputs for the current car state.
    """
    return [
        *car.vision(bg_array),
        car.speed,
        car.angle,
        *car.pos_relative_to_next_cp(checkpoints[next_checkpoint_idx]),
    ]

def eval_genomes(
    genomes: list[tuple[int, neat.DefaultGenome]],
    config: neat.Config
) -> None:
    """
    The evaluation function for each genome in the NEAT population.
    """
    global FITNESS_DATA, VISUALIZE  # For analytics tracking and visualization
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = Car(STARTING_POSITION, angle=STARTING_ANGLE)
        fitness = 0
        running = True
        curr_cp = 0
        max_frames = 400  # Reduced from 1000 to force progression
        frames = 0
        last_distance = None  # Track distance to checkpoint
        starting_distance = None  # Track initial distance to measure overall progress

        while running and frames < max_frames:
            frames += 1
            next_cp = (curr_cp + 1) % len(checkpoints)
            inputs = get_inputs(car, next_cp)
            output = net.activate(inputs)
            steering: float = output[0] * 2
            acceleration: float = output[1] * 5
            car.update(steering, acceleration)

            # TEMPORARILY DISABLED: Allow off-track exploration for initial learning
            # Once AI learns checkpoints, re-enable this penalty
            # if frames > 5 and car.check_off_track(bg_array):
            #     fitness -= 1000
            #     running = False

            # Calculate distance to next checkpoint for progress reward
            cp_mid_x = (checkpoints[next_cp][0][0] + checkpoints[next_cp][1][0]) / 2
            cp_mid_y = (checkpoints[next_cp][0][1] + checkpoints[next_cp][1][1]) / 2
            current_distance = np.sqrt((car.x - cp_mid_x)**2 + (car.y - cp_mid_y)**2)

            # Initialize starting distance on first frame
            if starting_distance is None:
                starting_distance = current_distance

            # Very strong reward for getting closer to checkpoint
            if last_distance is not None:
                distance_improvement = last_distance - current_distance
                fitness += distance_improvement * 5.0  # Massively increased

            # Additional reward based on proximity to checkpoint (inverse distance)
            # Being close to checkpoint is good even if not moving toward it
            proximity_reward = (200 - min(current_distance, 200)) / 10.0
            fitness += proximity_reward

            last_distance = current_distance

            prev_cp = (curr_cp - 1) % len(checkpoints)
            # Passed next checkpoint - TEMPORARILY using larger threshold for initial learning
            if car.get_collide_checkpoint(checkpoints[next_cp], threshold=35.0):  # Increased from 20 to 35
                curr_cp = next_cp
                fitness += 500
                frames = 0  # Reset frame counter for this checkpoint
                last_distance = None  # Reset distance tracking
                starting_distance = None  # Reset starting distance

            # Penalty for going back
            if car.get_collide_checkpoint(checkpoints[prev_cp], threshold=35.0):
                curr_cp = prev_cp
                fitness -= 50

            # Small reward for forward speed
            fitness += max(0, car.speed) * 0.2

            if VISUALIZE:
                screen.blit(background_image, (0, 0))

                # Draw checkpoints
                for idx, cp in enumerate(checkpoints):
                    color = (0, 255, 255) if idx == next_cp else (0, 255, 0)
                    pygame.draw.line(screen, color, cp[0], cp[1], 3 if idx == next_cp else 2)

                car.draw(screen)
                car.vision(bg_array, screen=screen)
                car.dist_to_checkpoint(checkpoints[next_cp], screen)

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
    try:
        with open(filename, "rb") as f:
            winner = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please train a model first or check the file path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading trained model: {e}")
        sys.exit(1)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    car = Car(STARTING_POSITION, angle=STARTING_ANGLE)

    running = True
    start_time = time.time()
    curr_cp = 0
    laps_completed = 0
    max_laps = 3  # Stop after completing 3 laps

    while running:
        next_cp = (curr_cp + 1) % len(checkpoints)
        inputs = get_inputs(car, next_cp)
        output = net.activate(inputs)
        car.update(output[0] * 2, output[1] * 5)

        screen.blit(background_image, (0, 0))

        # Check if car went off track
        if car.check_off_track(bg_array):
            print("Car went off track!")
            running = False

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # Check checkpoint progression
        if car.get_collide_checkpoint(checkpoints[next_cp]):
            curr_cp = next_cp
            # Check if we completed a lap
            if curr_cp == 0:
                laps_completed += 1
                print(f"Lap {laps_completed} completed!")
                if laps_completed >= max_laps:
                    print(f"Finished {max_laps} laps!")
                    running = False

        car.draw(screen)
        car.vision(bg_array, screen=screen)
        car.dist_to_checkpoint(checkpoints[next_cp], screen)

        elapsed = time.time() - start_time
        draw_text(screen, f"Time: {elapsed:.2f}s | Lap: {laps_completed + 1} | CP: {curr_cp}/{len(checkpoints)}", (10, 10))

        # Draw all checkpoints
        for idx, cp in enumerate(checkpoints):
            # Highlight the next checkpoint
            color = (0, 255, 255) if idx == next_cp else (0, 255, 0)
            pygame.draw.line(screen, color, cp[0], cp[1], 3 if idx == next_cp else 2)

        pygame.display.flip()
        clock.tick(60)  # Reduced from 200 to 60 for smoother visualization

    pygame.quit()
    print("Replay finished")
    duration = time.time() - start_time
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Laps completed: {laps_completed}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 NEAT Racer - Train or play with AI")
    parser.add_argument(
        "mode",
        choices=["train", "play"],
        help="Mode: 'train' to train a new model, 'play' to replay a trained model"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations to train (default: 100)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the best car during training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="winner.pkl",
        help="Model filename to save/load (default: winner.pkl)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        VISUALIZE = args.visualize
        print(f"Training for {args.generations} generations...")
        print(f"Visualization: {'ON' if VISUALIZE else 'OFF'}")
        train(args.generations, args.model, analytics=True)
    else:
        load_play(args.model)