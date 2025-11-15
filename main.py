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
    ((393, 376), (401, 391)),  # start/finish
    ((298, 426), (297, 446)),
    ((244, 450), (249, 473)),
    ((210, 503), (208, 528)),
    ((126, 496), (101, 502)),
    ((56, 415), (37, 422)),
    ((51, 384), (29, 393)),
    ((62, 383), (77, 369)),
    ((121, 428), (110, 444)),
    ((149, 437), (159, 452)),
    ((243, 393), (247, 409)),
    ((263, 348), (254, 325)),
    ((290, 319), (291, 337)),
    ((330, 333), (336, 348)),
    ((404, 312), (419, 322)),
    ((455, 290), (466, 302)),
    ((462, 271), (479, 276)),
    ((441, 250), (438, 269)),
    ((405, 275), (402, 286)),
    ((380, 282), (369, 295)),
    ((288, 281), (282, 291)),
    ((240, 279), (230, 292)),
    ((209, 307), (194, 306)),
    ((213, 327), (200, 335)),
    ((217, 348), (203, 358)),
    ((219, 373), (204, 371)),
    ((184, 374), (181, 386)),
    ((150, 372), (148, 383)),
    ((124, 338), (113, 350)),
    ((120, 309), (102, 319)),
    ((126, 281), (113, 277)),
    ((158, 262), (137, 257)),
    ((314, 181), (294, 171)),
    ((260, 209), (234, 204)),
    ((224, 229), (194, 227)),
    ((384, 141), (361, 138)),
    ((423, 115), (398, 110)),
    ((455, 98), (434, 92)),
    ((489, 90), (469, 73)),
    ((501, 98), (506, 79)),
    ((517, 114), (529, 108)),
    ((526, 138), (542, 130)),
    ((508, 156), (504, 172)),
    ((477, 137), (475, 156)),
    ((452, 163), (436, 162)),
    ((465, 203), (452, 210)),
    ((481, 204), (487, 219)),
    ((515, 196), (515, 210)),
    ((545, 192), (543, 203)),
    ((579, 203), (572, 211)),
    ((597, 239), (579, 251)),
    ((594, 269), (567, 274)),
    ((580, 288), (539, 296)),
    ((544, 311), (491, 320)),
    ((490, 342), (454, 341)),
    ((353, 399), (339, 423)),
]

# Starting position - on the grid before the start/finish line
STARTING_POSITION = (422, 369)  # Grid position set via checkpoint helper
STARTING_ANGLE = 241.0  # Calculated angle pointing toward checkpoint 0

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
        max_frames = 1000
        frames = 0

        while running and frames < max_frames:
            frames += 1
            next_cp = (curr_cp + 1) % len(checkpoints)
            inputs = get_inputs(car, next_cp)
            output = net.activate(inputs)
            steering: float = output[0] * 2
            acceleration: float = output[1] * 5
            car.update(steering, acceleration)

            # Give car a few frames to start moving before checking off-track
            if frames > 5 and car.check_off_track(bg_array):
                fitness -= 1000
                running = False

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

            # Reward for staying on track (small positive reward per frame)
            fitness += 0.5

            # Small reward for moving forward (based on speed)
            fitness += abs(car.speed) * 0.1

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