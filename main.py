import pygame, sys, pickle, pandas as pd, math
from pygame.locals import *
from car import Car
from PIL import Image
import neat, time
import numpy as np

WIDTH, HEIGHT = 800,600
FPS = 100

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 RACE")
running = True
clock = pygame.time.Clock()

pygame.init()

def find_midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
def find_midpoint_line(cp):
    return ((cp[0][0] + cp[1][0]) // 2, (cp[0][1] + cp[1][1]) // 2)

checkpoints = [((9, 376), (93, 380)), ((117, 290), (189, 345)), ((112, 219), (195, 211)), ((185, 123), (219, 212)), ((255, 190), (313, 145)), ((356, 232), (299, 307)), ((594, 235), (659, 293)), ((557, 201), (632, 173)), ((581, 115), (629, 174)), ((677, 120), (629, 174)), ((727, 269), (792, 240)), ((729, 474), (788, 536)), ((645, 437), (675, 522)), ((600, 520), (670, 578)), ((327, 514), (324, 591))]

current_checkpoint = 0

pilImage = Image.open("circuit.png")
background_image = pygame.image.fromstring(pilImage.tobytes(), pilImage.size, pilImage.mode)
bg_array = np.array(pilImage)

VISUALIZE = False
FITNESS_DATA = [] # for analytics purposes
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    "config-feedforward.txt")

def get_inputs(car): # experiment with these, many unused functions in the Snake class 
    return [
        *car.vision(bg_array),
        car.speed,
        car.angle,
        *car.pos_relative_to_next_cp(checkpoints[(current_checkpoint+1)%len(checkpoints)]),

    ]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        car = Car(find_midpoint(checkpoints[0][0], checkpoints[0][1]))

        fitness = 0
        running = True
        current_checkpoint = 0
        max_frames = 1000
        frames = 0

        while running and frames < max_frames:
            frames+=1
            # Get inputs for NN
            inputs = get_inputs(car)

            # Network decision
            output = net.activate(inputs)
            decision = output
            steering = decision[0]*2
            acceleration = decision[1]*5
            
            car.update(steering, acceleration)
            #print(decision)

            if car.check_off_track(bg_array):
                fitness -= 1000
                running = False

            # reach next checkpoint
            if car.get_collide_checkpoint(checkpoints[(current_checkpoint+1) % len(checkpoints)]):
                current_checkpoint = (current_checkpoint + 1) % len(checkpoints)
                fitness += 500
                frames = 0
            # penalty for going back
            if car.get_collide_checkpoint(checkpoints[(current_checkpoint-1) % len(checkpoints)]):
                current_checkpoint = (current_checkpoint - 1) % len(checkpoints)
                fitness -= 50

            # encourage shorter games
            fitness -= 1

            if VISUALIZE:
                car.draw(screen)
                car.vision(pilImage, screen=screen)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                pygame.display.flip()
                clock.tick(FPS)
                
        genome.fitness = fitness
        FITNESS_DATA.append(fitness)
        #print(f"Genome {genome_id} fitness: {fitness}")

def train(generations, filename, analytics = True):
    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))

    winner = p.run(eval_genomes, generations) # generations

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

    # Save winner to file
    with open(filename, "wb") as f:
        pickle.dump(winner, f)

    print("Winner saved to winner.pkl")

def load_play(filename):
    with open(filename, "rb") as f:
        winner = pickle.load(f)

    # Rebuild network from genome
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    car = Car(find_midpoint(checkpoints[0][0], checkpoints[0][1]))
    if current_checkpoint == len(checkpoints)-2 and car.get_collide_checkpoint(checkpoints[0], threshold = 20):
        running = False

    running = True
    start = time.time()
    while running:
        inputs = get_inputs(car)
        
        output = net.activate(inputs)
        decision = output

        car.update(decision[0]*2, decision[1]*5)


        screen.blit(background_image, (0,0))
        if car.check_off_track(bg_array):
            running = False
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        car.draw(screen)
        car.vision(bg_array, screen=screen)
        car.dist_to_checkpoint(checkpoints[(current_checkpoint+1)%len(checkpoints)], screen)

        screen.blit(pygame.font.SysFont("Arial", 18).render(f"time: {time.time()-start}", True, (0,0,0)), (10,10))

        for cp in checkpoints:
            pygame.draw.line(screen, (0,255,0), cp[0], cp[1], 2)
        
        
        pygame.display.flip()
        clock.tick(200)
    pygame.quit() 

    print("Replay finished")
    end = time.time()
    print(f"Time taken: {end-start} seconds")


#train(100, "winner.pkl", analytics=True)
load_play("winner.pkl")