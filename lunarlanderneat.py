#200 fitness score is considered as a good score for the lunar lander game. Network graph
#and fitness score graph is plotted.


import neat
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
import graphviz
from plots import draw_neural_network_graphviz, plot_training_stats


def create_environment(render_mode=None):
    return gym.make(
        "LunarLander-v3",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode=render_mode,
    )


def run_genome(genome, config, render_mode = None):

    # Create neural network from genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create environment
    env = create_environment(render_mode=render_mode)

    observation, info = env.reset(seed=42) # Set a fixed seed for consistent behavior

    # Run simulation and calculate fitness
    total_reward = 0
    for _ in range(1000):
        action = net.activate(observation)
        action = action.index(max(action)) 
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            env.close()
            break
    return total_reward


def eval_genomes(genomes, config, render_mode = 'none'):
    for genome_id, genome in genomes:
        genome.fitness = run_genome(genome, config)




def run_neat():
    config_path = "config-feedforward.txt"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 70)
    print("\nBest Genome:\n", winner)
    with open("1LunarLanderwinner.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Winner genome saved.")

    plot_training_stats(stats)

    



def visualize_winner(winner, config_path, config):
    
    # Recreate the neural network from the winner
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Create the environment
    env = create_environment(render_mode="human")  # Enable rendering
    observation, info = env.reset(seed=42)  # Set a fixed seed for consistent behavior
    done = False
    total_reward = 0

    while not done:
        # Get the action from the neural network
        action = np.argmax(net.activate(observation))
        
        # Take the action in the environment
        observation, reward, done, truncated, info = env.step(action)
        
        # Accumulate rewards
        total_reward += reward
    
    env.close()
    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    run_neat()

    with open("1LunarLanderwinner.pkl", "rb") as f:
        winner = pickle.load(f)
    
    config_path = "config-feedforward.txt"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )


    visualize_winner(winner, config_path, config)
    draw_neural_network_graphviz(winner, config, filename="nn_lunar_lander")



