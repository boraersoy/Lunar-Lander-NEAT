import graphviz
import matplotlib.pyplot as plt
import numpy as np

def draw_neural_network_graphviz(winner, config, filename="neural_net"):

    input_labels = [
        "X Position", "Y Position", "X Velocity", "Y Velocity",
        "Angle", "Angular Velocity", "Left Leg Contact", "Right Leg Contact"
    ]
    output_labels = ["Do Nothing", "Fire Left", "Fire Main", "Fire Right"]
    
    dot = graphviz.Digraph(format="png", comment="Neural Network")
    
    for i, label in enumerate(input_labels, start=-len(input_labels)):
        dot.node(f"Input {i}", label=label, shape="circle", color="lightblue")
    
    for i, label in enumerate(output_labels):
        dot.node(f"Output {i}", label=label, shape="circle", color="lightgreen")
    
    for node in winner.nodes:
        if node >= 0:  # Hidden nodes have non-negative keys
            dot.node(f"Hidden {node}", label=f"Hidden {node}", shape="circle", color="lightyellow")
    
    # Add edges
    for conn_key, conn in winner.connections.items():
        if conn.enabled:
            style = "solid"
        else:
            style = "dashed"
        from_node = (
            f"Input {conn_key[0]}" if conn_key[0] < 0 else f"Hidden {conn_key[0]}"
        )
        to_node = (
            f"Output {conn_key[1]}" if conn_key[1] >= 0 else f"Hidden {conn_key[1]}"
        )
        dot.edge(from_node, to_node, label=f"{conn.weight:.2f}", style=style)
    
    dot.render(filename, view=True)



def plot_training_stats(stats, filename="training_stats.png"):

    generations = range(len(stats.most_fit_genomes))
    best_fitness = [genome.fitness for genome in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()
    stdev_fitness = stats.get_fitness_stdev()

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, avg_fitness, label="Average Fitness")
    plt.fill_between(
        generations,
        [avg - std for avg, std in zip(avg_fitness, stdev_fitness)],
        [avg + std for avg, std in zip(avg_fitness, stdev_fitness)],
        color="gray",
        alpha=0.3,
        label="Fitness Std Dev"
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

