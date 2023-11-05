import random
import pandas as pd
import math
import time

def crossover(parent1, parent2):
    """Order 1 Crossover."""
    child = [''] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]

    for i, city in enumerate(parent2):
        if city not in child:
            for j in range(len(child)):
                if child[j] == '':
                    child[j] = city
                    break
    return child

def mutate(route, mutation_rate=0.05):
    """Swap Mutation."""
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route
# Improved Genetic Algorithm for TSP
def initialize_population(pop_size, data):
    """Create an initial population of diverse routes."""
    population = [random.sample(list(range(len(data))), len(data)) for _ in range(pop_size - 1)]
    # Add a nearest-neighbor route to start with a good candidate
    nn_route = [0] + random.sample(list(range(1, len(data))), len(data) - 1)
    population.append(nn_route)
    return population

def tournament_selection(population, tournament_size):
    """Select parents using tournament selection."""
    selected = []
    for _ in range(tournament_size):
        contenders = random.sample(population, tournament_size)
        winner = min(contenders, key=graph.evaluation)
        selected.append(winner)
    return selected

# We can use the same crossover and mutate functions as before
# Modify the genetic_algorithm function to use improved selection and initialization
def genetic_algorithm_improved(pop_size=50, time_limit=60):
    start_time = time.time()
    population = initialize_population(pop_size, coordinates)
    best_distance = float('inf')
    best_route = None

    while time.time() - start_time < time_limit:
        # Tournament selection
        tournament_size = max(2, int(0.2 * len(population)))  # At least 2 and 20% of population size
        parents = tournament_selection(population, tournament_size)
        
        # Generate children from parents using crossover and mutation
        children = []
        while len(children) < len(population) - len(parents):
            parent1, parent2 = random.sample(parents, 2)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            children.append(mutate(child1))
            children.append(mutate(child2))
        
        # Combine parents and children to form new population
        population = parents + children
        
        # Check for new best route
        current_best_route = min(population, key=graph.evaluation)
        current_best_distance = graph.evaluation(current_best_route)
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route

    return best_route, best_distance
# Since we don't have access to the original CSV file in this context, I will redefine the Graph class
# and the coordinates from the data we loaded earlier.


# Redefine the Graph class to use the coordinates
class Graph:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def get_distance(self, index1, index2):
        x1, y1 = self.coordinates[index1]
        x2, y2 = self.coordinates[index2]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def evaluation(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.get_distance(route[i], route[i+1])
        # Add distance from last to first to complete the cycle
        distance += self.get_distance(route[-1], route[0])
        return distance
df = pd.read_csv('ulysses16.csv')
coordinates = [(row.x, row.y) for index, row in df.iterrows()]
# Instantiate the Graph with the new coordinates
graph = Graph(coordinates)

# Now, we can attempt to run the improved genetic algorithm again.
best_route, best_distance = genetic_algorithm_improved(pop_size=50, time_limit=3)
print(best_route, best_distance)

# Running the improved GA
best_route, best_distance = genetic_algorithm_improved()
best_route, best_distance
print(best_route, best_distance)
