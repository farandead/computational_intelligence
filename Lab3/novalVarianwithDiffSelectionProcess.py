import random
import pandas as pd
import math
import time
# Load the CSV data into a pandas DataFrame
df = pd.read_csv('ulysses16.csv')
coordinates = [(row.x, row.y) for index, row in df.iterrows()]
def initialize_population(pop_size, data):
    """Create an initial population of diverse routes."""
    population = [random.sample(list(range(len(data))), len(data)) for _ in range(pop_size - 1)]
    # Add a nearest-neighbor route to start with a good candidate
    nn_route = [0] + random.sample(list(range(1, len(data))), len(data) - 1)
    population.append(nn_route)
    return population

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
        return distance

graph = Graph(coordinates)

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

def selection(population, num_parents):
    # First, let's implement elitism
    elite_size = int(0.1 * len(population))  # 10% of the population
    population_sorted = sorted(population, key=graph.evaluation)
    elites = population_sorted[:elite_size]
    
    # Rest of the slots will be filled using Roulette Wheel Selection
    fitness_values = [1 / graph.evaluation(route) for route in population] # inverse because we want to minimize distance
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    
    selected = elites[:]
    
    while len(selected) < num_parents:
        pick = random.uniform(0, 1)
        current = 0
        for i, individual in enumerate(population):
            current += selection_probs[i]
            if current > pick:
                if individual not in selected: # Ensure unique selection
                    selected.append(individual)
                break
                
    return selected


def genetic_algorithm(num_generations=10000000, pop_size=50, time_limit=3):
    start_time = time.time()  # Record the start time

    num_cities = len(coordinates)
    population = [random.sample(list(range(num_cities)), num_cities) for _ in range(pop_size)]
  

    for generation in range(num_generations):
        current_time = time.time()
        if (current_time - start_time) > time_limit:
            print("Time limit reached. Stopping at generation:", generation)
            break 
        # Ensure that the number of parents selected is even
        num_parents = (pop_size // 2) if (pop_size // 2) % 2 == 0 else (pop_size // 2) - 1
        parents = selection(population, num_parents)
        
        children = []
        for i in range(0, len(parents) - 1, 2):
            child1 = crossover(parents[i], parents[i+1])
            child2 = crossover(parents[i+1], parents[i])
            children.extend([mutate(child1), mutate(child2)])

        # If the length of the population is odd, choose a random individual from parents to add to children
        if (len(parents) + len(children)) % 2 != 0:
            children.append(random.choice(parents))

        population = parents + children

    best_route = min(population, key=graph.evaluation)
    return best_route, graph.evaluation(best_route)

# Running the GA
best_route, best_distance = genetic_algorithm()
print(f"Best Route: {best_route}")
print(f"Distance: {best_distance}")