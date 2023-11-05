import random
import pandas as pd
import math
# from Lab2.LocalSearchWith2Opt import Graph
# Load the CSV data into a pandas DataFrame
df = pd.read_csv('ulysses16.csv')
coordinates = [(row.x, row.y) for index, row in df.iterrows()]

class Graph:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def set_distance(self, index1, index2, distance):
        pass

    def get_distance(self, index1, index2):
        x1, y1 = self.coordinates[index1]
        x2, y2 = self.coordinates[index2]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def get_cost_of_route(self, route):
        total_cost = 0
        for i in range(len(route.route)):
            index1 = route.route[i-1]
            index2 = route.route[i]
            total_cost += self.get_distance(index1, index2)
        return total_cost

graph = Graph(coordinates)

def evaluation(route):
    """Calculate the total distance of a route."""
    distance = 0
    for i in range(len(route) - 1):
        distance += graph.get_distance(route[i], route[i+1])
    return distance

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
    """Tournament Selection."""
    selected = []
    while len(selected) < num_parents:
        individuals = random.sample(population, 3)
        best_individual = min(individuals, key=evaluation)
        if best_individual not in selected: # Ensure unique selection
            selected.append(best_individual)
    return selected


def genetic_algorithm(num_generations=1000, pop_size=50):
    num_cities = len(coordinates)
    population = [random.sample(list(range(num_cities)), num_cities) for _ in range(pop_size)]

    for generation in range(num_generations):
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

    best_route = min(population, key=evaluation)
    return best_route, evaluation(best_route)
# Running the GA
best_route, best_distance = genetic_algorithm()
print(f"Best Route: {best_route}")
print(f"Distance: {best_distance}")