import pandas as pd
import math
import random
# Load the data from the CSV file
df = pd.read_csv('ulysses16.csv')

# Display the first few rows of the dataframe
print(df.head())



class Graph:
    def __init__(self, coordinates):
        num_cities = len(coordinates)
        self.matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
        self.cities = coordinates

    def calculate_distances(self):
        for i, (x1, y1) in enumerate(self.cities):
            for j, (x2, y2) in enumerate(self.cities):
                if i != j:
                    self.matrix[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def set_distance(self, city1, city2, distance):
        i, j = self.cities.index(city1), self.cities.index(city2)
        self.matrix[i][j] = distance
        self.matrix[j][i] = distance

    def get_distance(self, index1, index2):
        x1, y1 = self.coordinates[index1]
        x2, y2 = self.coordinates[index2]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    
    def get_cost_of_route(self, route):
        total_cost = 0
        for i in range(len(route.route) - 1):
            city1 = route.route[i]
            city2 = route.route[i+1]
            total_cost += self.get_distance(city1, city2)
        return total_cost
class Route:
    def __init__(self, num_cities):
        self.route = [i for i in range(num_cities)]
        random.shuffle(self.route)

    def __str__(self):
        return "->".join(str(city) for city in self.route)
    
# Convert the dataframe to a list of tuples (x, y) coordinates
coordinates = [(row['x'], row['y']) for index, row in df.iterrows()]

# Create a graph with these coordinates
g = Graph(coordinates)



    
# Suppose we want to generate 10 random routes
num_random_routes = 10

for _ in range(num_random_routes):
    random_route = Route(len(coordinates))
    
    cost = g.get_cost_of_route(random_route)
    
    print(f"Random Route: {random_route}, Cost: {cost}")