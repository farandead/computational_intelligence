import random

class Graph:
    def __init__(self, num_cities):
        self.matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
        self.cities = [chr(i + ord('A')) for i in range(num_cities)]

    def set_distance(self, city1, city2, distance):
        i, j = self.cities.index(city1), self.cities.index(city2)
        self.matrix[i][j] = distance
        self.matrix[j][i] = distance

    def get_distance(self, city1, city2):
        i, j = self.cities.index(city1), self.cities.index(city2)
        return self.matrix[i][j]
    
    def get_cost_of_route(self, route):
        total_cost = 0
        for i in range(len(route.route) - 1):
            city1 = route.route[i]
            city2 = route.route[i+1]
            total_cost += self.get_distance(city1, city2)
        return total_cost
    
class Route:
    def __init__(self):
        self.route = []

    def add_city(self, city):
        if city not in self.route:
            self.route.append(city)   
    
    @classmethod
    def generate_random_route(cls, cities):
        route = cls()
        route.route.append(cities[0])  # Starting with 'A'
        route.route.extend(random.sample(cities[1:], len(cities) - 1))
        return route
    def __str__(self):
        return "->".join(self.route)   
    

num_random_routes = 10


g = Graph(4)
g.set_distance('A', 'B', 20)
g.set_distance('A', 'C', 42)
g.set_distance('A', 'D', 35)
g.set_distance('B', 'C', 30)
g.set_distance('B', 'D', 34)
g.set_distance('C', 'D', 12)

r = Route()
r.add_city('A')
r.add_city('B')
r.add_city('C')
r.add_city('D')


for _ in range(num_random_routes):
    random_route = Route.generate_random_route(g.cities)
    print(f"Random Route: {random_route}, Cost: {g.get_cost_of_route(random_route)}")


# Code Breakdown
# construct(adjacency_matrix): Takes an adjacency matrix as input and constructs a TSP graph using the networkx library.

# construct_from_city_coords(city_coords): Constructs a TSP instance from a given list of city coordinates by calculating the distances between cities to form the adjacency matrix, then calling construct.

# display_instance(...): Visualizes the TSP instance, with optional labeling of edges and custom titles.

# is_valid_route(tsp_instance, route): Checks whether a given route is valid for a TSP instance, following the rules of visiting all cities and returning to the start.

# display_route(...): Visualizes a specific route on a TSP instance, highlighting the path of the route.

# evaluate(tsp_instance, route): Calculates the total distance of a given route in the TSP instance.

# random_route(tsp_instance): Generates a random route for the TSP instance, starting and ending at the first city.

# Comparison to Your Code
# Your code represents a basic version of the TSP without the use of an external library for graph representation. It uses a simple class structure (Graph and Route) to define the TSP problem and calculate route costs.

# The new code you provided is a more advanced and robust implementation that leverages the networkx library, which provides a wide range of functions for working with graphs. Additionally, it uses matplotlib for visualization, which your code did not include.

# Here are some specific differences:

# Graph Representation: Your code uses a 2D list to represent the adjacency matrix, while the provided code uses networkx's graph representation.
# Distance Calculation: Your code does not include distance calculation as it seems to assume that the distances are provided. The new code calculates Euclidean distances based on city coordinates.
# Validation: The new code includes a function to validate the routes, which your code lacks.
# Visualization: The provided code can display both the TSP instance and routes using matplotlib, offering a visual way to understand the problem and solutions.
# Random Route Generation: Both codes generate random routes, but the provided code does so in the context of networkx graphs.
# Modularity and Use of Libraries: The provided code is more modular and makes extensive use of the networkx library, which can simplify many tasks associated with graph handling and analysis.