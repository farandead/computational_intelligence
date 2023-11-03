import random
import math
import pandas as pd
class Graph:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def set_distance(self, index1, index2, distance):
        # This is not necessary with coordinates; distances are calculated on the fly.
        pass

    def get_distance(self, index1, index2):
        x1, y1 = self.coordinates[index1]
        x2, y2 = self.coordinates[index2]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def get_cost_of_route(self, route):
        total_cost = 0
        for i in range(len(route.route) - 1):
            index1 = route.route[i]
            index2 = route.route[i+1]
            total_cost += self.get_distance(index1, index2)
        # Optionally close the loop:
        # total_cost += self.get_distance(route.route[-1], route.route[0])
        return total_cost
    
class Route:
    def __init__(self, num_cities):
        self.route = list(range(num_cities))  # List of indices

    @classmethod
    def generate_random_route(cls, num_cities):
        route = cls(num_cities)
        random.shuffle(route.route)
        return route

    def __str__(self):
        return "->".join(str(i) for i in self.route)     
    

# num_random_routes = 10


# g = Graph(4)
# g.set_distance('A', 'B', 20)
# g.set_distance('A', 'C', 42)
# g.set_distance('A', 'D', 35)
# g.set_distance('B', 'C', 30)
# g.set_distance('B', 'D', 34)
# g.set_distance('C', 'D', 12)

# r = Route()
# r.add_city('A')
# r.add_city('B')
# r.add_city('C')
# r.add_city('D')


# for _ in range(num_random_routes):
#     random_route = Route.generate_random_route(g.cities)
#     print(f"Random Route: {random_route}, Cost: {g.get_cost_of_route(random_route)}")

df = pd.read_csv('ulysses16.csv')
coordinates = [(row.x, row.y) for index, row in df.iterrows()]

# Initialize graph with coordinates
g = Graph(coordinates)

# Generate random routes and calculate costs
num_random_routes = 10

for _ in range(num_random_routes):
    random_route = Route.generate_random_route(len(coordinates))
    print(f"Random Route: {random_route}, Cost: {g.get_cost_of_route(random_route)}")