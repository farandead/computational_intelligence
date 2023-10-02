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
