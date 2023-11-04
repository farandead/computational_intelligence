import random
import math
import pandas as pd
import time
import copy
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
    
def random_search(graph, time_limit):
    start_time = time.time()
    best_route = None
    best_cost = float('inf')

    while (time.time() - start_time) < time_limit:
        # Generate a random route
        current_route = Route.generate_random_route(len(graph.coordinates))
        # Calculate the cost of the current route
        current_cost = graph.get_cost_of_route(current_route)
        print(f'{best_cost} best cost of the current best route {best_route} for random search')
        # Update the best route and cost if the current one is better
        if current_cost < best_cost:
            best_route = current_route
            best_cost = current_cost

    return best_route, best_cost

def nearest_neighbor_solution(graph):
    num_cities = len(graph.coordinates)
    start = random.choice(range(num_cities))
    unvisited = set(range(num_cities))
    unvisited.remove(start)
    tour = [start]
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda city: graph.get_distance(last, city))
        tour.append(next_city)
        unvisited.remove(next_city)
    return Route(tour)
def local_search(tsp_instance, time_limit):
    start_time = time.time()
    best_route = Route.generate_random_route(len(tsp_instance.coordinates))
    best_route_cost = tsp_instance.get_cost_of_route(best_route)
  
    while time.time() - start_time < time_limit:
        current_route = copy.deepcopy(best_route)
        neighbourhood = city_swap_neighbourhood(current_route)
        local_optimum, local_optimum_cost = find_best_neighbour(tsp_instance, neighbourhood)
        
        print(f'{best_route_cost} best cost of the current best route {best_route} for local search search')

        if local_optimum_cost < best_route_cost:
            best_route = local_optimum
            best_route_cost = local_optimum_cost
        else:
            # Local optimum reached, generate a new random route
            best_route = Route.generate_random_route(len(tsp_instance.coordinates))
            best_route_cost = tsp_instance.get_cost_of_route(best_route)


    return best_route, best_route_cost

def two_opt_swap(route, i, k):
    """Perform a 2-opt swap by reversing the order of the nodes between i and k inclusive."""
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route
def local_search_2_opt(tsp_instance, time_limit):
    start_time = time.time()
    best_route = Route.generate_random_route(len(tsp_instance.coordinates))
    best_route_cost = tsp_instance.get_cost_of_route(best_route)

    while time.time() - start_time < time_limit:
        improved = False
        for i in range(1, len(best_route.route) - 2):
            for k in range(i + 1, len(best_route.route)):
                new_route = two_opt_swap(best_route.route, i, k)
                new_route_cost = tsp_instance.get_cost_of_route(new_route)
                if new_route_cost < best_route_cost:
                    best_route.route = new_route
                    best_route_cost = new_route_cost
                    improved = True
                    break  # Improvement found, exit inner loop
            if improved:
                break  # Restart scanning for 2-opt swaps

        if not improved:
            break  # No improvement found, exit the search

    return best_route, best_route_cost

def local_search(tsp_instance, time_limit, no_improvement_limit):
    start_time = time.time()
    no_improvement_iterations = 0
    best_route = nearest_neighbor_solution(tsp_instance)
    best_route_cost = tsp_instance.get_cost_of_route(best_route)
            

    while time.time() - start_time < time_limit and no_improvement_iterations < no_improvement_limit:
        current_route = copy.deepcopy(best_route)
        neighbourhood = two_opt_neighbourhood(current_route)
        local_optimum, local_optimum_cost = find_best_neighbour(tsp_instance, neighbourhood)
        print(f'{best_route_cost} best cost of the current best route {best_route} for random search')
        if local_optimum_cost < best_route_cost:
            best_route = local_optimum
            best_route_cost = local_optimum_cost
            no_improvement_iterations = 0  # reset the count as we found a better solution
        else:
            no_improvement_iterations += 1  # increment as no better solution was found

    return best_route, best_route_cost
def city_swap_neighbourhood(route):
    # Assuming the route is a list of cities where the first and last are the same and fixed
    neighbouring_routes = []
    route_length = len(route.route) - 1  # Exclude the last city because it's fixed (same as the first)
    
    for i in range(1, route_length - 1):
        for j in range(i + 1, route_length):
            # Make a deep copy of the route to avoid modifying the original
            new_route = copy.deepcopy(route)
            # Swap the two cities
            new_route.route[i], new_route.route[j] = new_route.route[j], new_route.route[i]
            neighbouring_routes.append(new_route)
    
    return neighbouring_routes

def two_opt_neighbourhood(route):
    neighbouring_routes = []
    for i in range(1, len(route.route) - 2):
        for j in range(i + 1, len(route.route) - 1):
            new_route = copy.deepcopy(route)
            new_route.route[i:j] = reversed(new_route.route[i:j])  # This is the 2-opt Swap
            neighbouring_routes.append(new_route)
    return neighbouring_routes

def find_best_neighbour(graph, neighbourhood):
    best_tour = None
    best_cost = float('inf')
    
    for neighbour in neighbourhood:
        cost = graph.get_cost_of_route(neighbour)
        if cost < best_cost:
            best_cost = cost
            best_tour = neighbour
            
    return best_tour, best_cost
def calculate_percentage_improvement(cost_random, cost_local):
    return ((cost_random - cost_local) / cost_random) * 100


# Load the CSV data into a pandas DataFrame and then into the Graph
df = pd.read_csv('ulysses16.csv')
coordinates = [(row.x, row.y) for index, row in df.iterrows()]

# Initialize the graph with coordinates
graph = Graph(coordinates)

# # Perform random search and print the best route found and its cost
time_limit = 3 # Set the time limit for the search here
# best_route, best_cost = random_search(graph, time_limit)
# print(f"Best Route within {time_limit} seconds: {best_route}, Cost: {best_cost}")

# # Convert the neighbouring routes to a list of lists for easy printing
# neighbouring_routes_list = [neighbour.route for neighbour in neighbouring_routes]

# print("All neighboring routes:")
# for neighbour_route in neighbouring_routes_list:
#     print(neighbour_route)
    
# best_neighbour, best_neighbour_cost = find_best_neighbour(graph, neighbouring_routes)

# print(f"The best neighbour tour is: {best_neighbour}, with a cost of: {best_neighbour_cost}")

best_random_route, best_random_route_cost = random_search(graph, time_limit)
best_local_route, best_local_route_cost = local_search(graph, time_limit,100)
percentage_improvement = calculate_percentage_improvement(best_random_route_cost, best_local_route_cost)


print(f"Best route from random search: {best_random_route}, Cost: {best_random_route_cost}")
print(f"Best route from local search: {best_local_route}, Cost: {best_local_route_cost}")
print(f"Local search is {percentage_improvement:.2f}% better than random search.")
