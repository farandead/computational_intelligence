import random
import math
import pandas as pd
import time
import copy
from openpyxl import load_workbook
from os import path 
import os
import csv

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
    
class Route:
    def __init__(self, route):
            if isinstance(route, int):
                # If an integer is passed, create a route with that many cities
                self.route = list(range(route))
            elif isinstance(route, list):
                # If a list is passed, use it directly as the route
                self.route = route
            else:
                raise ValueError("Invalid input: route must be an integer or a list")
            
    

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
    
    best_route = nearest_neighbor_solution(tsp_instance)  
    best_route_cost = tsp_instance.get_cost_of_route(best_route)
  
    while time.time() - start_time < time_limit:
        current_route = copy.deepcopy(best_route)
        neighbourhood = city_swap_neighbourhood(current_route)
        local_optimum, local_optimum_cost = find_best_neighbour(tsp_instance, neighbourhood)
        
        if local_optimum_cost < best_route_cost:
            best_route = local_optimum
            best_route_cost = local_optimum_cost
        else:
            break  

    return best_route, best_route_cost
def local_search_with_statistics(tsp_instance, time_limit):
    start_time = time.time()
    
    # Initialize statistics
    lowest_cost = float("inf")
    times_searched = 0
    iterations = 0
    
    while time.time() - start_time < time_limit:
        # Start with a random route
        current_route = nearest_neighbor_solution(tsp_instance)
        current_cost = tsp_instance.get_cost_of_route(current_route)
        
        local_best = False
        while not local_best:
            neighbourhood = city_swap_neighbourhood(current_route)
            new_route, new_cost = find_best_neighbour(tsp_instance, neighbourhood)
           

            if new_cost < current_cost:
                current_route = new_route
                current_cost = new_cost
            else:
                local_best = True
        
        if current_cost < lowest_cost:
            best_route = current_route
            lowest_cost = current_cost
        
        iterations += 1
        
    return best_route, lowest_cost
def local_search_no_improvement_limit(tsp_instance, time_limit, no_improvement_limit):
    start_time = time.time()
    no_improvement_iterations = 0
    best_route = nearest_neighbor_solution(tsp_instance)
    best_route_cost = tsp_instance.get_cost_of_route(best_route)
            

    while time.time() - start_time < time_limit and no_improvement_iterations < no_improvement_limit:
        current_route = nearest_neighbor_solution(tsp_instance)
        current_cost = tsp_instance.get_cost_of_route(current_route)
        neighbourhood = city_swap_neighbourhood(current_route)
        local_optimum, local_optimum_cost = find_best_neighbour(tsp_instance, neighbourhood)
        local_best = False
        while not local_best:
            neighbourhood = city_swap_neighbourhood(current_route)
            new_route, new_cost = find_best_neighbour(tsp_instance, neighbourhood)
            
            if new_cost < current_cost:
                current_route = new_route
                current_cost = new_cost
            else:
                local_best = True
                
        if local_optimum_cost < best_route_cost:
            best_route = local_optimum
            best_route_cost = local_optimum_cost
            no_improvement_iterations = 0  # reset the count as we found a better solution
        else:
            no_improvement_iterations += 1  # increment as no better solution was found

    return best_route, best_route_cost


def two_opt_with_no_improvement_limit(tsp_instance, initial_route, time_limit, no_improvement_limit):
    start_time = time.time()
    best_route = initial_route
    best_cost = tsp_instance.get_cost_of_route(best_route)
    no_improvement_iterations = 0  # Initialize counter for no improvement iterations

    while (time.time() - start_time) < time_limit and no_improvement_iterations < no_improvement_limit:
        improved = False
        for i in range(1, len(best_route.route) - 2):
            for k in range(i + 1, len(best_route.route)):
                new_route = two_opt_swap(best_route.route, i, k)
                new_route_obj = Route(new_route)  # Create a new Route object
                new_cost = tsp_instance.get_cost_of_route(new_route_obj)
                local_best= False
                while not local_best:
                    if new_cost < best_cost:
                        best_route.route = new_route  # Update the route in the Route object
                        best_cost = new_cost
                        improved = True
                        no_improvement_iterations = 0  # Reset the counter as an improvement was found
                    
                    else:
                        local_best = True
                

        if not improved:
            no_improvement_iterations += 1  # Increment the counter as no better solution was found

        # Optionally, you can add a print statement here to show progress.
        # It will execute after each full pass through the route.
        if no_improvement_iterations > 0:
            no_improvement_iterations

    return best_route, best_cost
def city_swap_neighbourhood(route):
    # Assuming the route is a list of cities where the first and last are the same and fixed
    neighbouring_routes = []
    route_length = len(route.route)   # Exclude the last city because it's fixed (same as the first)
    
    for i in range(1, route_length ):
        for j in range(i + 1, route_length):
            # Make a copy of the route object. You might need a custom copy method if it's a complex object
            new_route = copy.deepcopy(route)
            # Swap two cities in the route
            new_route.route[i], new_route.route[j] = new_route.route[j], new_route.route[i]
            # Add the new route configuration to the list of neighbouring routes
            neighbouring_routes.append(new_route)

    
    return neighbouring_routes

def two_opt_swap(route, i, k):
    """Take route[0] to route[i-1] and add them in order to new_route
    Take route[i] to route[k] and add them in reverse order to new_route
    Take route[k+1] to end and add them in order to new_route"""
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route
def two_opt_neighbourhood(route,tspinstance):
    best_neighbour = route[:]
    best_cost = float('inf')
    made_improvement = False

    # Generate all 2-opt neighbors
    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            if j - i == 1: continue  # This would be a no-op since the edges are adjacent
            new_neighbour = route[:i] + route[i:j][::-1] + route[j:]
            new_cost = tspinstance.get_cost_of_route(new_neighbour)  # You need to implement this function

            # Check if the new neighbour is the best so far
            if new_cost < best_cost:
                best_cost = new_cost
                best_neighbour = new_neighbour
                made_improvement = True

    return best_neighbour, best_cost, made_improvement
def two_opt(tsp_instance, route, time_limit):
    start_time = time.time()
    best_route =  nearest_neighbor_solution(tsp_instance)
   
    best_cost = tsp_instance.get_cost_of_route(best_route)
    improved = True

    # Adjust the range to use len(best_route.route) instead of len(route)
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Update the range function to use the correct length
        for i in range(1, len(best_route.route) ):  
            for k in range(i + 1, len(best_route.route)):
                new_route = two_opt_swap(best_route.route, i, k)
                new_route_obj = Route(new_route)  # Create a new Route object
                new_cost = tsp_instance.get_cost_of_route(new_route_obj)
                local_best = False
                while not local_best:
                    if new_cost < best_cost:
                        best_route.route = new_route  # Update the route in the Route object
                        best_cost = new_cost
                        improved = True
                    else:
                        local_best = True
                
        if not improved:
            break

    return best_route, best_cost



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



    
def append_results_to_csv(file_path, data):
    # Convert the data to a DataFrame
    df = pd.DataFrame([data])
    
    # Check if the CSV file already exists
    if path.isfile(file_path):
        # If the file exists, append the data without headers
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # If the file does not exist, create it with headers
        df.to_csv(file_path, mode='w', header=True, index=False)

  
def compare_search_algorithms(tsp_instance, time_limit, no_improvement_limit):
    # Run Random Search
    best_random_route, best_random_route_cost = random_search(tsp_instance, time_limit)

    # Run Simple Local Search
    best_simple_local_route, best_simple_local_route_cost = local_search_with_statistics(tsp_instance, time_limit)

    # Run 2-opt Local Search
    best_2_opt_route, best_2_opt_route_cost = two_opt(tsp_instance,best_random_route, time_limit)

    # Run Local Search with No Improvement Limit
    best_no_improv_route, best_no_improv_route_cost = local_search_no_improvement_limit(tsp_instance, time_limit, no_improvement_limit)
    
    best_route_no_improv_route_and_2_opt_route, best_cost_no_improv_route_and_2_opt_route = two_opt_with_no_improvement_limit(graph, best_random_route, time_limit, no_improvement_limit)


    # Calculate improvements
    simple_improvement = calculate_percentage_improvement(best_random_route_cost, best_simple_local_route_cost)
    two_opt_improvement = calculate_percentage_improvement(best_random_route_cost, best_2_opt_route_cost)
    no_improv_improvement = calculate_percentage_improvement(best_random_route_cost, best_no_improv_route_cost) 
    no_improv_and_two_opt_improvement = calculate_percentage_improvement(best_random_route_cost, best_cost_no_improv_route_and_2_opt_route) 

    # Print the results
    print("Comparison of Search Algorithms:")
    print("Random Search:", best_random_route, "Cost:", best_random_route_cost)
    print("Simple Local Search:", best_simple_local_route, f"Cost: {best_simple_local_route_cost} (Improvement: {simple_improvement:.2f}%)")
    print("2-opt Local Search:", best_2_opt_route, f"Cost: {best_2_opt_route_cost} (Improvement: {two_opt_improvement:.2f}%)")
    print("Local Search with No Improvement Limit:", best_no_improv_route, f"Cost: {best_no_improv_route_cost} (Improvement: {no_improv_improvement:.2f}%)")
    print("Local Search with 2-opt Local Search and No Improvement Limit:", best_route_no_improv_route_and_2_opt_route, f"Cost: {best_cost_no_improv_route_and_2_opt_route} (Improvement: {no_improv_and_two_opt_improvement:.2f}%)")
     # Prepare results for Excel
   
    results_for_csv = {
    'Algorithm': [
        'Random Search',
        'Simple Local Search',
        '2-opt Local Search',
        'Local Search with No Improvement Limit',
        'Local Search with 2-opt and No Improvement Limit'
    ],
    'Cost': [
        best_random_route_cost,
        best_simple_local_route_cost,
        best_2_opt_route_cost,
        best_no_improv_route_cost,
        best_cost_no_improv_route_and_2_opt_route
    ],
    'Improvement': [
        0,
        simple_improvement,
        two_opt_improvement,
        no_improv_improvement,
        no_improv_and_two_opt_improvement
    ]
}
    file_path = 'algorithm_results.csv'

    # Check if file exists and is empty to decide whether to write headers
    file_exists = os.path.isfile(file_path)
    write_header = not file_exists or os.stat(file_path).st_size == 0

    with open(file_path, mode='a', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the header row only if needed
        if write_header:
            writer.writerow(['Algorithm', 'Cost', 'Improvement', 'Time Limit', 'No Improvement Limit'])

        # Write the data rows
        for i in range(len(results_for_csv['Algorithm'])):
            writer.writerow([
                results_for_csv['Algorithm'][i],
                results_for_csv['Cost'][i],
                results_for_csv['Improvement'][i],
                time_limit,
                no_improvement_limit
            ])
    

    # Append results to Excel file
    append_results_to_csv('search_algorithm_results.csv', results_for_csv)
   

    # You can also add a more detailed comparison, like statistical analysis or graphical representations if needed.

# Example usage
time_limit =10 # or whatever is suitable
no_improvement_limit = 2000000 # or suitable value based on your experimentation
compare_search_algorithms(graph, time_limit, no_improvement_limit)


