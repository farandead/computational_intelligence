import math
import numpy as np
import random

random.seed(42)  # Set the random seed for reproducibility

# Define the AntennaArray class with all the necessary methods
class AntennaArray:
    MIN_SPACING = 0.25
    
    def __init__(self, n_antennae, steering_angle):
        self.n_antennae = n_antennae
        self.steering_angle = steering_angle

    def bounds(self):
        return [[0, self.n_antennae/2] for i in range(self.n_antennae)]

    def generate_valid_design(self):
        design = [0] * self.n_antennae
        design[-1] = self.n_antennae / 2  # Set the last antenna at the aperture size limit
        for i in range(1, self.n_antennae - 1):
            min_pos = design[i - 1] + AntennaArray.MIN_SPACING
            max_pos = design[-1] - AntennaArray.MIN_SPACING * (self.n_antennae - i)
            design[i] = random.uniform(min_pos, max_pos)
        middle_antennae = design[1:-1]
        random.shuffle(middle_antennae)
        design[1:-1] = middle_antennae
        return design

    def is_valid(self, design):
        # ... Validation checks (omitted for brevity)
        return True  # Assuming the design is always valid as per the generate_valid_design method

    def evaluate(self, design):
        # ... Evaluation logic (omitted for brevity)
        if not self.is_valid(design): 
            return float('inf')
        return random.uniform(0, 1)  # Placeholder for actual evaluation logic

    def initialize_swarm(self, num_particles):
        self.swarm = [Particle(self) for _ in range(num_particles)]
        self.global_best_position = min(self.swarm, key=lambda p: p.personal_best_cost).personal_best_position
        self.global_best_cost = self.evaluate(self.global_best_position)

    def optimize(self, iterations):
        for _ in range(iterations):
            for particle in self.swarm:
                particle.update_velocity(self.global_best_position)
                particle.update_position()
                particle.update_personal_best()
                if particle.personal_best_cost < self.global_best_cost:
                    self.global_best_position = list(particle.personal_best_position)
                    self.global_best_cost = particle.personal_best_cost

class Particle:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance
        self.position = problem_instance.generate_valid_design()
        self.velocity = [0.0] * problem_instance.n_antennae
        self.personal_best_position = list(self.position)
        self.personal_best_cost = problem_instance.evaluate(self.position)

    def update_velocity(self, global_best_position, w=0.729, c1=1.49445, c2=1.49445):
        for i in range(len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.personal_best_position[i] - self.position[i])
            social_velocity = c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive_velocity + social_velocity

    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            if self.position[i] < 0 or self.position[i] > self.problem_instance.n_antennae / 2:
                self.velocity[i] = -self.velocity[i]
                self.position[i] = max(min(self.position[i], self.problem_instance.n_antennae / 2), 0)

    def update_personal_best(self):
        cost = self.problem_instance.evaluate(self.position)
        if cost < self.personal_best_cost:
            self.personal_best_position = list(self.position)
            self.personal_best_cost = cost

# Define the number of particles and iterations
NUM_PARTICLES = 30
ITERATIONS = 100

# Create an instance of the problem
antenna_array_problem = AntennaArray(3, 90)

# Initialize the swarm
antenna_array_problem.initialize_swarm(NUM_PARTICLES)

# Perform optimization
antenna_array_problem.optimize(ITERATIONS)

# Output the result
print(antenna_array_problem.global_best_position, antenna_array_problem.global_best_cost)
