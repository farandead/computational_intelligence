import math
import random
from antennaarray import AntennaArray
from antennaarray import Particle

class AdaptiveParticle(Particle):
    def __init__(self, problem_instance):
        super().__init__(problem_instance)
        # Initialize adaptive inertia weight
        self.w = 0.9

    def update_velocity(self, global_best_position, c1=2.05, c2=2.05):
        phi = c1 + c2
        # Ensure phi is greater than 4 to avoid math domain error
        phi = max(phi, 4.1)
        chi = 2 / abs(2 - phi - math.sqrt(phi**2 - 4*phi))  # Constriction factor

        for i in range(len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.personal_best_position[i] - self.position[i])
            social_velocity = c2 * r2 * (global_best_position[i] - self.position[i])
            # Apply the constriction factor chi to the velocity update
            self.velocity[i] = chi * (self.w * self.velocity[i] + cognitive_velocity + social_velocity)

        # Adaptive inertia weight update
        # Decrease inertia weight linearly from 0.9 to 0.4 over all iterations
        self.w = 0.9 - (self.problem_instance.iteration * (0.5 / self.problem_instance.max_iterations))

# Extend the AntennaArray class to include the adaptive inertia weight in the swarm optimization
class AdaptiveAntennaArray(AntennaArray):
    def __init__(self, n_antennae, steering_angle, max_iterations):
        super().__init__(n_antennae, steering_angle)
        self.max_iterations = max_iterations
        self.iteration = 0

    def initialize_swarm(self, num_particles):
        self.swarm = [AdaptiveParticle(self) for _ in range(num_particles)]
        self.global_best_position = min(self.swarm, key=lambda p: p.personal_best_cost).personal_best_position
        self.global_best_cost = self.evaluate(self.global_best_position)

    def optimize(self):
        for self.iteration in range(self.max_iterations):
            for particle in self.swarm:
                particle.update_velocity(self.global_best_position)
                particle.update_position()
                particle.update_personal_best()
                if particle.personal_best_cost < self.global_best_cost:
                    self.global_best_position = list(particle.personal_best_position)
                    self.global_best_cost = particle.personal_best_cost
NUM_PARTICLES = 30
ITERATIONS = 16
# Now let's run the optimization with the new AdaptiveAntennaArray class
adaptive_antenna_array_problem = AdaptiveAntennaArray(3, 90, ITERATIONS)
adaptive_antenna_array_problem.initialize_swarm(NUM_PARTICLES)
adaptive_antenna_array_problem.optimize()

# Output the result
print(adaptive_antenna_array_problem.global_best_position, adaptive_antenna_array_problem.global_best_cost)