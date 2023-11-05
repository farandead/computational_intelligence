import math
import numpy as np
import random
# Car price prediction problem
class AntennaArray:
    # The minimum distance between antennae.
    MIN_SPACING = 0.25
    
    # Construct an antenna design problem instance.
    # The parameter "n_antennae" specifies the number of antennae in our array.
    # The parameter "steering_angle" specifies the desired direction of the
    # main beam in degrees.
    def __init__(self,n_antennae,steering_angle):    
      self.n_antennae = n_antennae
      self.steering_angle = steering_angle
   
    # Rectangular bounds on the search space.
    # Returns a 2D array b such that b[i][0] is the minimum permissible value
    # of the ith solution component and b[i][1] is the maximum.    
    def bounds(self):
        return [[0,self.n_antennae/2] for i in range(self.n_antennae)]
    def generate_valid_design(self):
        design = [0] * self.n_antennae
        design[-1] = self.n_antennae / 2  # Set the last antenna at the aperture size limit

        # Generate random positions for the remaining antennae
        for i in range(1, self.n_antennae - 1):
            min_pos = design[i - 1] + AntennaArray.MIN_SPACING
            max_pos = design[-1] - AntennaArray.MIN_SPACING * (self.n_antennae - i)
            design[i] = random.uniform(min_pos, max_pos)

        # Shuffle the internal antennae to avoid any order bias
        middle_antennae = design[1:-1]
        random.shuffle(middle_antennae)
        design[1:-1] = middle_antennae

        return design
    # Check whether an antenna design lies within the problem's feasible
    # A design is a vector of n_antennae anntena placements.
    # A placement is a distance from the left hand side of the antenna array.
    # A valid placement is one in which
    #   1) all antennae are separated by at least MIN_SPACING
    #   2) the aperture size (the maximum element of the array) is exactly
    #      n_antennae/2.
    def is_valid(self, design):
        # The design has the correct number of antennae
        if len(design) != self.n_antennae:
          return False
        
        des = design.copy()
        des.sort()
        
        # Apperture size is exactly n_antennae/2
        if abs(des[-1] - self.n_antennae/2) > 1e-10:
          return False
        # All antennae lie within the problem bounds
        for placement, bound in zip(des,self.bounds()):
          if placement < bound[0] or placement > bound[1]:
            return False
        
        # All antennae are separated by at least MIN_SPACING        
        for i in range(len(des)-1):
          if des[i+1] - des[i] < AntennaArray.MIN_SPACING:
            return False

        # If none of the above checks have been failed, the design must be
        # valid.
        return True

    # Evaluate an antenna design returning peak SSL.
    # Designs which violate problem constraints will be penalised with extremely
    # high costs.
    # The parameter "design" is a valid antenna array design.
    # If the design is not valid, it will be penalised by being assigned the
    # maximum possible numerical cost, float('inf').
    def evaluate(self,design):
        if(not self.is_valid(design)): return float('inf');

        class PowerPeak:
            def __init__(self,elevation,power):
                self.elevation = elevation;
                self.power = power;
        
        # Find all the peaks in power
        peaks = []
        prev = PowerPeak(0.0,float('-inf'))
        current = PowerPeak(0.0,self.__array_factor(design,0.0));
        for elevation in np.arange(0.01,180.01,0.01):
            nxt = PowerPeak(elevation,self.__array_factor(design,elevation))
            if current.power >= prev.power and current.power >= nxt.power:
                peaks.append(current)
            prev = current
            current = nxt
        peaks.append(PowerPeak(180.0,self.__array_factor(design,180.0)))
        peaks.sort(reverse = True,key = lambda peak:peak.power)

        # No side-lobes case
        if len(peaks)<2:
          return float('-inf')
        
        # Filter out main lobe and then return highest lobe level
        distance_from_steering = abs(peaks[0].elevation - self.steering_angle);
        for i in range(1,len(peaks)):
          # If the peak with the highest power is not the closest peak to the
          # steering angle, it is a side lobe and, therefore, must be the peak
          # side lobe.
          if abs(peaks[i].elevation - self.steering_angle) < distance_from_steering:
            return peaks[0].power
        # Otherwise, the peak with the highest power is not a side lobe. The
        # peak with the second highest power must be the peak side lobe.
        return peaks[1].power

    def __array_factor(self,design,elevation):
        steering = 2.0*math.pi*self.steering_angle/360.0;
        elevation = 2.0*math.pi*elevation/360.0;
        sum = 0.0
        for placement in design:
            sum += math.cos(2 * math.pi * placement * (math.cos(elevation) - math.cos(steering)))
        return 20.0*math.log(abs(sum))
    def random_search(antenna_array, iterations=1000):
      best_design = None
      lowest_peak_SSL = float('inf')
      for _ in range(iterations):
          design = antenna_array.generate_valid_design()
          peak_SSL = antenna_array.evaluate(design)
          if peak_SSL < lowest_peak_SSL:
              lowest_peak_SSL = peak_SSL
              best_design = design
      return best_design, lowest_peak_SSL
    
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
                
                # Update the global best if necessary
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
            
            # Implement the invisible wall strategy for constraint handling
            if self.position[i] < 0 or self.position[i] > self.problem_instance.n_antennae / 2:
                self.velocity[i] = -self.velocity[i]  # Bounce back
                self.position[i] = max(min(self.position[i], self.problem_instance.n_antennae / 2), 0)
                
    def update_personal_best(self):
        cost = self.problem_instance.evaluate(self.position)
        if cost < self.personal_best_cost:
            self.personal_best_position = list(self.position)
            self.personal_best_cost = cost