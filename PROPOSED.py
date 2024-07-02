import time

import numpy as np
import random
import numpy.linalg as la

def bounce_at_boundaries(position, velocity, lower_bound, upper_bound):
  """
  Implements bouncing behavior for positions reaching boundaries.

  Args:
      position: A numpy array of shape (n_variables,) representing the current position.
      velocity: A numpy array of shape (n_variables,) representing the velocity.
      lower_bound: A float or numpy array representing the lower boundary value(s).
      upper_bound: A float or numpy array representing the upper boundary value(s).

  Returns:
      A numpy array of shape (n_variables,) representing the updated position after bouncing.
  """
  updated_position = position + velocity

  # Check for positions exceeding lower or upper bounds
  exceeded_lower = updated_position < lower_bound
  exceeded_upper = updated_position > upper_bound

  # Reverse velocity for dimensions exceeding boundaries
  updated_position[exceeded_lower] = lower_bound + (position[exceeded_lower] - lower_bound) * (-1)  # Reflect at lower bound
  updated_position[exceeded_upper] = upper_bound - (upper_bound - position[exceeded_upper]) * (-1)  # Reflect at upper bound
  velocity[exceeded_lower] *= -1  # Reverse velocity for dimensions exceeding lower bound
  velocity[exceeded_upper] *= -1  # Reverse velocity for dimensions exceeding upper bound

  return updated_position

def PROPOSED(population,objective_function,lb,ub, max_iterations):
    # Initialize population
    lb = lb[0,:]
    ub = ub[1,:]
    n_individuals, n_variables = population.shape[0],population.shape[1]
    learning_rate = 0.01
    random_chance = 0.2
    # Loop for iterations
    convergece = np.zeros((max_iterations))
    intelligence = np.ones(n_individuals)
    memory_depth = 2
    guesses = np.zeros(n_individuals, dtype=int)
    past_fitness=([np.zeros(n_individuals)] * (memory_depth - len(n_individuals)))
    past_fitness = past_fitness[-memory_depth:]
    velocity = np.zeros_like(population)
    velocity_weight =0.7
    boundary_handling = "clip"
    ct = time.time()
    for iter in range(max_iterations):
        # Update positions based on guesses
        accuracy = np.zeros(n_individuals)
        for i in range(n_individuals):
            # Fitness evaluation
            fitness = objective_function(population)
            selection_strategy = "best_and_random"
            # Shell selection (replace with your SGO strategy)
            if selection_strategy == "best_and_random":
                # Select the best individual based on fitness
                best_index = np.argmin(fitness)
                # Randomly select two other individuals (excluding the best)
                remaining_indices = np.arange(n_individuals)
                remaining_indices = np.delete(remaining_indices, best_index)
                selected_shells = random.choice(remaining_indices, size=2, replace=False)
            elif selection_strategy == "tournament":
                # Implement tournament selection (replace with your logic)
                # This involves randomly selecting a subset of individuals and choosing the best from that subset.
                # Repeat this process to select two more individuals.
                pass
                # Calculate distance between individual and "ball"
            distance = la.norm(population[i] - ball_position)

            # Calculate accuracy based on distance (higher proximity leads to higher accuracy)
            accuracy[i] = 1.0 / (1.0 + distance)  # Example formula (adjust as needed)
            fitness_improvement = np.mean(fitness[i] - past_fitness[:, i])

            # Update intelligence based on fitness improvement (higher improvement implies higher intelligence)
            intelligence[i] = 1.0 + fitness_improvement  # Example formula (adjust as needed)

            # State guessing (implement probabilistic approach)
            exploitation_factor = accuracy[i] * intelligence[i]
            exploration_factor = 1 - exploitation_factor
            r = -iter * ((-1) / max_iterations)
            # Random guess with probability based on random_chance
            if r < random_chance:
                guesses[i] = random.randint(0, n_individuals - 1)
            else:
                # Choose guess based on weighted exploitation (accuracy * intelligence)
                guesses[i] = np.argmax(exploitation_factor)

            # Update positions (based on guesses and actual ball position)
            n_individuals, n_variables = population.shape
            ball_position =objective_function[population[i,:]]
            # Move towards the chosen shell (assuming guesses are valid indices)
            movement = (population[i] - population[guesses[i]]) * learning_rate
            # Check if guess was correct (picked the shell with the ball)
            if np.all(population[i] == ball_position):
                # Stay put or move slightly closer (optional)
                movement *= 0.5  # Reduce movement for correct guesses
            # Apply movement with boundary check (assuming values between 0 and 1)
            population[i] = np.clip(population[i] + movement, 0, 1)
            movement = (population[i] - population[guesses[i]]) * learning_rate
            # Update velocity with momentum (consider both movement direction and past velocity)
            velocity[i] = velocity_weight * velocity[i] + movement
            # Apply velocity update (consider boundary handling)
            if boundary_handling == "clip":
                updated_position = np.clip(population[i] + velocity[i], 0, 1)
            elif boundary_handling == "bounce":
                updated_position = bounce_at_boundaries(population[i], velocity[i], 0,1)  # Implement bounce_at_boundaries function (replace with desired behavior)
            else:
                raise ValueError("Invalid boundary handling method. Choose 'clip' or 'bounce'.")

            population[i] = updated_position
        best_fitness_index = np.argmin(fitness)
        convergece[iter] = best_fitness_index

    # Update best solution

    best_solution = population[best_fitness_index, :]
    ct = time.time()-ct

    return best_solution, convergece,fitness[best_fitness_index],ct
