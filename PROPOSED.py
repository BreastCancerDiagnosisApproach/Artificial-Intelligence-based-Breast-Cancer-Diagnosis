import numpy as np
import time


#Improved Garter Snake Optimization Algorithm
def PROPOSED(agents, objective_function, lb, ub, max_iterations):
    num_agents, dim = agents.shape

    # Initialize the best solution and its corresponding fitness
    best_solution = np.zeros(dim)
    best_fitness = np.zeros(num_agents)

    for i in range(num_agents):
        best_fitness[i] = objective_function(agents[i,:])

    Convergence_curve = np.zeros((1, max_iterations))
    ct = time.time()

    for iteration in range(max_iterations):
        for i in range(num_agents):
            # Generate a random direction vector
            Ps = np.min(best_fitness) / (np.max(best_fitness) * np.mean(best_fitness))  #// Traditional update

            # Update the agent's position
            agents[i,:] += Ps

            # Clip the position to ensure it stays within the search space bounds
            agents[i,:] = np.clip(agents[i,:], lb[i,:], ub[i,:])

            # Calculate the fitness of the new position
            fitness = objective_function(agents[i,:])

            # Update the best solution if a better one is found
            if fitness < best_fitness[i]:
                best_solution = agents[i,:]
                best_fitness[i] = fitness

        # # Print the best fitness value at each iteration
        Convergence_curve[:, iteration] = np.min(best_fitness)
    best_fit = np.min(best_fitness)
    ct -= time.time()
    return best_fit, Convergence_curve, best_solution, ct


