import random
import time
import numpy as np


# Tomtit Flock Metaheuristic Optimization Algorithm (TFMOA)
def TFMOA(positions, objective_func, Lb, Ub, max_iter):
    sub_flocks = 4,
    a = 1.5,
    b = 1.5,
    c = 1.0
    sub_flocks = sub_flocks
    lb = Lb[0, :]
    ub = Ub[1, :]
    n_agents, n_variables = positions.shape[0], positions.shape[1]
    for i in range(n_agents):
        positions[i, :] = np.random.uniform(lb, ub, n_variables)

    # Divide agents into sub-flocks
    sub_flock_size = int(n_agents / sub_flocks)
    sub_flocks = [positions[i:i + sub_flock_size, :] for i in range(0, n_agents, sub_flock_size)]

    # Initialize global best position
    global_best_pos = np.zeros(n_variables)
    global_best_fit = float("inf")

    Convergence = np.zeros(max_iter)
    ct = time.time()
    # Iterate over all generations
    for t in range(max_iter):
        # Update sub-flock leaders
        sub_flock_bests = []
        for i in range(sub_flocks):
            sub_flock = sub_flocks[i]
            sub_flock_best_pos = np.zeros(n_variables)
            sub_flock_best_fit = float("inf")
            for j in range(sub_flock_size):
                positions = sub_flock[j, :]
                fitness = objective_func(positions)
                if fitness < sub_flock_best_fit:
                    sub_flock_best_fit = fitness
                    sub_flock_best_pos = positions
            sub_flock_bests.append(sub_flock_best_pos)

        # Update global best position
        for i in range(sub_flocks):
            sub_flock_best_pos = sub_flock_bests[i]
            fitness = objective_func(sub_flock_best_pos)
            if fitness < global_best_fit:
                global_best_fit = fitness
                global_best_pos = sub_flock_best_pos

        # Update sub-flock agents
        for i in range(sub_flocks):
            sub_flock = sub_flocks[i]
            sub_flock_best_pos = sub_flock_bests[i]
            for j in range(sub_flock_size):
                positions = sub_flock[j, :]
                leader_pos = sub_flock_best_pos
                neighbor_pos = sub_flock[random.randint(0, sub_flock_size - 1), :]
                velocity = a * (leader_pos - positions) + b * (neighbor_pos - positions) + c * (
                            global_best_pos - positions)
                positions += velocity

                # Ensure agents stay within bounds
                positions = np.clip(positions, lb, ub)
                sub_flock[j, :] = positions
                Convergence[i] = global_best_fit
    ct = time.time() - ct
    return global_best_fit, fitness, global_best_pos, ct
