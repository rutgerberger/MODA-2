import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.core.variable import Binary
import matplotlib.pyplot as plt

class ContainerLoadingProblem(Problem):

    def __init__(self, bays, rows, tiers, containers, container_weights, delta_w=0):
        # Store problem parameters
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
        self.containers = containers
        self.container_weights = container_weights
        self.delta_w = delta_w

        self.num_bays = len(bays)
        self.num_rows = len(rows)
        self.num_tiers = len(tiers)
        self.num_containers = len(containers)

        # Mapping for x_b_r_t_i variables
        self.x_flat_indices = {} # We use this for easy unique access to every unique container - coordinate combination
        idx_counter = 0
        for b_idx, b in enumerate(bays):
            for r_idx, r in enumerate(rows):
                for t_idx, t in enumerate(tiers):
                    for c_idx, c in enumerate(containers):
                        self.x_flat_indices[(b,r,t,c)] = idx_counter
                        idx_counter += 1

        self.num_x_vars = idx_counter

        # Mapping for o_i_j variables
        self.o_flat_indices = {}
        self.container_to_idx = {c: i for i, c in enumerate(containers)}

        o_idx_counter = 0
        for i_idx, i in enumerate(containers):
            for j_idx, j in enumerate(containers):
                if i != j:
                    self.o_flat_indices[(i,j)] = o_idx_counter # Use the new counter
                    o_idx_counter += 1


        self.num_o_vars = o_idx_counter # This holds the count of o variables

        n_var = self.num_x_vars + self.num_o_vars

        # Define the number of objectives (e.g., maximize containers loaded, minimize imbalance)
        n_obj = 3

        # Define the number of constraints
        num_constr1 = self.num_containers * (self.num_containers - 1)
        num_constr2 = self.num_containers * (self.num_containers - 1)
        num_constr3 = self.num_bays * self.num_rows * (self.num_tiers - 1) * self.num_containers * (self.num_containers - 1)
        num_constr_container_once = self.num_containers
        num_constr_spot_once = self.num_bays * self.num_rows * self.num_tiers
        num_constr_no_floating = self.num_bays * self.num_rows * (self.num_tiers - 1)

        n_constr = num_constr1 + num_constr2 + num_constr3 + \
                   num_constr_container_once + num_constr_spot_once + num_constr_no_floating

        xl = np.zeros(n_var)
        xu = np.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, type_var=Binary)

    def _evaluate(self, X, out, *args, **kwargs):
        # X is a 2D numpy array where each row is an individual (solution)
        # and columns are the variables. X has shape (pop_size, n_var).

        # For each individual in the population
        obj_values = []
        constraint_violations = []

        for k in range(X.shape[0]): # Iterate through each solution in the population
            x_sol = X[k, :self.num_x_vars]
            o_sol = X[k, self.num_x_vars:]

            # Objective function 1
            # Maximize the number of containers loaded (=minimize the negative count)
            num_containers_loaded = 0
            # Variables for new objectives
            longitudinal_weighted_sum = 0
            latitudinal_weighted_sum = 0

            for container_id in self.containers:
                container_is_loaded = False
                for b in self.bays:
                    for r in self.rows:
                        for t in self.tiers:
                            x_val = x_sol[self.x_flat_indices[(b,r,t,container_id)]]
                            if x_val > 0.5: # If container_id is placed at (b,r,t)
                                container_is_loaded = True
                                # Add to longitudinal and latitudinal sums
                                weight = self.container_weights[container_id]
                                longitudinal_weighted_sum += weight * (b - 4.5)
                                latitudinal_weighted_sum += weight * (r - 2.5)
                if container_is_loaded:
                    num_containers_loaded += 1
            
            # Objective 1: Minimize negative number of containers loaded
            obj_f1 = -num_containers_loaded
            # Objective 2: Minimize longitudinal disbalance
            obj_f2 = abs(longitudinal_weighted_sum)
            # Objective 3: Minimize latitudinal disbalance
            obj_f3 = abs(latitudinal_weighted_sum)
            obj_values.append([obj_f1, obj_f2, obj_f3]) # Append all three objectives

            # --- Constraints ---
            # Current constraints array keeps track of how many
            # violations were made to the constraints.
            current_constraints = []

            # 1. Single container per cell
            for b in self.bays:
                for r in self.rows:
                    for t in self.tiers:
                        spot_occupancy_sum = 0
                        for i in self.containers:
                            spot_occupancy_sum += x_sol[self.x_flat_indices[(b,r,t,i)]]
                        # Violation if sum > 1: (sum - 1)
                        current_constraints.append(spot_occupancy_sum - 1)

            # 2. Single cell per container
            for i in self.containers:
                container_placement_sum = 0
                for b in self.bays:
                    for r in self.rows:
                        for t in self.tiers:
                            container_placement_sum += x_sol[self.x_flat_indices[(b,r,t,i)]]
                # Violation if sum > 1: (sum - 1)
                current_constraints.append(container_placement_sum - 1)

            # 3. Containers must be stacked on another container
            for b in self.bays:
                for r in self.rows:
                    for t_idx, t in enumerate(self.tiers):
                        if t > min(self.tiers):
                            upper_tier_occupied = 0
                            lower_tier_occupied = 0
                            for i in self.containers:
                                upper_tier_occupied += x_sol[self.x_flat_indices[(b,r,t,i)]]
                                lower_tier_occupied += x_sol[self.x_flat_indices[(b,r,t-1,i)]]
                            current_constraints.append(upper_tier_occupied - lower_tier_occupied)

            # 4. o_i_j <= sum(x_b_r_t_i) (o_i_j can only be true of it i is at a lower tier 2/3)
            # Rewrite as: o_i_j - sum(x_b_r_t_i) <= 0
            for i in self.containers:
                for j in self.containers:
                    if i != j:
                        sum_x_i = 0
                        for b in self.bays:
                            for r in self.rows:
                                for t_idx, t in enumerate(self.tiers):
                                    if t >= 2: # Tiers 2, 3
                                        sum_x_i += x_sol[self.x_flat_indices[(b,r,t,i)]]
                        o_val = o_sol[self.o_flat_indices[(i,j)]]
                        current_constraints.append(o_val - sum_x_i)

            # 5. o_i_j <= sum(x_b_r_t_j) (o_i_j can only be true of it j is at a lower tier 1/2)
            # Rewrite as: o_i_j - sum(x_b_r_t_j) <= 0
            for i in self.containers:
                for j in self.containers:
                    if i != j:
                        sum_x_j = 0
                        for b in self.bays:
                            for r in self.rows:
                                # Tiers 1 and 2 (assuming tiers are 1-indexed)
                                for t_idx, t in enumerate(self.tiers):
                                    if t <= max(self.tiers) - 1: # Tiers 1, 2
                                        sum_x_j += x_sol[self.x_flat_indices[(b,r,t,j)]]
                        o_val = o_sol[self.o_flat_indices[(i,j)]]
                        current_constraints.append(o_val - sum_x_j)

            # 6. x_b_r_t_i + x_b_r_t-1_j - 1 <= o_i_j (o_i_j cannot be zero if i is on top of j)
            # Rewrite as: x_b_r_t_i + x_b_r_t-1_j - 1 - o_i_j <= 0
            for b in self.bays:
                for r in self.rows:
                    for t_idx, t in enumerate(self.tiers):
                        if t > min(self.tiers): # For tiers where t-1 exists
                            for i in self.containers:
                                for j in self.containers:
                                    if i != j:
                                        x_i_val = x_sol[self.x_flat_indices[(b,r,t,i)]]
                                        x_j_val = x_sol[self.x_flat_indices[(b,r,t-1,j)]]
                                        o_val = o_sol[self.o_flat_indices[(i,j)]]
                                        current_constraints.append(x_i_val + x_j_val - 1 - o_val)

            constraint_violations.append(current_constraints)

            # 7. Weight constraint
            if self.delta_w > 0:
                for b in self.bays:
                    for r in self.rows:
                        for t_idx, t in enumerate(self.tiers):
                            if t > min(self.tiers):
                                for i in self.containers:
                                    for j in self.containers:
                                        if i != j:
                                            if (self.container_weights[i] - self.container_weights[j]) > self.delta_w:
                                                x_i_val = x_sol[self.x_flat_indices[(b,r,t,i)]]
                                                x_j_val = x_sol[self.x_flat_indices[(b,r,t-1,j)]]
                                                # If x_i_val is 1 and x_j_val is 1, this configuration is invalid
                                                current_constraints.append(x_i_val + x_j_val - 1) # Should be <= 1 (so (sum - 1) <= 0)



        out["F"] = np.array(obj_values, dtype=float)
        out["G"] = np.array(constraint_violations, dtype=float)

if __name__ == "__main__":
    # Define problem parameters
    bays = [1, 2, 3] # Two bays
    rows = [1, 2] # Two rows
    tiers = [1, 2, 3] # Three tiers
    containers = ['C' + str(i) for i in range(1, 7)]  # 6 containers
    container_weights = {f'C{i}': np.random.randint(8, 16) for i in range(1, 7)}

    # Create the problem instance
    problem = ContainerLoadingProblem(bays, rows, tiers, containers, container_weights)

    # --- Define the algorithm ---
    algorithm = NSGA2(
        pop_size=100, # Population size
        sampling=BinaryRandomSampling(), # Initial random binary population
        crossover=BinomialCrossover(), # Crossover for binary variables
        mutation=BitflipMutation(), # Mutation for binary variables
        eliminate_duplicates=True # Remove duplicate solutions in the population
    )

    # --- Define termination criterion ---
    termination = ('n_gen', 500) # Stop after 500 generations

    # --- Solve the problem ---
    print("Solving the problem with pymoo...")
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1, # For reproducibility
                   verbose=True)

    print(f"\nOptimization finished with status: {res.message}")

if res.X is not None:
    print("\n--- Pareto Front Solutions Found ---")
    print(f"Number of non-dominated solutions found: {len(res.X)}")

    # Print details for all solutions on the Pareto front
    for sol_idx in range(len(res.X)):
        current_x_sol = res.X[sol_idx, :problem.num_x_vars]
        current_o_sol = res.X[sol_idx, problem.num_x_vars:]
        current_f_vals = res.F[sol_idx]
        current_g_vals = res.G[sol_idx] # Constraints for this solution

        print(f"\n--- Solution {sol_idx + 1} ---")
        print(f"  Objective Values: (Loaded Containers: {-current_f_vals[0]:.0f}, "
              f"Longitudinal Disbalance: {current_f_vals[1]:.2f}, "
              f"Latitudinal Disbalance: {current_f_vals[2]:.2f})")

        # Check feasibility for this specific solution
        total_violations = np.sum(np.maximum(0, current_g_vals))
        if total_violations < 1e-6:
            print("  Status: Feasible")
        else:
            print(f"  Status: NOT Feasible (Total Violation: {total_violations:.4f})")
            

    # --- Visualization  ---
    import matplotlib.pyplot as plt

    # Extract objective values
    f_loaded = -res.F[:, 0] # Convert back to positive loaded containers
    f_long = res.F[:, 1]
    f_lat = res.F[:, 2]

    # For 3 objectives, a 3D scatter plot is useful
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(f_loaded, f_long, f_lat, c='blue', marker='o')

    ax.set_xlabel('Loaded Containers (Maximize)')
    ax.set_ylabel('Longitudinal Disbalance (Minimize)')
    ax.set_zlabel('Latitudinal Disbalance (Minimize)')
    ax.set_title('Pareto Front for Container Loading')
    plt.grid(True)
    plt.show()

    # plot 2D projections
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(f_loaded, f_long, c='red')
    plt.xlabel('Loaded Containers')
    plt.ylabel('Longitudinal Disbalance')
    plt.title('Loaded vs. Longitudinal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(f_loaded, f_lat, c='green')
    plt.xlabel('Loaded Containers')
    plt.ylabel('Latitudinal Disbalance')
    plt.title('Loaded vs. Latitudinal')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

else:
    print("No solution found.")
