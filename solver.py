import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.core.variable import Binary # For defining variable types

class ContainerLoadingProblem(Problem):

    def __init__(self, bays, rows, tiers, containers, delta_w=0):
        # Store problem parameters
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
        self.containers = containers
        self.delta_w = delta_w
        self.destinations = []

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
        #print(self.x_flat_indices)

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

        """
        o_i_j is defined for i != j.
        If we have N_C containers, then for each i, there are (N_C - 1) possible j values.
        So, the total number of o_i_j variables should be N_C * (N_C - 1).
        """
        self.num_o_vars = o_idx_counter # This holds the count of o variables

        n_var = self.num_x_vars + self.num_o_vars

        # Define the number of objectives (e.g., maximize containers loaded, minimize imbalance)
        n_obj = 1

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
        breakpoint()
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
            for container_id in self.containers:
                container_x_vars = []
                for b in self.bays:
                    for r in self.rows:
                        for t in self.tiers:
                            container_x_vars.append(x_sol[self.x_flat_indices[(b,r,t,container_id)]])
                if np.sum(container_x_vars) > 0: # if the container is placed at some place
                    num_containers_loaded += 1
            obj_values.append(-num_containers_loaded) # Minimize the negative count

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



        out["F"] = np.array(obj_values, dtype=float)
        out["G"] = np.array(constraint_violations, dtype=float)

if __name__ == "__main__":
    # Define your problem parameters
    bays = [1, 2] # Two bays
    rows = [1, 2] # Two rows
    tiers = [1, 2, 3] # Three tiers
    containers = ['C1', 'C2', 'C3', 'C4'] # Four containers

    # Create the problem instance
    problem = ContainerLoadingProblem(bays, rows, tiers, containers)

    # --- Define the algorithm ---
    # For binary problems, you often need specific operators
    algorithm = GA(
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
        best_x_sol = res.X[:problem.num_x_vars]
        best_o_sol = res.X[problem.num_x_vars:]

        print("\n--- Best Solution Found ---")

        # Reconstruct container placement
        print("\nContainer Placement (x variables):")
        containers_placed = 0
        for b in bays:
            for r in rows:
                for t in tiers:
                    for c in containers:
                        flat_idx = problem.x_flat_indices[(b,r,t,c)]
                        if best_x_sol[flat_idx] > 0.5: # Consider 0.5 as threshold for binary
                            print(f"  Container {c} at Bay {b}, Row {r}, Tier {t}")
                            containers_placed += 1
        print(f"\nTotal containers loaded: {containers_placed}")

        # Reconstruct o_i_j relationships
        print("\nOn-Top Of Relationships (o variables):")
        for i in containers:
            for j in containers:
                if i != j:
                    flat_idx = problem.o_flat_indices[(i,j)]
                    if best_o_sol[flat_idx] > 0.5:
                        print(f"  Container {i} is on top of Container {j}")

        # Check constraint violations
        print("\nConstraint Violations (G values):")
        G_val = res.G
        if G_val is not None:
            # G values are the raw constraint violations. If any are > 0, the solution is infeasible.
            # In pymoo, a solution is considered feasible if all G values are <= 0.
            total_violations = np.sum(np.maximum(0, G_val)) # Sum of only positive violations
            print(f"Total constraint violation: {total_violations}")
            if total_violations < 1e-6: # Using a small tolerance for floating point comparisons
                print("The best found solution is feasible.")
            else:
                print("The best found solution is NOT fully feasible.")
                # You might want to print individual violations for debugging
                # print(G_val)

    else:
        print("No solution found.")
