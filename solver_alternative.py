import numpy as np
from pulp import *

# Problem dimensions
B = list(range(8))  # Bays
R = list(range(4))  # Rows
T = list(range(3))  # Tiers
C = 150             # Number of containers
D = list(range(1, 5))  # Destinations (1 to 4)

delta_w = 5  # Max allowed weight difference between stacked containers

# Generate container data
weights = 100 * np.random.rand(C)
destinations = np.random.randint(1, 5, size=C)

# Problem
prob = LpProblem("Container_Stowage", LpMaximize)

# Decision variables: X[r][b][t][c] == 1 if container c is placed at (r, b, t)
X = LpVariable.dicts("X", (R, B, T, range(C)), cat="Binary")

# Objective: maximize number of containers placed
prob += lpSum([X[r][b][t][c] for r in R for b in B for t in T for c in range(C)]), "TotalContainersPlaced"

# G1: One cell per container
for c in range(C):
    prob += lpSum([X[r][b][t][c] for r in R for b in B for t in T]) <= 1, f"OneCellPerContainer_{c}"

# G2: One container per cell
for r in R:
    for b in B:
        for t in T:
            prob += lpSum([X[r][b][t][c] for c in range(C)]) <= 1, f"OneContainerPerCell_{r}_{b}_{t}"

# G3: Stack only on another container or bottom
for r in R:
    for b in B:
        for t in range(1, len(T)):  # Skip bottom layer
            for c in range(C):
                # If container c is at (r, b, t), then there must be a container at (r, b, t-1)
                prob += X[r][b][t][c] <= lpSum([X[r][b][t-1][cc] for cc in range(C)]), f"StackingConstraint_{r}_{b}_{t}_{c}"

# G4: Unloading constraint: if i is above j, then destination[i] < destination[j]
for r in R:
    for b in B:
        for t in range(1, len(T)):
            for i in range(C):
                for j in range(C):
                    if i != j:
                        # If i is placed above j in the same column
                        prob += X[r][b][t][i] + X[r][b][t-1][j] <= 1 + (destinations[j] <= destinations[i]), f"Unload_{r}_{b}_{t}_{i}_{j}"

# G5: Weight constraint: container i above j must be lighter
for r in R:
    for b in B:
        for t in range(1, len(T)):
            for i in range(C):
                for j in range(C):
                    if i != j:
                        # If container i is above j, enforce weight constraint
                        prob += lpSum([weights[i]]) - lpSum([weights[j]]) <= delta_w + 1000 * (2 - X[r][b][t][i] - X[r][b][t-1][j]), f"Weight_{r}_{b}_{t}_{i}_{j}"


# Solve the problem
prob.solve()

# Output results
print(f"Status: {LpStatus[prob.status]}")
print(f"Containers loaded: {value(prob.objective)}")

# Show placements
placements = []
for r in R:
    for b in B:
        for t in T:
            for c in range(C):
                if value(X[r][b][t][c]) == 1:
                    placements.append((c, r, b, t, destinations[c], round(weights[c], 2)))

# Sort and display placements
placements.sort(key=lambda x: (x[2], x[1], x[3]))  # Sort by bay, row, tier
for c, r, b, t, dest, w in placements:
    print(f"Container {c}: Row {r}, Bay {b}, Tier {t}, Destination {dest}, Weight {w}")
