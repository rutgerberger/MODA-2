import numpy as np

B = [1,2,3,4,5,6,7,8] # Number of bays
R = [1,2,3,4] # Number of rows
T = [1,2,3] # Number of tiers
D = [1,2,3,4] # Destinations
C = 150 # Number of available containers

delta_w = 5 # Delta constraint
weights = 100 * np.random.rand(C) # Array with random weights of containers initialized by 0-100
destinations = np.random.randint(1,4,size=weights.shape) # Fill with destinations randomly

X = np.zeros((len(B),len(R),len(T),C)) # Search space

F1 = np.sum(X)

def O(i: int, j: int):
    return True

def G1(i: int):
    """
    One cell per container
    """
    print(X.shape)
    print(X[:,:,:,i].shape)
    return np.sum(X[:,:,:,i]) - 1

def G2(r: int, b: int, t: int):
    """
    One container per cell
    """
    return np.sum(X[r,b,t,:]) - 1

def G3(r: int, b: int, t: int):
    """
    Containers must be stacked on another container
    """
    return np.sum(X[r,b,t,:]) - np.sum(X[r,b,t-1,:])

def G4(i: int, j: int):
    """
    Unloading constraint - the destination of container i must be
    before container j, if container i is stacked on cont. j
    """
    return destinations[j] - destinations[i] - 5 * O(i,j) 

def G5(i: int, j: int):
    """
    Container i may not weight more than container j, if i is stacked
    upon container j
    """
    return weights[i] - weights[j] - delta_w - weights[i]*(1-O(i,j))


print(G1(2))
print(G2(1,1,2))

print(F1)