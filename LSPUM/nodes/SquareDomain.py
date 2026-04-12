import numpy as np
from rbf.pde.nodes import poisson_disc_nodes, min_energy_nodes

####################################
# A simple square domain with Poisson disc nodes.
# Only admits 1 boundary group
####################################
def PoissonSquareOne(r):
    vert = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    nodes, groups, normals = poisson_disc_nodes(r, (vert, edges))


    return nodes, normals, groups

def MinEnergySquareOne(N):
    vert = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    nodes, groups, normals = min_energy_nodes(N, (vert, edges))

    return nodes, normals, groups
