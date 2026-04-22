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

####################################
# A simple square domain with minimum energy nodes.
# Only admits 1 boundary group
####################################

def MinEnergySquareOne(N):
    vert = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    nodes, groups, normals = min_energy_nodes(N, (vert, edges))

    return nodes, normals, groups

###############################
# A simple square domain with uniform nodes.
# Only admits 1 boundary group
###############################

def UniformSquareOne(N):
    x = np.linspace(0, 1, N, endpoint=True)
    y = np.linspace(0, 1, N, endpoint=True)
    
    xv, yv = np.meshgrid(x, y)
    nodes = np.column_stack([xv.ravel(), yv.ravel()])

    boundary_idx = np.where((nodes[:, 0] == 0) | (nodes[:, 0] == 1) | (nodes[:, 1] == 0) | (nodes[:, 1] == 1))[0]
    interior_idx = np.where((nodes[:, 0] > 0) & (nodes[:, 0] < 1) & (nodes[:, 1] > 0) & (nodes[:, 1] < 1))[0]
    groups = {"boundary:all": boundary_idx, "interior": interior_idx}
    normals = []
    for bidx in boundary_idx:
        if nodes[bidx, 0] == 0:
            normal = np.array([-1, 0])
        elif nodes[bidx, 0] == 1:
            normal = np.array([1, 0])
        elif nodes[bidx, 1] == 0:
            normal = np.array([0, -1])
        elif nodes[bidx, 1] == 1:
            normal = np.array([0, 1])
        else:
            raise ValueError("Node is not on the boundary")
        
        normals.append(normal)
    
    return nodes, np.array(normals), groups