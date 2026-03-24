import numpy as np
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Patch import Patch
from source.Setup import Setup
from source.Operators import ApplyLap
from source.Solver import gmres
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.01)
    print(f"Rank {rank} generated {nodes.shape[0]} nodes.")
else:
    nodes = None
    normals = None
    groups = None

nodes = comm.bcast(nodes, root=0)
normals = comm.bcast(normals, root=0)
groups = comm.bcast(groups, root=0)

patches, patches_for_rank = Setup(comm, nodes, normals, 30)
print(f"Rank {rank} setup complete with {len(patches)} patches.")
BCs = np.array(["dirichlet"])
bc_groups = np.array([groups['boundary:all']])
Lap = ApplyLap(comm, patches, nodes.shape[0], bc_groups, BCs)

rhs = -2*np.pi**2*np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
#rhs = -8*np.pi**2*np.sin(2*np.pi * nodes[:, 0]) * np.sin(2*np.pi * nodes[:, 1])
rhs[bc_groups[0]] = 0.0

print(f"Rank {rank} starting GMRES solve...")
if rank == 0:
    t_start = MPI.Wtime()
solution, num_iters = gmres(comm, Lap, rhs, tol=1e-4, restart=100, maxiter=100)

if rank == 0:
    t_end = MPI.Wtime()
    print(f"GMRES solve complete in {t_end - t_start:.2f} seconds.")
    print(f"GMRES converged in {num_iters} iterations.")
    print("GMRES solve complete. ")
    u_exact = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    #u_exact = np.sin(2*np.pi * nodes[:, 0]) * np.sin(2*np.pi * nodes[:, 1])
    error = np.linalg.norm(solution - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.2e}")
    PlotSolution(nodes, solution)

