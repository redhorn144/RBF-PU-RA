import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup
from source.Galerkin import assemble_poisson, apply_dirichlet_distributed
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.01)
    print(f"Rank {rank} generated {nodes.shape[0]} nodes.")
else:
    nodes   = None
    normals = None
    groups  = None

nodes   = comm.bcast(nodes,   root=0)
normals = comm.bcast(normals, root=0)
groups  = comm.bcast(groups,  root=0)

bdy_indices = groups['boundary:all']

patches, _ = Setup(comm, nodes, normals, 80, bdy_indices=bdy_indices)
print(f"Rank {rank} setup complete with {len(patches)} patches.")

# Poisson problem:  -Delta u = f  with u = 0 on boundary
# Exact solution:   u = sin(2pi x) sin(2pi y)
# RHS:              f = 8 pi^2 sin(2pi x) sin(2pi y)
def rhs_fn(x):
    return 8.0 * np.pi**2 * np.sin(2*np.pi * x[:, 0]) * np.sin(2*np.pi * x[:, 1])

comm.Barrier()
if rank == 0:
    t_start = MPI.Wtime()

# Each rank assembles its sparse partial K_local; f is globally AllReduced
N = nodes.shape[0]
K_local, f = assemble_poisson(comm, patches, N, rhs_fn)

comm.Barrier()
if rank == 0:
    t_assemble = MPI.Wtime()
    print(f"Assembly complete in {t_assemble - t_start:.2f} s")

# Apply homogeneous Dirichlet BCs to distributed K_local and global f
K_local, f = apply_dirichlet_distributed(comm, K_local, f,
                                          bdy_indices, np.zeros(len(bdy_indices)))

# Distributed matvec: each rank applies its K_local, then AllReduce
def matvec(x):
    y_local = K_local @ x
    y = np.zeros_like(y_local)
    comm.Allreduce(y_local, y, op=MPI.SUM)
    return y

K_op = LinearOperator((N, N), matvec=matvec)

# CG is collective — all ranks participate every iteration
solution, info = cg(K_op, f, rtol=1e-10)

comm.Barrier()
if rank == 0:
    t_end = MPI.Wtime()
    if info == 0:
        print(f"CG converged in {t_end - t_assemble:.2f} s")
    else:
        print(f"CG did not converge (info={info})")

    u_exact = np.sin(2*np.pi * nodes[:, 0]) * np.sin(2*np.pi * nodes[:, 1])
    error = np.linalg.norm(solution - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.2e}")

    PlotSolution(nodes, solution)