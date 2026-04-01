import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup
from source.Operators import ApplyDeriv, ApplyDerivAdj
from source.Quadrature import PatchLocalWeights
from source.Solver import gmres
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.025)
else:
    nodes = normals = groups = None

nodes  = comm.bcast(nodes,   root=0)
normals = comm.bcast(normals, root=0)
groups  = comm.bcast(groups,  root=0)

N = nodes.shape[0]
bc_nodes = groups['boundary:all']

patches, patches_for_rank = Setup(comm, nodes, normals, 50)
print(f"Rank {rank} setup complete with {len(patches)} patches.")

# Quadrature weights
q = PatchLocalWeights(comm, patches, N)
if rank == 0:
    print(f"Quadrature weights: min={q.min():.2e}, max={q.max():.2e}, "
          f"all positive={np.all(q > 0)}")

# Weak Laplacian: A u = sum_l D_l^T Q D_l u  with Dirichlet row replacement
d   = nodes.shape[1]
Dk  = [ApplyDeriv(comm, patches, N, k)    for k in range(d)]
DkT = [ApplyDerivAdj(comm, patches, N, k) for k in range(d)]

def WeakLap(u):
    result = np.zeros(N)
    for l in range(d):
        g = Dk[l](u)
        g = q * g
        result += DkT[l](g)
    result[bc_nodes] = u[bc_nodes]
    return result

# Solve the Poisson problem first as a sanity check
rhs = 2*np.pi**2 * np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
rhs_weak = q * rhs
rhs_weak[bc_nodes] = 0.0

print(f"Rank {rank} starting GMRES solve...")
if rank == 0:
    t_start = MPI.Wtime()
solution, num_iters = gmres(comm, WeakLap, rhs_weak, tol=1e-4, restart=100, maxiter=100)
if rank == 0:
    t_end = MPI.Wtime()
    print(f"GMRES converged in {num_iters} iterations ({t_end - t_start:.2f} s).")
    u_exact = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    error = np.linalg.norm(solution - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.2e}")
    PlotSolution(nodes, solution)

# --- Spectrum computation ---
if rank == 0:
    print(f"Building dense operator matrix ({N}x{N}) for spectrum analysis...")

A_dense = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A_dense[:, j] = WeakLap(e_j)

if rank == 0:
    print("Computing eigenvalues...")
    eigenvalues, eigenvectors = np.linalg.eig(A_dense)

    problem_eig = np.pi**2
    closest_idx = np.argmin(np.abs(eigenvalues - problem_eig))
    closest_eig = eigenvalues[closest_idx]
    print(f"Problem eigenvalue: {problem_eig:.4e}")
    print(f"Closest computed eigenvalue: {closest_eig:.4e}")
    print(f"Distance: {np.abs(closest_eig - problem_eig):.4e}")

    closest_eigvec = np.real(eigenvectors[:, closest_idx])

    from matplotlib.tri import Triangulation
    tri_plot = Triangulation(nodes[:, 0], nodes[:, 1])

    fig_eig = plt.figure(figsize=(10, 8))
    ax = fig_eig.add_subplot(111, projection='3d')
    ax.plot_trisurf(tri_plot, closest_eigvec, cmap='RdBu_r',
                    edgecolor='none', antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Eigenvector value")
    ax.set_title(f"Eigenvector for eigenvalue {closest_eig:.4e}")
    plt.tight_layout()
    plt.savefig("closest_eigenvector.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=8, alpha=0.7,
               label="Computed eigenvalues")
    ax.scatter(np.real(closest_eig), np.imag(closest_eig), c='r', s=35,
               label="Closest to $\\pi^2$")
    ax.scatter(problem_eig, 0.0, c='k', marker='x', s=45,
               label="Reference $\\pi^2$")

    ax.set_xlabel("Re($\\lambda$)")
    ax.set_ylabel("Im($\\lambda$)")
    ax.set_title("Spectrum of weak $-\\Delta$ in the complex plane")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig("laplacian_spectrum.png", dpi=150)

    eig_nonzero = eigenvalues[np.abs(eigenvalues) > 1e-12]
    cond_estimate = np.max(np.abs(eig_nonzero)) / np.min(np.abs(eig_nonzero))
    print(f"Eigenvalue range: [{np.real(eigenvalues).min():.4e}, "
          f"{np.real(eigenvalues).max():.4e}]")
    print(f"Spectral condition number estimate: {cond_estimate:.4e}")
