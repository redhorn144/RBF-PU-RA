import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nodes.SquareDomain import PoissonSquareOne
from source.Patch import Patch
from source.Setup import Setup
from source.Operators import ApplyLap
from source.Solver import gmres
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

eval_epsilon = 0.02

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.025)
else:
    nodes = None
    normals = None
    groups = None

nodes = comm.bcast(nodes, root=0)
normals = comm.bcast(normals, root=0)
groups = comm.bcast(groups, root=0)

patches, patches_for_rank = Setup(comm, nodes, normals, 40, eval_epsilon=eval_epsilon)
if rank == 0:
    print(f"Setup complete with ~{len(patches)} patches per rank.")
BCs = np.array(["dirichlet"])
bc_groups = np.array([groups['boundary:all']])
Lap = ApplyLap(comm, patches, nodes.shape[0], bc_groups, BCs)

rhs = -2*np.pi**2*np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
rhs[bc_groups[0]] = 0.0
if rank == 0:
    print(f"Starting GMRES solve...")
if rank == 0:
    t_start = MPI.Wtime()
solution, num_iters = gmres(comm, Lap, rhs, tol=1e-4, restart=100, maxiter=10)

if rank == 0:
    t_end = MPI.Wtime()
    print(f"GMRES solve complete in {t_end - t_start:.2f} seconds.")
    print(f"GMRES converged in {num_iters} iterations.")
    #print("GMRES solve complete. ")
    u_exact = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    error = np.linalg.norm(solution - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.2e}")
    PlotSolution(nodes, solution)

# --- Spectrum computation ---
N = nodes.shape[0]
if rank == 0:
    print(f"Building dense operator matrix ({N}x{N}) for spectrum analysis...")

# Build the full matrix by applying Lap to each basis vector
A_dense = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A_dense[:, j] = Lap(e_j)

if rank == 0:
    print("Computing eigenvalues...")
    eigenvalues, eigenvectors = np.linalg.eig(A_dense)  
    problem_eig = -2*np.pi**2
    closest_idx = np.argmin(np.abs(eigenvalues - problem_eig))
    closest_eig = eigenvalues[closest_idx]
    #print(f"Problem eigenvalue: {problem_eig:.4e}")
    #print(f"Closest computed eigenvalue: {closest_eig:.4e}")
    #print(f"Distance: {np.abs(closest_eig - problem_eig):.4e}")
    print(f"maximum real part of eigenvalues: {np.max(np.real(eigenvalues)):.4e}")

    # Compute and plot the eigenvector associated with the closest eigenvalue
    
    closest_eigvec = np.real(eigenvectors[:, closest_idx])

    from matplotlib.tri import Triangulation
    tri = Triangulation(nodes[:, 0], nodes[:, 1])

    fig_eig = plt.figure(figsize=(10, 8))
    ax = fig_eig.add_subplot(111, projection='3d')
    ax.plot_trisurf(tri, closest_eigvec, cmap='RdBu_r', edgecolor='none', antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Eigenvector value")
    ax.set_title(f"Eigenvector for eigenvalue {closest_eig:.4e}")
    plt.tight_layout()
    plt.savefig("closest_eigenvector.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=5, alpha=0.7)
    ax.set_xlabel("Re($\\lambda$)")
    ax.set_ylabel("Im($\\lambda$)")
    ax.set_title("Spectrum of $-\\Delta$ in the complex plane")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"laplacian_spectrum_e{eval_epsilon:.2f}.png", dpi=150)

    # Print conditioning info
    eig_nonzero = eigenvalues[np.abs(eigenvalues) > 1e-12]
    cond_estimate = np.max(np.abs(eig_nonzero)) / np.min(np.abs(eig_nonzero))
    #print(f"Eigenvalue range of -Delta: [{eigenvalues.min():.4e}, {eigenvalues.max():.4e}]")
    print(f"Spectral condition number estimate: {cond_estimate:.4e}")