import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup, SetupGaussEval
from source.Operators import ApplyWeakLap, ApplyGaussMassMul, AssembleGaussRHS
from source.Quadrature import GaussPointsAndWeights
from source.Solver import gmres
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.025)
else:
    nodes = normals = groups = None

nodes   = comm.bcast(nodes,   root=0)
normals = comm.bcast(normals, root=0)
groups  = comm.bcast(groups,  root=0)

N = nodes.shape[0]
bc_nodes = groups['boundary:all']

patches, patches_for_rank = Setup(comm, nodes, normals, 50)
print(f"Rank {rank} setup complete with {len(patches)} patches.")

# Global Delaunay non-nodal Gauss quadrature (rank 0, then broadcast)
gauss_pts, gauss_wts = GaussPointsAndWeights(comm, patches, N, p=4)

# Per-patch PHS eval matrices at the global Gauss points
SetupGaussEval(comm, patches, gauss_pts)

WeakLap = ApplyWeakLap(comm, patches, N, gauss_wts, bc_nodes=bc_nodes)
MassMul = ApplyGaussMassMul(comm, patches, N, gauss_wts)

# Sanity-check Poisson solve: -Δu = 2π²sin(πx)sin(πy), u_exact = sin(πx)sin(πy)
def f_sanity(pts):
    return 2*np.pi**2 * np.sin(np.pi*pts[:, 0]) * np.sin(np.pi*pts[:, 1])

rhs_weak = AssembleGaussRHS(comm, patches, N, gauss_pts, gauss_wts, f_sanity)
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
# Generalised eigenvalue problem K φ = λ M φ:
#   K = stiffness (ApplyWeakLap without BC rows)
#   M = mass      (E_0^T diag(gauss_wts) E_0)
# Exact Dirichlet eigenvalues on [0,1]²: λ_mn = π²(m²+n²), λ₁₁ = 2π².

if rank == 0:
    print(f"Building dense K and M matrices ({N}x{N}) for spectrum analysis...")

int_nodes = np.array([i for i in range(N) if i not in set(bc_nodes.tolist())])
N_int = len(int_nodes)

K_dense = np.zeros((N_int, N_int))
M_dense = np.zeros((N_int, N_int))
for jj, j in enumerate(int_nodes):
    e_j    = np.zeros(N)
    e_j[j] = 1.0
    K_dense[:, jj] = WeakLap(e_j)[int_nodes]
    M_dense[:, jj] = MassMul(e_j)[int_nodes]

if rank == 0:
    from scipy.linalg import eig as geig
    print("Computing generalised eigenvalues (K φ = λ M φ)...")
    eigenvalues, eigenvectors = geig(K_dense, M_dense)

    real_mask = np.abs(np.imag(eigenvalues)) < 1e-6 * np.abs(np.real(eigenvalues))
    eig_real  = np.real(eigenvalues[real_mask])
    evec_real = np.real(eigenvectors[:, real_mask])

    sort_idx  = np.argsort(eig_real)
    eig_real  = eig_real[sort_idx]
    evec_real = evec_real[:, sort_idx]

    exact_eigs = sorted(
        np.pi**2 * (m**2 + n**2)
        for m in range(1, 6) for n in range(1, 6)
    )
    exact_eigs = np.array(exact_eigs[:20])

    print(f"\nFirst 10 computed eigenvalues vs exact π²(m²+n²):")
    print(f"  {'Computed':>14}  {'Exact':>14}  {'Rel. error':>12}")
    for i in range(min(10, len(eig_real))):
        ex  = exact_eigs[i] if i < len(exact_eigs) else float('nan')
        rel = abs(eig_real[i] - ex) / ex if ex > 0 else float('nan')
        print(f"  {eig_real[i]:14.6f}  {ex:14.6f}  {rel:12.2e}")

    # Eigenmode surface plot (fundamental mode λ₁₁ = 2π²)
    mode_vec = np.zeros(N)
    mode_vec[int_nodes] = evec_real[:, 0]
    centre = np.argmin(np.linalg.norm(nodes - 0.5, axis=1))
    if mode_vec[centre] < 0:
        mode_vec = -mode_vec

    from matplotlib.tri import Triangulation
    tri_plot = Triangulation(nodes[:, 0], nodes[:, 1])

    fig_eig = plt.figure(figsize=(10, 8))
    ax3 = fig_eig.add_subplot(111, projection='3d')
    ax3.plot_trisurf(tri_plot, mode_vec, cmap='RdBu_r',
                     edgecolor='none', antialiased=True)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("Eigenvector value")
    ax3.set_title(f"Fundamental eigenmode  "
                  f"(computed $\\lambda_{{11}}$ = {eig_real[0]:.4f}, "
                  f"exact $2\\pi^2$ = {2*np.pi**2:.4f})")
    plt.tight_layout()
    plt.savefig("closest_eigenvector.png", dpi=150)

    # Spectrum index plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(np.arange(1, len(eig_real) + 1), eig_real,
               s=18, color='steelblue', label="Computed $\\lambda_k$", zorder=3)
    exact_in_range = exact_eigs[exact_eigs <= eig_real.max() * 1.05]
    for ex in exact_in_range:
        ax.axhline(ex, color='k', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.axhline(exact_in_range[0], color='k', linewidth=0.6, linestyle='--',
               alpha=0.5, label="Exact $\\pi^2(m^2+n^2)$")
    ax.set_xlabel("Eigenvalue index $k$")
    ax.set_ylabel("$\\lambda_k$")
    ax.set_title("Generalised spectrum of weak $-\\Delta$  (K$\\phi$ = $\\lambda$M$\\phi$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("laplacian_spectrum.png", dpi=150)

    eig_pos = eig_real[eig_real > 1e-10]
    cond_estimate = eig_pos.max() / eig_pos.min() if len(eig_pos) > 1 else float('nan')
    print(f"\nEigenvalue range: [{eig_real.min():.4e}, {eig_real.max():.4e}]")
    print(f"Spectral condition number (K relative to M): {cond_estimate:.4e}")
