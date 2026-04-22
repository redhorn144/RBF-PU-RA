from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import solve, eigh
from scipy.spatial import cKDTree
from source.PatchTiling import BoxGridTiling2D, LarssonBox2D
from nodes.SquareDomain import PoissonSquareOne, MinEnergySquareOne
from source.LSSetup import Setup
from source.Operators import PoissonRowMatrices, InterpolationRowMatrices, assemble_dense
from source.PUWeights import NormalizeWeights

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plotfolder = "figures"

M = 2500
n_interp = 60
eval_eps = 0
bc_scale = 10


if rank == 0:
    print("Generating nodes...")
    #eval_nodes, normals, groups = PoissonSquareOne(r=0.02)
    eval_nodes, normals, groups = MinEnergySquareOne(M)

    print("Tiling domain...")
    #centers, r = BoxGridTiling2D(eval_nodes, n_interp=90, oversample_factor=beta, overlap=alpha)
    centers, r = LarssonBox2D(H=0.2, xrange=(0, 1), yrange=(0, 1), delta=0.2)

    # Count eval_nodes per patch
    tree = cKDTree(eval_nodes)
    node_counts = np.array([len(idx) for idx in tree.query_ball_point(centers, r)])
    print(f"Nodes per patch — min: {node_counts.min()}, max: {node_counts.max()}, mean: {node_counts.mean():.1f}")

    # Plot patches and nodes
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(eval_nodes[:, 0], eval_nodes[:, 1], s=2, color='steelblue', zorder=2, label='eval nodes')
    for c in centers:
        circle = mpatches.Circle(c, r, fill=False, edgecolor='tomato', linewidth=0.8, alpha=0.6)
        ax.add_patch(circle)
    ax.scatter(centers[:, 0], centers[:, 1], s=15, color='tomato', zorder=3, label='patch centers')
    ax.set_aspect('equal')
    ax.set_title(f'{len(centers)} patches, r={r:.4f}  |  nodes/patch: min={node_counts.min()}, max={node_counts.max()}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(plotfolder + '/patches.png', dpi=150)
    #plt.show()
    print("Saved patches.png")
else:
    eval_nodes = None
    normals = None
    groups = None
    centers = None
    r = None
eval_nodes = comm.bcast(eval_nodes, root=0)
normals    = comm.bcast(normals, root=0)
groups     = comm.bcast(groups, root=0)
centers    = comm.bcast(centers, root=0)
r          = comm.bcast(r, root=0)


bc_flags = np.empty(len(eval_nodes), dtype=str)
bc_flags[groups["boundary:all"]] = 'd'

bc_flags[groups["interior"]] = 'i'

local_patches = Setup(
    comm, eval_nodes, normals, bc_flags, centers, r,
    n_interp=n_interp, node_layout='vogel', assignment='round_robin',
    K=64, n=16, m=48, eval_epsilon=eval_eps,
)

NormalizeWeights(comm, local_patches, M)

# ---------------------------------------------------------------------------
# Spectrum computation
#
# The LS-PUM system for Poisson is A x = f (overdetermined, M rows, N_col
# cols where N_col = N_patches * n_interp).  Interior rows of A encode
# (Δu)(x_i) and boundary rows encode u(x_i).
#
# For the generalised eigenvalue problem  Δu = λ u  with homogeneous
# Dirichlet BCs we use interior rows only:
#
#   A_L : (N_int, N_col)  — PUM Laplacian rows at interior nodes
#   A_I : (N_int, N_col)  — PUM interpolation rows at interior nodes
#
# Normal equations give the N_col × N_col symmetric definite system:
#
#   K v = λ M v   where  K = A_L^T A_L,  M = A_I^T A_I
#
# The positive eigenvalues approximate the Dirichlet Laplacian spectrum.
# ---------------------------------------------------------------------------

N_patches = len(centers)

# Build per-patch row matrices for each operator.
# bc_scale doesn't affect interior rows so the value is irrelevant here.
Rs_lap    = PoissonRowMatrices(local_patches)
Rs_interp = InterpolationRowMatrices(local_patches)

if rank == 0:
    print("Assembling dense operator matrices...")

A_lap    = assemble_dense(comm, local_patches, Rs_lap,    M, N_patches, n_interp)
A_interp = assemble_dense(comm, local_patches, Rs_interp, M, N_patches, n_interp)

if rank == 0:
    int_idx = groups["interior"]

    A_L = A_lap[int_idx, :]     # (N_int, N_col)  Laplacian rows
    A_I = A_interp[int_idx, :]  # (N_int, N_col)  interpolation rows

    # Form the non-symmetric effective operator:
    #   A_L c ≈ λ A_I c  (LS sense)
    # → multiply through by A_I^+ = (A_I^T A_I)^{-1} A_I^T
    # → L_eff c = λ c   where  L_eff = (A_I^T A_I)^{-1} (A_I^T A_L)
    #
    # This is NOT symmetric and admits complex eigenvalues, giving a
    # full picture of the spectrum in the complex plane.
    G = A_I.T @ A_I          # (N_col, N_col) Gram matrix for interpolation
    C = A_I.T @ A_L          # (N_col, N_col) cross Gram matrix

    reg = 1e-12 * np.trace(G) / G.shape[0]
    G  += reg * np.eye(G.shape[0])

    print("Forming effective Laplacian operator  L_eff = G^{-1} C ...")
    L_eff = solve(G, C)      # (N_col, N_col), non-symmetric

    print("Computing eigenvalues of L_eff (unstabilized) ...")
    evals = np.linalg.eig(L_eff)[0]   # complex in general

    # Squared LS Laplacian: K = A_L^T A_L, always PSD.
    # K v = μ G v  →  μ = ||Δu_v||²/||u_v||² ≥ 0 by construction.
    # For a true eigenfunction Δu = λu: μ = λ² → λ = -√μ (Dirichlet BCs → λ < 0).
    # This guarantees all eigenvalues are real and in the left half-plane, unlike the
    # asymmetric Petrov-Galerkin C_sym approach which requires the discrete
    # quadrature identity ∑ u·Δu ≈ -∫|∇u|² to hold exactly.
    K = A_L.T @ A_L

    print("Computing eigenvalues of stabilized operator (K v = μ G v) ...")
    mu = eigh(K, G, eigvals_only=True)          # μ ≥ 0, ascending
    evals_stab = np.sort(-np.sqrt(np.maximum(mu, 0.0)))  # real, ≤ 0, ascending

    # Analytic Dirichlet eigenvalues for -Δ on unit square: π²(m²+n²)  (positive)
    analytic = sorted(
        np.pi**2 * (m**2 + n**2)
        for m in range(1, 40) for n in range(1, 40)
    )
    # Signed version for Δ (negative): λ_mn = -π²(m²+n²)
    analytic_neg = sorted(
        -np.pi**2 * (m**2 + n**2)
        for m in range(1, 40) for n in range(1, 40)
    )  # ascending (most negative first)

    print(f"\nFirst 6 analytic eigenvalues (Δ): {np.array(analytic_neg[:6])}")
    print(f"Unstabilized real range : [{evals.real.min():.3e}, {evals.real.max():.3e}]")
    print(f"Unstabilized imag range : [{evals.imag.min():.3e}, {evals.imag.max():.3e}]")
    print(f"Stabilized range        : [{evals_stab.min():.4e}, {evals_stab.max():.4e}]")
    print(f"All stabilized evals ≤ 0: {np.all(evals_stab <= 1e-8)}")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel 0: unstabilized — full complex plane scatter
    im_ratio = np.abs(evals.imag) / (np.abs(evals.real) + 1e-30)
    sc = axes[0].scatter(evals.real, evals.imag, c=np.log10(im_ratio + 1e-10),
                         cmap='plasma', s=6, alpha=0.8)
    for lam in analytic[:60]:
        axes[0].axvline(lam, color='cyan', lw=0.4, alpha=0.4)
    plt.colorbar(sc, ax=axes[0], label='log₁₀(|Im|/|Re|)')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xlabel('Re(λ)')
    axes[0].set_ylabel('Im(λ)')
    axes[0].set_title('Unstabilized L_eff — complex plane')

    # Panel 1: stabilized — all eigenvalues on the negative real axis
    axes[1].scatter(evals_stab, np.zeros_like(evals_stab),
                    c='steelblue', s=6, alpha=0.7, label='stabilized evals')
    for lam in analytic_neg[:80]:
        axes[1].axvline(lam, color='tomato', lw=0.4, alpha=0.5)
    axes[1].axvline(np.nan, color='tomato', lw=0.8, label='analytic −π²(m²+n²)')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_xlabel('Re(λ)')
    axes[1].set_ylabel('Im(λ)')
    axes[1].set_title('Stabilized — complex plane\n(real, left half-plane)')
    axes[1].legend(fontsize=8)

    # Panel 2: sorted accuracy comparison on the real axis
    K = min(len(evals_stab), len(analytic_neg), 100)
    idx = np.arange(K)
    axes[2].plot(idx, np.array(analytic_neg[:K]), 'o-', color='tomato',
                 ms=3, lw=1.0, label='analytic −π²(m²+n²)')
    axes[2].plot(idx, evals_stab[:K], 's--', color='steelblue',
                 ms=3, lw=1.0, label='stabilized LS-RBF-PUM')
    axes[2].set_xlabel('Eigenvalue index')
    axes[2].set_ylabel('λ')
    axes[2].set_title(f'Stabilized vs analytic — first {K}')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plotfolder + '/laplacian_spectrum.png', dpi=150)
    print("Saved laplacian_spectrum.png")
