# LS-RBF-PUM Solver

A distributed-memory solver for 2D PDEs using a **Least-Squares Radial Basis Function Partition of Unity Method (LS-RBF-PUM)**. The domain is covered by overlapping RBF patches weighted by a Wendland C² partition of unity; the resulting overdetermined system is solved iteratively with LSQR and a block-Jacobi preconditioner.

Two parallel backends are provided:

| Backend | Location | Communication |
|---------|----------|---------------|
| Allreduce | `source/` | Global `MPI_Allreduce` per matvec — simple, optimal at small rank counts |
| Halo exchange | `source_halo/` | Point-to-point to ≤8 neighbours — better strong scaling at large P or M |

---

## Method overview

Each patch $p$ has a centre $c_p$, radius $r$, and a set of scattered **interpolation nodes** inside the patch. The solution is approximated as

$$u(x) \approx \sum_p \bar{w}_p(x)\, \Phi_p(x)\, \mathbf{c}_p$$

where $\bar{w}_p$ is the normalised Wendland C² weight (partition of unity), $\Phi_p$ is the local RBF basis evaluated at the interpolation nodes, and $\mathbf{c}_p$ are the per-patch coefficients.

Collocating the PDE at all evaluation nodes produces an overdetermined least-squares system assembled from three per-patch matrices:

| Matrix | Shape | Meaning |
|--------|-------|---------|
| `E[i,j]` | `(n_eval, n_interp)` | RBF evaluation $\Phi(x_i, s_j)$ |
| `D[k,i,j]` | `(d, n_eval, n_interp)` | Partial derivative $\partial_k\Phi(x_i, s_j)$ |
| `L[i,j]` | `(n_eval, n_interp)` | Laplacian $\Delta\Phi(x_i, s_j)$ |

RBF matrices are computed via a **contour-integral rational approximation** (Wright & Fornberg 2017) that removes the ill-conditioning associated with the flat-limit shape parameter.

The overdetermined system is solved with **LSQR** (Paige & Saunders 1982), optionally preconditioned by a block-Jacobi factorisation of the per-patch normal-equation blocks. Full Lanczos reorthogonalisation is also available for stagnating problems.

---

## Repository layout

```
LSPUM/
├── source/                  # Allreduce backend
│   ├── LSSetup.py           # Setup() — builds patches, normalises PU weights
│   ├── Operators.py         # PoissonRowMatrices, AdvectionDiffusionRowMatrices,
│   │                        #   InterpolationRowMatrices, GenMatFreeOps, assemble_dense
│   ├── Solvers.py           # GenIterativeSolver() factory
│   ├── LSQR.py              # Distributed LSQR (optional reorthogonalisation)
│   ├── Preconditioners.py   # Block-Jacobi, column equilibration
│   ├── PUWeights.py         # NormalizeWeights() — Allreduce-based
│   ├── Patch.py             # Patch dataclass
│   ├── PatchTiling.py       # LarssonBox2D, BoxGridTiling2D, ManualTiling2D
│   ├── PatchNodes.py        # GenPatchNodes (Vogel / polar-GLL layouts)
│   ├── RAHelpers.py         # PhiFactors, StableMatricesLS (contour-RA)
│   └── BaseHelpers.py       # Low-level Gaussian RBF kernels
│
├── source_halo/             # Halo-exchange backend (same API as source/)
│   ├── LSSetup.py           # Setup() — returns (patches, halo)
│   ├── HaloComm.py          # build_halo_comm, HaloComm dataclass,
│   │                        #   mv_exchange / rmv_exchange_1d / rmv_exchange_nd
│   ├── Operators.py         # GenMatFreeOps — halo-exchange matvec
│   ├── Solvers.py           # GenIterativeSolver — takes halo instead of M
│   ├── LSQR.py              # Distributed LSQR over owned-node slices
│   └── ...                  # PUWeights, Patch, PatchTiling, RAHelpers identical
│
├── nodes/
│   └── SquareDomain.py      # MinEnergySquareOne, PoissonSquareOne, UniformSquareOne
│
├── IterativeTest.py         # LSQR convergence benchmark, four preconditioner variants
├── HaloSanityCheck.py       # Correctness test suite for source_halo/ (5 tests)
├── HaloProfile.py           # Timing comparison: allreduce vs rr-halo vs block-halo
├── ScaleProbe.py            # Memory and setup-time scaling vs M
├── HeatEquation.py          # Demo: du/dt = Δu, backward Euler, GIF output
├── AdvectionDiffusion.py    # Demo: du/dt = νΔu + a·∇u, GIF output
└── LapSpectrum.py           # Dirichlet Laplacian spectrum via generalised EVP
```

---

## Quick start

### Dependencies

```
numpy  scipy  mpi4py  matplotlib  rbf
```

Install with conda/pip as appropriate. The `rbf` package provides minimum-energy and Poisson-disc node generation.

### Solving Poisson on the unit square

The two backends share a nearly identical call pattern. The only visible difference is that `source_halo.Setup` returns a `(patches, halo)` tuple and the solver takes `halo` instead of `M`.

**Allreduce backend (`source/`)**

```python
from mpi4py import MPI
import numpy as np
from nodes.SquareDomain import MinEnergySquareOne
from source.PatchTiling import LarssonBox2D
from source.LSSetup    import Setup
from source.Operators  import PoissonRowMatrices, GenMatFreeOps
from source.Solvers    import GenIterativeSolver

comm = MPI.COMM_WORLD

# --- domain (generate on rank 0, broadcast) ---
eval_nodes, normals, groups = MinEnergySquareOne(2000)   # rank-0 only
eval_nodes = comm.bcast(eval_nodes, root=0)
normals    = comm.bcast(normals,    root=0)
groups     = comm.bcast(groups,     root=0)
centers, r = comm.bcast(LarssonBox2D(H=0.2, xrange=(0,1), yrange=(0,1), delta=0.2), root=0)

M = len(eval_nodes)
bc_flags = np.empty(M, dtype=str)
bc_flags[groups["boundary:all"]] = 'd'
bc_flags[groups["interior"]]     = 'i'

# --- setup (PU weights normalised internally) ---
patches = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                n_interp=40, node_layout='vogel', assignment='round_robin',
                K=64, n=16, m=48, eval_epsilon=0)

# --- right-hand side ---
f = np.zeros(M)
ii = groups["interior"]
f[ii] = -2*np.pi**2 * np.sin(np.pi*eval_nodes[ii,0]) * np.sin(np.pi*eval_nodes[ii,1])

# --- solve ---
solve = GenIterativeSolver(comm, patches, M, n_interp=40,
                           bc_scale=100.0, preconditioner='block_jacobi',
                           atol=1e-10, btol=1e-10, maxiter=5000)
local_cs, itn, rnorm = solve(f)
```

**Halo-exchange backend (`source_halo/`)**

```python
from source_halo.LSSetup   import Setup
from source_halo.Operators import PoissonRowMatrices, GenMatFreeOps
from source_halo.Solvers   import GenIterativeSolver

# Setup returns (patches, halo); assignment defaults to 'block_grid_2d'
patches, halo = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                      n_interp=40, node_layout='vogel',
                      K=64, n=16, m=48, eval_epsilon=0)

f_owned = f[halo.owned_indices]   # each rank holds its slice of f

solve = GenIterativeSolver(comm, patches, halo, n_interp=40,
                           bc_scale=100.0, preconditioner='block_jacobi',
                           atol=1e-10, btol=1e-10, maxiter=5000)
local_cs, itn, rnorm = solve(f_owned)
```

---

## Setup parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_interp` | 40 | Interpolation nodes per patch (DOFs per patch) |
| `node_layout` | `'vogel'` | Layout of patch nodes: `'vogel'` (Fibonacci spiral) or `'polar_gll'` |
| `assignment` | `'round_robin'` (`source/`), `'block_grid_2d'` (`source_halo/`) | Patch-to-rank assignment strategy |
| `K` | 64 | Number of contour points for rational approximation (shape-parameter stability) |
| `n` | 16 | Denominator degree of rational approximant |
| `m` | 48 | Numerator degree of rational approximant |
| `eval_epsilon` | 0 | Shape parameter (0 = flat limit via RA) |

### Patch tiling (`LarssonBox2D`)

| Parameter | Meaning |
|-----------|---------|
| `H` | Patch spacing; produces `(1/H)²` patches on the unit square |
| `delta` | Overlap factor; patch radius `r = (1+delta)·√2·H/2` |

Smaller `H` → more patches, finer resolution. Larger `delta` → more overlap, smoother PU transitions but larger halo volumes.

### Patch assignment strategies

| Strategy | Best for |
|----------|----------|
| `round_robin` | Load balance when patch count is not a multiple of ranks |
| `block_grid_2d` | Minimises inter-rank halo communication by assigning contiguous rectangular blocks; reduces halo volume by ~4–8× compared to round-robin |

---

## Solver options

`GenIterativeSolver` parameters (shared by both backends):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `bc_scale` | 100.0 | Weight applied to boundary-condition rows |
| `preconditioner` | `'block_jacobi'` | `'block_jacobi'`, `'equilibrate'`, or `'none'` |
| `atol` / `btol` | 1e-10 | LSQR stopping tolerances |
| `maxiter` | None | Maximum LSQR iterations |
| `reorth` | False | Full Lanczos reorthogonalisation (reduces iterations on ill-conditioned problems at the cost of O(itn·M) memory and one extra Allreduce per iteration) |

The solver returns `(local_cs, itn, rnorm)`:

- `local_cs` — list of `n_interp`-length coefficient arrays, one per local patch
- `itn` — LSQR iteration count
- `rnorm` — final residual norm

---

## Running the scripts

```bash
# Correctness tests (source_halo/ backend)
mpirun -n 4 python HaloSanityCheck.py

# Performance comparison: allreduce vs halo variants
mpirun -n 1 python HaloProfile.py
mpirun -n 4 python HaloProfile.py
mpirun -n 8 python HaloProfile.py

# LSQR convergence, four preconditioner variants
mpirun -n 4 python IterativeTest.py

# Memory and setup-time scaling
python ScaleProbe.py

# PDE demos (output to figures/)
python HeatEquation.py           # heat equation GIF
python AdvectionDiffusion.py     # advection-diffusion GIF
python LapSpectrum.py            # Laplacian eigenvalue spectrum
```

---

## Reconstructing the global solution

After solving, the PUM interpolant is evaluated by summing weighted local contributions across all patches:

```python
U_local = np.zeros(M)
for p, c in zip(patches, local_cs):
    U_local[p.eval_node_indices] += p.w_bar * (p.E @ c)

U = np.zeros(M)
comm.Allreduce(U_local, U, op=MPI.SUM)   # U is the global solution on all ranks
```

---

## Supported equations

New equations are added by writing a `*RowMatrices` function (see `source/Operators.py`) that returns per-patch row matrices assembled from `E`, `D`, `L`, and the PU weights `w_bar`, `gw_bar`, `lw_bar` stored on each patch.

| Equation | Row-matrix function |
|----------|---------------------|
| Poisson $\Delta u = f$ | `PoissonRowMatrices` |
| Advection-diffusion $\nu\Delta u + \mathbf{a}\cdot\nabla u = f$ | `AdvectionDiffusionRowMatrices` |
| Interpolation $u = g$ | `InterpolationRowMatrices` |
