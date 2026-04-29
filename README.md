# RBF-PU-RA

Meshfree PDE solvers using **Radial Basis Function Partition of Unity (RBF-PU)** with the **Rational Approximation (RA)** method for stable flat-limit evaluation. Three MPI-parallel backends are provided:

| Backend | Formulation | Communication | Use case |
|---------|-------------|---------------|----------|
| [`strong/`](strong/) | strong-form collocation | global `Allreduce` | reference / small problems |
| [`LSPUMAllreduce/`](LSPUMAllreduce/) | least-squares PUM | global `Allreduce` | LS prototyping, easy to read |
| [`LSPUMHalo/`](LSPUMHalo/) | least-squares PUM | point-to-point halo + SAS | scalable production solver |

---

## Quickstart

### Platforms
| Platform | Status |
|----------|--------|
| Linux | recommended |
| Windows | via WSL |
| macOS | untested |

### Prerequisites

Python 3.10+ and an MPI implementation (OpenMPI or MPICH). Each backend ships its own virtual environment under `<backend>/myenv/`. To build one from scratch:

```bash
cd LSPUMHalo                       # or strong/  or LSPUMAllreduce/
python -m venv myenv
source myenv/bin/activate
pip install numpy scipy matplotlib mpi4py numba treverhines-rbf
```

### Run the strong-form Poisson solver

`-Δu = f` on the unit square with homogeneous Dirichlet BCs:

```bash
cd strong
mpiexec -n 4 python PoissonDriver.py
```

Generates ~10k nodes, partitions into overlapping patches of 30 nodes, assembles the PU Laplacian, and solves via GMRES. Solution image is written to [strong/solution.png](strong/solution.png).

Star domain with non-homogeneous Dirichlet BCs and manufactured solution `u = exp(x+y)`:

```bash
mpiexec -n 4 python PoissonStarDomain.py
```

Spectral convergence study (error vs. nodes per patch):

```bash
mpiexec -n 4 python PoissonSpectralConv.py
```

### Run the LS-PUM Poisson solver (Allreduce backend)

Compares plain LSQR against three preconditioners (column equilibration, block Jacobi, block Jacobi + reorthogonalisation):

```bash
cd LSPUMAllreduce
mpiexec -n 4 python IterativeTest.py
```

### Run the LS-PUM Poisson solver (halo backend)

Same problem, halo-exchange matvec, SAS-preconditioned PCG. Star domain with manufactured Dirichlet BCs:

```bash
cd LSPUMHalo
mpiexec -n 4 python PoissonStarDomain.py
```

Output: [LSPUMHalo/figures/poisson_star_domain.png](LSPUMHalo/figures/poisson_star_domain.png).

### Time-dependent problems (halo backend)

Heat equation, backward Euler with SAS-PCG:

```bash
mpiexec -n 4 python HeatEquation.py            # → figures/heat.gif
```

Advection-diffusion, BDF3/EXT3 IMEX (implicit diffusion + explicit advection):

```bash
mpiexec -n 4 python AdvectionDiffusion.py      # → figures/advection_diffusion.gif
```

### Scaling and diagnostics

```bash
cd LSPUMHalo
mpiexec -n 4 python HaloSanityCheck.py         # ownership, PU=1, adjointness, solve correctness
mpiexec -n 4 python HaloProfile.py             # allreduce vs halo (rr / block_grid_2d) timings
bash run_scaling.sh                            # strong + weak scaling sweeps
```

Results land in [LSPUMHalo/results/](LSPUMHalo/results/).

---

## Theory

### RBF collocation

A radial basis function (RBF) interpolant has the form

$$s(x) = \sum_{j=1}^{N} \lambda_j \, \phi(\|x - x_j\|), \qquad \phi(r) = e^{-(\varepsilon r)^2}.$$

Differentiation matrices (gradient, Laplacian) follow analytically from the kernel derivatives and the interpolation system $\Phi \lambda = u$, giving operators of the form $L_\phi \Phi^{-1}$.

In **strong-form collocation** (the [`strong/`](strong/) backend), a PDE $\mathcal{L}u = f$ is enforced directly at each node and the resulting global linear system is solved (here via GMRES).

### Partition of Unity (PU)

Global RBF interpolation costs $O(N^3)$ and is severely ill-conditioned. PU decomposes the domain into overlapping patches, each containing $n \ll N$ nodes, and blends local approximations using compactly-supported weights $\{w_p\}$ (Wendland $C^2$ functions) satisfying $\sum_p w_p(x) = 1$:

$$u(x) = \sum_p w_p(x) \, s_p(x).$$

Differential operators on the global approximation follow from the product rule, e.g.

$$\Delta u(x_i) = \sum_p \Big[ w_p \, (L_p \, u_p) + 2\, \nabla w_p \cdot (D_p \, u_p) + (\Delta w_p) \, u_p \Big].$$

Each patch is independent, so setup is embarrassingly parallel across MPI ranks.

### Least-squares PU (LS-RBF-PUM)

The collocation variant requires the patch nodes to coincide with the evaluation points; the **least-squares** variant decouples them. Per patch:

- A **fixed set of `n_interp` interpolation nodes** (e.g. Vogel/golden-spiral disk points) — identical in every patch up to translation, which lets us factor $\Phi(\text{nodes}, \text{nodes})$ once and reuse it across patches.
- A larger set of **`n_eval_p` evaluation nodes** lying inside the patch ball (typically $n_{\text{eval},p} \approx 2\text{–}3 \cdot n_{\text{interp}}$).

The discrete operator is rectangular ($M \times N$ with $M > N$, $N = $ patches $\times \, n_{\text{interp}}$), giving an over-determined least-squares system $\min_x \|A x - b\|_2$. Boundary rows are weighted by a `bc_scale` factor (default 100) so Dirichlet conditions are enforced strongly relative to interior residuals.

Compared to collocation, LS-PUM:

- Tolerates patches that overhang the domain (interp nodes that are outside still contribute coverage).
- Decouples patch density from node density — patches can be tiled regularly even on irregular domains.
- Requires an iterative least-squares solver (LSQR, or PCG on the normal equations).

### RBF-RA (Rational Approximation) for the flat limit

Gaussian RBFs are most accurate as $\varepsilon \to 0$, but $\Phi$ becomes singular. The RBF-RA method (Wright & Fornberg, 2017) bypasses this by:

1. Evaluating the RBF matrices at complex shape parameters $\varepsilon_k = e^{i\theta_k}$ along a contour where $\Phi$ remains well-conditioned.
2. Fitting an even-power Padé-type rational approximant in $\varepsilon$ to each matrix entry.
3. Extracting the $\varepsilon \to 0$ limit as the leading numerator coefficient $a_0$.

This yields stable flat-limit interpolation, gradient, and Laplacian matrices for each patch, combining spectral accuracy with numerical stability. The implementation lives in `RAHelpers.py` (one in each backend) — `PhiFactors` LU-factors $\Phi$ at every contour point once, then `StableMatricesLS` reuses those factors for every patch.

### Halo-exchange parallelism

The Allreduce backends do an `Allreduce` of an $M$-vector inside every matvec, which dominates communication cost as ranks grow. The halo backend ([`LSPUMHalo/source_halo/`](LSPUMHalo/source_halo/)):

- Assigns each evaluation node to a unique **owner rank** via Voronoi on the patch centres.
- Per matvec: each patch computes its row contributions, accumulates them locally to owned nodes, and ships only the small **halo** subset to neighbour ranks via point-to-point `Isend`/`Irecv`.
- Replaces the $O(M)$ Allreduce with $O(\text{halo})$ neighbour exchanges plus a single scalar Allreduce for the LSQR/PCG norms.

Two patch-to-rank assignment policies are supported:
- `round_robin` — balanced load, irregular halo graph.
- `block_grid_2d` — contiguous rectangular blocks on the patch grid; minimises halo perimeter for Larsson box tilings.

### Preconditioning

LS-PUM normal-equation systems are stiff. The available preconditioners (`source/Preconditioners.py` and `source_halo/Preconditioners.py`):

- **`none`** — plain LSQR baseline.
- **`equilibrate`** — column scaling $P = \text{diag}(\|A_{:,j}\|)$; patch-local, no communication.
- **`block_jacobi`** — Cholesky of each patch's diagonal block $R_p^T R_p$; right-preconditioned LSQR.
- **`sas`** (halo backend only) — symmetric additive Schwarz on the normal equations. For each patch $p$ builds an extended local system over its 1-ring neighbourhood $N(p)$ and Cholesky-factors it; applied as a left preconditioner to PCG. Restores robust convergence on hard problems where LSQR stagnates.

### Time integration

Time-dependent problems in [`LSPUMHalo/`](LSPUMHalo/) use IMEX schemes built from the same row-matrix machinery:

- `HeatStepRowMatrices` — backward-Euler heat: $(I - \Delta t \, \Delta) u^{n+1} = u^n$.
- `HelmholtzStepRowMatrices` — generic BDF diffusion step: $(\alpha I - \Delta t \, \nu \, \Delta) u^{n+1} = f$, used for BDF1/BDF2/BDF3.
- `AdvectionRowMatrices` — explicit `−a·∇u` evaluator for IMEX RHS.
- `ADStepRowMatrices` — fully implicit advection-diffusion step.

Each Helmholtz/heat step calls `GenIterativeSolver` once per timestep with the SAS preconditioner factored once up-front.

---

## Code structure & API

### `strong/` — strong-form collocation backend

```
strong/
  source/
    BaseHelpers.py          Gaussian RBF kernel and derivative-matrix generators
    RAHelpers.py            RBF-RA contour method for stable flat-limit matrices
    Patch.py                Patch dataclass (nodes, matrices, PU weights)
    PUWeights.py            Wendland C2 weights and PU normalisation
    Setup.py                Patch generation, matrix computation, MPI distribution
    Operators.py            Matrix-free PU Laplacian and gradient operators
    Solver.py               Parallel GMRES
    Plotter.py              Visualisation utilities
  nodes/
    SquareDomain.py         Node generation for the unit square
    StrangeDomain.py        Node generation for a star-shaped domain
  PoissonDriver.py          Poisson on [0,1]^2
  PoissonStarDomain.py      Poisson on star domain (manufactured u = e^{x+y})
  PoissonSpectralConv.py    Spectral convergence study
  PoissonSpectrum.py        Eigenvalue analysis of the discrete Laplacian
  AdvecDriver.py            Time-dependent advection demo
  ConditionDriver.py        Conditioning experiments
```

**Entry point:**

```python
from source.Setup import Setup
from source.Operators import GenPoissonOps      # matvec, rmatvec
from source.Solver import gmres

patches = Setup(comm, eval_nodes, normals, n_interp, overlap=3)
matvec  = GenPoissonOps(comm, patches, M)
u, info = gmres(comm, matvec, rhs, tol=1e-4)
```

### `LSPUMAllreduce/` — least-squares backend, Allreduce matvec

```
LSPUMAllreduce/
  source/
    BaseHelpers.py          Gaussian kernels & evaluation matrices (eval ≠ nodes)
    RAHelpers.py            PhiFactors / StableMatricesLS — factor Φ once, reuse per patch
    Patch.py                LS-PUM patch dataclass
    PatchNodes.py           Vogel-disk and polar-GLL interpolation node layouts
    PatchTiling.py          BoxGridTiling2D / ManualTiling2D / LarssonBox2D
    PUWeights.py            Wendland C2 weights with global-Allreduce normalisation
    LSSetup.py              Setup() — builds local Patches, broadcasts Φ factors, normalises weights
    Operators.py            Row-matrix constructors + GenMatFreeOps (Allreduce matvec)
    Preconditioners.py      GenBlockJacobi, GenDiagEquil  (no communication)
    LSQR.py                 Distributed LSQR (Paige & Saunders), Allreduce norms
    Solvers.py              GenIterativeSolver — wraps preconditioner + LSQR
  nodes/                    SquareDomain, StrangeDomain (shared with halo backend)
  IterativeTest.py          Compare LSQR variants on a Poisson square problem
```

**Entry point:**

```python
from source.LSSetup import Setup
from source.Operators import PoissonRowMatrices
from source.Solvers import GenIterativeSolver

patches = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                n_interp=40, node_layout='vogel', K=64, n=16, m=48)

solve = GenIterativeSolver(comm, patches, M, n_interp,
                           bc_scale=100.0, preconditioner='block_jacobi',
                           atol=1e-10, btol=1e-10, maxiter=20000)
local_cs, itn, rnorm = solve(rhs)            # rhs is the global M-vector
```

### `LSPUMHalo/` — least-squares backend, halo-exchange matvec

```
LSPUMHalo/
  source_halo/
    BaseHelpers.py          (shared with Allreduce backend, vendored copy)
    RAHelpers.py            (shared)
    Patch.py                (shared)
    PatchNodes.py           Vogel / polar-GLL layouts
    PatchTiling.py          LarssonBox2D regular tiling (used by all LSPUMHalo drivers)
    HaloComm.py             build_halo_comm — Voronoi ownership, neighbour graph,
                            mv_exchange / rmv_exchange (point-to-point Isend/Irecv)
    PUWeights.py            Wendland C2 normalisation via halo exchange (no Allreduce)
    LSSetup.py              Setup() — also computes patch→rank map and HaloComm;
                            assignment ∈ {'block_grid_2d', 'round_robin'}
    Operators.py            Row-matrix constructors (Poisson, AdvecDiff, Heat,
                            Helmholtz, Interp, AD step) + halo-aware GenMatFreeOps
    Preconditioners.py      GenBlockJacobi, GenDiagEquil, GenSAS (Schwarz on normal eqns)
    LSQR.py                 Halo-exchange LSQR (no O(M) Allreduce)
    PCG.py                  Preconditioned CG on normal equations (for SAS preconditioner)
    Solvers.py              GenIterativeSolver — dispatches LSQR vs PCG by preconditioner
  nodes/                    SquareDomain, StrangeDomain
  PoissonStarDomain.py      LS-PUM Poisson on star domain with non-homogeneous Dirichlet
  HeatEquation.py           Backward-Euler heat (SAS-PCG)
  AdvectionDiffusion.py     BDF3/EXT3 IMEX advection-diffusion
  HaloSanityCheck.py        Correctness tests (ownership, PU, adjointness, vs Allreduce)
  HaloProfile.py            allreduce vs halo (rr / block_grid_2d) timing comparison
  ScalingTest.py            Component-level scaling analysis (size / rank / weak sweeps)
  run_scaling.sh            Strong + weak scaling driver script
  results/                  Scaling output (txt)
  figures/                  Generated plots and animations
```

**Entry point:**

```python
from source_halo.LSSetup import Setup
from source_halo.PatchTiling import LarssonBox2D
from source_halo.Operators import PoissonRowMatrices
from source_halo.Solvers import GenIterativeSolver

centers, r = LarssonBox2D(H=0.1, xrange=(0,1), yrange=(0,1), delta=0.2)

patches, halo = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                      n_interp=40, node_layout='vogel',
                      assignment='block_grid_2d',
                      K=64, n=16, m=48, eval_epsilon=0)

Rs    = PoissonRowMatrices(patches, bc_scale=100.0)
solve = GenIterativeSolver(comm, patches, halo, n_interp, Rs,
                           preconditioner='sas', atol=1e-10, maxiter=2000)

f_owned = f[halo.owned_indices]              # rank-local slice of RHS
local_cs, itn, rnorm = solve(f_owned)        # returns one (n_interp,) per local patch
```

To reconstruct the global solution at every eval node:

```python
def reconstruct(comm, patches, local_cs, M):
    U_local = np.zeros(M)
    for p, c in zip(patches, local_cs):
        U_local[p.eval_node_indices] += p.w_bar * (p.E @ c)
    U = np.zeros(M)
    comm.Allreduce(U_local, U, op=MPI.SUM)
    return U
```

### Key parameters (LS backends)

| Parameter | Where | Effect |
|-----------|-------|--------|
| `M` | driver | Total number of evaluation nodes |
| `n_interp` | `Setup(...)` | Interpolation DOFs per patch (40 typical) |
| `H` | `LarssonBox2D(H=...)` | Patch centre spacing; smaller → more patches |
| `delta` | `LarssonBox2D(..., delta=...)` | Halo factor on patch radius $r = (1+\delta)\sqrt{2}\,H/2$ |
| `bc_scale` | `*RowMatrices(..., bc_scale=...)` | Weight on Dirichlet rows (typically 100) |
| `node_layout` | `Setup(..., node_layout=...)` | `'vogel'` (recommended) or `'polar_gll'` |
| `assignment` | `Setup(..., assignment=...)` | `'block_grid_2d'` (halo-friendly) or `'round_robin'` |
| `preconditioner` | `GenIterativeSolver(...)` | `'none'`, `'equilibrate'`, `'block_jacobi'`, or `'sas'` (halo only) |
| `K`, `n`, `m` | `Setup(...)` | RA contour points, denom degree, numer degree |
| `eval_epsilon` | `Setup(..., eval_epsilon=...)` | `0` for flat limit, else evaluate RA at this $\varepsilon$ |
