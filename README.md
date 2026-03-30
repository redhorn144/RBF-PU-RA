# RBF-PU-RA

Meshfree PDE solvers using **Radial Basis Function Partition of Unity (RBF-PU)** with the **Rational Approximation (RA)** method for stable flat-limit evaluation. Supports MPI parallelism.

## Theory

### RBF Collocation

Radial basis function (RBF) methods approximate a function as a linear combination of translates of a radially-symmetric kernel. Given nodes $\{ x_j \}_{j=1}^{N}$, an interpolant takes the form

$$s(x) = \sum_{j=1}^{N} \lambda_j \phi(\|x - x_j\|)$$

where $\phi(r) = e^{-(\varepsilon r)^2}$ is the Gaussian RBF with shape parameter $\varepsilon$. Differentiation matrices (gradient, Laplacian) are obtained analytically from the kernel derivatives and the interpolation system $\Phi \lambda = u$, yielding operators of the form $L_\phi \Phi^{-1}$.

In the **strong-form collocation** approach, a PDE like $\mathcal{L}u = f$ is enforced directly at each node: the discrete operator $L_\phi \Phi^{-1}$ replaces $\mathcal{L}$, and the resulting linear system is solved globally (here via GMRES).

### Partition of Unity (PU)

Global RBF interpolation with $N$ nodes requires $O(N^3)$ work and produces ill-conditioned systems. The partition of unity method decomposes the domain into overlapping patches, each containing $n \ll N$ nodes. A set of compactly-supported weight functions $\{w_p\}$ (Wendland $C^2$ functions) satisfying $\sum_p w_p(x) = 1$ blends the local approximations:

$$u(x) = \sum_p w_p(x) \, s_p(x)$$

Differential operators on the global approximation follow from the product rule. For example the PU Laplacian is:

$$\Delta u(x_i) = \sum_p \Big[ w_p \, (L_p \, u_p) + 2\, \nabla w_p \cdot (D_p \, u_p) + (\Delta w_p) \, u_p \Big]$$

where $L_p$, $D_p$ are the local Laplacian and gradient matrices on patch $p$. Each patch is independent, so the setup is embarrassingly parallel across MPI ranks.

### RBF-RA (Rational Approximation) for the Flat Limit

Gaussian RBFs are most accurate in the flat limit ($\varepsilon \to 0$), but the interpolation matrix $\Phi$ becomes severely ill-conditioned. The RBF-RA method (Wright & Fornberg, 2017) bypasses this by:

1. Evaluating the RBF matrices at complex-valued shape parameters $\varepsilon_k = e^{i\theta_k}$ along a contour in the complex plane, where the system remains well-conditioned.
2. Fitting a rational approximant (Pad&eacute;-type, in even powers of $\varepsilon$) to the matrix entries as functions of $\varepsilon$.
3. Extracting the $\varepsilon \to 0$ limit as the leading coefficient $a_0$ of the numerator.

This yields the stable flat-limit interpolation, gradient, and Laplacian matrices for each patch, combining spectral accuracy with numerical stability.

## Project Structure

```
strong/                    # Strong-form (collocation) solvers
  source/
    BaseHelpers.py         # Gaussian RBF kernel and derivative matrix generation
    RAHelpers.py           # RBF-RA contour method for stable flat-limit matrices
    Patch.py               # Patch dataclass (nodes, matrices, PU weights)
    PUWeights.py           # Wendland C2 weight functions and PU normalization
    Setup.py               # Patch generation, matrix computation, MPI distribution
    Operators.py           # Matrix-free PU Laplacian and gradient operators
    Solver.py              # Parallel GMRES
    Plotter.py             # Visualization utilities
  nodes/
    SquareDomain.py        # Node generation for the unit square
    StrangeDomain.py       # Node generation for a star-shaped domain
  PoissonDriver.py         # Poisson equation on [0,1]^2
  PoissonStarDomain.py     # Poisson on a star-shaped domain (manufactured solution)
  PoissonSpectralConv.py   # Spectral convergence study (error vs. nodes per patch)
  PoissonSpectrum.py       # Eigenvalue analysis of the discrete Laplacian
weak/                      # Weak-form (Galerkin) solvers (in development)
```

## Quickstart

### Platforms
| Platform | Status | Notes |
|----------|--------|-------|
| Linux | ✓ Recommended | Best supported and tested |
| Windows | ✓ Supported | Use Windows Subsystem for Linux (WSL) |
| macOS | ⚠ Untested | Should work but not officially tested |


### Prerequisites

Python 3.10+ and an MPI implementation (e.g. OpenMPI or MPICH). Install dependencies:

```bash
cd strong
python -m venv myenv
source myenv/bin/activate
pip install numpy scipy matplotlib mpi4py numba treverhines-rbf
```

### Running the Poisson Solver

Solve $-\Delta u = f$ on the unit square with homogeneous Dirichlet BCs:

```bash
cd strong
mpiexec -n 4 python PoissonDriver.py
```

This generates ~10k nodes (spacing 0.01), partitions them into overlapping patches of 30 nodes each, assembles the PU Laplacian operator, and solves via GMRES. The solution and relative $L_2$ error are printed, and a plot is saved to `solution.png`.

### Running on a Non-trivial Domain

Solve the Poisson equation on a 5-pointed star domain with non-homogeneous Dirichlet BCs and a manufactured exact solution $u = e^{x+y}$:

```bash
mpiexec -n 4 python PoissonStarDomain.py
```

Produces a three-panel figure (exact / computed / error) saved to `figures/poisson_star_domain.png`.

### Spectral Convergence Study

Verify that error decreases spectrally (exponentially) as the number of nodes per patch increases:

```bash
mpiexec -n 4 python PoissonSpectralConv.py
```

Saves a convergence plot to `figures/poisson_spectral_conv.png`.


### Key Parameters

| Parameter | Where | Effect |
|-----------|-------|--------|
| Node spacing | Driver files (e.g. `0.01`) | Controls global resolution ($N \propto h^{-2}$) |
| `nodes_per_patch` | `Setup(comm, nodes, normals, 30)` | Local patch size; larger = more accurate but slower per patch |
| `overlap` | `Setup(..., overlap=3)` | Patch overlap factor; higher = more patches, better coverage |
| GMRES tolerance | `gmres(..., tol=1e-4)` | Solver convergence threshold |
