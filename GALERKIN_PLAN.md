# Plan: Galerkin RBF-PU-RA Method

## Motivation

The current strong-form (collocation) PU-RBF method produces spurious positive-real eigenvalues in the discrete operators — approximately λ ≈ 17 s⁻¹ for the advection operator and λ ≈ 1600 s⁻¹ for the Laplacian. These force implicit time-stepping and damp physical modes (e.g., the Gaussian bump decays to ~0.3× peak after one rotation). Moving to a Galerkin (weak) form eliminates the non-physical eigenvalue pollution and produces a symmetric-positive-definite stiffness matrix for self-adjoint problems.

---

## Mathematical Formulation

### Weak Form

For a PDE of the form **L u = f** on Ω with Dirichlet data on ∂Ω, the Galerkin statement is:

> Find u_h ∈ V_h such that  a(u_h, v) = ℓ(v)  for all v ∈ V_h

For the Poisson problem −Δu = f:
```
a(u, v) = ∫_Ω ∇u_h · ∇v dΩ
ℓ(v)    = ∫_Ω f v dΩ
```

For the advection problem ∂_t u + c·∇u = 0 (semi-discrete in space):
```
a(u, v) = ∫_Ω (c·∇u_h) v dΩ
```

### PU-RBF Trial/Test Space

The approximation on each patch p uses an RBF expansion:

```
u_p^loc(x) = Σ_j  c_j^p  φ_j^p(x)        (local RBF fit on patch p)
u_h(x)     = Σ_p  w̄_p(x) u_p^loc(x)      (global PU reconstruction)
```

Test functions are taken from the same space (Galerkin). The global degrees of freedom are the nodal values c = u_h evaluated at the scattered nodes, consistent with the existing data layout.

### Patch Stiffness and Mass Contributions

For the Poisson bilinear form, patch p contributes:

```
a_p(u, v) = ∫_{supp(w̄_p)} ∇[w̄_p u_p^loc] · ∇[w̄_p v_p^loc] dΩ_p
```

Expanding via the product rule:

```
∇(w̄_p u_p^loc) = w̄_p ∇u_p^loc + u_p^loc ∇w̄_p
```

This yields three types of integral per patch:
1. **Stiffness core**:     ∫ w̄_p² (∇φ_i · ∇φ_j) dΩ_p
2. **Cross terms**:        ∫ w̄_p (∇w̄_p · ∇φ_j) φ_i dΩ_p  +  transpose
3. **Mass-like term**:     ∫ |∇w̄_p|² φ_i φ_j dΩ_p

All integrals are evaluated by quadrature over the disk supp(w̄_p) ∩ Ω.

---

## Quadrature: GLL Polar Nodes

### Construction

Each circular patch of radius R centered at x_c carries a polar quadrature rule built from **Gauss-Lobatto-Legendre (GLL)** points in the radial direction and uniform points in angle:

```
ξ_k  ∈ [-1, 1],  k = 1…n_r      (GLL nodes, includes endpoints ±1)
r_k  = R (ξ_k + 1) / 2          (mapped to [0, R])
w_k^r = R/2 w_k^GLL             (scaled GLL weights)

θ_m  = 2π m / n_θ,  m = 0…n_θ-1
w^θ  = 2π / n_θ

Quadrature point:  q_{k,m} = x_c + r_k [cos θ_m, sin θ_m]
Quadrature weight: W_{k,m} = w_k^r · w^θ · r_k               (polar Jacobian r dr dθ)
```

The endpoint r_k = 0 (from ξ_k = -1) has Jacobian factor r = 0 and does not contribute. The endpoint r_k = R (from ξ_k = +1) lies on the patch boundary where w̄_p = 0, so it contributes nothing to the weighted integrals — but it is retained for completeness and to handle the degenerate case.

Typical choices: n_r = 8–16 radial points, n_θ = 2n_r angular points.

### New Module: `source/GLLQuad.py`

Responsibilities:
- `gll_nodes_weights(n)` — compute GLL nodes/weights on [−1, 1] via eigenvalue method on the Jacobi tridiagonal matrix
- `polar_quad(center, radius, n_r, n_theta)` — returns `(points, weights)` for the full polar rule (shape `(n_r * n_theta, d)` and `(n_r * n_theta,)`)
- `eval_rbf_at_quad(quad_pts, patch_nodes, e)` — evaluate the RBF basis functions and their gradients at the quadrature points (dense, used during assembly)

---

## Boundary Patch Handling

### Identifying Boundary Patches

A patch p overlaps the domain boundary if any of its quadrature points lies outside Ω. Since the domain is defined implicitly by scattered boundary nodes with outward normals, we use the following test.

### In/Out Test

For each quadrature point q:

1. Find the nearest boundary node x_b with outward unit normal n_b (using the global boundary KDTree).
2. Compute the signed distance: `s = (q − x_b) · n_b`
3. **Inside** criterion: `s ≤ 0` (point is on the interior side of the boundary tangent plane).
4. Quadrature points with `s > tol` (outside) are excluded; their weights are set to zero.

This is a first-order in/out test that is exact for flat boundaries and gives O(h) accuracy near curved boundaries (sufficient since the nodes already resolve the boundary to O(h)).

### Implementation Note

The test is applied once during setup and stored as a boolean mask on the quadrature points. No per-solve cost is incurred.

---

## New and Modified Components

### 1. `source/GLLQuad.py` *(new)*

```
gll_nodes_weights(n)           -> (xi, w)
polar_quad(center, R, n_r, n_theta) -> (pts, weights)
filter_interior(pts, weights, bdy_nodes, bdy_normals, tol=0.0) -> (pts, weights)
eval_phi_at_pts(pts, patch_nodes, e)     -> Phi_q     shape (n_q, n_loc)
eval_grad_phi_at_pts(pts, patch_nodes, e) -> dPhi_q   shape (d, n_q, n_loc)
```

### 2. `source/Patch.py` *(modified)*

Add fields to the `Patch` dataclass:

```python
quad_pts     : np.ndarray   # (n_q, d)  quadrature points (interior only)
quad_weights : np.ndarray   # (n_q,)    quadrature weights (Jacobian included)
Phi_q        : np.ndarray   # (n_q, n_loc)      RBF basis evaluated at quad pts
dPhi_q       : np.ndarray   # (d, n_q, n_loc)   RBF grad at quad pts
w_bar_q      : np.ndarray   # (n_q,)             normalized PU weight at quad pts
gw_bar_q     : np.ndarray   # (n_q, d)           grad of PU weight at quad pts
is_boundary  : bool         # True if any quad pts were filtered out
```

### 3. `source/Setup.py` *(modified)*

After the existing `GenPatches` and `StableFlatMatrices` calls, add a new stage:

```
GenQuadrature(patches, bdy_nodes, bdy_normals, n_r, n_theta)
```

For each patch on the local rank:
1. Call `polar_quad` to get raw polar quadrature.
2. Call `filter_interior` to zero out exterior points (sets is_boundary flag).
3. Evaluate `Phi_q`, `dPhi_q` at surviving quadrature points using the RA-stable shape parameter already computed for the patch.
4. Evaluate and normalize `w_bar_q`, `gw_bar_q` at quad pts via MPI AllReduce (same normalization pass used for node weights, extended to quad pts).

### 4. `source/Galerkin.py` *(new)*

Assembles the global sparse stiffness matrix K and load vector f patch by patch.

```
assemble_stiffness(patches, nodes, ...) -> K (scipy sparse CSR, or MPI-distributed)
assemble_load(patches, nodes, rhs_fn)   -> f (dense vector)
apply_dirichlet(K, f, bdy_indices, bdy_values)
```

**Per-patch stiffness loop** (pseudo-code):

```python
for patch in local_patches:
    W  = patch.quad_weights            # (n_q,)
    wb = patch.w_bar_q                 # (n_q,)
    gw = patch.gw_bar_q                # (n_q, d)
    P  = patch.Phi_q                   # (n_q, n_loc)
    dP = patch.dPhi_q                  # (d, n_q, n_loc)

    # ∇(w̄ φ_j) at each quad point: shape (d, n_q, n_loc)
    grad_wphi = wb[None,:,None] * dP + gw.T[:,:,None] * P[None,:,:]

    # Local stiffness: K_loc[i,j] = Σ_q W_q (∇w̄φ_i · ∇w̄φ_j)
    K_loc = einsum('dqi,dqj,q->ij', grad_wphi, grad_wphi, W)

    # Scatter K_loc into global K at patch.node_indices
    scatter_add(K_global, patch.node_indices, K_loc)
```

After all patches, do MPI AllReduce on K_global.

**Per-patch load loop**:

```python
f_loc[i] = Σ_q W_q  w̄(q) φ_i(q) f(q)
         = (P.T * (W * wb * f_at_q)) summed over q
```

### 5. `source/Operators.py` *(kept for reference, not used in Galerkin path)*

The existing matrix-free `ApplyLap` / `ApplyDeriv` remain for the strong-form spectrum diagnostics. The Galerkin path builds an explicit sparse matrix instead.

### 6. `PoissonDriver.py` *(modified)*

Replace the GMRES + matrix-free path with:

```python
K, f = assemble_stiffness(...), assemble_load(...)
apply_dirichlet(K, f, bdy_indices, bdy_values)
u = spsolve(K, f)          # or distributed CG (K is SPD)
```

---

## Implementation Sequence

### Phase 1 — GLL Quadrature (`GLLQuad.py`)
- [ ] Implement `gll_nodes_weights` (Jacobi eigenvalue method)
- [ ] Implement `polar_quad` with correct Jacobian r dr dθ
- [ ] Unit test: verify ∫_{disk} 1 dA = π R² to 10 digits with n_r=8, n_θ=16
- [ ] Unit test: verify ∫_{disk} r² dA = π R⁴/2

### Phase 2 — Boundary Filter
- [ ] Implement `filter_interior` using nearest-boundary-node normal test
- [ ] Test on a square: half-disk patch at corner; verify only interior quarter survives

### Phase 3 — Patch Augmentation
- [ ] Extend `Patch` dataclass with quadrature fields
- [ ] Extend `Setup.py` → `GenQuadrature` to populate these fields
- [ ] Verify `Phi_q` reproduces known functions at quad pts

### Phase 4 — Galerkin Assembly (`Galerkin.py`)
- [ ] Implement `assemble_stiffness` (per-patch einsum loop, scatter)
- [ ] Implement `assemble_load`
- [ ] Implement `apply_dirichlet` (zero rows/cols + identity diagonal)
- [ ] MPI AllReduce on assembled K, f

### Phase 5 — Driver Integration
- [ ] Update `PoissonDriver.py` to use Galerkin assembly
- [ ] Verify Poisson solution error O(h^p) convergence (compare with exact sin/sin solution)
- [ ] Run `PoissonSpectrum.py` with Galerkin K to confirm all eigenvalues are negative real
- [ ] Update `AdvecDriver.py` — replace ApplyDeriv matvec with Galerkin advection operator (M^{-1} A, or solve M x = A u at each step)

---

## Parameter Choices

| Parameter     | Symbol  | Suggested default | Notes |
|---------------|---------|-------------------|-------|
| Radial GLL pts| n_r     | 12                | Increase for high-order accuracy |
| Angular pts   | n_θ     | 24                | 2 × n_r rule of thumb |
| In/out tol    | tol     | 0.0               | Set to small ε (~1e-10) if noisy |
| Nodes per patch | K     | 80 (unchanged)    | Existing setting |
| Patch overlap | —       | 3× (unchanged)    | Existing setting |

---

## Expected Outcomes

1. **Symmetric positive-definite K** for Poisson — enables CG instead of GMRES.
2. **No spurious positive-real eigenvalues** — explicit time-stepping becomes stable.
3. **No physical mode damping** — Gaussian bump in advection retains amplitude.
4. **Consistent order of accuracy** — Galerkin error matches strong-form order for smooth solutions.
