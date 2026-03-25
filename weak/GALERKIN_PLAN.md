# Weak Form Poisson via RBF-PU with Patch-Local Quadrature

## Context

The current framework solves **strong form** Poisson: it enforces $-\Delta u = f$ pointwise at each node using the PU-RBF collocation operator. This works but produces spurious positive eigenvalues in the discrete operator (as noted in `AdvecDriver.py`), which cause stability issues for time-dependent problems. A **weak (Galerkin) formulation** eliminates these spurious eigenvalues because the resulting operator is symmetric positive definite by construction.

The goal is to convert the Poisson solve to a weak form using the existing scattered (Poisson disc) nodes as quadrature points.

---

## 1. The Weak Form of Poisson

### Strong form
$$-\Delta u = f \quad \text{in } \Omega, \qquad u = 0 \quad \text{on } \partial\Omega$$

### Weak form
Multiply by a test function $v \in H^1_0(\Omega)$ and integrate by parts:

$$\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f \, v \, dx$$

The boundary term $\int_{\partial\Omega} v \, \frac{\partial u}{\partial n} \, ds$ vanishes because $v = 0$ on $\partial\Omega$ (for homogeneous Dirichlet).

### Discrete weak form in the PU-RBF basis

The PU-RBF basis functions $\{\Psi_i\}$ are cardinal: $\Psi_i(x_j) = \delta_{ij}$. They are built from:
$$\Psi_i(x) = \sum_{p: i \in p} \bar{w}_p(x) \, \chi_{l(i,p)}^p(x)$$

where $\chi_j^p$ is the $j$-th cardinal RBF interpolant on patch $p$, and $\bar{w}_p$ is the normalized PU weight.

Expanding $u_h = \sum_j u_j \Psi_j$ and testing with $v = \Psi_i$:

$$\sum_j \underbrace{\left(\int_\Omega \nabla \Psi_j \cdot \nabla \Psi_i \, dx\right)}_{A_{ij}} u_j = \underbrace{\int_\Omega f \, \Psi_i \, dx}_{b_i}$$

In matrix form: $A \mathbf{u} = \mathbf{b}$.

### What "cardinal" means and why it matters

A set of basis functions $\{\Psi_i\}$ is **cardinal** with respect to nodes $\{x_j\}$ if:
$$\Psi_i(x_j) = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$
Each basis function equals 1 at "its own" node and 0 at every other node — the same property that Lagrange interpolation polynomials have.

#### Local cardinal interpolants on each patch

On each patch $p$, the RBF interpolation matrix $\Phi$ maps coefficient vectors to function values at the patch nodes. Its inverse $\Phi^{-1}$ goes the other direction: given values at nodes, it recovers the RBF coefficients. The $j$-th row of $\Phi^{-1}$ defines the **$j$-th cardinal interpolant** $\chi_j^p$ — the unique RBF combination that equals 1 at node $j$ and 0 at the other patch nodes.

In the code, $\chi_j^p$ is never formed explicitly as a function. Instead, the derivative matrices $D_k$ and $L$ (computed via `StableFlatMatrices` in `RAHelpers.py`) encode the action of interpolating then differentiating. Computing `D[k] @ u_local` implicitly:
1. Expands $u$ in the cardinal basis: $u(x) = \sum_j u_j \chi_j^p(x)$
2. Differentiates: $\partial_k u(x_i) = \sum_j (\partial_k \chi_j^p(x_i)) u_j$

The matrix $D_k^p$ has entries $(D_k^p)_{ij} = \partial_k \chi_j^p(x_i)$ — derivatives of the cardinal functions evaluated at the nodes.

#### Global cardinal basis via partition of unity

Each patch gives local cardinal functions, but they only live on one patch. The PU weights $\bar{w}_p(x)$ (computed in `NormalizeWeights`) stitch them into a global basis:
$$\Psi_i(x) = \sum_{p \ni i} \bar{w}_p(x) \, \chi_{l(i,p)}^p(x)$$
where $l(i,p)$ is the local index of global node $i$ within patch $p$. The global function $\Psi_i$ is still cardinal because:
- At node $x_i$: the PU weights sum to 1 ($\sum_p \bar{w}_p(x_i) = 1$), and each $\chi_{l(i,p)}^p(x_i) = 1$, so $\Psi_i(x_i) = 1$.
- At node $x_j \neq x_i$: each $\chi_{l(i,p)}^p(x_j) = 0$ (cardinal on the patch), so $\Psi_i(x_j) = 0$.

This is exactly what `ApplyDeriv` in `Operators.py` implements — the product rule on $\bar{w}_p \cdot \chi^p$.

#### Practical consequences for the weak form

The cardinal property gives two important simplifications:

**1. The coefficient vector IS the solution values.** Since $u_h(x_j) = \sum_i u_i \Psi_i(x_j) = u_j$, the unknowns $u_j$ are directly the function values at the nodes. No mass-matrix solve is needed to extract physical values from coefficients.

**2. The load vector is trivial.** The load vector integral
$$b_i = \int_\Omega f(x) \Psi_i(x) \, dx \approx \sum_k q_k f(x_k) \Psi_i(x_k)$$
collapses to $b_i = q_i f(x_i)$ because $\Psi_i(x_k) = 0$ for all $k \neq i$. Every term in the quadrature sum vanishes except $k = i$. So instead of assembling a full mass matrix $M_{ij} = \int \Psi_j \Psi_i \, dx$, the load vector is just an elementwise multiply: $\mathbf{b} = Q\mathbf{f}$.

Note that the stiffness matrix $A_{ij} = \int \nabla\Psi_j \cdot \nabla\Psi_i \, dx$ does **not** simplify this way — the *derivatives* $\nabla\Psi_i(x_k)$ are generally nonzero at nodes other than $x_i$. Cardinal means the functions interpolate, not their derivatives.

---

## 2. Quadrature with Existing Nodes

### The quadrature approximation

Approximate all integrals using the $N$ existing nodes $\{x_k\}$ with weights $\{q_k\}$:

$$\int_\Omega g(x) \, dx \approx \sum_{k=1}^N q_k \, g(x_k)$$

### Stiffness matrix with quadrature

$$A_{ij} \approx \sum_{k=1}^N q_k \, \nabla\Psi_j(x_k) \cdot \nabla\Psi_i(x_k)$$

In matrix notation, defining the PU derivative matrix $\mathbf{D}_l$ with entries $(\mathbf{D}_l)_{kj} = \frac{\partial \Psi_j}{\partial x_l}(x_k)$:

$$\boxed{A = \sum_{l=1}^d \mathbf{D}_l^T \, Q \, \mathbf{D}_l}$$

where $Q = \mathrm{diag}(q_1, \ldots, q_N)$.

### Load vector with quadrature

Since $\Psi_i(x_k) = \delta_{ik}$ (cardinal property):

$$b_i \approx \sum_k q_k f(x_k) \Psi_i(x_k) = q_i f(x_i)$$

$$\boxed{\mathbf{b} = Q \mathbf{f}}$$

This is trivial — just multiply the RHS by quadrature weights.

---

## 3. Computing Quadrature Weights

### Design requirements

The quadrature weights $\{q_k\}$ must satisfy two requirements:

1. **Positive definite:** All $q_k > 0$, so that $Q = \mathrm{diag}(q)$ is SPD and $A = \sum_l D_l^T Q D_l$ is SPD on interior nodes.
2. **Accuracy matching:** The quadrature error must not exceed the approximation error. The Gaussian RBF flat limit on $n$ nodes reproduces polynomials of degree $m$ (the unisolvent degree, $m \sim \sqrt{n}$ in 2D). The quadrature must be exact for polynomials of degree $m$ to avoid being the bottleneck.

### Why naive approaches fail

- **Minimum-norm global polynomial-exact weights** ($q = V(V^T V)^{-1} M$ globally): almost certainly produces negative weights for scattered nodes, breaking positive-definiteness.
- **Voronoi cell areas**: always positive, but only $O(h^2)$ accurate. For the Gaussian RBF with unisolvent degree $m \gg 2$, the quadrature error dominates and caps overall accuracy at $O(h^2)$.

### Chosen approach: patch-local constrained QP

Decompose the global integral using the partition of unity:
$$\int_\Omega g \, dx = \sum_p \int_\Omega \bar{w}_p \, g \, dx \approx \sum_p \sum_{k \in p} \hat{q}_k^p \, \bar{w}_p(x_k) \, g(x_k)$$

On each patch $p$, find positive local weights $\hat{q}^p$ that are polynomial-exact of degree $m$ over the patch support disk. Then assemble:
$$\boxed{q_k = \sum_{p \ni k} \hat{q}^p_{l(k,p)} \cdot \bar{w}_p(x_k)}$$

**Positivity:** $\hat{q}^p_k > 0$ (enforced by the QP) and $\bar{w}_p(x_k) > 0$ for all patch nodes (nodes lie strictly inside the support), so every $q_k > 0$.

**Accuracy:** Polynomial-exact quadrature of degree $m$ applied to an analytic integrand gives error $O((h/\rho)^{m+1})$ for some $\rho > 1$, which is spectrally convergent in $m$. Since the flat-limit Gaussian approximation also converges at this rate in $m$, the quadrature does not degrade the method's accuracy.

**Patch-parallel:** Each patch's QP is solved independently. The global assembly requires one `Allreduce`.

**Feasibility:** By a Tchakaloff-type argument, for well-distributed Poisson disc nodes with $n \gg \binom{m+2}{2}$, positive polynomial-exact weights almost certainly exist. Fall back to Voronoi weights for any patch where the QP is infeasible (degenerate configurations).

### Step-by-step local QP on patch $p$

**1. Determine the polynomial degree** $m$: the largest $m$ such that $\binom{m+2}{2} \leq n$ and the local Vandermonde has full rank. For $n = 50$: $m = 8$ (since $\binom{10}{2} = 45 \leq 50 < 55 = \binom{11}{2}$).

**2. Build the local Vandermonde** in patch-centered coordinates $\xi_k = x_k^p - c_p$:
$$(P^p)_{k,\alpha} = \xi_k^a \eta_k^b, \quad |\alpha| = a + b \leq m, \quad P^p \in \mathbb{R}^{n \times M}$$

**3. Compute disk moments analytically** over $D(0, R_p)$:
$$M^p_\alpha = \int_{D(0,R_p)} \xi^a \eta^b \, d\xi \, d\eta = \begin{cases} \dfrac{2 R_p^{a+b+2} \,\Gamma\!\left(\frac{a+1}{2}\right)\Gamma\!\left(\frac{b+1}{2}\right)}{\Gamma\!\left(\frac{a+b}{2}+2\right)} & a, b \text{ both even} \\[6pt] 0 & \text{otherwise} \end{cases}$$

**4. Solve the QP:**
$$\hat{q}^p = \arg\min_{q \geq 0} \left\| q - \frac{\pi R_p^2}{n}\mathbf{1} \right\|_2^2 \quad \text{subject to} \quad (P^p)^T q = M^p$$

This is a small convex QP: $n \sim 50$ unknowns, $M \leq 45$ equality constraints. Use `scipy.optimize.lsq_linear`, `quadprog`, or `cvxpy`.

---

## 4. Matrix-Free Weak Laplacian Operator

### The forward derivative (existing)

`ApplyDeriv` computes $v = D_l u$. Per patch $p$, contributing to nodes $i \in \text{idx}_p$:

```
result_local[idx] += w_bar * (D[l] @ u[idx]) + gw_bar[:, l] * u[idx]
```

then `Allreduce`.

### Deriving the adjoint derivative

The global derivative matrix has entries:
$$(\mathbf{D}_l)_{ij} = \sum_{p:\, i,j \in p} \bar{w}_p(x_i)\,(D_l^p)_{l(i,p),\,l(j,p)} + \delta_{ij} \sum_{p \ni i} (\partial_l \bar{w}_p)(x_i)$$

Transposing and collecting by patch:
$$(D_l^T g)_j = \sum_{p \ni j} \left[ \bigl((D_l^p)^T (\bar{w}_p \odot g_p)\bigr)_{l(j,p)} + (\partial_l \bar{w}_p)(x_j)\, g_j \right]$$

`ApplyDerivAdj` per patch, contributing to nodes $j \in \text{idx}_p$:

```
result_local[idx] += D[l].T @ (w_bar * g[idx]) + gw_bar[:, l] * g[idx]
```

then `Allreduce`.

Key differences from the forward:
1. `w_bar` multiplies the **input** $g_p$ before the matrix multiply, not the output after
2. The local matrix is **transposed**: $(D_l^p)^T$
3. The `gw_bar` term is structurally identical to the forward

### Full weak Laplacian application: $Au = \sum_l D_l^T Q D_l u$

```python
def ApplyWeakLap(u):
    result = np.zeros(N)
    for l in range(d):
        g = ApplyDeriv(l, u)           # g = D_l u          [Allreduce]
        g = q * g                       # g = Q D_l u        [local, no comm]
        result += ApplyDerivAdj(l, g)  # result += D_l^T g  [Allreduce]
    result[bc_nodes] = u[bc_nodes]     # Dirichlet BCs
    return result
```

Total MPI communication: $2d$ `Allreduce` calls per operator application — the same as the strong-form Laplacian.

### Symmetry and solver choice

$A = \sum_l D_l^T Q D_l$ is symmetric positive definite on interior nodes. The row-replacement boundary condition enforcement breaks exact symmetry of the assembled operator, so **GMRES is the safe default**. CG is possible if BCs are symmetrized (zeroing the corresponding columns as well as rows).

---

## 5. Implementation Plan

### Files to modify
- `source/Quadrature.py` — implement patch-local QP weight computation and global assembly
- `source/Operators.py` — add `ApplyDerivAdj` and `ApplyWeakLap`
- `PoissonDriver.py` — switch to weak form operator and load vector

### Step 1: Quadrature weights in `Quadrature.py`

Implement `PatchLocalWeights(comm, patches, N)`:
- For each local patch: determine $m$, build $P^p$, compute $M^p$ analytically, solve QP
- Assemble global weights: `q_local[idx] += q_hat * w_bar`, then `Allreduce`
- Store `patch.q_hat` on each patch object for the assembly step

### Step 2: Adjoint derivative in `Operators.py`

Implement `ApplyDerivAdj(comm, patches, N, k, boundary_groups, BCs)`:
- Same loop and `Allreduce` structure as `ApplyDeriv`
- Per patch: `result_local[idx] += D[k].T @ (w_bar * g[idx]) + gw_bar[:, k] * g[idx]`
- No BC enforcement needed on the adjoint (BCs are enforced only in `ApplyWeakLap`)

### Step 3: Weak Laplacian in `Operators.py`

Implement `ApplyWeakLap(comm, patches, N, q, boundary_groups, BCs)`:
- For each direction $l$: forward deriv → pointwise weight multiply → adjoint deriv
- Sum over directions, then enforce Dirichlet BCs by row replacement

### Step 4: Update `PoissonDriver.py`
- Call `PatchLocalWeights` after `Setup`
- Use `ApplyWeakLap` instead of `ApplyLap`
- Set `rhs = q * f` instead of `rhs = f`

### Verification
- Solve Poisson with $u = \sin(2\pi x)\sin(2\pi y)$, compare errors to strong form
- Check symmetry: $\langle Au, v \rangle = \langle u, Av \rangle$ for random interior $u, v$
- Inspect eigenvalue spectrum: should be real negative with no spurious positive eigenvalues
- Confirm $q_k > 0$ for all nodes after assembly
