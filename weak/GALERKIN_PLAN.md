# Weak Form Poisson via RBF-PU with Node-Based Quadrature

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

### Chosen approach: Polynomial-exact weights

Find minimum-norm weights $q$ satisfying:

$$\sum_{i=1}^N q_i \, x_i^a y_i^b = \int_0^1\int_0^1 x^a y^b \, dx\,dy = \frac{1}{(a+1)(b+1)}$$

for all monomials $x^a y^b$ with $a + b \leq p$. This is an underdetermined system $V^T q = m$ solved by:

$$q = V (V^T V)^{-1} m$$

Polynomial degree $p \sim 4$–$6$ is sufficient. Moments are analytic on the unit square.

---

## 4. Matrix-Free Weak Laplacian Operator

### The forward derivative (existing)
`ApplyDeriv` computes $\mathbf{D}_k u$. Per patch $p$:

```
(D_k u)_i += w_bar_p[i] * (D_k^p @ u_p)[i] + gw_bar_p[i,k] * u_p[i]
```

### The adjoint derivative (new — needed for $\mathbf{D}_k^T$)
Per patch $p$, the adjoint acting on vector $g$:

```
(D_k^T g)_j += (D_k^p)^T @ (w_bar_p * g_p)[j] + gw_bar_p[j,k] * g_p[j]
```

Key differences from forward:
1. PU weight multiplies the **input** $g$ (not the output of $D_k^p$)
2. Local derivative matrix is **transposed**: $(D_k^p)^T$

### Full weak Laplacian application: $Au = \sum_l D_l^T Q D_l u$

```python
def weak_lap(u):
    result = zeros(N)
    for l in range(d):
        g = D_l(u)              # forward derivative (existing ApplyDeriv)
        g = q * g               # multiply by quadrature weights
        result += D_l_T(g)      # adjoint derivative (new)
    # boundary conditions
    result[bc_nodes] = u[bc_nodes]
    return result
```

---

## 5. Implementation Plan

### Files to modify
- `source/Quadrature.py` — implement quadrature weight computation
- `source/Operators.py` — add `ApplyDerivAdj` and `ApplyWeakLap`
- `PoissonDriver.py` — switch to weak form operator and load vector

### Step 1: Quadrature weights in `Quadrature.py`
Implement polynomial-exact quadrature weight computation:
- Build Vandermonde matrix for monomials up to degree $p$
- Compute moments analytically for unit square
- Solve minimum-norm system

### Step 2: Adjoint derivative in `Operators.py`
Implement `ApplyDerivAdj(comm, patches, N, k, boundary_groups, BCs)`:
- Same loop structure as `ApplyDeriv`
- Transpose the local matrix, swap the weight multiplication order

### Step 3: Weak Laplacian in `Operators.py`
Implement `ApplyWeakLap(comm, patches, N, quad_weights, boundary_groups, BCs)`:
- For each direction: forward deriv -> weight multiply -> adjoint deriv
- Sum over directions
- Enforce Dirichlet BCs

### Step 4: Update `PoissonDriver.py`
- Compute quadrature weights after setup
- Use `ApplyWeakLap` instead of `ApplyLap`
- Set `rhs = q * f` instead of `rhs = f`

### Verification
- Solve Poisson with $u = \sin(2\pi x)\sin(2\pi y)$, compare errors to strong form
- Check that the weak operator is symmetric: $\langle Au, v \rangle = \langle u, Av \rangle$ for random $u, v$
- Inspect eigenvalue spectrum (should be real negative — no spurious positive eigenvalues)
