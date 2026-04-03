"""
ConditionDriver.py

Plots condition number of the Laplacian operator vs N for:
  - RBF-PU-RA  (flat limit, eval_epsilon=0.01)
  - Direct-RBF (global, epsilon = 1/h where h = mean nearest-neighbor spacing)

Condition numbers are computed on interior nodes only (no BCs applied) so
we compare the differential operator itself rather than the algebraic system.

Run with: mpirun -n <P> python ConditionDriver.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup
from source.BaseHelpers import GenMatrices

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

NODES_PER_PATCH  = 35
EVAL_EPSILON     = 0.01   # flat-limit shape parameter for PU-RA
DIRECT_EPSILON   = 2.0    # fixed shape parameter for Direct-RBF

# Node spacings to sweep — N grows roughly as (1/r)^2 on the unit square
spacings = [0.10, 0.085, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_pura_lap_matrix(comm, nodes, normals, nodes_per_patch, eval_epsilon):
    """Build the dense (N x N) PU-RA Laplacian matrix (no BCs)."""
    patches, _ = Setup(comm, nodes, normals, nodes_per_patch, eval_epsilon=eval_epsilon)
    N = nodes.shape[0]

    def apply_lap(u):
        local = np.zeros(N)
        for patch in patches:
            idx      = patch.node_indices
            u_loc    = u[idx]
            grad     = np.column_stack([D @ u_loc for D in patch.D])
            lap_loc  = patch.L @ u_loc
            local[idx] += (patch.w_bar  * lap_loc
                           + 2.0 * np.sum(patch.gw_bar * grad, axis=1)
                           + patch.lw_bar * u_loc)
        result = np.zeros(N)
        comm.Allreduce(local, result)
        return result

    A = np.zeros((N, N))
    for j in range(N):
        e_j = np.zeros(N)
        e_j[j] = 1.0
        A[:, j] = apply_lap(e_j)
    return A


def build_direct_rbf_lap_matrix(nodes, e=DIRECT_EPSILON):
    """Build the dense Direct-RBF Laplacian on the given nodes."""
    _, _, Lap = GenMatrices(nodes, e)
    return Lap

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

Ns          = []
cond_pura   = []
cond_direct = []

for r in spacings:

    # Generate nodes on rank 0 and broadcast
    if rank == 0:
        nodes, normals, groups = PoissonSquareOne(r)
        interior = groups['interior']
        print(f"\nr = {r:.3f}  →  N = {nodes.shape[0]}", flush=True)
    else:
        nodes    = None
        normals  = None
        groups   = None
        interior = None

    nodes    = comm.bcast(nodes,   root=0)
    normals  = comm.bcast(normals, root=0)
    groups   = comm.bcast(groups,  root=0)
    interior = groups['interior']

    N_total   = nodes.shape[0]
    int_nodes = nodes[interior]

    # ---- PU-RA --------------------------------------------------------
    if rank == 0:
        print("  Building PU-RA matrix ...", flush=True)
    A_pura = build_pura_lap_matrix(comm, nodes, normals, NODES_PER_PATCH, EVAL_EPSILON)

    if rank == 0:
        # Restrict to interior block
        A_int  = A_pura[np.ix_(interior, interior)]
        c_pura = np.linalg.cond(A_int)
        print(f"  PU-RA   cond = {c_pura:.3e}", flush=True)

    # ---- Direct RBF ---------------------------------------------------
    if rank == 0:
        print("  Building Direct-RBF matrix ...", flush=True)
        A_direct = build_direct_rbf_lap_matrix(int_nodes)
        c_direct = np.linalg.cond(A_direct)
        print(f"  Direct  cond = {c_direct:.3e}  (ε = {DIRECT_EPSILON})", flush=True)

        Ns.append(N_total)
        cond_pura.append(c_pura)
        cond_direct.append(c_direct)

# ---------------------------------------------------------------------------
# Plot (rank 0 only)
# ---------------------------------------------------------------------------
if rank == 0:
    Ns         = np.array(Ns)
    cond_pura  = np.array(cond_pura)
    cond_direct = np.array(cond_direct)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogy(Ns, cond_pura,   'o-',  color='steelblue',  label=r'RBF-PU-RA  (flat limit, $\varepsilon \to 0$)')
    ax.semilogy(Ns, cond_direct, 's--', color='firebrick',   label=rf'Direct RBF  ($\varepsilon = {DIRECT_EPSILON}$)')

    ax.set_xlabel('$N$ (number of nodes)',  fontsize=12)
    ax.set_ylabel('Condition number',        fontsize=12)
    ax.set_title('Laplacian operator condition number\n'
                 f'({NODES_PER_PATCH} nodes/patch)',   fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    plt.tight_layout()

    outpath = 'figures/condition_comparison.png'
    plt.savefig(outpath, dpi=150)
    print(f"\nSaved {outpath}")
