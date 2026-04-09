"""
test_tiling.py — Quality tests for grid_tiling on simple and non-trivial domains.

Metrics reported per domain:
  coverage     : fraction of nodes covered by ≥1 patch (must be 1.0)
  n_patches    : number of active patches
  nodes/patch  : mean, std, min, max, and coefficient of variation (CV = std/mean)
                 CV close to 0 means patches see nearly equal numbers of eval nodes.
  patches/node : mean overlap (relevant for PUM conditioning)

Node distributions use Halton quasi-uniform sequences.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree

from source.PatchTiling import grid_tiling


# ---------------------------------------------------------------------------
# Quasi-uniform node generation
# ---------------------------------------------------------------------------

def halton_in_domain(n, bbox, mask):
    """
    Return up to n Halton points inside the domain described by mask.

    Parameters
    ----------
    n    : target number of interior nodes
    bbox : list of (lo, hi) pairs, one per dimension
    mask : callable (N,d) -> bool array, True = inside domain
    """
    d   = len(bbox)
    lo  = np.array([b[0] for b in bbox])
    hi  = np.array([b[1] for b in bbox])
    sampler = Halton(d=d, scramble=False)
    raw  = sampler.random(n * 12) * (hi - lo) + lo
    keep = mask(raw)
    return raw[keep][:n]


def boundary_circle(n, cx=0.0, cy=0.0, r=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def boundary_rect(n, lo, hi):
    """Roughly n points uniformly distributed along rectangle perimeter."""
    lx, ly = hi[0] - lo[0], hi[1] - lo[1]
    perim   = 2 * (lx + ly)
    n_per   = max(2, int(n * lx / perim))
    n_per_y = max(2, int(n * ly / perim))
    edges = []
    edges.append(np.column_stack([np.linspace(lo[0], hi[0], n_per),     np.full(n_per,   lo[1])]))
    edges.append(np.column_stack([np.full(n_per_y,  hi[0]),              np.linspace(lo[1], hi[1], n_per_y)]))
    edges.append(np.column_stack([np.linspace(hi[0], lo[0], n_per),     np.full(n_per,   hi[1])]))
    edges.append(np.column_stack([np.full(n_per_y,  lo[0]),              np.linspace(hi[1], lo[1], n_per_y)]))
    return np.vstack(edges)


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

def make_unit_square(n_int, n_bnd):
    bbox = [(0.0, 1.0), (0.0, 1.0)]
    xi = halton_in_domain(n_int, bbox,
                          lambda p: (p[:,0] > 0) & (p[:,0] < 1) &
                                    (p[:,1] > 0) & (p[:,1] < 1))
    xb = boundary_rect(n_bnd, np.array([0., 0.]), np.array([1., 1.]))
    return xi, xb


def make_unit_disk(n_int, n_bnd):
    bbox = [(-1.0, 1.0), (-1.0, 1.0)]
    xi = halton_in_domain(n_int, bbox,
                          lambda p: np.sum(p**2, axis=1) < 1.0)
    xb = boundary_circle(n_bnd)
    return xi, xb


def make_annulus(n_int, n_bnd, r_inner=0.4, r_outer=1.0):
    bbox = [(-r_outer, r_outer), (-r_outer, r_outer)]
    r2   = np.sum
    xi = halton_in_domain(n_int, bbox,
                          lambda p: (np.sum(p**2, axis=1) < r_outer**2) &
                                    (np.sum(p**2, axis=1) > r_inner**2))
    xb = np.vstack([boundary_circle(n_bnd // 2, r=r_outer),
                    boundary_circle(n_bnd // 2, r=r_inner)])
    return xi, xb


def make_l_shape(n_int, n_bnd):
    """L-shape: unit square minus the top-right quadrant [0.5,1]x[0.5,1]."""
    bbox = [(0.0, 1.0), (0.0, 1.0)]
    def mask(p):
        x, y = p[:,0], p[:,1]
        in_sq   = (x > 0) & (x < 1) & (y > 0) & (y < 1)
        in_hole = (x > 0.5) & (y > 0.5)
        return in_sq & ~in_hole

    xi = halton_in_domain(n_int, bbox, mask)

    # Boundary: outer L perimeter (6 segments)
    n6 = n_bnd // 6
    segs = [
        np.column_stack([np.linspace(0,   1,   n6), np.zeros(n6)]),   # bottom
        np.column_stack([np.ones(n6),               np.linspace(0, 0.5, n6)]),  # right bottom
        np.column_stack([np.linspace(1, 0.5, n6),   np.full(n6, 0.5)]),         # inner horiz
        np.column_stack([np.full(n6, 0.5),           np.linspace(0.5, 1, n6)]), # inner vert
        np.column_stack([np.linspace(0.5, 0, n6),   np.ones(n6)]),              # top
        np.column_stack([np.zeros(n6),               np.linspace(1, 0, n6)]),   # left
    ]
    xb = np.vstack(segs)
    return xi, xb


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

def tiling_metrics(xi, xb, centers, r):
    """
    Compute coverage and load-balance metrics for a grid tiling.

    Returns a dict with:
      coverage       : fraction of all nodes covered by ≥1 patch
      n_patches      : number of patches
      npp_mean/std/min/max/cv : nodes-per-patch statistics
      ppn_mean       : mean patches per node (overlap)
    """
    all_nodes = np.vstack([xi, xb])

    # patches → nodes  (load balance)
    node_tree = cKDTree(all_nodes)
    npp = np.array(node_tree.query_ball_point(centers, r=r, return_length=True))

    # nodes → patches  (coverage)
    center_tree = cKDTree(centers)
    ppn = np.array(center_tree.query_ball_point(all_nodes, r=r, return_length=True))

    covered  = np.sum(ppn >= 1)
    coverage = covered / len(all_nodes)

    return dict(
        coverage  = coverage,
        n_patches = len(centers),
        npp_mean  = npp.mean(),
        npp_std   = npp.std(),
        npp_min   = npp.min(),
        npp_max   = npp.max(),
        npp_cv    = npp.std() / npp.mean() if npp.mean() > 0 else np.inf,
        ppn_mean  = ppn.mean(),
    )


def print_metrics(name, m):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    cov_str = "OK" if np.isclose(m['coverage'], 1.0) else f"FAIL ({m['coverage']:.4f})"
    print(f"  coverage       : {cov_str}")
    print(f"  n_patches      : {m['n_patches']}")
    print(f"  nodes/patch    : mean={m['npp_mean']:.1f}  std={m['npp_std']:.1f}  "
          f"min={m['npp_min']}  max={m['npp_max']}")
    print(f"  CV (std/mean)  : {m['npp_cv']:.3f}  "
          f"({'good' if m['npp_cv'] < 0.3 else 'moderate' if m['npp_cv'] < 0.6 else 'poor'})")
    print(f"  patches/node   : {m['ppn_mean']:.2f}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_tiling(ax, name, xi, xb, centers, r, npp):
    ax.set_aspect('equal')
    ax.set_title(name, fontsize=10)

    # Colour patches by node count (load balance)
    vmin, vmax = npp.min(), npp.max()
    cmap = plt.cm.RdYlGn
    for c, count in zip(centers, npp):
        t  = (count - vmin) / max(vmax - vmin, 1)
        fc = cmap(t)
        circle = mpatches.Circle(c, r, facecolor=fc, alpha=0.25, linewidth=0.5,
                                 edgecolor='steelblue')
        ax.add_patch(circle)

    ax.scatter(xi[:,0], xi[:,1], s=4,  color='k',      label='interior', zorder=3)
    ax.scatter(xb[:,0], xb[:,1], s=10, color='tomato', marker='x', label='boundary', zorder=4)
    ax.scatter(centers[:,0], centers[:,1], s=15, color='steelblue',
               marker='+', label='centers', zorder=5)

    ax.autoscale_view()
    ax.legend(fontsize=6, loc='upper right')

    # Colour bar hint via text
    ax.text(0.01, 0.01, f"colour: nodes/patch  [{vmin}–{vmax}]",
            transform=ax.transAxes, fontsize=6, color='dimgray')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DOMAINS = [
    ("Unit square",  make_unit_square),
    ("Unit disk",    make_unit_disk),
    ("Annulus",      make_annulus),
    ("L-shape",      make_l_shape),
]

N_INT = 400   # interior nodes per domain
N_BND = 80    # boundary nodes per domain
OVERLAP = 1.5

def main():
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4 * len(DOMAINS), 4))
    print(f"\nGrid tiling quality  (n_interior={N_INT}, n_boundary={N_BND}, overlap={OVERLAP})")

    for ax, (name, make_fn) in zip(axes, DOMAINS):
        xi, xb = make_fn(N_INT, N_BND)
        centers, r = grid_tiling(xi, xb, overlap=OVERLAP)
        node_tree  = cKDTree(np.vstack([xi, xb]))
        npp = np.array(node_tree.query_ball_point(centers, r=r, return_length=True))

        m = tiling_metrics(xi, xb, centers, r)
        print_metrics(name, m)
        plot_tiling(ax, name, xi, xb, centers, r, npp)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "tiling_test.png")
    plt.savefig(out, dpi=150)
    print(f"\nFigure saved to {out}\n")
    plt.show()


if __name__ == "__main__":
    main()
