"""
test_compare.py — Convergence tests, positivity checks, and comparison plots
                  for all five scattered-node quadrature methods.

Usage
-----
    python test_compare.py            # full benchmark (2D, all methods)
    python test_compare.py --3d       # 3D benchmark
    python test_compare.py --quick    # fast run with fewer nodes/trials
    python test_compare.py --noplot   # skip matplotlib output

Outputs
-------
  • Console: comparison table (method, positivity rate, estimated order, time)
  • Figures:  convergence plots saved as  convergence_2d.png / convergence_3d.png
"""

import argparse
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')    # headless rendering; change to 'TkAgg' for interactive
import matplotlib.pyplot as plt

from quad_utils import (
    random_unit_square, random_unit_cube,
    TEST_INTEGRANDS_2D, TEST_INTEGRANDS_3D,
    check_positivity, compute_error
)

from quad_delaunay       import DelaunayGaussQuadrature
from quad_moment_fitting import MomentFittingQuadrature
from quad_compressed     import CompressedQuadrature
from quad_rbf            import RBFQuadrature
from quad_sph            import SPHQuadrature


# ---------------------------------------------------------------------------
# Method factories
# ---------------------------------------------------------------------------

def _methods_2d():
    return {
        'Delaunay-p3':        lambda: DelaunayGaussQuadrature(p=3),
        'Delaunay-p4':        lambda: DelaunayGaussQuadrature(p=4),
        'Delaunay-p5':        lambda: DelaunayGaussQuadrature(p=5),
        'Delaunay-p4-nodal':  lambda: DelaunayGaussQuadrature(p=4, nodal=True),
        'MomFit-d4':          lambda: MomentFittingQuadrature(degree=4),
        'MomFit-d6':          lambda: MomentFittingQuadrature(degree=6),
        'Compressed-d4':      lambda: CompressedQuadrature(degree=4),
        'Compressed-d6':      lambda: CompressedQuadrature(degree=6),
        'RBF-phs3':           lambda: RBFQuadrature(rbf='phs3'),
        'RBF-phs5':           lambda: RBFQuadrature(rbf='phs5'),
        'SPH-order0':         lambda: SPHQuadrature(order=0),
        'SPH-order1':         lambda: SPHQuadrature(order=1),
        'SPH-rkpm3':          lambda: SPHQuadrature(order=2, rkpm_degree=3),
    }

def _methods_3d():
    return {
        'Delaunay-p3':        lambda: DelaunayGaussQuadrature(p=3),
        'Delaunay-p4':        lambda: DelaunayGaussQuadrature(p=4),
        'Delaunay-p4-nodal':  lambda: DelaunayGaussQuadrature(p=4, nodal=True),
        'MomFit-d4':          lambda: MomentFittingQuadrature(degree=4),
        'Compressed-d4':      lambda: CompressedQuadrature(degree=4),
        'RBF-phs3':           lambda: RBFQuadrature(rbf='phs3'),
        'SPH-order0':         lambda: SPHQuadrature(order=0),
        'SPH-order1':         lambda: SPHQuadrature(order=1),
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_convergence_test(method_factory, test_func, exact, node_counts,
                         dim=2, trials=5, seed_offset=0):
    """
    For each node count, run `trials` random node sets and record the mean
    absolute quadrature error.

    Returns
    -------
    results : list of (n, mean_error, std_error, mean_time_ms, pos_rate)
    """
    node_gen = random_unit_square if dim == 2 else random_unit_cube
    domain_bounds_kw = {} if dim == 2 else {}

    results = []
    for n in node_counts:
        errors, times, pos_flags = [], [], []
        for trial in range(trials):
            nodes = node_gen(n, seed=seed_offset + n * 100 + trial)
            try:
                t0 = time.perf_counter()
                quad = method_factory()
                # SPHQuadrature accepts optional domain_bounds
                if isinstance(quad, SPHQuadrature):
                    if dim == 2:
                        pts, wts = quad.fit(nodes, domain_bounds=(0, 1, 0, 1))
                    else:
                        pts, wts = quad.fit(nodes, domain_bounds=(0,1,0,1,0,1))
                else:
                    pts, wts = quad.fit(nodes)
                elapsed = (time.perf_counter() - t0) * 1e3   # ms

                err = compute_error(test_func, pts, wts, exact)
                errors.append(err)
                times.append(elapsed)
                pos_flags.append(check_positivity(wts))
            except Exception as exc:
                warnings.warn(f"n={n} trial={trial}: {exc}")
                errors.append(np.nan)
                times.append(np.nan)
                pos_flags.append(False)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mean_e = float(np.nanmean(errors))
            std_e  = float(np.nanstd(errors))
            mean_t = float(np.nanmean(times))
            pos_r  = float(np.mean(pos_flags))

        results.append((n, mean_e, std_e, mean_t, pos_r))
        print(f"    n={n:5d}: err={mean_e:.3e} ± {std_e:.1e}  "
              f"pos={pos_r:.0%}  t={mean_t:.1f}ms")

    return results


def _estimate_order(results):
    """Estimate convergence order from the last two non-nan results."""
    valid = [(n, e) for n, e, *_ in results if np.isfinite(e) and e > 0]
    if len(valid) < 2:
        return float('nan')
    (n1, e1), (n2, e2) = valid[-2], valid[-1]
    # err ~ h^p, h ~ n^{-1/d}.  For d=2: err ~ n^{-p/2}, so p = -log(e2/e1)/log(n2/n1)*2.
    return -np.log(e2 / e1) / np.log(n2 / n1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(all_results, title, filename, node_counts, ref_orders=(2,4,6,8)):
    """
    Log-log plot: error vs. N for all methods.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.tab20
    colors = [cmap(i / max(len(all_results) - 1, 1)) for i in range(len(all_results))]

    for (name, results), color in zip(all_results.items(), colors):
        ns   = [r[0] for r in results]
        errs = [r[1] for r in results]
        valid = [(n, e) for n, e in zip(ns, errs) if np.isfinite(e) and e > 0]
        if not valid:
            continue
        ns_v, errs_v = zip(*valid)
        ax.loglog(ns_v, errs_v, 'o-', label=name, color=color, linewidth=1.5,
                  markersize=5)

    # Reference slopes
    n_ref = np.array(node_counts, dtype=float)
    for p in ref_orders:
        # Anchor line at first point of first method with data
        anchor_err = 1e-2
        line = anchor_err * (n_ref / n_ref[0]) ** (-p / 2)
        ax.loglog(n_ref, line, '--', color='gray', linewidth=0.8, alpha=0.6,
                  label=f'O(N^{{-{p}/2}})')

    ax.set_xlabel("Number of nodes N")
    ax.set_ylabel("Absolute error")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_table(all_results, all_pos_rates, all_times, all_orders):
    col_w = 22
    header = (f"{'Method':<{col_w}} {'PosRate':>8} {'Order':>7} {'Time(ms)':>10}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name in all_results:
        pos  = all_pos_rates.get(name, float('nan'))
        ord_ = all_orders.get(name, float('nan'))
        t    = all_times.get(name, float('nan'))
        pos_str  = f"{pos:.0%}" if np.isfinite(pos) else "  N/A"
        ord_str  = f"{ord_:.1f}" if np.isfinite(ord_) else " N/A"
        t_str    = f"{t:.1f}" if np.isfinite(t) else "  N/A"
        print(f"  {name:<{col_w-2}} {pos_str:>8} {ord_str:>7} {t_str:>10}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(dim=2, quick=False, noplot=False):
    node_counts = [50, 100, 200, 400] if quick else [50, 100, 200, 400, 800]
    trials = 3 if quick else 5

    methods = _methods_2d() if dim == 2 else _methods_3d()
    integrands = TEST_INTEGRANDS_2D if dim == 2 else TEST_INTEGRANDS_3D

    # Use 'sin_product' as the primary convergence test function
    test_name = 'sin_product'
    f, exact = integrands[test_name]

    print(f"\n{'='*60}")
    print(f"  Benchmark: dim={dim}, integrand='{test_name}', exact={exact:.6f}")
    print(f"  node_counts={node_counts}, trials={trials}")
    print(f"{'='*60}\n")

    all_results = {}
    all_pos_rates = {}
    all_times = {}
    all_orders = {}

    for name, factory in methods.items():
        print(f"\n[{name}]")
        try:
            results = run_convergence_test(
                factory, f, exact, node_counts, dim=dim, trials=trials
            )
            all_results[name] = results
            all_pos_rates[name] = float(np.mean([r[4] for r in results]))
            all_times[name]     = float(np.mean([r[3] for r in results]))
            all_orders[name]    = _estimate_order(results)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_results[name] = []
            all_pos_rates[name] = 0.0
            all_times[name]     = float('nan')
            all_orders[name]    = float('nan')

    print_table(all_results, all_pos_rates, all_times, all_orders)

    if not noplot and all_results:
        fname = f"convergence_{dim}d.png"
        plot_convergence(
            all_results,
            title=f"{dim}D Convergence: {test_name} over unit {'square' if dim==2 else 'cube'}",
            filename=fname,
            node_counts=node_counts
        )

    return all_results, all_orders


# ---------------------------------------------------------------------------
# Additional: per-integrand error table
# ---------------------------------------------------------------------------

def run_integrand_comparison(n=300, dim=2, trials=3):
    """
    Test each method on all integrands for a fixed node count n.
    Prints a wide table: methods × integrands.
    """
    methods = _methods_2d() if dim == 2 else _methods_3d()
    integrands = TEST_INTEGRANDS_2D if dim == 2 else TEST_INTEGRANDS_3D
    node_gen = random_unit_square if dim == 2 else random_unit_cube

    print(f"\n--- Per-integrand comparison: n={n}, dim={dim} ---")
    int_names = list(integrands.keys())
    print(f"{'Method':<22}", end='')
    for iname in int_names:
        print(f"  {iname[:10]:>10}", end='')
    print()
    print('-' * (22 + 12 * len(int_names)))

    for mname, factory in methods.items():
        row = f"{mname:<22}"
        for iname in int_names:
            f, exact = integrands[iname]
            errs = []
            for trial in range(trials):
                nodes = node_gen(n, seed=trial * 999 + 1)
                try:
                    quad = factory()
                    if isinstance(quad, SPHQuadrature):
                        bds = (0,1,0,1) if dim==2 else (0,1,0,1,0,1)
                        pts, wts = quad.fit(nodes, domain_bounds=bds)
                    else:
                        pts, wts = quad.fit(nodes)
                    errs.append(compute_error(f, pts, wts, exact))
                except Exception:
                    errs.append(np.nan)
            row += f"  {np.nanmean(errs):>10.2e}"
        print(row)


# ---------------------------------------------------------------------------
# Positivity stress test
# ---------------------------------------------------------------------------

def positivity_stress_test(n_list=(50, 100, 200), trials=20, dim=2):
    """
    For each method and node count, report the fraction of random node sets
    that yield all-positive weights.
    """
    methods = _methods_2d() if dim == 2 else _methods_3d()
    node_gen = random_unit_square if dim == 2 else random_unit_cube

    print(f"\n--- Positivity stress test: dim={dim}, trials={trials} ---")
    ns = list(n_list)
    header = f"{'Method':<22}" + "".join(f"  n={n:4d}" for n in ns)
    print(header)
    print('-' * len(header))

    for mname, factory in methods.items():
        row = f"{mname:<22}"
        for n in ns:
            pos_count = 0
            for trial in range(trials):
                nodes = node_gen(n, seed=trial * 31 + n)
                try:
                    quad = factory()
                    if isinstance(quad, SPHQuadrature):
                        bds = (0,1,0,1) if dim==2 else (0,1,0,1,0,1)
                        _, wts = quad.fit(nodes, domain_bounds=bds)
                    else:
                        _, wts = quad.fit(nodes)
                    if check_positivity(wts):
                        pos_count += 1
                except Exception:
                    pass
            row += f"  {pos_count/trials:>6.0%}"
        print(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quadrature method benchmark")
    parser.add_argument('--3d',      dest='do_3d',   action='store_true',
                        help='run 3D benchmark instead of 2D')
    parser.add_argument('--quick',   action='store_true',
                        help='fewer nodes and trials for a fast preview')
    parser.add_argument('--noplot',  action='store_true',
                        help='skip saving convergence plots')
    parser.add_argument('--positivity', action='store_true',
                        help='run positivity stress test only')
    parser.add_argument('--integrands', action='store_true',
                        help='run per-integrand comparison table')
    args = parser.parse_args()

    dim = 3 if args.do_3d else 2

    if args.positivity:
        positivity_stress_test(dim=dim)
    elif args.integrands:
        run_integrand_comparison(n=300, dim=dim)
    else:
        run_benchmark(dim=dim, quick=args.quick, noplot=args.noplot)
        if not args.do_3d:
            positivity_stress_test(n_list=(50, 200, 500), trials=10, dim=2)
        print("\nDone.")
