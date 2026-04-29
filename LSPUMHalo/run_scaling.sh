#!/usr/bin/env bash
# run_scaling.sh — strong and weak scaling sweeps for ScalingTest.py
#
# Usage:
#   bash run_scaling.sh          # both sweeps, default rank counts
#   bash run_scaling.sh strong   # strong scaling only
#   bash run_scaling.sh weak     # weak scaling only
#
# Results land in results/strong_scaling.txt and results/weak_scaling.txt.
# SCALING_ROW lines are also extracted to *_rows.txt for easy parsing.
#
# Tune RANKS, STRONG_M, STRONG_H, WEAK_M1, WEAK_H_BASE below if needed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANKS=(1 2 4 8)          # rank counts to sweep
SCRIPT=ScalingTest.py

# Strong scaling: fixed problem, increasing ranks
STRONG_M=10000
STRONG_H=0.09

# Weak scaling: both M and P grow with ranks so each rank always owns the
# same number of patches and eval nodes.
#   M at P ranks  = WEAK_M1 * P
#   H at P ranks  = WEAK_H_BASE / sqrt(P)   (patch count ∝ 1/H² ∝ P)
# This keeps n_eval_p ≈ constant and avoids underdetermined patches.
# WEAK_H_BASE=0.20 gives ~25 patches at 1 rank with n_eval_p >> n_interp=40.
WEAK_M1=2500
WEAK_H_BASE=0.20

RESULTS_DIR=results
mkdir -p "$RESULTS_DIR"

STRONG_OUT="$RESULTS_DIR/strong_scaling.txt"
WEAK_OUT="$RESULTS_DIR/weak_scaling.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date +%H:%M:%S)] $*"; }
sep()  { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

run_case() {
    local mode="$1"   # "strong" or "weak"
    local nranks="$2"
    local M="$3"
    local H="$4"
    local outfile="$5"

    log "$mode  ranks=$nranks  M=$M  H=$H"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        mpirun -n "$nranks" python "$SCRIPT" --single "$M" "$H" 2>&1 | tee -a "$outfile"
}

# ---------------------------------------------------------------------------
# Strong scaling
# ---------------------------------------------------------------------------
run_strong() {
    sep
    log "Strong scaling  M=$STRONG_M  H=$STRONG_H  ranks: ${RANKS[*]}"
    sep

    : > "$STRONG_OUT"   # truncate / create

    for P in "${RANKS[@]}"; do
        run_case strong "$P" "$STRONG_M" "$STRONG_H" "$STRONG_OUT"
    done

    grep "^SCALING_ROW" "$STRONG_OUT" > "$RESULTS_DIR/strong_scaling_rows.txt" || true

    sep
    log "Strong scaling complete."
    log "Full output  : $STRONG_OUT"
    log "SCALING_ROWs : $RESULTS_DIR/strong_scaling_rows.txt"
}

# ---------------------------------------------------------------------------
# Weak scaling
# ---------------------------------------------------------------------------
run_weak() {
    sep
    log "Weak scaling  M1=$WEAK_M1  H_base=$WEAK_H_BASE  ranks: ${RANKS[*]}"
    log "  M = WEAK_M1 * P,  H = WEAK_H_BASE / sqrt(P)"
    sep

    : > "$WEAK_OUT"

    for P in "${RANKS[@]}"; do
        M=$(( WEAK_M1 * P ))
        H=$(python3 -c "print(f'{$WEAK_H_BASE / ($P ** 0.5):.4f}')")
        run_case weak "$P" "$M" "$H" "$WEAK_OUT"
    done

    grep "^SCALING_ROW" "$WEAK_OUT" > "$RESULTS_DIR/weak_scaling_rows.txt" || true

    sep
    log "Weak scaling complete."
    log "Full output  : $WEAK_OUT"
    log "SCALING_ROWs : $RESULTS_DIR/weak_scaling_rows.txt"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MODE="${1:-both}"

case "$MODE" in
    strong) run_strong ;;
    weak)   run_weak   ;;
    both)   run_strong; run_weak ;;
    *)
        echo "Usage: $0 [strong|weak|both]"
        exit 1
        ;;
esac

sep
log "Done.  Results in $RESULTS_DIR/"
