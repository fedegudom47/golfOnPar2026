"""
cross_seed_jaccard.py – Test 2: pairwise cross-seed Jaccard matrices.

For a fixed grid point and sample size N, every seed has produced its own
1-SE equivalence set E*(g, seed, N). This script builds the S x S matrix of
pairwise Jaccard (|intersection| / |union|) between those per-seed sets, so
you can see whether the seeds agree on the tied-best strategy set — and watch
that agreement tighten as N grows (hopefully toward the true answer).

Matrices are written per grid point at:
  * N = n_max (always), and
  * every N at which >=1 seed's within-seed stabilisation kicked in for that
    grid point (i.e. the distinct converged_N values across seeds) — so you
    can inspect agreement exactly where seeds started claiming stability.

Inputs: the local ./outputs/seed{SEED}/seed{SEED}_N{N}_equivset.csv files
(after fetch_hpc_results.sh).

Outputs (under --results-dir, default cross_seed_jaccard/):
  gridpoints.csv                         – gp_id -> (x, y)
  matrices/gp{ID}_N{N}.csv               – S x S Jaccard matrix (seed-labelled)
  cross_seed_jaccard_summary.csv         – per (gp_id, N): mean/median/min off-diagonal
                                           Jaccard, frac identical pairs, n_seeds
  cross_seed_jaccard_vs_N.csv / .png     – mean off-diagonal Jaccard vs N,
                                           averaged over all grid points
  heatmaps/gp{ID}_N{N}.png               – only with --heatmaps

Usage:
    python cross_seed_jaccard.py --output-dir outputs --n-seeds 100
    python cross_seed_jaccard.py --n-values 500 --heatmaps
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equivalence import (
    EquivSet,
    SequentialState,
    load_equivset,
    pairwise_jaccard_matrix,
)

logger = logging.getLogger(__name__)


def _equivset_path(output_dir: Path, seed: int, N: int) -> Path:
    return output_dir / f"seed{seed:04d}" / f"seed{seed:04d}_N{N:04d}_equivset.csv"


def _offdiag(mat: np.ndarray) -> np.ndarray:
    m = mat.shape[0]
    if m < 2:
        return np.array([], dtype=float)
    iu = np.triu_indices(m, k=1)
    return mat[iu]


def load_all_sets(
    output_dir: Path, n_seeds: int, n_values: list[int],
) -> dict[int, dict[tuple, EquivSet]]:
    """seed_sets[N][gp][seed] -> EquivSet, only for seeds/files that exist."""
    by_N: dict[int, dict[tuple, dict[int, EquivSet]]] = {N: {} for N in n_values}
    for seed in range(n_seeds):
        for N in n_values:
            p = _equivset_path(output_dir, seed, N)
            if not p.exists():
                continue
            for gp, eqset in load_equivset(p).items():
                by_N[N].setdefault(gp, {})[seed] = eqset
    return by_N


def per_gp_stabilisation_Ns(
    output_dir: Path, n_seeds: int, n_values: list[int],
    k_consecutive: int, jaccard_threshold: float,
) -> dict[tuple, set[int]]:
    """For each grid point, the distinct within-seed converged_N across seeds."""
    states: dict[tuple, SequentialState] = {}
    for N in n_values:
        for seed in range(n_seeds):
            p = _equivset_path(output_dir, seed, N)
            if not p.exists():
                continue
            for gp, eqset in load_equivset(p).items():
                st = states.setdefault((gp, seed), SequentialState())
                st.update(N, eqset, k_consecutive, jaccard_threshold)
    out: dict[tuple, set[int]] = {}
    for (gp, _seed), st in states.items():
        if st.converged_N is not None:
            out.setdefault(gp, set()).add(int(st.converged_N))
    return out


def run(
    output_dir: Path,
    results_dir: Path,
    n_seeds: int,
    n_start: int,
    n_step: int,
    n_max: int,
    forced_n_values: list[int] | None,
    k_consecutive: int,
    jaccard_threshold: float,
    make_heatmaps: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "matrices").mkdir(exist_ok=True)
    if make_heatmaps:
        (results_dir / "heatmaps").mkdir(exist_ok=True)

    all_N = list(range(n_start, n_max + 1, n_step))
    by_N = load_all_sets(output_dir, n_seeds, all_N)

    # Stable grid-point index
    grid_points = sorted({gp for N in all_N for gp in by_N[N]})
    gp_id = {gp: i for i, gp in enumerate(grid_points)}
    pd.DataFrame(
        [{"gp_id": i, "x": gp[0], "y": gp[1]} for gp, i in gp_id.items()]
    ).sort_values("gp_id").to_csv(results_dir / "gridpoints.csv", index=False)

    # Which (gp, N) matrices to actually write out
    if forced_n_values:
        targets: dict[tuple, set[int]] = {gp: set(forced_n_values) for gp in grid_points}
    else:
        stab_Ns = per_gp_stabilisation_Ns(
            output_dir, n_seeds, all_N, k_consecutive, jaccard_threshold
        )
        targets = {gp: {n_max} | stab_Ns.get(gp, set()) for gp in grid_points}

    summary_rows: list[dict] = []
    vs_N: dict[int, list[float]] = {N: [] for N in all_N}

    for gp in grid_points:
        gid = gp_id[gp]
        for N in sorted(targets[gp]):
            seed_sets = by_N.get(N, {}).get(gp, {})
            if len(seed_sets) < 2:
                continue
            seeds, mat = pairwise_jaccard_matrix(seed_sets)
            pd.DataFrame(mat, index=seeds, columns=seeds).to_csv(
                results_dir / "matrices" / f"gp{gid:03d}_N{N:04d}.csv"
            )
            od = _offdiag(mat)
            summary_rows.append({
                "gp_id": gid, "x": gp[0], "y": gp[1], "N": N,
                "n_seeds": len(seeds),
                "mean_jaccard": float(od.mean()),
                "median_jaccard": float(np.median(od)),
                "min_jaccard": float(od.min()),
                "frac_pairs_identical": float((od >= 1.0).mean()),
                "is_n_max": N == n_max,
            })
            if make_heatmaps:
                _heatmap(mat, seeds, gp, N, results_dir / "heatmaps" / f"gp{gid:03d}_N{N:04d}.png")

        # convergence-over-N curve (all N, cheap)
        for N in all_N:
            seed_sets = by_N.get(N, {}).get(gp, {})
            if len(seed_sets) < 2:
                continue
            _, mat = pairwise_jaccard_matrix(seed_sets)
            od = _offdiag(mat)
            if od.size:
                vs_N[N].append(float(od.mean()))

    pd.DataFrame(summary_rows).to_csv(
        results_dir / "cross_seed_jaccard_summary.csv", index=False
    )

    vs_N_df = pd.DataFrame([
        {"N": N, "mean_jaccard_over_gridpoints": float(np.mean(v)),
         "n_gridpoints": len(v)}
        for N, v in vs_N.items() if v
    ])
    vs_N_df.to_csv(results_dir / "cross_seed_jaccard_vs_N.csv", index=False)

    if not vs_N_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(vs_N_df["N"], vs_N_df["mean_jaccard_over_gridpoints"], marker="o")
        ax.set_xlabel("N (shots per grid point)")
        ax.set_ylabel("Mean pairwise cross-seed Jaccard\n(averaged over grid points)")
        ax.set_ylim(0, 1.02)
        ax.set_title("Cross-seed agreement of the equivalence set vs N")
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        fig.savefig(results_dir / "cross_seed_jaccard_vs_N.png", dpi=120)
        plt.close(fig)

    logger.info("Wrote %d matrices + summary to %s", len(summary_rows), results_dir)


def _heatmap(mat: np.ndarray, seeds: list[int], gp: tuple, N: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(f"Pairwise cross-seed Jaccard — grid point {gp}, N={N}")
    ax.set_xlabel("seed"); ax.set_ylabel("seed")
    step = max(1, len(seeds) // 20)
    ax.set_xticks(range(0, len(seeds), step)); ax.set_xticklabels(seeds[::step], rotation=90, fontsize=6)
    ax.set_yticks(range(0, len(seeds), step)); ax.set_yticklabels(seeds[::step], fontsize=6)
    fig.colorbar(im, ax=ax, label="Jaccard")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise cross-seed Jaccard matrices (Test 2).")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--results-dir", type=Path, default=Path("cross_seed_jaccard"))
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-start", type=int, default=10)
    p.add_argument("--n-step", type=int, default=10)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--n-values", type=int, nargs="+", default=None,
                   help="Explicit N list for the matrices (overrides the "
                        "n_max + per-gridpoint-stabilisation-N default).")
    p.add_argument("--k-consecutive", type=int, default=3)
    p.add_argument("--jaccard-threshold", type=float, default=1.0)
    p.add_argument("--heatmaps", action="store_true", help="Also save a PNG per matrix.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    run(
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        n_seeds=args.n_seeds,
        n_start=args.n_start,
        n_step=args.n_step,
        n_max=args.n_max,
        forced_n_values=args.n_values,
        k_consecutive=args.k_consecutive,
        jaccard_threshold=args.jaccard_threshold,
        make_heatmaps=args.heatmaps,
    )
