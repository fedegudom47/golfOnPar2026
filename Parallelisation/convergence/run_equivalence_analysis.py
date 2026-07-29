"""
run_equivalence_analysis.py – Post-processing: equivalence-set convergence.

Consumes the per-(seed, N) candidate parquet files written by
convergence_worker.py (one row per grid-point/club/aim combo, with R(s,theta)
= mean and n_total, so SE(s,theta) = sqrt(var/n_total)) across the existing
100-seed x N=10..500-step-10 x M=280-grid-point sweep. Does NOT resimulate —
purely reads the already-computed sweep and applies the equivalence-set
definitions in equivalence.py, swept over e in {1.0, 1.645, 2.0}.

The sweep always runs to n_max (default 500) — there is no live stopping rule.
A (grid point, seed) pair that never hits k_consecutive stable Jaccard steps
by n_max is reported, not silently dropped: its `stop_reason` is
"reached_n_max_no_stability" (vs "converged"), and it shows up in
equiv_non_converged_gridpoints_e{e}.csv.

Outputs, per e:
  (a) equiv_seq_e{e}.csv                    – per-grid-point/seed: converged_N (or None),
                                               |E*| at convergence/n_max, stop_reason
  (b) equiv_cross_seed_e{e}.csv             – per-grid-point core/union set size + full-agreement vs N
  (c) equiv_match_rate_summary.png          – mean sequential Jaccard match rate vs N, one curve per e
  (d) equiv_size_at_convergence_e{e}.png    – distribution of |E*| at convergence, converged pairs only
  (e) equiv_non_converged_gridpoints_e{e}.csv – which grid points didn't stabilise by n_max, how many
                                               seeds, and how large |E*| was stuck at there
  (f) equiv_spatial_map_e{e}.png            – hole-layout scatter, one point per grid point, coloured
                                               by mean |E*| across seeds — WHERE the ambiguity is on the hole

Plus one overall table: equiv_overall_summary.csv (% converged / not converged by n_max,
mean/median |E*| at convergence, per e).

Usage:
    python run_equivalence_analysis.py --output-dir outputs --n-seeds 100 \
        --results-dir equivalence_results
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
    EquivalenceConfig,
    SequentialState,
    compute_equivalence_sets,
    cross_seed_core_stats,
    grid_level_summary,
    load_candidates,
    non_converged_report,
    summarize_sequential,
)

logger = logging.getLogger(__name__)


def _candidate_path(output_dir: Path, seed: int, N: int) -> Path:
    return output_dir / f"seed{seed:04d}" / f"seed{seed:04d}_N{N:04d}_candidates.parquet"


def plot_equivalence_spatial_map(
    e_seq: pd.DataFrame,
    hole,
    e: float,
    n_max: int,
    output_path: Path,
) -> None:
    """Hole-layout scatter of grid points coloured by mean |E*| across seeds.

    This is the key spatial finding: not just how many grid points are
    ambiguous, but WHERE on the hole the tied-best-strategy sets are large
    (e.g. a wide fairway landing zone where many club/aim combos score the
    same) vs small (a tight pin position with one clearly-best play).
    """
    from core import plot_hole_layout

    per_gp = (
        e_seq.groupby(["x", "y"])["equiv_set_size_final"]
        .mean()
        .reset_index(name="mean_equiv_set_size")
    )

    fig, ax = plt.subplots(figsize=(11, 13))
    plot_hole_layout(hole, title="", plot_strategy_points=False, ax=ax)

    sizes = per_gp["mean_equiv_set_size"].to_numpy()
    sc = ax.scatter(
        per_gp["x"], per_gp["y"],
        c=sizes, cmap="RdYlGn_r", s=60, alpha=0.9,
        edgecolors="black", linewidths=0.5, zorder=20,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean |E*| across seeds (# statistically tied-best club/aim combos)")
    ax.set_title(
        f"Strategic ambiguity by grid point — e={e}, at convergence or N={n_max}\n"
        "(green = one clearly-best play, red = many genuinely-tied options)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved spatial equivalence map → %s", output_path)


def run_analysis(
    output_dir: Path,
    results_dir: Path,
    config: EquivalenceConfig,
    n_seeds: int,
    data_dir: Path | None = None,
    make_spatial_map: bool = True,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    hole = None
    if make_spatial_map:
        try:
            from core import build_hole
            # gp_training_iter=1: geometry/polygons are all this plot needs —
            # the putting GPR itself is never queried, so skip real training cost.
            hole = build_hole(data_dir, gp_training_iter=1)
        except Exception:
            logger.warning(
                "Could not build hole geometry (data files unavailable?) — "
                "skipping spatial equivalence maps.", exc_info=True,
            )

    seq_states: dict[tuple, SequentialState] = {}
    cross_rows: list[dict] = []
    n_values = list(range(config.n_start, config.n_max + 1, config.n_step))

    for N in n_values:
        # (grid point, e) -> list of theta-binned (cross-seed tolerance) sets, one per seed
        pooled_cross: dict[tuple, list] = {}
        n_seeds_loaded = 0

        for seed in range(n_seeds):
            path = _candidate_path(output_dir, seed, N)
            if not path.exists():
                continue
            n_seeds_loaded += 1
            df = load_candidates(path)

            sets_within = compute_equivalence_sets(df, config.e_values, config.aim_tol_within)
            sets_cross = compute_equivalence_sets(df, config.e_values, config.aim_tol_cross)

            for gp, per_e in sets_within.items():
                for e, info in per_e.items():
                    key = (gp, seed, e)
                    st = seq_states.setdefault(key, SequentialState())
                    st.update(N, info["set"], config.k_consecutive, config.jaccard_threshold)

            for gp, per_e in sets_cross.items():
                for e, info in per_e.items():
                    pooled_cross.setdefault((gp, e), []).append(info["set"])

        logger.info("N=%d: loaded %d/%d seeds", N, n_seeds_loaded, n_seeds)

        for (gp, e), seed_sets in pooled_cross.items():
            stats = cross_seed_core_stats(seed_sets)
            cross_rows.append({"x": gp[0], "y": gp[1], "e": e, "N": N, **stats})

    seq_df = summarize_sequential(seq_states)
    cross_df = pd.DataFrame(cross_rows)

    seq_df.to_csv(results_dir / "equiv_sequential_all_e.csv", index=False)
    cross_df.to_csv(results_dir / "equiv_cross_seed_all_e.csv", index=False)

    overall_rows = []
    match_rate_curves: dict[float, pd.Series] = {}

    for e in config.e_values:
        e_seq = seq_df[seq_df["e"] == e]
        e_cross = cross_df[cross_df["e"] == e]

        e_seq.to_csv(results_dir / f"equiv_seq_e{e}.csv", index=False)
        e_cross.to_csv(results_dir / f"equiv_cross_seed_e{e}.csv", index=False)

        n_grid_points = e_seq["x"].astype(str).str.cat(e_seq["y"].astype(str), sep="_").nunique()
        summary = grid_level_summary(e_seq, n_grid_points)
        summary["e"] = e
        overall_rows.append(summary)

        # Which grid points drove non-convergence, and how badly (test 3 diagnostic)
        non_conv = non_converged_report(e_seq)
        non_conv.to_csv(results_dir / f"equiv_non_converged_gridpoints_e{e}.csv", index=False)
        logger.info(
            "e=%s: %d/%d grid points have >=1 seed that reached n_max=%d without stabilising "
            "(reason='reached_n_max_no_stability'); see equiv_non_converged_gridpoints_e%s.csv",
            e, len(non_conv), n_grid_points, config.n_max, e,
        )

        # (d) distribution of |E*| at convergence, per e — for converged (grid point, seed)
        # pairs only; non-converged pairs are broken out separately above since their final
        # |E*| reflects "still ambiguous at n_max", not a stabilised value.
        converged = e_seq[e_seq["converged"]]
        fig, ax = plt.subplots(figsize=(7, 5))
        if len(converged):
            ax.hist(converged["equiv_set_size_final"], bins=range(1, int(converged["equiv_set_size_final"].max()) + 2))
        ax.set_xlabel("|E*| at convergence (# statistically tied-best (club, aim) combos)")
        ax.set_ylabel("Count of (grid point, seed) pairs")
        ax.set_title(f"Equivalence-set size at convergence — e={e}  "
                     f"({len(converged)}/{len(e_seq)} pairs converged by N={config.n_max})")
        fig.tight_layout()
        fig.savefig(results_dir / f"equiv_size_at_convergence_e{e}.png", dpi=120)
        plt.close(fig)

        # (f) spatial map: where on the hole is |E*| large vs small
        if hole is not None:
            plot_equivalence_spatial_map(
                e_seq, hole, e, config.n_max,
                results_dir / f"equiv_spatial_map_e{e}.png",
            )

        # (c) mean sequential Jaccard-based match rate vs N — the fraction of
        # (grid point, seed) pairs already converged by each N.
        curve = pd.Series(
            {N: float((e_seq["converged_N"].fillna(np.inf) <= N).mean()) for N in n_values}
        )
        match_rate_curves[e] = curve

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(results_dir / "equiv_overall_summary.csv", index=False)

    # (c) summary plot: match rate vs N, one curve per e
    fig, ax = plt.subplots(figsize=(8, 5))
    for e, curve in match_rate_curves.items():
        ax.plot(curve.index, curve.values * 100, marker="o", label=f"e={e}")
    ax.set_xlabel("N (shots per grid point)")
    ax.set_ylabel("% (grid point, seed) pairs converged by N")
    ax.set_title("Equivalence-set convergence vs N, by e")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "equiv_match_rate_summary.png", dpi=120)
    plt.close(fig)

    logger.info("Done. Outputs in %s", results_dir)
    logger.info("Overall summary:\n%s", overall_df.to_string(index=False))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Equivalence-set convergence post-processing.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Directory containing seed{NNNN}/ subfolders with candidate parquet files.")
    p.add_argument("--results-dir", type=Path, default=Path("equivalence_results"))
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-start", type=int, default=10)
    p.add_argument("--n-step", type=int, default=10)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--e-values", type=float, nargs="+", default=[1.0, 1.645, 2.0])
    p.add_argument("--aim-tol-within", type=float, default=2.0)
    p.add_argument("--aim-tol-cross", type=float, default=3.0)
    p.add_argument("--jaccard-threshold", type=float, default=1.0)
    p.add_argument("--k-consecutive", type=int, default=3)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Directory with hole_9_data.csv etc, for the spatial map's hole geometry "
                        "(defaults to Parallelisation/data/, same as core.py's default).")
    p.add_argument("--no-spatial-map", action="store_true",
                   help="Skip the hole-layout spatial map (e.g. if geometry data isn't available here).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    cfg = EquivalenceConfig(
        e_values=tuple(args.e_values),
        aim_tol_within=args.aim_tol_within,
        aim_tol_cross=args.aim_tol_cross,
        jaccard_threshold=args.jaccard_threshold,
        k_consecutive=args.k_consecutive,
        n_start=args.n_start,
        n_step=args.n_step,
        n_max=args.n_max,
    )

    run_analysis(
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        config=cfg,
        n_seeds=args.n_seeds,
        data_dir=args.data_dir,
        make_spatial_map=not args.no_spatial_map,
    )
