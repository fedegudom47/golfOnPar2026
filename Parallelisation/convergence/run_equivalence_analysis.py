"""
run_equivalence_analysis.py – Post-processing: aggregate the per-seed
equivalence-set / stabilisation results the HPC workers now compute live.

The workers (convergence_worker.py) already:
  * store E*(g, seed, N) — the 1-SE equivalence set — per (seed, N) as
    seed{SEED}/seed{SEED}_N{N}_equivset.parquet
  * track within-seed stabilisation (Test 1) live and log it to
    seed{SEED}/seed{SEED}_stabilisation.tsv + seed{SEED}_result.json

This script does NOT resimulate. It:
  (a) re-derives per-(grid point, seed) stabilisation from the stored equivset
      series (so you get converged_N and |E*| at n_max in one table)
  (b) pools the stabilisation logs across seeds -> % of (grid point, seed)
      pairs stabilised vs N, and per-seed "reached 100%?" booleans
  (c) computes cross-seed core/union agreement per (grid point, N)
  (d) draws a spatial map of mean |E*| across seeds

For the pairwise cross-seed Jaccard matrices, see cross_seed_jaccard.py.

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
    cross_seed_core_stats,
    grid_level_summary,
    load_equivset,
    non_converged_report,
    summarize_sequential,
)

logger = logging.getLogger(__name__)


def _equivset_path(output_dir: Path, seed: int, N: int) -> Path:
    return output_dir / f"seed{seed:04d}" / f"seed{seed:04d}_N{N:04d}_equivset.csv"


def collect_stabilisation_logs(output_dir: Path, n_seeds: int) -> pd.DataFrame:
    """Concatenate every seed's seed{SEED}_stabilisation.tsv."""
    frames = []
    for seed in range(n_seeds):
        p = output_dir / f"seed{seed:04d}" / f"seed{seed:04d}_stabilisation.tsv"
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t")
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_equivalence_spatial_map(seq_df: pd.DataFrame, hole, n_max: int, output_path: Path) -> None:
    from core import plot_hole_layout

    per_gp = (
        seq_df.groupby(["x", "y"])["equiv_set_size_final"]
        .mean().reset_index(name="mean_equiv_set_size")
    )
    fig, ax = plt.subplots(figsize=(11, 13))
    plot_hole_layout(hole, title="", plot_strategy_points=False, ax=ax)
    sc = ax.scatter(
        per_gp["x"], per_gp["y"], c=per_gp["mean_equiv_set_size"],
        cmap="RdYlGn_r", s=60, alpha=0.9, edgecolors="black", linewidths=0.5, zorder=20,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean |E*| across seeds (# statistically tied-best club/aim combos)")
    ax.set_title(
        f"Strategic ambiguity by grid point — e=1, at convergence or N={n_max}\n"
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
    n_values = list(range(config.n_start, config.n_max + 1, config.n_step))

    # ---- (b) pooled stabilisation logs -----------------------------------
    stab = collect_stabilisation_logs(output_dir, n_seeds)
    if not stab.empty:
        stab.to_csv(results_dir / "stabilisation_all_seeds.csv", index=False)
        per_N = stab.groupby("N").agg(
            mean_pct_stable_ever=("pct_stable_ever", "mean"),
            mean_pct_stable_now=("pct_stable_now", "mean"),
            mean_pct_equiv_size1=("pct_equiv_size1", "mean"),
            mean_jaccard_vs_prev=("mean_jaccard_vs_prev", "mean"),
            n_seeds=("seed", "nunique"),
        ).reset_index()
        per_N.to_csv(results_dir / "stabilisation_vs_N.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(per_N["N"], per_N["mean_pct_stable_ever"] * 100, marker="o",
                label="% points ever stabilised")
        ax.plot(per_N["N"], per_N["mean_pct_equiv_size1"] * 100, marker="s",
                label="% points with |E*| = 1")
        ax.plot(per_N["N"], per_N["mean_jaccard_vs_prev"] * 100, marker="^",
                label="mean Jaccard vs prev N (%)")
        ax.set_xlabel("N (shots per grid point)")
        ax.set_ylabel("% (mean across seeds)")
        ax.set_title("Within-seed stabilisation vs N")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "stabilisation_vs_N.png", dpi=120)
        plt.close(fig)

    # ---- (a) re-derive per-(grid point, seed) stabilisation -------------
    seq_states: dict[tuple, SequentialState] = {}
    cross_rows: list[dict] = []

    for N in n_values:
        pooled: dict[tuple, list] = {}
        loaded = 0
        for seed in range(n_seeds):
            p = _equivset_path(output_dir, seed, N)
            if not p.exists():
                continue
            loaded += 1
            sets = load_equivset(p)
            for gp, eqset in sets.items():
                st = seq_states.setdefault((gp, seed), SequentialState())
                st.update(N, eqset, config.k_consecutive, config.jaccard_threshold)
                pooled.setdefault(gp, []).append(eqset)
        logger.info("N=%d: loaded %d/%d seeds", N, loaded, n_seeds)
        for gp, seed_sets in pooled.items():
            cross_rows.append({
                "x": gp[0], "y": gp[1], "N": N, **cross_seed_core_stats(seed_sets),
            })

    seq_df = summarize_sequential(seq_states)
    cross_df = pd.DataFrame(cross_rows)
    seq_df.to_csv(results_dir / "equiv_sequential.csv", index=False)
    cross_df.to_csv(results_dir / "equiv_cross_seed.csv", index=False)

    n_grid_points = seq_df[["x", "y"]].drop_duplicates().shape[0]
    summary = grid_level_summary(seq_df, n_grid_points)
    pd.DataFrame([summary]).to_csv(results_dir / "equiv_overall_summary.csv", index=False)

    non_conv = non_converged_report(seq_df)
    non_conv.to_csv(results_dir / "equiv_non_converged_gridpoints.csv", index=False)
    logger.info(
        "%d/%d grid points have >=1 seed that reached n_max=%d without stabilising.",
        len(non_conv), n_grid_points, config.n_max,
    )

    converged = seq_df[seq_df["converged"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(converged):
        ax.hist(converged["equiv_set_size_final"],
                bins=range(1, int(converged["equiv_set_size_final"].max()) + 2))
    ax.set_xlabel("|E*| at stabilisation (# tied-best (club, aim) combos)")
    ax.set_ylabel("Count of (grid point, seed) pairs")
    ax.set_title(f"Equivalence-set size at stabilisation  "
                 f"({len(converged)}/{len(seq_df)} pairs stabilised by N={config.n_max})")
    fig.tight_layout()
    fig.savefig(results_dir / "equiv_size_at_convergence.png", dpi=120)
    plt.close(fig)

    if make_spatial_map:
        try:
            from core import build_hole
            hole = build_hole(data_dir, gp_training_iter=1)
            plot_equivalence_spatial_map(seq_df, hole, config.n_max,
                                        results_dir / "equiv_spatial_map.png")
        except Exception:
            logger.warning("Spatial map skipped (hole geometry unavailable).", exc_info=True)

    logger.info("Done. Outputs in %s", results_dir)
    logger.info("Overall summary: %s", summary)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Equivalence-set convergence post-processing.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--results-dir", type=Path, default=Path("equivalence_results"))
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-start", type=int, default=10)
    p.add_argument("--n-step", type=int, default=10)
    p.add_argument("--n-max", type=int, default=500)
    p.add_argument("--e", type=float, default=1.0)
    p.add_argument("--jaccard-threshold", type=float, default=1.0)
    p.add_argument("--k-consecutive", type=int, default=3)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--no-spatial-map", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    cfg = EquivalenceConfig(
        e=args.e,
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
