"""
convergence_worker.py – Simulation sweep for one random seed.

This worker no longer declares convergence itself. A single arg-min
(club*, aim*) per grid point is not a well-defined statistic when several
(club, aim) combinations are statistically tied in expected outcome — at
many grid points the arg-min oscillates forever even after the model has
converged in every meaningful sense. Convergence is instead assessed as a
post-processing step (see equivalence.py / run_equivalence_analysis.py)
using equivalence sets of tied-best combinations, computed from the full
per-candidate R(s,theta)/SE(s,theta) data this worker persists below.

This worker sweeps N = n_start, n_start+n_step, ..., n_max shots per grid
point (always to n_max — no early stopping) and, at every N, saves:
  - the arg-min snapshot (optimal_results) as before, for plotting/back-compat
  - ONLY the 1-SE equivalence set E*(g) per grid point, as a CSV file
    (seed{SEED}_N{N}_equivset.csv). The full per-candidate table is not
    persisted.
  - a per-seed stabilisation log (seed{SEED}_stabilisation.tsv): for each N,
    the % of grid points whose equivalence set has been unchanged (Jaccard==1)
    for k_consecutive snapshots, plus the mean Jaccard vs the previous N and a
    boolean flag once 100% of points have stabilised. The N at which 100% was
    first reached (if ever) is recorded in seed{SEED}_result.json.
Cross-seed agreement (Test 2) is separate post-processing: cross_seed_jaccard.py.

Usage (direct):
    python convergence_worker.py --seed 0 --data-dir ../data --output-dir ./outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Local import – workers are run from inside the convergence/ directory
from core import HoleData, build_hole, plot_optimal_approaches, results_to_dataframe, simulate_approach_shots
from equivalence import SeedStabilityTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """All tunable parameters for the simulation sweep."""
    n_start: int   = 10       # initial shots per grid point
    n_step: int    = 10       # additional shots per iteration
    n_max: int     = 500      # sweep runs to this N (no early stopping — see equivalence.py)
    aim_range: tuple[float, float] = (-40.0, 40.0)
    aim_step: float = 5.0
    gp_training_iter: int = 100
    early_stop_N: Optional[int] = None  # cut short (for quick tests)
    carry_shift_yards: float = 0.0      # added to mean carry of all clubs
    variance_scale: float = 1.0         # multiplier on all club covariance matrices
    # --- Equivalence-set / stabilisation tracking (computed live, see equivalence.py) ---
    equiv_e: float = 1.0               # SE multiplier for the equivalence band
    k_consecutive: int = 3            # consecutive Jaccard==1 snapshots => grid point stabilised
    jaccard_threshold: float = 1.0


# ---------------------------------------------------------------------------
# Sweep result
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    seed: int
    n_iterations: int
    wall_time_s: float
    stopped_early: bool
    n_grid_points: int = 0
    n_points_stabilised: int = 0
    final_pct_stable: float = 0.0
    reached_100pct_stable: bool = False
    first_N_100pct_stable: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AIM_TOLERANCE: float = 1.0  # yards — two aim offsets are "equal" within this (arg-min match-rate diagnostic only)


def _compute_match_rate(
    current: list[dict],
    previous: list[dict],
    aim_tolerance: float = AIM_TOLERANCE,
) -> float:
    """Fraction of grid points where club is identical AND aim is within tolerance.

    Points present in one snapshot but not the other count as non-matches.
    Returns a value in [0, 1].
    """
    prev_map = {r["start"]: r for r in previous}
    all_points = {r["start"] for r in current} | set(prev_map.keys())
    if not all_points:
        return 1.0

    matches = 0
    for r in current:
        p = prev_map.get(r["start"])
        if p is None:
            continue
        if r["club"] == p["club"] and abs(r["aim_offset"] - p["aim_offset"]) <= aim_tolerance:
            matches += 1

    return matches / len(all_points)


def _save_csv(
    seed: int,
    N: int,
    optimal_results: list[dict],
    output_dir: Path,
    converged: bool = False,
) -> None:
    """Save per-iteration results as a CSV alongside the PNG."""
    tag = "CONVERGED" if converged else f"N{N:04d}"
    path = output_dir / f"seed{seed:04d}_{tag}.csv"
    df = results_to_dataframe(optimal_results, seed=seed, N=N)
    df.to_csv(path, index=False)
    logger.info("Saved CSV → %s", path)


def _append_match_log(
    seed: int,
    N: int,
    match_rate: Optional[float],
    output_dir: Path,
) -> None:
    """Append one line to the per-seed match-rate log (TSV)."""
    log_path = output_dir / f"seed{seed:04d}_match_rate.tsv"
    header_needed = not log_path.exists()
    with open(log_path, "a") as f:
        if header_needed:
            f.write("N\tmatch_rate_pct\n")
        rate_str = f"{match_rate * 100:.2f}" if match_rate is not None else "N/A"
        f.write(f"{N}\t{rate_str}\n")


def _save_snapshot_plot(
    seed: int,
    N: int,
    optimal_results: list[dict],
    hole: HoleData,
    output_dir: Path,
    converged: bool = False,
    match_rate: Optional[float] = None,
) -> None:
    tag = "CONVERGED" if converged else f"N{N:04d}"
    fname = output_dir / f"seed{seed:04d}_{tag}.png"
    status = "Converged" if converged else f"N = {N}"
    plot_optimal_approaches(
        optimal_results,
        hole,
        title=f"Seed {seed} | {status}",
        output_path=fname,
        match_rate=match_rate,
    )


def _candidates_dataframe(all_candidates: list[dict]) -> pd.DataFrame:
    """Tidy frame of every (grid-point, club, aim) candidate for one (seed, N)."""
    return pd.DataFrame([
        {
            "x": float(c["start"][0]),
            "y": float(c["start"][1]),
            "club": c["club"],
            "aim_offset": float(c["aim_offset"]),
            "mean": float(c["mean"]),
            "var": float(c["var"]),
            "n_total": int(c["n_total"]),
        }
        for c in all_candidates
    ])


def _save_equivset(
    seed: int,
    N: int,
    sets: dict,
    output_dir: Path,
) -> None:
    """Save ONLY the equivalence set for each grid point this N.

    One row per (grid point, club, aim) that is within `equiv_e` * SE_min of
    the arg-min ESHO. Columns: x, y, club, aim_offset, esho_mean, esho_se,
    n_total, is_argmin, R_min, SE_min, equiv_set_size, seed, N.
    The full per-candidate table is deliberately NOT persisted.
    """
    path = output_dir / f"seed{seed:04d}_N{N:04d}_equivset.csv"
    frames = []
    for (x, y), info in sets.items():
        d = info["members_df"].copy()
        d.insert(0, "x", float(x))
        d.insert(1, "y", float(y))
        d["R_min"] = info["R_min"]
        d["SE_min"] = info["SE_min"]
        d["equiv_set_size"] = info["size"]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["x", "y", "club", "aim_offset", "mean", "se", "n_total", "is_argmin",
                 "R_min", "SE_min", "equiv_set_size"]
    )
    df = df.rename(columns={"mean": "esho_mean", "se": "esho_se"})
    df["seed"] = seed
    df["N"] = N
    df.to_csv(path, index=False)
    logger.info("Saved equivalence-set CSV → %s (%d rows)", path, len(df))


def _append_stab_log(seed: int, row: dict, output_dir: Path) -> None:
    """Append one line to the per-seed stabilisation log (TSV).

    Columns: N, n_points, n_stable_ever, pct_stable_ever, n_stable_now,
    pct_stable_now, n_equiv_size1, pct_equiv_size1, mean_jaccard_vs_prev,
    all_points_stable.
    """
    log_path = output_dir / f"seed{seed:04d}_stabilisation.tsv"
    cols = ["N", "n_points", "n_stable_ever", "pct_stable_ever", "n_stable_now",
            "pct_stable_now", "n_equiv_size1", "pct_equiv_size1",
            "mean_jaccard_vs_prev", "all_points_stable"]
    header_needed = not log_path.exists()
    with open(log_path, "a") as f:
        if header_needed:
            f.write("\t".join(cols) + "\n")
        f.write("\t".join(_fmt(row[c]) for c in cols) + "\n")


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _save_result_json(result: ConvergenceResult, output_dir: Path) -> None:
    path = output_dir / f"seed{result.seed:04d}_result.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info("Result JSON → %s", path)


# ---------------------------------------------------------------------------
# Main convergence loop
# ---------------------------------------------------------------------------

def run_convergence(
    seed: int,
    config: WorkerConfig,
    data_dir: Path,
    output_dir: Path,
) -> ConvergenceResult:
    """Run the full N-sweep simulation for `seed`.

    Sweeps N = n_start, n_start+n_step, ..., n_max (no convergence stopping —
    that is now assessed as post-processing, see equivalence.py). Saves, at
    every N: the arg-min snapshot CSV/PNG (as before, for plotting/back-compat)
    and a full per-candidate parquet (R(s,theta), n_total for every club/aim
    combo) that the equivalence-set analysis consumes.
    """
    seed_dir = output_dir / f"seed{seed:04d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed log file (INFO level)
    log_path = output_dir / "logs" / f"seed{seed:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)

    logger.info("=== Seed %d  START ===", seed)
    logger.info("Config: %s", config)

    t0 = time.monotonic()

    # Each worker sets its own RNG state
    np.random.seed(seed)

    # Build hole geometry + train putt GPR (one-time cost per worker)
    logger.info("Building hole geometry and training putt GPR...")
    hole = build_hole(
        data_dir,
        gp_training_iter=config.gp_training_iter,
        carry_shift_yards=config.carry_shift_yards,
        variance_scale=config.variance_scale,
    )
    logger.info("Hole ready. %d strategy points.", len(hole.strategy_points))

    prev_results: Optional[list[dict]] = None  # for the arg-min match-rate diagnostic only
    accumulator: Optional[dict] = None          # accumulated shot strokes across iterations

    # Live within-seed stabilisation tracker (Test 1). Keeps a rolling
    # equivalence set per grid point and records when each first stabilises.
    tracker = SeedStabilityTracker(
        e=config.equiv_e,
        k_consecutive=config.k_consecutive,
        jaccard_threshold=config.jaccard_threshold,
    )

    N = config.n_start   # total shots accumulated so far (for logging / CSV label)
    n_iterations = 0
    stopped_early = False

    while True:
        iter_t0 = time.monotonic()
        # On the first iteration simulate n_start shots; thereafter add n_step.
        n_new = config.n_start if n_iterations == 0 else config.n_step
        logger.info("--- Iteration %d  N_total=%d  (+%d new shots) ---", n_iterations, N, n_new)

        # Simulate ONLY the new shots; merge with accumulator internally.
        # return_all_candidates=True also gives us every (club, aim) candidate's
        # R(s,theta)/n_total, not just the arg-min, for the equivalence-set layer.
        optimal_results, accumulator, all_candidates = simulate_approach_shots(
            hole=hole,
            n_new=n_new,
            accumulator=accumulator,
            aim_range=config.aim_range,
            aim_step=config.aim_step,
            return_all_candidates=True,
        )

        logger.info(
            "Iteration %d  N=%d: %d/%d grid points returned valid results  (%.1fs)",
            n_iterations, N, len(optimal_results), len(hole.strategy_points),
            time.monotonic() - iter_t0,
        )

        # Arg-min match rate vs previous iteration — retained only as a diagnostic
        # (it is NOT used to declare convergence; see equivalence.py for that).
        match_rate: Optional[float] = None
        if prev_results is not None:
            match_rate = _compute_match_rate(optimal_results, prev_results)
            logger.info("  Arg-min match rate vs N=%d: %.1f%%", N - config.n_step, match_rate * 100)

        # Log club distribution
        club_counts = Counter(r["club"] for r in optimal_results)
        logger.info("  Club distribution: %s", dict(club_counts.most_common()))

        # --- Equivalence-set + stabilisation tracking (Test 1, computed live) ---
        cand_df = _candidates_dataframe(all_candidates)
        stab_row, equiv_sets = tracker.update(N, cand_df)
        logger.info(
            "  Stabilised: %d/%d points ever (%.1f%%), %d currently; "
            "mean Jaccard vs prev = %.3f%s",
            stab_row["n_stable_ever"], stab_row["n_points"],
            stab_row["pct_stable_ever"] * 100, stab_row["n_stable_now"],
            stab_row["mean_jaccard_vs_prev"],
            "  [ALL POINTS STABLE]" if stab_row["all_points_stable"] else "",
        )

        # Save per-iteration outputs
        _save_csv(seed, N, optimal_results, seed_dir)
        _save_equivset(seed, N, equiv_sets, seed_dir)
        _append_stab_log(seed, stab_row, seed_dir)
        _append_match_log(seed, N, match_rate, seed_dir)
        _save_snapshot_plot(seed, N, optimal_results, hole, seed_dir, match_rate=match_rate)

        prev_results = optimal_results

        # Check for early stop (test mode)
        if config.early_stop_N is not None and N >= config.early_stop_N:
            logger.info("Early stop triggered at N=%d (limit=%d).", N, config.early_stop_N)
            stopped_early = True
            break

        if N >= config.n_max:
            logger.info("Sweep complete at n_max=%d.", config.n_max)
            break

        N += config.n_step   # track total accumulated shots for logging
        n_iterations += 1

    wall_time = time.monotonic() - t0
    stab_summary = tracker.summary()
    result = ConvergenceResult(
        seed=seed,
        n_iterations=n_iterations,
        wall_time_s=wall_time,
        stopped_early=stopped_early,
        n_grid_points=stab_summary["n_grid_points"],
        n_points_stabilised=stab_summary["n_points_stabilised"],
        final_pct_stable=stab_summary["final_pct_stable"],
        reached_100pct_stable=stab_summary["reached_100pct_stable"],
        first_N_100pct_stable=stab_summary["first_N_100pct_stable"],
    )
    _save_result_json(result, seed_dir)
    logger.info(
        "Seed %d stabilisation: %d/%d points, 100%%-stable=%s (first at N=%s)",
        seed, stab_summary["n_points_stabilised"], stab_summary["n_grid_points"],
        stab_summary["reached_100pct_stable"], stab_summary["first_N_100pct_stable"],
    )

    logger.info("=== Seed %d  DONE  (%.1fs) ===", seed, wall_time)
    logger.removeHandler(fh)
    fh.close()

    return result


# ---------------------------------------------------------------------------
# CLI entry point (single seed, no parallelism)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convergence study for one seed.")
    p.add_argument("--seed",          type=int, required=True)
    p.add_argument("--data-dir",      type=Path, default=None)
    p.add_argument("--output-dir",    type=Path, default=Path("outputs"))
    p.add_argument("--n-start",       type=int,   default=10)
    p.add_argument("--n-step",        type=int,   default=10)
    p.add_argument("--n-max",         type=int,   default=500)
    p.add_argument("--aim-step",      type=float, default=5.0)
    p.add_argument("--gp-iter",       type=int,   default=100)
    p.add_argument("--equiv-e",       type=float, default=1.0,
                   help="SE multiplier for the equivalence band (E* = R<=R_min+e*SE_min).")
    p.add_argument("--k-consecutive", type=int,   default=3,
                   help="Consecutive Jaccard==1 snapshots for a grid point to count as stabilised.")
    p.add_argument("--early-stop-N",  type=int,   default=None,
                   help="Stop after this many shots (for quick tests).")
    p.add_argument("--log-level",     default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    cfg = WorkerConfig(
        n_start=args.n_start,
        n_step=args.n_step,
        n_max=args.n_max,
        aim_step=args.aim_step,
        gp_training_iter=args.gp_iter,
        equiv_e=args.equiv_e,
        k_consecutive=args.k_consecutive,
        early_stop_N=args.early_stop_N,
    )

    data_dir = Path(args.data_dir) if args.data_dir else None
    result = run_convergence(
        seed=args.seed,
        config=cfg,
        data_dir=data_dir or (Path(__file__).parent.parent / "data"),
        output_dir=Path(args.output_dir),
    )

    print(json.dumps(asdict(result), indent=2))
