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

So this worker simply sweeps N = n_start, n_start+n_step, ..., n_max shots
per grid point and, at every N, saves:
  - the arg-min snapshot (optimal_results) as before, for plotting/back-compat
  - every (grid-point, club, aim) candidate's R(s,theta)=mean and
    SE(s,theta)=sqrt(var/n_total), as a parquet file, for the equivalence-set
    post-processing layer.

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
    aim_range: tuple[float, float] = (-20.0, 20.0)
    aim_step: float = 2.0
    gp_training_iter: int = 100
    early_stop_N: Optional[int] = None  # cut short (for quick tests)
    carry_shift_yards: float = 0.0      # added to mean carry of all clubs
    variance_scale: float = 1.0         # multiplier on all club covariance matrices


# ---------------------------------------------------------------------------
# Sweep result
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    seed: int
    n_iterations: int
    wall_time_s: float
    stopped_early: bool


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


def _save_candidates(
    seed: int,
    N: int,
    all_candidates: list[dict],
    output_dir: Path,
) -> None:
    """Save every (grid-point, club, aim) candidate's R(s,theta)/n_total this N.

    This is the full per-candidate data the equivalence-set post-processing
    layer (equivalence.py) needs — SE(s,theta) = sqrt(var/n_total) is derived
    from these columns, not stored directly.
    """
    path = output_dir / f"seed{seed:04d}_N{N:04d}_candidates.parquet"
    df = pd.DataFrame([
        {
            "x": float(c["start"][0]),
            "y": float(c["start"][1]),
            "club": c["club"],
            "aim_offset": float(c["aim_offset"]),
            "mean": float(c["mean"]),
            "var": float(c["var"]),
            "n_total": int(c["n_total"]),
            "seed": seed,
            "N": N,
        }
        for c in all_candidates
    ])
    df.to_parquet(path, index=False)
    logger.info("Saved candidates parquet → %s (%d rows)", path, len(df))


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

        # Save per-iteration outputs
        _save_csv(seed, N, optimal_results, seed_dir)
        _save_candidates(seed, N, all_candidates, seed_dir)
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
    result = ConvergenceResult(
        seed=seed,
        n_iterations=n_iterations,
        wall_time_s=wall_time,
        stopped_early=stopped_early,
    )
    _save_result_json(result, seed_dir)

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
    p.add_argument("--aim-step",      type=float, default=2.0)
    p.add_argument("--gp-iter",       type=int,   default=100)
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
