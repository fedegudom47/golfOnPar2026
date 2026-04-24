"""
convergence_worker_birdie.py – Convergence study for birdie probability.

Same algorithm as convergence/convergence_worker.py but using P(birdie)
instead of ESHO:
  - Simulation calls simulate_approach_shots_birdie()
  - Convergence: k=3 identical (club, aim_offset) snapshots per grid point
  - CSVs save mean_birdie_prob / var_birdie_prob instead of esho_mean / esho_var

Usage (direct):
    python convergence_worker_birdie.py --seed 0 --data-dir ../data
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core_birdie import (
    BirdieHoleData,
    build_hole_birdie,
    plot_optimal_approaches_birdie,
    results_to_dataframe,
    simulate_approach_shots_birdie,
    BirdieAccumulator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------

@dataclass
class BirdieWorkerConfig:
    """All tunable parameters for the birdie convergence study."""
    n_start: int   = 10
    n_step: int    = 10
    n_max: int     = 300
    k: int         = 3
    aim_range: tuple[float, float] = (-20.0, 20.0)
    aim_step: float = 2.0
    gp_training_iter: int = 200
    early_stop_N: Optional[int] = None
    carry_shift_yards: float = 0.0
    variance_scale: float = 1.0


# ---------------------------------------------------------------------------
# Convergence result
# ---------------------------------------------------------------------------

@dataclass
class BirdieConvergenceResult:
    seed: int
    convergence_N: Optional[int]
    n_iterations: int
    wall_time_s: float
    did_not_converge: bool
    stopped_early: bool


# ---------------------------------------------------------------------------
# Helpers (identical in logic to convergence_worker.py)
# ---------------------------------------------------------------------------

AIM_TOLERANCE: float = 1.0


def _opt_key(result: dict) -> tuple[str, int]:
    return (result["club"], int(round(result["aim_offset"])))


def _build_snapshot(optimal_results: list[dict]) -> dict[tuple, tuple[str, int]]:
    return {r["start"]: _opt_key(r) for r in optimal_results}


def _snapshots_agree(history: deque) -> bool:
    first = history[0]
    for snap in history:
        if snap != first:
            return False
    return True


def _compute_match_rate(
    current: list[dict],
    previous: list[dict],
    aim_tolerance: float = AIM_TOLERANCE,
) -> float:
    prev_map  = {r["start"]: r for r in previous}
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
    tag  = "CONVERGED" if converged else f"N{N:04d}"
    path = output_dir / f"seed{seed:04d}_{tag}.csv"
    df   = results_to_dataframe(optimal_results, seed=seed, N=N)
    df.to_csv(path, index=False)
    logger.info("Saved CSV → %s", path)


def _append_match_log(
    seed: int,
    N: int,
    match_rate: Optional[float],
    output_dir: Path,
) -> None:
    log_path     = output_dir / f"seed{seed:04d}_match_rate.tsv"
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
    hole: BirdieHoleData,
    output_dir: Path,
    converged: bool = False,
    match_rate: Optional[float] = None,
) -> None:
    tag    = "CONVERGED" if converged else f"N{N:04d}"
    fname  = output_dir / f"seed{seed:04d}_{tag}.png"
    status = "Converged" if converged else f"N = {N}"
    plot_optimal_approaches_birdie(
        optimal_results,
        hole,
        title=f"Seed {seed} | {status}",
        output_path=fname,
        match_rate=match_rate,
    )


def _save_result_json(result: BirdieConvergenceResult, output_dir: Path) -> None:
    path = output_dir / f"seed{result.seed:04d}_result.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info("Result JSON → %s", path)


# ---------------------------------------------------------------------------
# Main convergence loop
# ---------------------------------------------------------------------------

def run_convergence_birdie(
    seed: int,
    config: BirdieWorkerConfig,
    data_dir: Path,
    output_dir: Path,
) -> BirdieConvergenceResult:
    """Run birdie convergence study for `seed`."""
    seed_dir = output_dir / f"seed{seed:04d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "logs" / f"seed{seed:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)

    logger.info("=== Birdie Seed %d  START ===", seed)
    logger.info("Config: %s", config)

    t0 = time.monotonic()
    np.random.seed(seed)

    logger.info("Building hole geometry and training birdie GPR...")
    hole = build_hole_birdie(
        data_dir,
        gp_training_iter=config.gp_training_iter,
        carry_shift_yards=config.carry_shift_yards,
        variance_scale=config.variance_scale,
    )
    logger.info("Hole ready. %d strategy points.", len(hole.strategy_points))

    history: deque[dict] = deque(maxlen=config.k)
    prev_results: Optional[list[dict]] = None
    accumulator: Optional[BirdieAccumulator] = None

    N = config.n_start
    n_iterations       = 0
    did_not_converge   = False
    stopped_early      = False
    convergence_N: Optional[int] = None

    while True:
        iter_t0 = time.monotonic()
        n_new = config.n_start if n_iterations == 0 else config.n_step
        logger.info(
            "--- Iteration %d  N_total=%d  (+%d new shots) ---",
            n_iterations, N, n_new,
        )

        optimal_results, accumulator = simulate_approach_shots_birdie(
            hole=hole,
            n_new=n_new,
            accumulator=accumulator,
            aim_range=config.aim_range,
            aim_step=config.aim_step,
        )

        logger.info(
            "Iteration %d  N=%d: %d/%d grid points (%.1fs)",
            n_iterations, N, len(optimal_results), len(hole.strategy_points),
            time.monotonic() - iter_t0,
        )

        match_rate: Optional[float] = None
        if prev_results is not None:
            match_rate = _compute_match_rate(optimal_results, prev_results)
            logger.info("  Match rate vs N=%d: %.1f%%", N - config.n_step, match_rate * 100)

        club_counts = Counter(r["club"] for r in optimal_results)
        logger.info("  Club distribution: %s", dict(club_counts.most_common()))

        mean_birdie = np.mean([r["mean_birdie_prob"] for r in optimal_results])
        logger.info("  Mean P(birdie) across grid: %.4f", mean_birdie)

        _save_csv(seed, N, optimal_results, seed_dir)
        _append_match_log(seed, N, match_rate, seed_dir)
        _save_snapshot_plot(seed, N, optimal_results, hole, seed_dir, match_rate=match_rate)

        snap = _build_snapshot(optimal_results)
        history.append(snap)
        prev_results = optimal_results

        if config.early_stop_N is not None and N >= config.early_stop_N:
            logger.info("Early stop at N=%d (limit=%d).", N, config.early_stop_N)
            stopped_early = True
            break

        if len(history) == config.k and _snapshots_agree(history):
            convergence_N = N
            logger.info(
                "CONVERGED at N=%d after %d iterations  (%.1fs total).",
                N, n_iterations, time.monotonic() - t0,
            )
            _save_snapshot_plot(seed, N, optimal_results, hole, seed_dir,
                                 converged=True, match_rate=match_rate)
            _save_csv(seed, N, optimal_results, seed_dir, converged=True)
            break

        if N >= config.n_max:
            logger.warning("Did NOT converge within n_max=%d.", config.n_max)
            did_not_converge = True
            break

        N += config.n_step
        n_iterations += 1

    wall_time = time.monotonic() - t0
    result = BirdieConvergenceResult(
        seed=seed,
        convergence_N=convergence_N,
        n_iterations=n_iterations,
        wall_time_s=wall_time,
        did_not_converge=did_not_converge,
        stopped_early=stopped_early,
    )
    _save_result_json(result, seed_dir)

    logger.info("=== Birdie Seed %d  DONE  (%.1fs) ===", seed, wall_time)
    logger.removeHandler(fh)
    fh.close()

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Birdie convergence study for one seed.")
    p.add_argument("--seed",           type=int,   required=True)
    p.add_argument("--data-dir",       type=Path,  default=None)
    p.add_argument("--output-dir",     type=Path,  default=Path("outputs"))
    p.add_argument("--n-start",        type=int,   default=10)
    p.add_argument("--n-step",         type=int,   default=10)
    p.add_argument("--n-max",          type=int,   default=300)
    p.add_argument("--k",              type=int,   default=3)
    p.add_argument("--aim-step",       type=float, default=2.0)
    p.add_argument("--gp-iter",        type=int,   default=200)
    p.add_argument("--early-stop-N",   type=int,   default=None)
    p.add_argument("--carry-shift",    type=float, default=0.0)
    p.add_argument("--variance-scale", type=float, default=1.0)
    p.add_argument("--log-level",      default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    cfg = BirdieWorkerConfig(
        n_start=args.n_start,
        n_step=args.n_step,
        n_max=args.n_max,
        k=args.k,
        aim_step=args.aim_step,
        gp_training_iter=args.gp_iter,
        early_stop_N=args.early_stop_N,
        carry_shift_yards=args.carry_shift,
        variance_scale=args.variance_scale,
    )

    data_dir = Path(args.data_dir) if args.data_dir else (
        Path(__file__).parent.parent / "data"
    )
    result = run_convergence_birdie(
        seed=args.seed,
        config=cfg,
        data_dir=data_dir,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(asdict(result), indent=2))
