"""
convergence_worker.py – Convergence study for one random seed.

Algorithm
---------
Starting from N = n_start shots per grid-point, we increase N by n_step
each iteration and recompute the optimal strategy OPT_N(x, y) = (club,
aim_offset) for every strategy point.  Convergence is declared when the
last k consecutive OPT snapshots are identical for every grid point.

A PNG snapshot of the hole coloured by OPT is saved after every iteration
so you can see the strategy stabilise visually.

Usage (direct):
    python convergence_worker.py --seed 0 --data-dir ../data --output-dir ./outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Local import – workers are run from inside the convergence/ directory
from core import HoleData, build_hole, plot_optimal_approaches, simulate_approach_shots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """All tunable parameters for the convergence study."""
    n_start: int   = 10       # initial shots per grid point
    n_step: int    = 10       # additional shots per iteration
    n_max: int     = 300      # give up if N exceeds this
    k: int         = 3        # consecutive identical snapshots required
    aim_range: tuple[float, float] = (-20.0, 20.0)
    aim_step: float = 2.0
    gp_training_iter: int = 100
    early_stop_N: Optional[int] = None  # cut short (for quick tests)


# ---------------------------------------------------------------------------
# Convergence result
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    seed: int
    convergence_N: Optional[int]   # None → did not converge within n_max
    n_iterations: int
    wall_time_s: float
    did_not_converge: bool
    stopped_early: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opt_key(result: dict) -> tuple[str, int]:
    """Hashable key for one grid-point's optimal decision."""
    return (result["club"], int(round(result["aim_offset"])))


def _build_snapshot(optimal_results: list[dict]) -> dict[tuple, tuple[str, int]]:
    """Map start_point → (club, aim_offset_int) for the full grid."""
    return {r["start"]: _opt_key(r) for r in optimal_results}


def _snapshots_agree(history: deque) -> bool:
    """Return True iff all snapshots in the deque are identical."""
    first = history[0]
    for snap in history:
        if snap != first:
            return False
    return True


def _save_snapshot_plot(
    seed: int,
    N: int,
    optimal_results: list[dict],
    hole: HoleData,
    output_dir: Path,
    converged: bool = False,
) -> None:
    tag = "CONVERGED" if converged else f"N{N:04d}"
    fname = output_dir / f"seed{seed:04d}_{tag}.png"
    status = "Converged" if converged else f"N = {N}"
    plot_optimal_approaches(
        optimal_results,
        hole,
        title=f"Seed {seed} | {status}",
        output_path=fname,
    )


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
    """Run the convergence study for `seed`.

    Returns a ConvergenceResult and saves PNGs + a JSON result file under
    `output_dir / seed{seed:04d}/`.
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
    hole = build_hole(data_dir, gp_training_iter=config.gp_training_iter)
    logger.info("Hole ready. %d strategy points.", len(hole.strategy_points))

    history: deque[dict] = deque(maxlen=config.k)

    N = config.n_start
    n_iterations = 0
    did_not_converge = False
    stopped_early = False
    convergence_N: Optional[int] = None

    while True:
        iter_t0 = time.monotonic()
        logger.info("--- Iteration %d  N=%d ---", n_iterations, N)

        # Simulate approach shots with N samples per grid point
        optimal_results = simulate_approach_shots(
            hole=hole,
            n_samples=N,
            aim_range=config.aim_range,
            aim_step=config.aim_step,
        )

        logger.info(
            "Iteration %d  N=%d: %d/%d grid points returned valid results  (%.1fs)",
            n_iterations, N, len(optimal_results), len(hole.strategy_points),
            time.monotonic() - iter_t0,
        )

        # Save snapshot image
        _save_snapshot_plot(seed, N, optimal_results, hole, seed_dir)

        # Log a brief summary of strategy distribution
        from collections import Counter
        club_counts = Counter(r["club"] for r in optimal_results)
        logger.info("  Club distribution: %s", dict(club_counts.most_common()))

        # Store snapshot in rolling history
        snap = _build_snapshot(optimal_results)
        history.append(snap)

        # Check for early stop (test mode)
        if config.early_stop_N is not None and N >= config.early_stop_N:
            logger.info("Early stop triggered at N=%d (limit=%d).", N, config.early_stop_N)
            _save_snapshot_plot(seed, N, optimal_results, hole, seed_dir, converged=False)
            stopped_early = True
            break

        # Check convergence
        if len(history) == config.k and _snapshots_agree(history):
            convergence_N = N
            logger.info(
                "CONVERGED at N=%d after %d iterations  (%.1fs total).",
                N, n_iterations, time.monotonic() - t0,
            )
            _save_snapshot_plot(seed, N, optimal_results, hole, seed_dir, converged=True)
            break

        # Check whether we have exhausted our budget
        if N >= config.n_max:
            logger.warning("Did NOT converge within n_max=%d.", config.n_max)
            did_not_converge = True
            break

        N += config.n_step
        n_iterations += 1

    wall_time = time.monotonic() - t0
    result = ConvergenceResult(
        seed=seed,
        convergence_N=convergence_N,
        n_iterations=n_iterations,
        wall_time_s=wall_time,
        did_not_converge=did_not_converge,
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
    p.add_argument("--n-max",         type=int,   default=300)
    p.add_argument("--k",             type=int,   default=3)
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
        k=args.k,
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
