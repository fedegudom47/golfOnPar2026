"""
run_local_birdie.py – Local parallel birdie convergence study.

Mirrors convergence/run_local.py but uses the birdie simulation.

Quick test (2 seeds, early stop at N=30):
    python run_local_birdie.py --mode test

Full run (100 seeds, up to N=300):
    python run_local_birdie.py --mode full --n-workers 4
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import logging
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from convergence_worker_birdie import (
    BirdieConvergenceResult,
    BirdieWorkerConfig,
    run_convergence_birdie,
)

logger = logging.getLogger(__name__)


def _worker(
    seed: int,
    config: BirdieWorkerConfig,
    data_dir: Path,
    output_dir: Path,
) -> BirdieConvergenceResult:
    return run_convergence_birdie(
        seed=seed,
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
    )


def _write_summary(results: list[BirdieConvergenceResult], output_dir: Path) -> None:
    path = output_dir / "convergence_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    logger.info("Summary CSV → %s", path)


def _print_summary(results: list[BirdieConvergenceResult]) -> None:
    converged = [r for r in results if r.convergence_N is not None]
    not_conv  = [r for r in results if r.did_not_converge]
    early     = [r for r in results if r.stopped_early]

    print("\n" + "=" * 60)
    print(f"  Seeds run       : {len(results)}")
    print(f"  Converged       : {len(converged)}")
    print(f"  Did not converge: {len(not_conv)}")
    print(f"  Stopped early   : {len(early)}")

    if converged:
        import statistics
        ns = [r.convergence_N for r in converged]
        print(f"\n  Convergence N — min={min(ns)}  max={max(ns)}  "
              f"mean={statistics.mean(ns):.1f}  median={statistics.median(ns):.1f}")
    print("=" * 60 + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local parallel birdie convergence study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",         choices=["test", "full"], default="test")
    p.add_argument("--seeds",        type=int,   default=None)
    p.add_argument("--n-workers",    type=int,   default=2)
    p.add_argument("--n-start",      type=int,   default=10)
    p.add_argument("--n-step",       type=int,   default=10)
    p.add_argument("--n-max",        type=int,   default=300)
    p.add_argument("--k",            type=int,   default=3)
    p.add_argument("--aim-step",     type=float, default=2.0)
    p.add_argument("--gp-iter",      type=int,   default=200)
    p.add_argument("--early-stop-N", type=int,   default=None)
    p.add_argument("--carry-shift",  type=float, default=0.0,
                   help="Yards to add to mean carry of all clubs.")
    p.add_argument("--variance-scale", type=float, default=1.0,
                   help="Multiplier on all club covariance matrices.")
    p.add_argument("--data-dir",     type=Path,
                   default=Path(__file__).parent.parent / "data")
    p.add_argument("--output-dir",   type=Path,  default=Path("outputs"))
    p.add_argument("--log-level",    default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run_local_birdie.log"),
        ],
    )

    if args.mode == "test":
        n_seeds    = args.seeds if args.seeds is not None else 2
        early_stop = args.early_stop_N if args.early_stop_N is not None else 30
        n_workers  = min(args.n_workers, n_seeds)
        logger.info("MODE=test  seeds=%d  early_stop_N=%d  workers=%d",
                    n_seeds, early_stop, n_workers)
    else:
        n_seeds    = args.seeds if args.seeds is not None else 100
        early_stop = args.early_stop_N
        n_workers  = args.n_workers
        logger.info("MODE=full  seeds=%d  workers=%d", n_seeds, n_workers)

    config = BirdieWorkerConfig(
        n_start=args.n_start,
        n_step=args.n_step,
        n_max=args.n_max,
        k=args.k,
        aim_step=args.aim_step,
        gp_training_iter=args.gp_iter,
        early_stop_N=early_stop,
        carry_shift_yards=args.carry_shift,
        variance_scale=args.variance_scale,
    )

    seeds         = list(range(n_seeds))
    bound_worker  = functools.partial(
        _worker,
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    logger.info("Launching %d worker(s) for %d seeds ...", n_workers, n_seeds)
    results: list[BirdieConvergenceResult] = []

    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(bound_worker, seeds), start=1):
            results.append(result)
            status = (
                "CONVERGED" if result.convergence_N
                else ("EARLY" if result.stopped_early else "NOT CONV")
            )
            logger.info(
                "[%d/%d] seed=%d  %s  N=%s  time=%.1fs",
                i, n_seeds, result.seed, status,
                result.convergence_N, result.wall_time_s,
            )

    results.sort(key=lambda r: r.seed)
    _write_summary(results, args.output_dir)
    _print_summary(results)

    json_path = args.output_dir / "convergence_summary.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Full JSON → %s", json_path)


if __name__ == "__main__":
    main()
