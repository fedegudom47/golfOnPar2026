"""
run_local.py – Local parallel convergence study using multiprocessing.

Each seed is handed to a separate subprocess so the heavy per-seed work
(hole geometry setup, putt GPR training, simulation loop) runs without
GIL contention.  No GPyTorch models or shapely objects are pickled across
boundaries — every worker calls build_hole() independently.

Quick local TEST (terminates after N=30, 2 seeds, 2 workers, small grid):
    python run_local.py --mode test

Full local run (100 seeds, up to N=300):
    python run_local.py --mode full --n-workers 4

Results are written to ./outputs/ with one sub-directory per seed plus a
summary CSV at ./outputs/convergence_summary.csv.
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

# Workers run inside the convergence/ directory
sys.path.insert(0, str(Path(__file__).parent))
from convergence_worker import ConvergenceResult, WorkerConfig, run_convergence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker entry point (must be module-level for multiprocessing on macOS/Windows)
# ---------------------------------------------------------------------------

def _worker(
    seed: int,
    config: WorkerConfig,
    data_dir: Path,
    output_dir: Path,
) -> ConvergenceResult:
    return run_convergence(
        seed=seed,
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _write_summary(results: list[ConvergenceResult], output_dir: Path) -> None:
    path = output_dir / "convergence_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    logger.info("Summary CSV → %s", path)


def _print_summary(results: list[ConvergenceResult]) -> None:
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
        print(f"  Distribution   : {sorted(ns)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local parallel convergence study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["test", "full"], default="test",
        help=(
            "test = 2 seeds, early stop at N=30, small grid for sanity check. "
            "full = 100 seeds, up to N=300."
        ),
    )
    p.add_argument("--seeds",       type=int, default=None,
                   help="Override number of seeds (ignores --mode default).")
    p.add_argument("--n-workers",   type=int, default=2,
                   help="Parallel worker processes.")
    p.add_argument("--n-start",     type=int,   default=10)
    p.add_argument("--n-step",      type=int,   default=10)
    p.add_argument("--n-max",       type=int,   default=300)
    p.add_argument("--k",           type=int,   default=3)
    p.add_argument("--aim-step",    type=float, default=2.0)
    p.add_argument("--gp-iter",     type=int,   default=100)
    p.add_argument("--early-stop-N", type=int,  default=None,
                   help="Force early stop at this N (test mode sets 30).")
    p.add_argument("--data-dir",    type=Path,
                   default=Path(__file__).parent.parent / "data")
    p.add_argument("--output-dir",  type=Path, default=Path("outputs"))
    p.add_argument("--log-level",   default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run_local.log"),
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve mode defaults -----------------------------------------
    if args.mode == "test":
        n_seeds = args.seeds if args.seeds is not None else 2
        early_stop = args.early_stop_N if args.early_stop_N is not None else 30
        n_workers = min(args.n_workers, n_seeds)
        logger.info("MODE=test  seeds=%d  early_stop_N=%d  workers=%d",
                    n_seeds, early_stop, n_workers)
    else:
        n_seeds = args.seeds if args.seeds is not None else 100
        early_stop = args.early_stop_N
        n_workers = args.n_workers
        logger.info("MODE=full  seeds=%d  workers=%d", n_seeds, n_workers)

    config = WorkerConfig(
        n_start=args.n_start,
        n_step=args.n_step,
        n_max=args.n_max,
        k=args.k,
        aim_step=args.aim_step,
        gp_training_iter=args.gp_iter,
        early_stop_N=early_stop,
    )

    seeds = list(range(n_seeds))

    logger.info("Launching %d worker(s) for %d seeds ...", n_workers, n_seeds)

    bound_worker = functools.partial(
        _worker,
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    results: list[ConvergenceResult] = []
    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(bound_worker, seeds), start=1):
            results.append(result)
            status = "CONVERGED" if result.convergence_N else ("EARLY" if result.stopped_early else "NOT CONV")
            logger.info(
                "[%d/%d] seed=%d  %s  N=%s  time=%.1fs",
                i, n_seeds, result.seed, status,
                result.convergence_N, result.wall_time_s,
            )

    results.sort(key=lambda r: r.seed)
    _write_summary(results, args.output_dir)
    _print_summary(results)

    # Also dump full JSON
    json_path = args.output_dir / "convergence_summary.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Full JSON → %s", json_path)


if __name__ == "__main__":
    main()
