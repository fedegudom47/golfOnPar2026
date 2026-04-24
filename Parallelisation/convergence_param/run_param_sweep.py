"""
run_param_sweep.py – Parameterised sweep over carry distance and variance.

Runs both the ESHO (approach shot) and Birdie convergence pipelines across
a grid of (carry_shift_yards, variance_scale) combinations.

Each combination × n_seeds workers are run in parallel.  Results land in:
    outputs/esho/cs{shift:+d}_vs{scale}/seed####/
    outputs/birdie/cs{shift:+d}_vs{scale}/seed####/

Quick test (1 combo at baseline, 2 seeds, early stop N=30):
    python run_param_sweep.py --mode test

Full sweep:
    python run_param_sweep.py --mode full --n-workers 8

Custom grid:
    python run_param_sweep.py --carry-shifts -10 -5 0 5 10 \\
                               --variance-scales 0.95 0.99 1.0 1.01 \\
                               --n-seeds 20 --n-workers 8
"""

from __future__ import annotations

import argparse
import csv
import functools
import itertools
import json
import logging
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

# Resolve sibling package paths
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "convergence"))
sys.path.insert(0, str(_HERE.parent / "convergence_birdie"))

from convergence_worker import WorkerConfig, run_convergence, ConvergenceResult
from convergence_worker_birdie import BirdieWorkerConfig, run_convergence_birdie, BirdieConvergenceResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameter grids
# ---------------------------------------------------------------------------

DEFAULT_CARRY_SHIFTS:    list[float] = [-10.0, -5.0, 0.0, 5.0, 10.0]
DEFAULT_VARIANCE_SCALES: list[float] = [0.95, 0.99, 1.0, 1.01]


def _combo_tag(carry_shift: float, variance_scale: float) -> str:
    return f"cs{carry_shift:+.0f}_vs{variance_scale:.2f}"


# ---------------------------------------------------------------------------
# ESHO worker
# ---------------------------------------------------------------------------

def _esho_worker(
    args: tuple[int, WorkerConfig, Path, Path],
) -> ConvergenceResult:
    seed, config, data_dir, output_dir = args
    return run_convergence(seed=seed, config=config, data_dir=data_dir, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Birdie worker
# ---------------------------------------------------------------------------

def _birdie_worker(
    args: tuple[int, BirdieWorkerConfig, Path, Path],
) -> BirdieConvergenceResult:
    seed, config, data_dir, output_dir = args
    return run_convergence_birdie(seed=seed, config=config, data_dir=data_dir, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _write_summary_csv(results: list, output_dir: Path, filename: str) -> None:
    if not results:
        return
    path = output_dir / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    logger.info("Summary → %s", path)


def _print_combo_summary(
    combo_tag: str,
    model: str,
    results: list,
) -> None:
    converged = [r for r in results if r.convergence_N is not None]
    print(f"  [{model.upper()}] {combo_tag}: "
          f"{len(converged)}/{len(results)} converged")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sweep(
    carry_shifts: list[float],
    variance_scales: list[float],
    n_seeds: int,
    n_workers: int,
    early_stop_N: int | None,
    n_start: int,
    n_step: int,
    n_max: int,
    gp_training_iter_esho: int,
    gp_training_iter_birdie: int,
    data_dir: Path,
    output_base: Path,
    run_esho: bool = True,
    run_birdie: bool = True,
) -> None:
    combos = list(itertools.product(carry_shifts, variance_scales))
    seeds  = list(range(n_seeds))

    # Build flat task lists for each model
    esho_tasks:   list[tuple] = []
    birdie_tasks: list[tuple] = []

    for carry_shift, var_scale in combos:
        tag = _combo_tag(carry_shift, var_scale)

        if run_esho:
            esho_out = output_base / "esho" / tag
            esho_out.mkdir(parents=True, exist_ok=True)
            cfg = WorkerConfig(
                n_start=n_start, n_step=n_step, n_max=n_max,
                gp_training_iter=gp_training_iter_esho,
                early_stop_N=early_stop_N,
                carry_shift_yards=carry_shift,
                variance_scale=var_scale,
            )
            for seed in seeds:
                esho_tasks.append((seed, cfg, data_dir, esho_out))

        if run_birdie:
            birdie_out = output_base / "birdie" / tag
            birdie_out.mkdir(parents=True, exist_ok=True)
            bcfg = BirdieWorkerConfig(
                n_start=n_start, n_step=n_step, n_max=n_max,
                gp_training_iter=gp_training_iter_birdie,
                early_stop_N=early_stop_N,
                carry_shift_yards=carry_shift,
                variance_scale=var_scale,
            )
            for seed in seeds:
                birdie_tasks.append((seed, bcfg, data_dir, birdie_out))

    total = len(esho_tasks) + len(birdie_tasks)
    logger.info(
        "Sweep: %d combos × %d seeds = %d ESHO + %d birdie tasks (%d total)",
        len(combos), n_seeds, len(esho_tasks), len(birdie_tasks), total,
    )

    all_esho_results:   list[ConvergenceResult]       = []
    all_birdie_results: list[BirdieConvergenceResult] = []

    # --- Run ESHO tasks ---
    if esho_tasks:
        logger.info("Running %d ESHO tasks ...", len(esho_tasks))
        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_esho_worker, esho_tasks), 1):
                all_esho_results.append(result)
                if i % 10 == 0 or i == len(esho_tasks):
                    logger.info("[ESHO] %d/%d done", i, len(esho_tasks))

        # Group and summarise per combo
        for carry_shift, var_scale in combos:
            tag = _combo_tag(carry_shift, var_scale)
            combo_results = [
                r for r in all_esho_results
                # We can't easily link results back to combo without tagging,
                # so write per-combo summary inline above instead
            ]
        _write_summary_csv(
            all_esho_results,
            output_base / "esho",
            "sweep_esho_summary.csv",
        )

    # --- Run Birdie tasks ---
    if birdie_tasks:
        logger.info("Running %d Birdie tasks ...", len(birdie_tasks))
        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_birdie_worker, birdie_tasks), 1):
                all_birdie_results.append(result)
                if i % 10 == 0 or i == len(birdie_tasks):
                    logger.info("[Birdie] %d/%d done", i, len(birdie_tasks))

        _write_summary_csv(
            all_birdie_results,
            output_base / "birdie",
            "sweep_birdie_summary.csv",
        )

    # Full JSON dump
    summary = {
        "carry_shifts":     carry_shifts,
        "variance_scales":  variance_scales,
        "n_seeds":          n_seeds,
        "esho_results":     [asdict(r) for r in all_esho_results],
        "birdie_results":   [asdict(r) for r in all_birdie_results],
    }
    json_path = output_base / "sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Full sweep JSON → %s", json_path)

    print("\n" + "=" * 60)
    print(f"  Sweep complete — {len(combos)} combos × {n_seeds} seeds")
    print(f"  ESHO total:   {len(all_esho_results)}")
    print(f"  Birdie total: {len(all_birdie_results)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parameterised sweep over carry shift and variance scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["test", "full"], default="test",
                   help="test = baseline combo only, 2 seeds, early stop N=30.")
    p.add_argument("--carry-shifts",    type=float, nargs="+",
                   default=DEFAULT_CARRY_SHIFTS,
                   help="Yards to add to mean carry (can be negative).")
    p.add_argument("--variance-scales", type=float, nargs="+",
                   default=DEFAULT_VARIANCE_SCALES,
                   help="Multiplier on club covariance matrices.")
    p.add_argument("--n-seeds",    type=int, default=20)
    p.add_argument("--n-workers",  type=int, default=4)
    p.add_argument("--n-start",    type=int, default=10)
    p.add_argument("--n-step",     type=int, default=10)
    p.add_argument("--n-max",      type=int, default=300)
    p.add_argument("--early-stop-N", type=int, default=None)
    p.add_argument("--gp-iter-esho",   type=int, default=100)
    p.add_argument("--gp-iter-birdie", type=int, default=200)
    p.add_argument("--no-esho",   action="store_true", help="Skip ESHO model.")
    p.add_argument("--no-birdie", action="store_true", help="Skip birdie model.")
    p.add_argument("--data-dir",  type=Path,
                   default=Path(__file__).parent.parent / "data")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--log-level",  default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run_param_sweep.log"),
        ],
    )

    if args.mode == "test":
        carry_shifts    = [0.0]
        variance_scales = [1.0]
        n_seeds         = 2
        early_stop_N    = args.early_stop_N or 30
        logger.info("MODE=test: baseline combo only, %d seeds, early_stop_N=%d",
                    n_seeds, early_stop_N)
    else:
        carry_shifts    = args.carry_shifts
        variance_scales = args.variance_scales
        n_seeds         = args.n_seeds
        early_stop_N    = args.early_stop_N
        logger.info(
            "MODE=full: %d carry shifts × %d variance scales × %d seeds",
            len(carry_shifts), len(variance_scales), n_seeds,
        )

    run_sweep(
        carry_shifts=carry_shifts,
        variance_scales=variance_scales,
        n_seeds=n_seeds,
        n_workers=args.n_workers,
        early_stop_N=early_stop_N,
        n_start=args.n_start,
        n_step=args.n_step,
        n_max=args.n_max,
        gp_training_iter_esho=args.gp_iter_esho,
        gp_training_iter_birdie=args.gp_iter_birdie,
        data_dir=args.data_dir,
        output_base=args.output_dir,
        run_esho=not args.no_esho,
        run_birdie=not args.no_birdie,
    )


if __name__ == "__main__":
    main()
