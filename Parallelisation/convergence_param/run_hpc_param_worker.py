"""
run_hpc_param_worker.py – Single-task entry point for the parameterised sweep.

Called by submit_hpc_param.sh for each Slurm array task.  The task ID is
decoded arithmetically into (model, carry_shift, variance_scale, seed):

    task_id = combo_idx * N_SEEDS + seed
    combo   = COMBOS[combo_idx]   # (model, carry_shift, variance_scale)

The COMBOS list and N_SEEDS must match whatever the submit script used — they
are passed as CLI args so the submit script is the single source of truth.

Example call from Slurm:
    python run_hpc_param_worker.py \\
        --task-id $SLURM_ARRAY_TASK_ID \\
        --n-seeds 20 \\
        --carry-shifts -10 -5 0 5 10 \\
        --variance-scales 0.95 0.99 1.0 1.01 \\
        --models esho birdie \\
        --output-dir /path/to/outputs_param \\
        --data-dir  /path/to/data
"""

from __future__ import annotations

import sys as _sys
if _sys.version_info < (3, 9):
    _sys.exit(f"ERROR: Python 3.9+ required, got {_sys.version}")

import argparse
import itertools
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).parent
_sys.path.insert(0, str(_HERE.parent / "convergence"))
_sys.path.insert(0, str(_HERE.parent / "convergence_birdie"))

from convergence_worker import WorkerConfig, run_convergence, ConvergenceResult
from convergence_worker_birdie import BirdieWorkerConfig, run_convergence_birdie, BirdieConvergenceResult


def _decode_task(
    task_id: int,
    n_seeds: int,
    models: list[str],
    carry_shifts: list[float],
    variance_scales: list[float],
) -> tuple[str, float, float, int]:
    """Return (model, carry_shift, variance_scale, seed) for this task_id."""
    combos = list(itertools.product(models, carry_shifts, variance_scales))
    combo_idx = task_id // n_seeds
    seed      = task_id  % n_seeds

    if combo_idx >= len(combos):
        raise ValueError(
            f"task_id={task_id} → combo_idx={combo_idx} exceeds "
            f"len(combos)={len(combos)}.  "
            f"Check --n-seeds / array size in the submit script."
        )

    model, carry_shift, variance_scale = combos[combo_idx]
    return model, carry_shift, variance_scale, seed


def _combo_tag(carry_shift: float, variance_scale: float) -> str:
    return f"cs{carry_shift:+.0f}_vs{variance_scale:.2f}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single Slurm task for the parameterised sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task-id",        type=int,   default=None,
                   help="Slurm array task ID (defaults to $SLURM_ARRAY_TASK_ID).")
    p.add_argument("--n-seeds",        type=int,   required=True,
                   help="Number of seeds per combo (must match submit script).")
    p.add_argument("--models",         nargs="+",  default=["esho", "birdie"],
                   choices=["esho", "birdie"],
                   help="Models to include in the combo list.")
    p.add_argument("--carry-shifts",   type=float, nargs="+",
                   default=[-10.0, -5.0, 0.0, 5.0, 10.0])
    p.add_argument("--variance-scales", type=float, nargs="+",
                   default=[0.95, 0.99, 1.0, 1.01])
    p.add_argument("--n-start",        type=int,   default=10)
    p.add_argument("--n-step",         type=int,   default=10)
    p.add_argument("--n-max",          type=int,   default=300)
    p.add_argument("--k",              type=int,   default=3)
    p.add_argument("--aim-step",       type=float, default=2.0)
    p.add_argument("--gp-iter-esho",   type=int,   default=100)
    p.add_argument("--gp-iter-birdie", type=int,   default=200)
    p.add_argument("--early-stop-N",   type=int,   default=None)
    p.add_argument("--data-dir",       type=Path,  default=None)
    p.add_argument("--output-dir",     type=Path,  default=Path("outputs_param"))
    p.add_argument("--log-level",      default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    task_id = args.task_id
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            _sys.exit("ERROR: --task-id not set and $SLURM_ARRAY_TASK_ID not found.")
        task_id = int(raw)

    model, carry_shift, variance_scale, seed = _decode_task(
        task_id=task_id,
        n_seeds=args.n_seeds,
        models=args.models,
        carry_shifts=args.carry_shifts,
        variance_scales=args.variance_scales,
    )

    tag        = _combo_tag(carry_shift, variance_scale)
    output_dir = args.output_dir / model / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    log_path = output_dir / "logs" / f"seed{seed:04d}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(_sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "HPC param worker | SLURM_JOB_ID=%s  task_id=%d  "
        "model=%s  carry_shift=%.1f  variance_scale=%.2f  seed=%d",
        os.environ.get("SLURM_JOB_ID", "N/A"),
        task_id, model, carry_shift, variance_scale, seed,
    )

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None
        else Path(__file__).parent.parent / "data"
    )

    if model == "esho":
        config = WorkerConfig(
            n_start=args.n_start,
            n_step=args.n_step,
            n_max=args.n_max,
            k=args.k,
            aim_step=args.aim_step,
            gp_training_iter=args.gp_iter_esho,
            early_stop_N=args.early_stop_N,
            carry_shift_yards=carry_shift,
            variance_scale=variance_scale,
        )
        result = run_convergence(
            seed=seed, config=config, data_dir=data_dir, output_dir=output_dir,
        )
    else:
        config = BirdieWorkerConfig(
            n_start=args.n_start,
            n_step=args.n_step,
            n_max=args.n_max,
            k=args.k,
            aim_step=args.aim_step,
            gp_training_iter=args.gp_iter_birdie,
            early_stop_N=args.early_stop_N,
            carry_shift_yards=carry_shift,
            variance_scale=variance_scale,
        )
        result = run_convergence_birdie(
            seed=seed, config=config, data_dir=data_dir, output_dir=output_dir,
        )

    print(json.dumps(asdict(result)))
    logger.info("Worker finished.  convergence_N=%s", result.convergence_N)


if __name__ == "__main__":
    main()
