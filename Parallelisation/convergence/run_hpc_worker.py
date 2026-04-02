"""
run_hpc_worker.py – Single-seed entry point for a Slurm array job.

Each task in the Slurm array calls this script with a unique --seed value
(typically $SLURM_ARRAY_TASK_ID).  The script runs the full convergence
loop for that seed, saves all outputs, and exits.

Example (called by submit_hpc.sh):
    python run_hpc_worker.py \\
        --seed $SLURM_ARRAY_TASK_ID \\
        --data-dir /path/to/repo/Parallelisation/data \\
        --output-dir /path/to/repo/Parallelisation/convergence/outputs \\
        --n-max 300 --k 3 --n-step 10

Example HPC test (called by submit_hpc_test.sh):
    python run_hpc_worker.py \\
        --seed $SLURM_ARRAY_TASK_ID \\
        --data-dir /path/to/repo/Parallelisation/data \\
        --output-dir /path/to/repo/Parallelisation/convergence/outputs_test \\
        --n-max 300 --early-stop-N 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Make sure we can import core/convergence_worker regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))
from convergence_worker import ConvergenceResult, WorkerConfig, run_convergence


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-seed convergence worker for Slurm job array.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed",          type=int, required=True,
                   help="Random seed (use $SLURM_ARRAY_TASK_ID in the submit script).")
    p.add_argument("--data-dir",      type=Path, default=None,
                   help="Directory containing the CSV/GeoJSON data files.")
    p.add_argument("--output-dir",    type=Path, default=Path("outputs"),
                   help="Root directory for outputs (seed sub-dirs + logs).")
    p.add_argument("--n-start",       type=int,   default=10)
    p.add_argument("--n-step",        type=int,   default=10)
    p.add_argument("--n-max",         type=int,   default=300)
    p.add_argument("--k",             type=int,   default=3,
                   help="Consecutive identical snapshots required for convergence.")
    p.add_argument("--aim-step",      type=float, default=2.0)
    p.add_argument("--gp-iter",       type=int,   default=100,
                   help="GPyTorch training iterations for the putting GPR.")
    p.add_argument("--early-stop-N",  type=int,   default=None,
                   help="Stop after this N (HPC test mode).")
    p.add_argument("--log-level",     default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Console + per-seed file logging
    log_path = output_dir / "logs" / f"seed{args.seed:04d}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("HPC worker started.  SLURM_JOB_ID=%s  SLURM_ARRAY_TASK_ID=%s  seed=%d",
                os.environ.get("SLURM_JOB_ID", "N/A"),
                os.environ.get("SLURM_ARRAY_TASK_ID", "N/A"),
                args.seed)

    data_dir = (
        Path(args.data_dir)
        if args.data_dir is not None
        else Path(__file__).parent.parent / "data"
    )

    config = WorkerConfig(
        n_start=args.n_start,
        n_step=args.n_step,
        n_max=args.n_max,
        k=args.k,
        aim_step=args.aim_step,
        gp_training_iter=args.gp_iter,
        early_stop_N=args.early_stop_N,
    )

    result: ConvergenceResult = run_convergence(
        seed=args.seed,
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
    )

    # Print a one-line summary to stdout (captured by Slurm .out file)
    print(json.dumps(asdict(result)))
    logger.info("Worker finished.  convergence_N=%s", result.convergence_N)


if __name__ == "__main__":
    main()
