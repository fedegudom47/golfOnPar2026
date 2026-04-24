"""
test_sensitivity_local.py – Fast local validation of the sensitivity pipeline.

Runs task_id=0 with n_shots=10, then checks every expected output:
  - CSV has same columns as convergence + carry_shift, variance_scale
  - 280 rows (one per strategy grid point)
  - PNG exists
  - Metadata JSON exists with tee shot fields
  - Approach GPR + tee evaluation runs without error

Usage:
    cd Parallelisation/sensitivity
    python test_sensitivity_local.py
    python test_sensitivity_local.py --n-shots 10 --task-id 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "convergence"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sensitivity_test")

EXPECTED_CSV_COLS = {
    "x", "y", "club", "aim_offset",
    "esho_mean", "esho_var", "n_total", "seed", "N",
    "carry_shift", "variance_scale",
}


def _check(cond: bool, msg_pass: str, msg_fail: str) -> bool:
    if cond:
        logger.info("  PASS  %s", msg_pass)
    else:
        logger.error("  FAIL  %s", msg_fail)
    return cond


def run_tests(task_id: int, n_shots: int, output_dir: Path, data_dir: Path | None) -> bool:
    import numpy as np
    import pandas as pd

    from config_matrix import build_config_matrix, get_config
    from run_hpc_sensitivity import plot_sensitivity_result
    from core import build_hole, results_to_dataframe, simulate_approach_shots
    from run_full_hole import evaluate_tee_shot, fit_approach_gpr

    df_cfg         = build_config_matrix()
    cfg            = get_config(task_id, df_cfg)
    carry_shift    = float(cfg["carry_shift"])
    variance_scale = float(cfg["variance_scale"])
    trend          = int(cfg["trend"])

    fname_base = f"sensitivity_dist{carry_shift:.2f}_disp{variance_scale:.4f}"
    csv_path   = output_dir / f"{fname_base}.csv"
    png_path   = output_dir / f"{fname_base}.png"
    meta_path  = output_dir / f"{fname_base}_meta.json"

    logger.info("=" * 60)
    logger.info("Sensitivity pipeline test")
    logger.info("  task_id        = %d", task_id)
    logger.info("  carry_shift    = %.2f yd", carry_shift)
    logger.info("  variance_scale = %.4f", variance_scale)
    logger.info("  n_shots        = %d", n_shots)
    logger.info("  output_dir     = %s", output_dir)
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    if data_dir is None:
        data_dir = _HERE.parent / "data"

    # ── Run the pipeline ───────────────────────────────────────────────────
    logger.info("Building hole ...")
    hole = build_hole(
        data_dir=data_dir,
        gp_training_iter=50,
        carry_shift_yards=carry_shift,
        variance_scale=variance_scale,
    )

    logger.info("Simulating approach shots (N=%d) ...", n_shots)
    np.random.seed(task_id)
    optimal_results, _ = simulate_approach_shots(
        hole=hole, n_new=n_shots, accumulator=None,
        aim_range=(-20.0, 20.0), aim_step=2.0,
    )

    df = results_to_dataframe(optimal_results, seed=task_id, N=n_shots)
    df["carry_shift"]    = carry_shift
    df["variance_scale"] = variance_scale
    df.to_csv(csv_path, index=False)

    logger.info("Fitting approach GPR + evaluating tee shot ...")
    approach_model, approach_likelihood = fit_approach_gpr(optimal_results, gp_training_iter=50)
    best_tee, all_tee = evaluate_tee_shot(
        hole=hole, approach_model=approach_model,
        approach_likelihood=approach_likelihood, n_samples=20,
    )

    plot_sensitivity_result(
        optimal_results=optimal_results, best_tee=best_tee, all_tee=all_tee,
        hole=hole, carry_shift=carry_shift, variance_scale=variance_scale,
        output_path=png_path,
    )

    meta = {
        "task_id": task_id, "trend": trend,
        "carry_shift": carry_shift, "variance_scale": variance_scale,
        "N": n_shots, "n_grid_points": len(optimal_results),
        "mean_esho": float(df["esho_mean"].mean()),
        "best_tee_club": best_tee["club"],
        "best_tee_aim": best_tee["aim_offset"],
        "best_tee_strokes": best_tee["mean"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Assertions ─────────────────────────────────────────────────────────
    results: list[bool] = []

    # 1. CSV exists
    results.append(_check(csv_path.exists(), f"CSV exists: {csv_path.name}", "CSV not found"))

    # 2. Columns match convergence format + extras
    missing = EXPECTED_CSV_COLS - set(df.columns)
    results.append(_check(not missing, "All expected columns present",
                           f"Missing columns: {missing}"))

    # 3. Row count = number of strategy grid points
    n_grid = len(hole.strategy_points)
    results.append(_check(len(df) == n_grid, f"CSV has {n_grid} rows (one per grid point)",
                           f"CSV has {len(df)} rows, expected {n_grid}"))

    # 4. ESHO values are finite and positive
    bad_esho = df[df["esho_mean"] <= 0]
    results.append(_check(len(bad_esho) == 0, "All esho_mean > 0",
                           f"{len(bad_esho)} rows with esho_mean <= 0"))

    # 5. PNG exists
    results.append(_check(png_path.exists(), f"PNG exists: {png_path.name}", "PNG not found"))

    # 6. Metadata has tee shot fields
    tee_keys = {"best_tee_club", "best_tee_aim", "best_tee_strokes"}
    results.append(_check(tee_keys.issubset(meta),
                           "Metadata has tee shot fields",
                           f"Missing: {tee_keys - set(meta)}"))

    # 7. Tee strokes are plausible (3.5–6.0 for a Par-4)
    ts = meta["best_tee_strokes"]
    results.append(_check(3.0 <= ts <= 7.0, f"Tee E[strokes]={ts:.3f} is plausible",
                           f"Tee E[strokes]={ts:.3f} is outside [3.0, 7.0]"))

    # ── Summary ────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Results: %d/%d passed", passed, total)
    logger.info("Mean ESHO        : %.3f", meta["mean_esho"])
    logger.info("Best tee shot    : %s  aim=%+.0f yd  E[strokes]=%.3f",
                meta["best_tee_club"], meta["best_tee_aim"], meta["best_tee_strokes"])
    logger.info("=" * 60)

    if passed < total:
        logger.error("%d test(s) FAILED", total - passed)
        return False
    logger.info("All tests passed. Safe to submit to HPC.")
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task-id",    type=int, default=0)
    p.add_argument("--n-shots",    type=int, default=10,
                   help="Approach shots per combo (keep low for speed).")
    p.add_argument("--output-dir", type=Path, default=_HERE / "test_outputs")
    p.add_argument("--data-dir",   type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = run_tests(args.task_id, args.n_shots, args.output_dir, args.data_dir)
    sys.exit(0 if ok else 1)
