"""
test_sensitivity_local.py – Fast local validation of the sensitivity pipeline.

Runs task_id=0 with n_sims=10, then checks every expected output:
  - CSV exists with correct columns
  - Driver never appears in non-tee shots
  - Tee shots have shot_num=1 and is_tee=True
  - Score distribution is plausible (2–15 strokes per hole for a Par-4)
  - Metadata JSON, tee summary JSON, and three PNGs all exist

Usage:
    cd Parallelisation/sensitivity
    python test_sensitivity_local.py
    python test_sensitivity_local.py --n-sims 5 --task-id 3   # spot-check another task
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sensitivity_test")

EXPECTED_CSV_COLUMNS = {
    "sim_id", "shot_num", "x", "y", "club",
    "lie", "is_tee", "aim_offset", "carry_shift", "variance_scale",
}
PLAUSIBLE_STROKES = (2, 15)


def _check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        logger.info("  PASS  %s", msg_pass)
    else:
        logger.error("  FAIL  %s", msg_fail)
    return condition


def run_tests(task_id: int, n_sims: int, output_dir: Path, data_dir: Path | None) -> bool:
    from config_matrix import build_config_matrix, get_config
    from sim_full_hole import generate_sensitivity_plots, get_tee_summary, output_filename, run_sensitivity

    # ── Resolve config ─────────────────────────────────────────────────────
    df_configs = build_config_matrix()
    cfg        = get_config(task_id, df_configs)
    carry_shift    = float(cfg["carry_shift"])
    variance_scale = float(cfg["variance_scale"])
    fname_base     = output_filename(carry_shift, variance_scale)
    csv_path       = output_dir / f"{fname_base}.csv"
    meta_path      = output_dir / f"{fname_base}_meta.json"
    tee_path       = output_dir / f"{fname_base}_tee.json"

    logger.info("=" * 60)
    logger.info("Sensitivity pipeline test")
    logger.info("  task_id       = %d", task_id)
    logger.info("  carry_shift   = %.2f yd", carry_shift)
    logger.info("  variance_scale= %.4f", variance_scale)
    logger.info("  n_sims        = %d", n_sims)
    logger.info("  output_dir    = %s", output_dir)
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run the pipeline ───────────────────────────────────────────────────
    logger.info("Running simulation ...")
    if data_dir is None:
        data_dir = _HERE.parent / "data"

    df, hole = run_sensitivity(
        carry_shift=carry_shift,
        variance_scale=variance_scale,
        n_sims=n_sims,
        seed=task_id,
        data_dir=data_dir,
        gp_training_iter=50,   # fast for testing
    )

    df.to_csv(csv_path, index=False)

    tee_summary = get_tee_summary(df)
    with open(tee_path, "w") as f:
        json.dump(tee_summary, f, indent=2)

    strokes_per_hole = df.groupby("sim_id")["shot_num"].max()
    meta = {
        "task_id":       task_id,
        "carry_shift":   carry_shift,
        "variance_scale": variance_scale,
        "n_sims":        n_sims,
        "n_shots_total": len(df),
        "mean_strokes":  float(strokes_per_hole.mean()),
        "tee_top_club":  tee_summary.get("top_club", "N/A"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    plot_paths = generate_sensitivity_plots(
        df=df, hole=hole,
        carry_shift=carry_shift, variance_scale=variance_scale,
        output_dir=output_dir, fname_base=fname_base,
    )

    # ── Assertions ─────────────────────────────────────────────────────────
    results: list[bool] = []

    # 1. CSV exists
    results.append(_check(csv_path.exists(), f"CSV exists: {csv_path.name}", "CSV not found"))

    # 2. CSV has expected columns
    actual_cols = set(df.columns)
    missing     = EXPECTED_CSV_COLUMNS - actual_cols
    results.append(_check(
        not missing,
        "CSV has all expected columns",
        f"CSV missing columns: {missing}",
    ))

    # 3. Driver never in non-tee shots
    non_tee_driver = df[(df["is_tee"] == False) & (df["club"] == "Driver")]
    results.append(_check(
        len(non_tee_driver) == 0,
        "Driver never used on non-tee shots",
        f"Driver found in {len(non_tee_driver)} non-tee shots — check ClubSelector!",
    ))

    # 4. All shot_num==1 rows are tee shots
    shot1 = df[df["shot_num"] == 1]
    results.append(_check(
        (shot1["is_tee"] == True).all(),
        "All shot_num=1 rows have is_tee=True",
        "Some shot_num=1 rows have is_tee=False",
    ))

    # 5. No tee shots after shot_num==1
    late_tee = df[(df["shot_num"] > 1) & (df["is_tee"] == True)]
    results.append(_check(
        len(late_tee) == 0,
        "No is_tee=True after shot_num=1",
        f"{len(late_tee)} tee-flag rows appear after shot 1",
    ))

    # 6. Plausible score range
    lo, hi = PLAUSIBLE_STROKES
    out_of_range = strokes_per_hole[(strokes_per_hole < lo) | (strokes_per_hole > hi)]
    results.append(_check(
        len(out_of_range) == 0,
        f"All hole scores in [{lo}, {hi}]",
        f"{len(out_of_range)} holes outside [{lo}, {hi}]: {out_of_range.tolist()[:5]}",
    ))

    # 7. Tee summary JSON exists and has expected keys
    tee_keys = {"n_tee_shots", "top_club", "landing_mean_x", "landing_mean_y"}
    results.append(_check(
        tee_path.exists() and tee_keys.issubset(tee_summary),
        "Tee summary JSON exists with required keys",
        f"Tee summary missing keys: {tee_keys - set(tee_summary)}",
    ))

    # 8. All three PNGs exist
    for name, path in plot_paths.items():
        results.append(_check(path.exists(), f"Plot exists: {path.name}", f"Plot missing: {path}"))

    # 9. Row count sanity (at least n_sims rows — each hole has ≥ 1 shot)
    results.append(_check(
        len(df) >= n_sims,
        f"CSV has ≥{n_sims} rows ({len(df)} rows)",
        f"CSV has only {len(df)} rows for {n_sims} sims",
    ))

    # ── Summary ────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Results: %d/%d passed", passed, total)
    logger.info("Mean strokes/hole : %.2f", strokes_per_hole.mean())
    logger.info("Tee top club      : %s", tee_summary.get("top_club", "N/A"))
    logger.info("Tee mean landing  : (%.1f, %.1f)",
                tee_summary.get("landing_mean_x", 0),
                tee_summary.get("landing_mean_y", 0))
    logger.info("=" * 60)

    if passed < total:
        logger.error("%d test(s) FAILED — fix before submitting to HPC", total - passed)
        return False
    logger.info("All tests passed. Safe to submit to HPC.")
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task-id",    type=int, default=0,
                   help="Config row to test (0–23).")
    p.add_argument("--n-sims",     type=int, default=10,
                   help="Number of holes to simulate.")
    p.add_argument("--output-dir", type=Path, default=_HERE / "test_outputs",
                   help="Where to save test outputs.")
    p.add_argument("--data-dir",   type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = run_tests(
        task_id=args.task_id,
        n_sims=args.n_sims,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    sys.exit(0 if ok else 1)
