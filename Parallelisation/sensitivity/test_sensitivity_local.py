"""
test_sensitivity_local.py – Fast local validation of both sensitivity pipelines.

Tests the ESHO and birdie pipelines end-to-end at low N, checking:
  - CSV is long format: one row per (x, y, club, aim_offset) with a `rank` column
  - rank=1 approach rows = one per strategy grid point
  - Both PNGs (scatter + heatmap) exist
  - Metadata JSON has tee-shot fields
  - ESHO / P(birdie) values are plausible

Usage:
    cd Parallelisation/sensitivity
    python test_sensitivity_local.py                        # ESHO pipeline, task 0
    python test_sensitivity_local.py --pipeline birdie      # birdie pipeline, task 0
    python test_sensitivity_local.py --n-shots 10 --task-id 5
    python test_sensitivity_local.py --pipeline both        # run both back-to-back
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
sys.path.insert(0, str(_HERE.parent / "convergence_birdie"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sensitivity_test")

EXPECTED_CSV_COLS_ESHO = {
    "x", "y", "rank", "club", "aim_offset",
    "esho_mean", "esho_var", "n_total",
    "seed", "N", "carry_shift", "variance_scale", "is_tee_shot",
}

EXPECTED_CSV_COLS_BIRDIE = {
    "x", "y", "rank", "club", "aim_offset",
    "birdie_prob_mean", "birdie_prob_var", "n_total",
    "seed", "N", "carry_shift", "variance_scale", "is_tee_shot",
}


def _check(cond: bool, msg_pass: str, msg_fail: str) -> bool:
    if cond:
        logger.info("  PASS  %s", msg_pass)
    else:
        logger.error("  FAIL  %s", msg_fail)
    return cond


def _check_long_format(df, n_grid: int, metric_col: str) -> list[bool]:
    """Shared structural checks for both pipelines' long-format CSVs."""
    results: list[bool] = []

    approach  = df[df["is_tee_shot"] == False]
    rank1_app = approach[approach["rank"] == 1]
    tee_rows  = df[df["is_tee_shot"] == True]

    results.append(_check(len(rank1_app) == n_grid,
                           f"rank=1 approach rows = {n_grid} (one per grid point)",
                           f"Got {len(rank1_app)}, expected {n_grid}"))
    results.append(_check(len(tee_rows) >= 1,
                           f"Tee rows present ({len(tee_rows)} ranked combos)",
                           "No tee rows found"))

    bad = rank1_app[rank1_app[metric_col] <= 0]
    results.append(_check(len(bad) == 0, f"All rank-1 {metric_col} > 0",
                           f"{len(bad)} rank-1 rows with {metric_col} <= 0"))

    n_rank2 = len(approach[approach["rank"] == 2])
    results.append(_check(n_rank2 > 0, f"rank=2 rows present ({n_rank2})",
                           "No rank=2 rows — fewer than 2 (club, aim) combos per point?"))

    max_rank = int(approach["rank"].max())
    results.append(_check(max_rank >= 2, f"Max rank = {max_rank}",
                           "Only rank=1 recorded — check accumulator extraction"))

    return results


# ---------------------------------------------------------------------------
# ESHO pipeline test
# ---------------------------------------------------------------------------

def run_tests_esho(task_id: int, n_shots: int, output_dir: Path, data_dir: Path | None) -> bool:
    import numpy as np
    import pandas as pd

    from config_matrix import build_config_matrix, get_config
    from run_hpc_sensitivity import (
        _extract_top3,
        _top3_tee,
        _build_dataframe,
        _fit_approach_gpr,
        _evaluate_tee_shot,
        _plot_heatmap,
        plot_sensitivity_result,
    )
    from core import build_hole, simulate_approach_shots

    df_cfg         = build_config_matrix()
    cfg            = get_config(task_id, df_cfg)
    carry_shift    = float(cfg["carry_shift"])
    variance_scale = float(cfg["variance_scale"])
    trend          = int(cfg["trend"])

    fname_base  = f"sensitivity_dist{carry_shift:.2f}_disp{variance_scale:.4f}"
    csv_path    = output_dir / f"{fname_base}.csv"
    png_scatter = output_dir / f"{fname_base}.png"
    png_heatmap = output_dir / f"{fname_base}_heatmap.png"
    meta_path   = output_dir / f"{fname_base}_meta.json"

    logger.info("=" * 60)
    logger.info("Sensitivity pipeline test  [ESHO]")
    logger.info("  task_id=%d  carry=%.2f yd  var=%.4f  N=%d", task_id, carry_shift, variance_scale, n_shots)
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    if data_dir is None:
        data_dir = _HERE.parent / "data"

    hole = build_hole(data_dir=data_dir, gp_training_iter=50,
                      carry_shift_yards=carry_shift, variance_scale=variance_scale)

    np.random.seed(task_id)
    optimal_results, accumulator = simulate_approach_shots(
        hole=hole, n_new=n_shots, accumulator=None,
        aim_range=(-20.0, 20.0), aim_step=2.0,
    )

    top3_by_point = _extract_top3(accumulator, hole.strategy_points)

    approach_model, approach_likelihood = _fit_approach_gpr(optimal_results, gp_training_iter=50)
    best_tee, all_tee = _evaluate_tee_shot(
        hole=hole, approach_model=approach_model, approach_likelihood=approach_likelihood,
        aim_range=(-20.0, 20.0), aim_step=2.0, n_samples=20,
    )
    top3_tee = _top3_tee(all_tee)

    df = _build_dataframe(top3_by_point=top3_by_point, top3_tee=top3_tee,
                          hole=hole, seed=task_id, N=n_shots,
                          carry_shift=carry_shift, variance_scale=variance_scale)
    df.to_csv(csv_path, index=False)

    plot_sensitivity_result(optimal_results=optimal_results, best_tee=best_tee, all_tee=all_tee,
                            hole=hole, carry_shift=carry_shift, variance_scale=variance_scale,
                            N=n_shots, output_path=png_scatter)

    _plot_heatmap(top3_by_point=top3_by_point, best_tee=best_tee,
                  carry_shift=carry_shift, variance_scale=variance_scale,
                  N=n_shots, output_path=png_heatmap)

    rank1_app = df[(df["is_tee_shot"] == False) & (df["rank"] == 1)]
    meta = {
        "task_id": task_id, "trend": trend,
        "carry_shift": carry_shift, "variance_scale": variance_scale,
        "N": n_shots, "n_approach_rows": int(len(rank1_app)),
        "mean_esho": float(rank1_app["esho_mean"].mean()),
        "best_tee_club": best_tee["club"], "best_tee_aim": best_tee["aim_offset"],
        "best_tee_strokes": best_tee["mean"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    results: list[bool] = []
    results.append(_check(csv_path.exists(),    f"CSV exists: {csv_path.name}",  "CSV not found"))
    results.append(_check(png_scatter.exists(), "Scatter PNG exists",            "Scatter PNG not found"))
    results.append(_check(png_heatmap.exists(), "Heatmap PNG exists",            "Heatmap PNG not found"))

    missing = EXPECTED_CSV_COLS_ESHO - set(df.columns)
    results.append(_check(not missing, "All expected columns present", f"Missing: {missing}"))

    results.extend(_check_long_format(df, len(hole.strategy_points), "esho_mean"))

    tee_keys = {"best_tee_club", "best_tee_aim", "best_tee_strokes"}
    results.append(_check(tee_keys.issubset(meta), "Metadata has tee fields",
                           f"Missing: {tee_keys - set(meta)}"))

    ts = meta["best_tee_strokes"]
    results.append(_check(3.0 <= ts <= 7.0, f"Tee E[strokes]={ts:.3f} plausible",
                           f"Tee E[strokes]={ts:.3f} outside [3.0, 7.0]"))

    return _summarise(results, label="ESHO",
                      detail=f"Mean ESHO={meta['mean_esho']:.3f}  best tee={meta['best_tee_club']} {meta['best_tee_aim']:+.0f} yd → {meta['best_tee_strokes']:.3f}")


# ---------------------------------------------------------------------------
# Birdie pipeline test
# ---------------------------------------------------------------------------

def run_tests_birdie(task_id: int, n_shots: int, output_dir: Path, data_dir: Path | None) -> bool:
    import numpy as np
    import pandas as pd

    from config_matrix import build_config_matrix, get_config
    from run_hpc_sensitivity_birdie import (
        _extract_top3_birdie,
        _top3_tee_birdie,
        _build_dataframe,
        _fit_birdie_approach_gpr,
        _evaluate_tee_shot_birdie,
        _plot_heatmap_birdie,
        plot_birdie_sensitivity_result,
    )
    from core_birdie import build_hole_birdie, simulate_approach_shots_birdie

    df_cfg         = build_config_matrix()
    cfg            = get_config(task_id, df_cfg)
    carry_shift    = float(cfg["carry_shift"])
    variance_scale = float(cfg["variance_scale"])
    trend          = int(cfg["trend"])

    fname_base  = f"birdie_sensitivity_dist{carry_shift:.2f}_disp{variance_scale:.4f}"
    csv_path    = output_dir / f"{fname_base}.csv"
    png_scatter = output_dir / f"{fname_base}.png"
    png_heatmap = output_dir / f"{fname_base}_heatmap.png"
    meta_path   = output_dir / f"{fname_base}_meta.json"

    logger.info("=" * 60)
    logger.info("Sensitivity pipeline test  [Birdie]")
    logger.info("  task_id=%d  carry=%.2f yd  var=%.4f  N=%d", task_id, carry_shift, variance_scale, n_shots)
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    if data_dir is None:
        data_dir = _HERE.parent / "data"

    hole = build_hole_birdie(data_dir=data_dir, gp_training_iter=50,
                             carry_shift_yards=carry_shift, variance_scale=variance_scale)

    np.random.seed(task_id)
    optimal_results, accumulator = simulate_approach_shots_birdie(
        hole=hole, n_new=n_shots, accumulator=None,
        aim_range=(-20.0, 20.0), aim_step=2.0,
    )

    top3_by_point = _extract_top3_birdie(accumulator, hole.strategy_points)

    approach_model, approach_likelihood = _fit_birdie_approach_gpr(optimal_results, gp_training_iter=50)
    best_tee, all_tee = _evaluate_tee_shot_birdie(
        hole=hole, approach_model=approach_model, approach_likelihood=approach_likelihood,
        aim_range=(-20.0, 20.0), aim_step=2.0, n_samples=20,
    )
    top3_tee = _top3_tee_birdie(all_tee)

    df = _build_dataframe(top3_by_point=top3_by_point, top3_tee=top3_tee,
                          hole=hole, seed=task_id, N=n_shots,
                          carry_shift=carry_shift, variance_scale=variance_scale)
    df.to_csv(csv_path, index=False)

    plot_birdie_sensitivity_result(optimal_results=optimal_results, best_tee=best_tee,
                                   hole=hole, carry_shift=carry_shift, variance_scale=variance_scale,
                                   N=n_shots, output_path=png_scatter)

    _plot_heatmap_birdie(top3_by_point=top3_by_point, best_tee=best_tee,
                         carry_shift=carry_shift, variance_scale=variance_scale,
                         N=n_shots, output_path=png_heatmap)

    rank1_app = df[(df["is_tee_shot"] == False) & (df["rank"] == 1)]
    meta = {
        "task_id": task_id, "trend": trend,
        "carry_shift": carry_shift, "variance_scale": variance_scale,
        "N": n_shots, "n_approach_rows": int(len(rank1_app)),
        "mean_birdie_prob": float(rank1_app["birdie_prob_mean"].mean()),
        "best_tee_club": best_tee["club"], "best_tee_aim": best_tee["aim_offset"],
        "best_tee_birdie_prob": best_tee["mean_birdie_prob"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    results: list[bool] = []
    results.append(_check(csv_path.exists(),    f"CSV exists: {csv_path.name}",  "CSV not found"))
    results.append(_check(png_scatter.exists(), "Scatter PNG exists",            "Scatter PNG not found"))
    results.append(_check(png_heatmap.exists(), "Heatmap PNG exists",            "Heatmap PNG not found"))

    missing = EXPECTED_CSV_COLS_BIRDIE - set(df.columns)
    results.append(_check(not missing, "All expected columns present", f"Missing: {missing}"))

    results.extend(_check_long_format(df, len(hole.strategy_points), "birdie_prob_mean"))

    tee_keys = {"best_tee_club", "best_tee_aim", "best_tee_birdie_prob"}
    results.append(_check(tee_keys.issubset(meta), "Metadata has tee fields",
                           f"Missing: {tee_keys - set(meta)}"))

    tp = meta["best_tee_birdie_prob"]
    results.append(_check(0.0 <= tp <= 1.0, f"Tee P(birdie)={tp:.4f} in [0,1]",
                           f"Tee P(birdie)={tp:.4f} out of range"))

    return _summarise(results, label="Birdie",
                      detail=f"Mean P(birdie)={meta['mean_birdie_prob']:.4f}  best tee={meta['best_tee_club']} {meta['best_tee_aim']:+.0f} yd → {meta['best_tee_birdie_prob']:.4f}")


# ---------------------------------------------------------------------------
# Shared summary helper
# ---------------------------------------------------------------------------

def _summarise(results: list[bool], label: str, detail: str) -> bool:
    passed = sum(results)
    total  = len(results)
    logger.info("")
    logger.info("=" * 60)
    logger.info("[%s]  %d/%d passed", label, passed, total)
    logger.info("%s", detail)
    logger.info("=" * 60)
    if passed < total:
        logger.error("%d test(s) FAILED", total - passed)
        return False
    logger.info("All tests passed. Safe to submit to HPC.")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pipeline",   choices=["esho", "birdie", "both"], default="esho")
    p.add_argument("--task-id",    type=int, default=0)
    p.add_argument("--n-shots",    type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=_HERE / "test_outputs")
    p.add_argument("--data-dir",   type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok_esho   = True
    ok_birdie = True

    if args.pipeline in ("esho", "both"):
        ok_esho = run_tests_esho(args.task_id, args.n_shots, args.output_dir, args.data_dir)

    if args.pipeline in ("birdie", "both"):
        ok_birdie = run_tests_birdie(args.task_id, args.n_shots, args.output_dir, args.data_dir)

    sys.exit(0 if (ok_esho and ok_birdie) else 1)
