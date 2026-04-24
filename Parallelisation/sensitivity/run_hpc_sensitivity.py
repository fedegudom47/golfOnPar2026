"""
run_hpc_sensitivity.py – Single Slurm task for the sensitivity analysis.

Mirrors the convergence pipeline (simulate_approach_shots + plot_optimal_approaches)
but with parameterised club distributions (carry_shift, variance_scale) and a
fixed N=280 instead of an incremental convergence loop.

Adds one extra step: fits an approach-GPR surface and evaluates the tee shot,
then overlays the best tee shot on the same approach strategy image.

Outputs per task (one per carry_shift × variance_scale config):
    outputs/{fname}.csv          approach grid, same columns as convergence +
                                   carry_shift, variance_scale
    outputs/{fname}.png          approach strategy + tee shot overlay (one image)
    outputs/{fname}_meta.json    config metadata

Usage (called by submit_hpc_sensitivity.sh):
    python run_hpc_sensitivity.py \\
        --task-id $SLURM_ARRAY_TASK_ID \\
        --configs-csv param_configs.csv \\
        --n-shots 280 \\
        --gp-iter 100 \\
        --data-dir /path/to/data \\
        --output-dir outputs/
"""

from __future__ import annotations

import sys as _sys
if _sys.version_info < (3, 9):
    _sys.exit(f"ERROR: Python 3.9+ required, got {_sys.version}")

import argparse
import json
import logging
import os
from pathlib import Path

import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

_HERE = Path(__file__).parent
_sys.path.insert(0, str(_HERE))
_sys.path.insert(0, str(_HERE.parent / "convergence"))

from config_matrix import build_config_matrix, get_config

# Reuse convergence building blocks directly
from core import (
    CLUB_STYLES,
    HoleData,
    build_hole,
    plot_hole_layout,
    results_to_dataframe,
    rotation_translator,
    simulate_approach_shots,
)

# Tee-shot evaluation from run_full_hole (reused verbatim)
from run_full_hole import (
    _ApproachGPR,
    evaluate_tee_shot,
    fit_approach_gpr,
)


# ---------------------------------------------------------------------------
# Combined plot: approach strategy grid + tee shot overlay (one image)
# ---------------------------------------------------------------------------

def plot_sensitivity_result(
    optimal_results: list[dict],
    best_tee: dict,
    all_tee: list[dict],
    hole: HoleData,
    carry_shift: float,
    variance_scale: float,
    output_path: Path,
) -> None:
    """Approach strategy grid (convergence style) + tee shot overlay."""
    title = (
        f"Sensitivity: carry +{carry_shift:.1f} yd | var ×{variance_scale:.2f}"
        f"\nBest tee: {best_tee['club']} aim {best_tee['aim_offset']:+.0f} yd"
        f"  →  E[total strokes] = {best_tee['mean']:.3f}"
    )

    fig, ax = plt.subplots(figsize=(14, 16))
    plot_hole_layout(hole, title=title, plot_strategy_points=False, ax=ax)

    # ── Approach strategy grid (identical to plot_optimal_approaches) ────────
    xs    = [r["start"][0] for r in optimal_results]
    ys    = [r["start"][1] for r in optimal_results]
    means = [r["mean"]     for r in optimal_results]

    face_colors = [
        CLUB_STYLES.get(r["club"], {"color": "#999999"})["color"]
        for r in optimal_results
    ]
    norm_approach = mpl.colors.Normalize(vmin=min(means), vmax=max(means))
    edge_colors   = [plt.get_cmap("viridis")(norm_approach(m)) for m in means]

    ax.scatter(xs, ys, c=face_colors, s=25, alpha=0.85, zorder=20,
               edgecolors=edge_colors, linewidths=1.5)

    for r, x, y in zip(optimal_results, xs, ys):
        short = CLUB_STYLES.get(r["club"], {"short": r["club"]})["short"]
        ax.text(x - 2, y + 2.5, f'{short},{int(r["aim_offset"]):+}',
                fontsize=5, color="black", zorder=21)

    sm_approach = mpl.cm.ScalarMappable(cmap="viridis", norm=norm_approach)
    sm_approach.set_array([])
    cbar = fig.colorbar(sm_approach, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("ESHO (Expected Strokes to Hole Out)")

    # ── Tee shot overlay ─────────────────────────────────────────────────────
    tee    = hole.tee_point
    target = hole.hole
    total_dist = float(np.linalg.norm(np.array(target) - np.array(tee)))

    # All tee options — small coloured dots
    tee_means = [r["mean"] for r in all_tee]
    norm_tee  = mpl.colors.Normalize(vmin=min(tee_means), vmax=max(tee_means))
    cmap_tee  = plt.get_cmap("RdYlGn_r")

    for r in all_tee:
        mu        = hole.club_distributions[r["club"]]["mean"]
        angle_deg = (float(np.degrees(np.arctan(r["aim_offset"] / total_dist)))
                     if total_dist > 0 else 0.0)
        lp = rotation_translator(float(mu[0]), float(mu[1]), angle_deg, tee, target)
        ax.scatter(lp[0], lp[1], color=cmap_tee(norm_tee(r["mean"])),
                   s=14, alpha=0.5, zorder=15)

    # Best tee shot — gold star
    mu_best   = hole.club_distributions[best_tee["club"]]["mean"]
    angle_best = (float(np.degrees(np.arctan(best_tee["aim_offset"] / total_dist)))
                  if total_dist > 0 else 0.0)
    lp_best = rotation_translator(
        float(mu_best[0]), float(mu_best[1]), angle_best, tee, target
    )
    ax.scatter(*lp_best, color="gold", s=250, zorder=35,
               edgecolors="black", linewidths=2,
               label=f"Best tee: {best_tee['club']} {best_tee['aim_offset']:+.0f} yd")
    ax.annotate(
        f"{best_tee['club']}\naim {best_tee['aim_offset']:+.0f} yd\n"
        f"E[strokes]={best_tee['mean']:.3f}",
        xy=lp_best, xytext=(lp_best[0] + 12, lp_best[1] + 12),
        fontsize=9, fontweight="bold", color="black", zorder=40,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    # Club legend
    legend_patches = [
        mpatches.Patch(facecolor=v["color"], edgecolor="k", label=v["short"])
        for v in CLUB_STYLES.values()
        if any(r["club"] == club for r in optimal_results
               for club in [list(CLUB_STYLES.keys())[list(CLUB_STYLES.values()).index(v)]])
    ]
    # simpler: show all clubs that appear in this run
    used_clubs = {r["club"] for r in optimal_results}
    legend_patches = [
        mpatches.Patch(facecolor=CLUB_STYLES[c]["color"], edgecolor="k",
                       label=CLUB_STYLES[c]["short"])
        for c in CLUB_STYLES if c in used_clubs
    ]
    ax.legend(handles=legend_patches, title="Club", loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-task sensitivity analysis worker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task-id",     type=int, default=None)
    p.add_argument("--configs-csv", type=Path, default=_HERE / "param_configs.csv")
    p.add_argument("--n-shots",     type=int, default=280,
                   help="Approach shots per (grid-point, club, aim). "
                        "Default 280 matches the convergence study.")
    p.add_argument("--gp-iter",     type=int, default=100,
                   help="GPR training iterations (putting model + approach GPR).")
    p.add_argument("--aim-range",   type=float, nargs=2, default=[-20.0, 20.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--aim-step",    type=float, default=2.0)
    p.add_argument("--tee-samples", type=int, default=50,
                   help="Samples per (club, aim) for tee shot evaluation.")
    p.add_argument("--data-dir",    type=Path, default=None)
    p.add_argument("--output-dir",  type=Path, default=Path("outputs"))
    p.add_argument("--log-level",   default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Resolve task ID ────────────────────────────────────────────────────
    task_id = args.task_id
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            _sys.exit("ERROR: --task-id not set and $SLURM_ARRAY_TASK_ID not found.")
        task_id = int(raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(_sys.stdout),
            logging.FileHandler(log_dir / f"task{task_id:04d}.log"),
        ],
    )

    logger.info("Sensitivity worker | SLURM_JOB_ID=%s  task_id=%d",
                os.environ.get("SLURM_JOB_ID", "N/A"), task_id)

    # ── Config lookup ──────────────────────────────────────────────────────
    if args.configs_csv.exists():
        import pandas as pd
        df_cfg = pd.read_csv(args.configs_csv)
    else:
        logger.warning("configs-csv not found; regenerating.")
        df_cfg = build_config_matrix()

    cfg            = get_config(task_id, df_cfg)
    carry_shift    = float(cfg["carry_shift"])
    variance_scale = float(cfg["variance_scale"])
    trend          = int(cfg["trend"])
    N              = args.n_shots

    logger.info("Config: task_id=%d  trend=%d  carry_shift=%.2f  variance_scale=%.4f  N=%d",
                task_id, trend, carry_shift, variance_scale, N)

    data_dir = Path(args.data_dir) if args.data_dir else _HERE.parent / "data"

    # ── 1. Build hole ──────────────────────────────────────────────────────
    logger.info("Building hole ...")
    hole = build_hole(
        data_dir=data_dir,
        gp_training_iter=args.gp_iter,
        carry_shift_yards=carry_shift,
        variance_scale=variance_scale,
    )
    logger.info("Hole ready. %d strategy points.", len(hole.strategy_points))

    # ── 2. Approach shot simulation (single batch, N shots per combo) ──────
    logger.info("Simulating approach shots (N=%d) ...", N)
    np.random.seed(task_id)
    optimal_results, _ = simulate_approach_shots(
        hole=hole,
        n_new=N,
        accumulator=None,
        aim_range=tuple(args.aim_range),
        aim_step=args.aim_step,
    )
    logger.info("Approach simulation done. %d grid points.", len(optimal_results))

    # ── 3. Save CSV (same format as convergence) ───────────────────────────
    fname_base = f"sensitivity_dist{carry_shift:.2f}_disp{variance_scale:.4f}"
    csv_path   = args.output_dir / f"{fname_base}.csv"

    import pandas as pd
    df = results_to_dataframe(optimal_results, seed=task_id, N=N)
    df["carry_shift"]    = carry_shift
    df["variance_scale"] = variance_scale
    df.to_csv(csv_path, index=False)
    logger.info("CSV saved → %s  (%d rows)", csv_path, len(df))

    # ── 4. Fit approach GPR for tee shot ───────────────────────────────────
    logger.info("Fitting approach GPR ...")
    approach_model, approach_likelihood = fit_approach_gpr(
        optimal_results, gp_training_iter=args.gp_iter
    )

    # ── 5. Evaluate tee shot ───────────────────────────────────────────────
    logger.info("Evaluating tee shot (%d samples per combo) ...", args.tee_samples)
    best_tee, all_tee = evaluate_tee_shot(
        hole=hole,
        approach_model=approach_model,
        approach_likelihood=approach_likelihood,
        aim_range=tuple(args.aim_range),
        aim_step=args.aim_step,
        n_samples=args.tee_samples,
    )
    logger.info("Best tee: %s  aim=%+.0f yd  E[strokes]=%.4f",
                best_tee["club"], best_tee["aim_offset"], best_tee["mean"])

    # ── 6. Combined plot ───────────────────────────────────────────────────
    png_path = args.output_dir / f"{fname_base}.png"
    logger.info("Generating plot ...")
    plot_sensitivity_result(
        optimal_results=optimal_results,
        best_tee=best_tee,
        all_tee=all_tee,
        hole=hole,
        carry_shift=carry_shift,
        variance_scale=variance_scale,
        output_path=png_path,
    )

    # ── 7. Metadata ────────────────────────────────────────────────────────
    meta = {
        "task_id":         task_id,
        "trend":           trend,
        "carry_shift":     carry_shift,
        "variance_scale":  variance_scale,
        "N":               N,
        "n_grid_points":   len(optimal_results),
        "mean_esho":       float(df["esho_mean"].mean()),
        "best_tee_club":   best_tee["club"],
        "best_tee_aim":    best_tee["aim_offset"],
        "best_tee_strokes": best_tee["mean"],
        "output_csv":      f"{fname_base}.csv",
        "output_png":      f"{fname_base}.png",
    }
    meta_path = args.output_dir / f"{fname_base}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Task %d done | mean ESHO=%.3f | best tee: %s %+.0f yd → %.3f strokes",
        task_id, meta["mean_esho"], best_tee["club"],
        best_tee["aim_offset"], best_tee["mean"],
    )
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
