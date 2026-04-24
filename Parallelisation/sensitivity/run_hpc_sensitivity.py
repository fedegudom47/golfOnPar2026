"""
run_hpc_sensitivity.py – Single Slurm task for the sensitivity analysis.

Produces outputs that match the convergence pipeline format exactly:
  - CSV: same columns as seed####_N####.csv (x, y, club, aim_offset,
         esho_mean, esho_var, n_total, seed, N) plus carry_shift,
         variance_scale, is_tee_shot.
         280 approach rows (is_tee_shot=False) + one row per evaluated
         tee-shot option (is_tee_shot=True, x/y = tee box position).
  - PNG: identical style to plot_optimal_approaches() in core.py,
         with the best tee shot overlaid as a gold star on the same image.
  - JSON: metadata sidecar.

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

_HERE = Path(__file__).parent
_sys.path.insert(0, str(_HERE))
_sys.path.insert(0, str(_HERE.parent / "convergence"))

# ── heavy imports after path setup ────────────────────────────────────────────
try:
    import gpytorch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import torch
    import pandas as pd
except ImportError as e:
    _sys.exit(f"ERROR: missing package — {e}\n"
              f"Run:  pip install gpytorch torch pandas numpy matplotlib")

try:
    from config_matrix import build_config_matrix, get_config
    from core import (
        CLUB_STYLES,
        HoleData,
        build_hole,
        plot_hole_layout,
        rotation_translator,
        simulate_approach_shots,
    )
except ImportError as e:
    _sys.exit(f"ERROR: failed to import local modules — {e}")


# ---------------------------------------------------------------------------
# Second-stage GPR: (x, y) → optimal ESHO  (inlined from run_full_hole.py)
# ---------------------------------------------------------------------------

class _ApproachGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = 15.0

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _fit_approach_gpr(optimal_results, gp_training_iter=100):
    X = torch.tensor([[r["start"][0], r["start"][1]] for r in optimal_results],
                     dtype=torch.float32)
    y = torch.tensor([r["mean"] for r in optimal_results], dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = _ApproachGPR(X, y, likelihood)
    model.train(); likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(gp_training_iter):
        optimizer.zero_grad()
        (-mll(model(X), y)).backward()
        optimizer.step()

    model.eval(); likelihood.eval()
    return model, likelihood


def _evaluate_tee_shot(hole, approach_model, approach_likelihood,
                       aim_range, aim_step, n_samples):
    """Return (best_result, all_results) for all (club, aim) from tee."""
    tee    = hole.tee_point
    target = hole.hole
    total_dist = float(np.linalg.norm(np.array(target) - np.array(tee)))
    aim_points = list(np.arange(aim_range[0], aim_range[1] + aim_step, aim_step))

    best = None
    all_results = []

    for club, stats in hole.club_distributions.items():
        mu, cov = stats["mean"], stats["cov"]
        for aim in aim_points:
            samples = np.random.multivariate_normal(mu, cov, size=n_samples)
            angle_deg = (float(np.degrees(np.arctan(aim / total_dist)))
                         if total_dist > 0 else 0.0)
            esho_vals = []
            for shot in samples:
                lp = rotation_translator(float(shot[0]), float(shot[1]),
                                         angle_deg, tee, target)
                inp = torch.tensor([[lp[0], lp[1]]], dtype=torch.float32)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred = approach_likelihood(approach_model(inp))
                    esho = float(pred.mean.item())
                if not np.isnan(esho):
                    esho_vals.append(esho)

            if not esho_vals:
                continue

            total_mean = 1.0 + float(np.mean(esho_vals))
            total_var  = float(np.var(esho_vals))
            result = {"club": club, "aim_offset": float(aim),
                      "mean": total_mean, "var": total_var,
                      "n_samples": len(esho_vals)}
            all_results.append(result)
            if best is None or total_mean < best["mean"]:
                best = result

    return best, all_results


# ---------------------------------------------------------------------------
# CSV builder
# ---------------------------------------------------------------------------

def _build_dataframe(
    optimal_results: list[dict],
    all_tee: list[dict],
    hole: HoleData,
    seed: int,
    N: int,
    carry_shift: float,
    variance_scale: float,
) -> pd.DataFrame:
    """Convergence-format CSV with tee shot rows appended.

    Columns identical to seed####_N####.csv:
        x, y, club, aim_offset, esho_mean, esho_var, n_total, seed, N
    Plus sensitivity extras:
        carry_shift, variance_scale, is_tee_shot
    """
    rows = []

    # ── Approach grid rows (280, one per strategy point) ──────────────────
    for r in optimal_results:
        rows.append({
            "x":             float(r["start"][0]),
            "y":             float(r["start"][1]),
            "club":          r["club"],
            "aim_offset":    float(r["aim_offset"]),
            "esho_mean":     float(r["mean"]),
            "esho_var":      float(r["var"]),
            "n_total":       int(r.get("n_total", N)),
            "seed":          seed,
            "N":             N,
            "carry_shift":   carry_shift,
            "variance_scale": variance_scale,
            "is_tee_shot":   False,
        })

    # ── Tee shot rows (one per evaluated club × aim combination) ──────────
    tx, ty = float(hole.tee_point[0]), float(hole.tee_point[1])
    for r in all_tee:
        rows.append({
            "x":             tx,
            "y":             ty,
            "club":          r["club"],
            "aim_offset":    float(r["aim_offset"]),
            "esho_mean":     float(r["mean"]),   # includes the +1 tee stroke
            "esho_var":      float(r["var"]),
            "n_total":       int(r["n_samples"]),
            "seed":          seed,
            "N":             N,
            "carry_shift":   carry_shift,
            "variance_scale": variance_scale,
            "is_tee_shot":   True,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot: identical to plot_optimal_approaches() + tee shot overlay
# ---------------------------------------------------------------------------

def _plot_result(
    optimal_results: list[dict],
    best_tee: dict,
    all_tee: list[dict],
    hole: HoleData,
    carry_shift: float,
    variance_scale: float,
    N: int,
    output_path: Path,
) -> None:
    title = (
        f"Sensitivity  |  carry +{carry_shift:.1f} yd  |  var ×{variance_scale:.2f}"
        f"  |  N={N}"
    )

    fig, ax = plt.subplots(figsize=(14, 16))
    plot_hole_layout(hole, title=title, plot_strategy_points=False, ax=ax)

    # ── Approach grid — IDENTICAL to plot_optimal_approaches() ──────────
    xs    = [r["start"][0] for r in optimal_results]
    ys    = [r["start"][1] for r in optimal_results]
    means = [r["mean"]     for r in optimal_results]

    face_colors = [
        CLUB_STYLES.get(r["club"], {"color": "#999999"})["color"]
        for r in optimal_results
    ]
    norm_a      = mpl.colors.Normalize(vmin=min(means), vmax=max(means))
    edge_colors = [plt.get_cmap("viridis")(norm_a(m)) for m in means]

    ax.scatter(xs, ys, c=face_colors, s=25, alpha=0.85, zorder=20,
               edgecolors=edge_colors, linewidths=1.5)

    for r, x, y in zip(optimal_results, xs, ys):
        short = CLUB_STYLES.get(r["club"], {"short": r["club"]})["short"]
        ax.text(x - 2, y + 2.5, f'{short},{int(r["aim_offset"]):+}',
                fontsize=5, color="black", zorder=21)

    sm_a = mpl.cm.ScalarMappable(cmap="viridis", norm=norm_a)
    sm_a.set_array([])
    cbar = fig.colorbar(sm_a, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("ESHO (Expected Strokes to Hole Out)")

    legend_patches = [
        mpatches.Patch(facecolor=CLUB_STYLES[c]["color"], edgecolor="k",
                       label=CLUB_STYLES[c]["short"])
        for c in CLUB_STYLES
        if any(r["club"] == c for r in optimal_results)
    ]
    ax.legend(handles=legend_patches, title="Club", loc="upper left", fontsize=7)

    # ── Tee shot — same style as approach grid points ────────────────────
    tx, ty      = float(hole.tee_point[0]), float(hole.tee_point[1])
    tee_face    = CLUB_STYLES.get(best_tee["club"], {"color": "#999999"})["color"]
    tee_edge    = plt.get_cmap("viridis")(norm_a(best_tee["mean"]))
    tee_short   = CLUB_STYLES.get(best_tee["club"], {"short": best_tee["club"]})["short"]

    ax.scatter(tx, ty, c=[tee_face], s=25, alpha=0.85, zorder=20,
               edgecolors=[tee_edge], linewidths=1.5)
    ax.text(tx - 2, ty + 2.5,
            f'{tee_short},{int(best_tee["aim_offset"]):+}',
            fontsize=5, color="black", zorder=21)

    # ESHO summary for the tee shot in the top-right corner
    std = float(np.sqrt(best_tee["var"]))
    ax.text(0.98, 0.98,
            f'Tee shot\n'
            f'{tee_short}  aim {best_tee["aim_offset"]:+.0f} yd\n'
            f'ESHO = {best_tee["mean"]:.3f} ± {std:.3f}',
            transform=ax.transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9),
            zorder=40)

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
                   help="Approach shots per (grid-point, club, aim).")
    p.add_argument("--gp-iter",     type=int, default=100,
                   help="GPR training iterations (putting + approach GPR).")
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

    # ── Config ────────────────────────────────────────────────────────────
    if args.configs_csv.exists():
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
    logger.info("Data dir: %s", data_dir)
    if not data_dir.exists():
        _sys.exit(f"ERROR: data-dir does not exist: {data_dir}")

    # ── 1. Build hole ──────────────────────────────────────────────────────
    logger.info("Building hole ...")
    hole = build_hole(
        data_dir=data_dir,
        gp_training_iter=args.gp_iter,
        carry_shift_yards=carry_shift,
        variance_scale=variance_scale,
    )
    logger.info("Hole ready. %d strategy points.", len(hole.strategy_points))

    # ── 2. Approach simulation ─────────────────────────────────────────────
    logger.info("Simulating approach shots (N=%d per combo) ...", N)
    np.random.seed(task_id)
    optimal_results, _ = simulate_approach_shots(
        hole=hole,
        n_new=N,
        accumulator=None,
        aim_range=tuple(args.aim_range),
        aim_step=args.aim_step,
    )
    logger.info("Approach done. %d grid points.", len(optimal_results))

    # ── 3. Approach GPR + tee shot evaluation ─────────────────────────────
    logger.info("Fitting approach GPR (%d iter) ...", args.gp_iter)
    approach_model, approach_likelihood = _fit_approach_gpr(
        optimal_results, gp_training_iter=args.gp_iter
    )

    logger.info("Evaluating tee shot (%d samples per combo) ...", args.tee_samples)
    best_tee, all_tee = _evaluate_tee_shot(
        hole=hole,
        approach_model=approach_model,
        approach_likelihood=approach_likelihood,
        aim_range=tuple(args.aim_range),
        aim_step=args.aim_step,
        n_samples=args.tee_samples,
    )
    logger.info("Best tee: %s  aim=%+.0f yd  E[strokes]=%.4f",
                best_tee["club"], best_tee["aim_offset"], best_tee["mean"])

    # ── 4. Save CSV (approach rows + tee rows) ─────────────────────────────
    fname_base = f"sensitivity_dist{carry_shift:.2f}_disp{variance_scale:.4f}"
    csv_path   = args.output_dir / f"{fname_base}.csv"

    df = _build_dataframe(optimal_results, all_tee, hole,
                          seed=task_id, N=N,
                          carry_shift=carry_shift, variance_scale=variance_scale)
    df.to_csv(csv_path, index=False)

    n_approach = (df["is_tee_shot"] == False).sum()
    n_tee      = (df["is_tee_shot"] == True).sum()
    logger.info("CSV saved → %s  (%d approach + %d tee rows)", csv_path, n_approach, n_tee)

    # ── 5. Plot ────────────────────────────────────────────────────────────
    png_path = args.output_dir / f"{fname_base}.png"
    logger.info("Generating plot ...")
    _plot_result(
        optimal_results=optimal_results,
        best_tee=best_tee,
        all_tee=all_tee,
        hole=hole,
        carry_shift=carry_shift,
        variance_scale=variance_scale,
        N=N,
        output_path=png_path,
    )

    # ── 6. Metadata ────────────────────────────────────────────────────────
    approach_df = df[df["is_tee_shot"] == False]
    meta = {
        "task_id":          task_id,
        "trend":            trend,
        "carry_shift":      carry_shift,
        "variance_scale":   variance_scale,
        "N":                N,
        "n_approach_rows":  int(n_approach),
        "n_tee_rows":       int(n_tee),
        "mean_esho":        float(approach_df["esho_mean"].mean()),
        "best_tee_club":    best_tee["club"],
        "best_tee_aim":     best_tee["aim_offset"],
        "best_tee_strokes": best_tee["mean"],
        "output_csv":       f"{fname_base}.csv",
        "output_png":       f"{fname_base}.png",
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
