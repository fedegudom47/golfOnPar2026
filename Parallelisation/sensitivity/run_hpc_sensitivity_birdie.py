"""
run_hpc_sensitivity_birdie.py – Single Slurm task for birdie-mechanism sensitivity.

Mirrors run_hpc_sensitivity.py but uses BirdieMechanism (VariationalGP +
BernoulliLikelihood) instead of ESHO optimisation.

Produces per-task outputs:
  CSV  – wide top-3 format per grid point:
           x, y, top_1_club, top_1_aim_offset, top_1_birdie_prob_mean, top_1_birdie_prob_var, top_1_n_total,
                 top_2_*, top_3_*, seed, N, carry_shift, variance_scale, is_tee_shot
  PNG1 – hole layout scatter coloured by P(birdie).
  PNG2 – heatmap: one cell per grid point, colour = top-1 P(birdie), text = club+aim.
  JSON – metadata sidecar.

Usage (called by submit_hpc_sensitivity_birdie.sh):
    python run_hpc_sensitivity_birdie.py \\
        --task-id $SLURM_ARRAY_TASK_ID \\
        --configs-csv param_configs.csv \\
        --n-shots 280 \\
        --gp-iter 100 \\
        --data-dir /path/to/data \\
        --output-dir outputs_birdie/
"""

from __future__ import annotations

import sys as _sys
if _sys.version_info < (3, 9):
    _sys.exit(f"ERROR: Python 3.9+ required, got {_sys.version}")

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

_HERE   = Path(__file__).parent
_BIRDIE = _HERE.parent / "convergence_birdie"
_sys.path.insert(0, str(_HERE))
_sys.path.insert(0, str(_BIRDIE))

try:
    import gpytorch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import torch
    import pandas as pd
except ImportError as e:
    _sys.exit(f"ERROR: missing package — {e}")

try:
    from config_matrix import build_config_matrix, get_config
    from core_birdie import (
        BirdieHoleData,
        CLUB_STYLES,
        build_hole_birdie,
        evaluate_birdie_prob,
        get_lie_category,
        rotation_translator,
        simulate_approach_shots_birdie,
        _plot_hole_layout,
    )
except ImportError as e:
    _sys.exit(f"ERROR: failed to import local modules — {e}")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Second-stage GPR: (x, y) → mean birdie probability
# ---------------------------------------------------------------------------

class _BirdieApproachGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = 15.0

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _fit_birdie_approach_gpr(
    optimal_results: list[dict],
    gp_training_iter: int = 100,
) -> tuple:
    X = torch.tensor([[r["start"][0], r["start"][1]] for r in optimal_results],
                     dtype=torch.float32)
    y = torch.tensor([r["mean_birdie_prob"] for r in optimal_results], dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = _BirdieApproachGPR(X, y, likelihood)
    model.train(); likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(gp_training_iter):
        optimizer.zero_grad()
        (-mll(model(X), y)).backward()
        optimizer.step()

    model.eval(); likelihood.eval()
    return model, likelihood


# ---------------------------------------------------------------------------
# Tee-shot evaluation (birdie)
# ---------------------------------------------------------------------------

def _evaluate_tee_shot_birdie(
    hole: BirdieHoleData,
    approach_model,
    approach_likelihood,
    aim_range: tuple[float, float],
    aim_step: float,
    n_samples: int,
) -> tuple[dict, list[dict]]:
    tee        = hole.tee_point
    target     = hole.hole
    total_dist = float(np.linalg.norm(np.array(target) - np.array(tee)))
    aim_points = list(np.arange(aim_range[0], aim_range[1] + aim_step, aim_step))

    best: dict | None = None
    all_results: list[dict] = []

    for club, stats in hole.club_distributions.items():
        mu, cov = stats["mean"], stats["cov"]
        for aim in aim_points:
            samples   = np.random.multivariate_normal(mu, cov, size=n_samples)
            angle_deg = float(np.degrees(np.arctan(aim / total_dist))) if total_dist > 0 else 0.0
            probs: list[float] = []

            for shot in samples:
                lp  = rotation_translator(float(shot[0]), float(shot[1]),
                                          angle_deg, tee, target)
                inp = torch.tensor([[lp[0], lp[1]]], dtype=torch.float32)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred = approach_likelihood(approach_model(inp))
                    p    = float(pred.mean.item())
                if not np.isnan(p):
                    probs.append(max(0.0, p))

            if not probs:
                continue

            mean_p = float(np.mean(probs))
            var_p  = float(np.var(probs))
            entry  = {"club": club, "aim_offset": float(aim),
                      "mean_birdie_prob": mean_p, "var_birdie_prob": var_p,
                      "n_samples": len(probs)}
            all_results.append(entry)
            if best is None or mean_p > best["mean_birdie_prob"]:
                best = entry

    return best, all_results


# ---------------------------------------------------------------------------
# Top-3 extraction from accumulator  (birdie: maximise)
# ---------------------------------------------------------------------------

def _extract_top3_birdie(
    accumulator: dict,
    strategy_points: list[tuple[float, float]],
) -> dict[tuple[float, float], list[dict]]:
    """Return top-3 (club, aim) combos per grid point, sorted by descending P(birdie)."""
    point_entries: dict[tuple, list[dict]] = defaultdict(list)
    for (x, y, club, aim), shots in accumulator.items():
        if len(shots) == 0:
            continue
        point_entries[(x, y)].append({
            "club":             club,
            "aim_offset":       float(aim),
            "mean_birdie_prob": float(np.mean(shots)),
            "var_birdie_prob":  float(np.var(shots)),
            "n_total":          int(len(shots)),
        })

    top3: dict[tuple, list[dict]] = {}
    for pt in strategy_points:
        key     = (pt[0], pt[1])
        entries = point_entries.get(key, [])
        top3[key] = sorted(entries, key=lambda r: -r["mean_birdie_prob"])[:15]

    return top3


def _top3_tee_birdie(all_tee: list[dict]) -> list[dict]:
    """Return top-3 tee-shot options sorted by descending P(birdie)."""
    return sorted(all_tee, key=lambda r: -r["mean_birdie_prob"])[:15]


# ---------------------------------------------------------------------------
# CSV builder  (long format: one row per (x, y, club, aim_offset) + rank column)
# ---------------------------------------------------------------------------

def _build_dataframe(
    top3_by_point: dict[tuple, list[dict]],
    top3_tee: list[dict],
    hole: BirdieHoleData,
    seed: int,
    N: int,
    carry_shift: float,
    variance_scale: float,
) -> pd.DataFrame:
    rows = []

    def _emit_rows(entries: list[dict], x: float, y: float, is_tee: bool) -> None:
        for rank, e in enumerate(entries, start=1):
            rows.append({
                "x":                x,
                "y":                y,
                "rank":             rank,
                "club":             e["club"],
                "aim_offset":       e["aim_offset"],
                "birdie_prob_mean": e["mean_birdie_prob"],
                "birdie_prob_var":  e["var_birdie_prob"],
                "n_total":          e.get("n_total", e.get("n_samples", N)),
                "seed":             seed,
                "N":                N,
                "carry_shift":      carry_shift,
                "variance_scale":   variance_scale,
                "is_tee_shot":      is_tee,
            })

    for (x, y), entries in top3_by_point.items():
        _emit_rows(entries, x, y, is_tee=False)

    tx, ty = float(hole.tee_point[0]), float(hole.tee_point[1])
    _emit_rows(top3_tee, tx, ty, is_tee=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1 – scatter on hole layout  (public for test script)
# ---------------------------------------------------------------------------

def plot_birdie_sensitivity_result(
    optimal_results: list[dict],
    best_tee: dict,
    hole: BirdieHoleData,
    carry_shift: float,
    variance_scale: float,
    N: int = 0,
    output_path: Path | None = None,
) -> None:
    title = (
        f"Birdie Sensitivity  |  carry {carry_shift:+.1f} yd"
        f"  |  var ×{variance_scale:.2f}  |  N={N}"
    )
    fig, ax = plt.subplots(figsize=(14, 16))
    _plot_hole_layout(hole, title, ax)

    probs       = [r["mean_birdie_prob"] for r in optimal_results]
    norm        = mpl.colors.Normalize(vmin=min(probs), vmax=max(probs))
    face_colors = [CLUB_STYLES.get(r["club"], {"color": "#999999"})["color"]
                   for r in optimal_results]
    edge_colors = [plt.get_cmap("plasma")(norm(p)) for p in probs]

    ax.scatter(
        [r["start"][0] for r in optimal_results],
        [r["start"][1] for r in optimal_results],
        c=face_colors, s=25, alpha=0.85, zorder=20,
        edgecolors=edge_colors, linewidths=1.5,
    )
    for r in optimal_results:
        short = CLUB_STYLES.get(r["club"], {"short": r["club"]})["short"]
        ax.text(r["start"][0] - 2, r["start"][1] + 2.5,
                f'{short},{int(r["aim_offset"]):+}',
                fontsize=5, color="black", zorder=21)

    sm = mpl.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("P(birdie)")

    tx, ty    = float(hole.tee_point[0]), float(hole.tee_point[1])
    tee_face  = CLUB_STYLES.get(best_tee["club"], {"color": "#999999"})["color"]
    tee_edge  = plt.get_cmap("plasma")(norm(best_tee["mean_birdie_prob"]))
    tee_short = CLUB_STYLES.get(best_tee["club"], {"short": best_tee["club"]})["short"]

    ax.scatter(tx, ty, c=[tee_face], s=25, alpha=0.85, zorder=20,
               edgecolors=[tee_edge], linewidths=1.5)
    ax.text(tx - 2, ty + 2.5, f'{tee_short},{int(best_tee["aim_offset"]):+}',
            fontsize=5, color="black", zorder=21)

    std = float(np.sqrt(best_tee["var_birdie_prob"]))
    ax.text(0.98, 0.98,
            f'Best tee shot\n{tee_short}  aim {best_tee["aim_offset"]:+.0f} yd\n'
            f'P(birdie) = {best_tee["mean_birdie_prob"]:.4f} ± {std:.4f}',
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9),
            zorder=40)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("Saved scatter plot → %s", output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 – heatmap grid  (one cell per approach grid point)
# ---------------------------------------------------------------------------

def _plot_heatmap_birdie(
    top3_by_point: dict[tuple, list[dict]],
    best_tee: dict,
    carry_shift: float,
    variance_scale: float,
    N: int,
    output_path: Path,
) -> None:
    xs_uniq = sorted(set(pt[0] for pt in top3_by_point))
    ys_uniq = sorted(set(pt[1] for pt in top3_by_point))
    n_x, n_y = len(xs_uniq), len(ys_uniq)

    xi = {x: i for i, x in enumerate(xs_uniq)}
    yi = {y: i for i, y in enumerate(ys_uniq)}

    values = np.full((n_y, n_x), np.nan)
    labels: list[list[str]] = [[""] * n_x for _ in range(n_y)]

    for (x, y), entries in top3_by_point.items():
        if not entries:
            continue
        top1  = entries[0]
        r, c  = yi[y], xi[x]
        values[r, c] = top1["mean_birdie_prob"]
        short = CLUB_STYLES.get(top1["club"], {"short": top1["club"][:4]})["short"]
        labels[r][c] = f"{short}\n{top1['aim_offset']:+.0f}"

    title = (
        f"Decision Heatmap – P(birdie)  |  carry {carry_shift:+.1f} yd"
        f"  |  var ×{variance_scale:.2f}  |  N={N}"
    )

    cell_h  = max(0.55, 14 / n_y)
    fig_h   = n_y * cell_h + 2.5
    fig, ax = plt.subplots(figsize=(14, fig_h))

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    im   = ax.imshow(
        values, origin="lower", aspect="auto",
        cmap="plasma", vmin=vmin, vmax=vmax,
        extent=[-.5, n_x - .5, -.5, n_y - .5],
    )

    fontsize = max(5, min(8, int(120 / max(n_x, n_y))))
    for r in range(n_y):
        for c in range(n_x):
            txt = labels[r][c]
            if not txt:
                continue
            brightness = (values[r, c] - vmin) / (vmax - vmin + 1e-9)
            txt_color  = "white" if brightness < 0.45 else "black"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=fontsize, color=txt_color, fontweight="bold",
                    multialignment="center")

    ax.set_xticks(range(n_x))
    ax.set_xticklabels([f"{x:.0f}" for x in xs_uniq], fontsize=8)
    ax.set_yticks(range(n_y))
    ax.set_yticklabels([f"{y:.0f}" for y in ys_uniq], fontsize=8)
    ax.set_xlabel("x (yards)", fontsize=9)
    ax.set_ylabel("y (yards)", fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("P(birdie) (higher = better)", fontsize=9)

    tee_short = CLUB_STYLES.get(best_tee["club"], {"short": best_tee["club"]})["short"]
    std       = float(np.sqrt(best_tee["var_birdie_prob"]))
    ax.text(0.01, 0.99,
            f"Tee: {tee_short} {best_tee['aim_offset']:+.0f} yd\n"
            f"P(birdie) = {best_tee['mean_birdie_prob']:.4f} ± {std:.4f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.9),
            zorder=40)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-task birdie sensitivity analysis worker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task-id",     type=int, default=None)
    p.add_argument("--configs-csv", type=Path, default=_HERE / "param_configs.csv")
    p.add_argument("--n-shots",     type=int, default=280)
    p.add_argument("--gp-iter",     type=int, default=100)
    p.add_argument("--aim-range",   type=float, nargs=2, default=[-20.0, 20.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--aim-step",    type=float, default=2.0)
    p.add_argument("--tee-samples", type=int, default=50)
    p.add_argument("--data-dir",    type=Path, default=None)
    p.add_argument("--output-dir",  type=Path, default=Path("outputs_birdie"))
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

    logger.info("Birdie sensitivity worker | SLURM_JOB_ID=%s  task_id=%d",
                os.environ.get("SLURM_JOB_ID", "N/A"), task_id)

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
    if not data_dir.exists():
        _sys.exit(f"ERROR: data-dir does not exist: {data_dir}")

    # ── 1. Build hole ──────────────────────────────────────────────────────
    logger.info("Building birdie hole ...")
    hole = build_hole_birdie(
        data_dir=data_dir,
        gp_training_iter=args.gp_iter,
        carry_shift_yards=carry_shift,
        variance_scale=variance_scale,
    )
    logger.info("Hole ready. %d strategy points.", len(hole.strategy_points))

    # ── 2. Approach simulation ─────────────────────────────────────────────
    logger.info("Starting simulate_approach_shots_birdie (N=%d) ...", N)
    np.random.seed(task_id)
    optimal_results, accumulator = simulate_approach_shots_birdie(
        hole=hole, n_new=N, accumulator=None,
        aim_range=tuple(args.aim_range), aim_step=args.aim_step,
    )
    logger.info("Approach done. %d grid points.", len(optimal_results))

    # ── 3. Extract top-3 per point ─────────────────────────────────────────
    top3_by_point = _extract_top3_birdie(accumulator, hole.strategy_points)

    # ── 4. Second-stage GPR + tee evaluation ──────────────────────────────
    logger.info("Fitting birdie approach GPR (%d iter) ...", args.gp_iter)
    approach_model, approach_likelihood = _fit_birdie_approach_gpr(
        optimal_results, gp_training_iter=args.gp_iter
    )

    logger.info("Evaluating tee shot (%d samples per combo) ...", args.tee_samples)
    best_tee, all_tee = _evaluate_tee_shot_birdie(
        hole=hole,
        approach_model=approach_model,
        approach_likelihood=approach_likelihood,
        aim_range=tuple(args.aim_range),
        aim_step=args.aim_step,
        n_samples=args.tee_samples,
    )
    top3_tee = _top3_tee_birdie(all_tee)
    logger.info("Best tee: %s  aim=%+.0f yd  P(birdie)=%.4f",
                best_tee["club"], best_tee["aim_offset"], best_tee["mean_birdie_prob"])

    # ── 5. Save CSV ────────────────────────────────────────────────────────
    fname_base = f"birdie_sensitivity_dist{carry_shift:.2f}_disp{variance_scale:.4f}"
    csv_path   = args.output_dir / f"{fname_base}.csv"

    df = _build_dataframe(
        top3_by_point=top3_by_point,
        top3_tee=top3_tee,
        hole=hole,
        seed=task_id, N=N,
        carry_shift=carry_shift,
        variance_scale=variance_scale,
    )
    df.to_csv(csv_path, index=False)

    n_approach = int(((df["is_tee_shot"] == False) & (df["rank"] == 1)).sum())
    logger.info("CSV saved → %s  (%d approach rows + 1 tee row)", csv_path, n_approach)

    # ── 6. Scatter plot ────────────────────────────────────────────────────
    png_scatter = args.output_dir / f"{fname_base}.png"
    logger.info("Generating scatter plot ...")
    plot_birdie_sensitivity_result(
        optimal_results=optimal_results,
        best_tee=best_tee,
        hole=hole,
        carry_shift=carry_shift,
        variance_scale=variance_scale,
        N=N,
        output_path=png_scatter,
    )

    # ── 7. Heatmap ─────────────────────────────────────────────────────────
    png_heatmap = args.output_dir / f"{fname_base}_heatmap.png"
    logger.info("Generating heatmap ...")
    _plot_heatmap_birdie(
        top3_by_point=top3_by_point,
        best_tee=best_tee,
        carry_shift=carry_shift,
        variance_scale=variance_scale,
        N=N,
        output_path=png_heatmap,
    )

    # ── 8. Metadata ────────────────────────────────────────────────────────
    approach_df = df[(df["is_tee_shot"] == False) & (df["rank"] == 1)]
    meta = {
        "task_id":              task_id,
        "trend":                trend,
        "carry_shift":          carry_shift,
        "variance_scale":       variance_scale,
        "N":                    N,
        "n_approach_rows":      n_approach,
        "mean_birdie_prob":     float(approach_df["birdie_prob_mean"].mean()),
        "best_tee_club":        best_tee["club"],
        "best_tee_aim":         best_tee["aim_offset"],
        "best_tee_birdie_prob": best_tee["mean_birdie_prob"],
        "output_csv":           f"{fname_base}.csv",
        "output_png":           f"{fname_base}.png",
        "output_heatmap":       f"{fname_base}_heatmap.png",
    }
    meta_path = args.output_dir / f"{fname_base}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Task %d done | mean P(birdie)=%.4f | best tee: %s %+.0f yd → %.4f",
        task_id, meta["mean_birdie_prob"], best_tee["club"],
        best_tee["aim_offset"], best_tee["mean_birdie_prob"],
    )
    print(json.dumps(meta))


if __name__ == "__main__":
    import traceback as _tb
    try:
        main()
    except Exception:
        _tb.print_exc()
        _sys.exit(1)
