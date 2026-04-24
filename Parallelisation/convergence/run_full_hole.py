"""
run_full_hole.py – Full Par-4 simulation: approach shots + tee shot.

Two modes:
  --load-csv   Load already-simulated N=300 results from a seed CSV and skip
               the approach-shot simulation (fast, ~2 min for GPR training).
  (default)    Simulate fresh approach shots to --n-shots (default 300) and
               then evaluate the tee shot on top.

Pipeline:
  1. Build hole geometry + train putting GPR.
  2. Simulate approach shots to N shots per (grid-point, club, aim).
  3. Fit a second GPR over the optimal ESHO values across the strategy grid.
  4. Evaluate all (club, aim) combos from the tee using that GPR.
  5. Save plots: approach strategy map + tee shot overlay.

Usage examples:

    # Fastest: load existing N=300 data for seed 0
    python run_full_hole.py --load-csv outputs/seed0000/seed0000_N0300.csv

    # Fresh simulation to N=300 (takes ~20-40 min)
    python run_full_hole.py --n-shots 300

    # Lighter test: simulate to N=50
    python run_full_hole.py --n-shots 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Run from inside convergence/ directory
sys.path.insert(0, str(Path(__file__).parent))
from core import (
    HoleData,
    _PuttGPModel,
    build_hole,
    plot_hole_layout,
    plot_optimal_approaches,
    rotation_translator,
    simulate_approach_shots,
    CLUB_STYLES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Second-stage GPR: (x, y) → optimal ESHO
# ---------------------------------------------------------------------------

class _ApproachGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.covar_module.base_kernel.lengthscale = 15.0

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def fit_approach_gpr(
    optimal_results: list[dict],
    gp_training_iter: int = 100,
) -> tuple[_ApproachGPR, gpytorch.likelihoods.GaussianLikelihood]:
    """Fit ExactGP to optimal ESHO values over the strategy grid."""
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
        loss = -mll(model(X), y)
        loss.backward()
        optimizer.step()

    model.eval(); likelihood.eval()
    logger.info("Approach GPR trained (%d iterations).", gp_training_iter)
    return model, likelihood


# ---------------------------------------------------------------------------
# Tee shot evaluation
# ---------------------------------------------------------------------------

def evaluate_tee_shot(
    hole: HoleData,
    approach_model: _ApproachGPR,
    approach_likelihood: gpytorch.likelihoods.GaussianLikelihood,
    aim_range: tuple[float, float] = (-30.0, 30.0),
    aim_step: float = 2.0,
    n_samples: int = 50,
) -> tuple[dict, list[dict]]:
    """Evaluate all (club, aim) combos from the tee.

    For each combo, simulate n_samples tee shots, look up predicted ESHO at
    each landing spot from the approach GPR, and add 1 stroke for the tee shot
    itself.  Returns (best_result, all_results).
    """
    tee_point  = hole.tee_point
    target     = hole.hole
    aim_points = list(np.arange(aim_range[0], aim_range[1] + aim_step, aim_step))

    total_distance = float(np.linalg.norm(np.array(target) - np.array(tee_point)))

    best: dict | None = None
    all_results: list[dict] = []

    for club, stats in hole.club_distributions.items():
        mu  = stats["mean"]
        cov = stats["cov"]

        for aim in aim_points:
            samples = np.random.multivariate_normal(mu, cov, size=n_samples)
            angle_deg = (
                float(np.degrees(np.arctan(aim / total_distance)))
                if total_distance > 0 else 0.0
            )

            esho_vals: list[float] = []
            for shot in samples:
                lp = rotation_translator(
                    float(shot[0]), float(shot[1]),
                    angle_deg, tee_point, target,
                )
                inp = torch.tensor([[lp[0], lp[1]]], dtype=torch.float32)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred = approach_likelihood(approach_model(inp))
                    esho = float(pred.mean.item())
                if not np.isnan(esho):
                    esho_vals.append(esho)

            if not esho_vals:
                continue

            # Total expected strokes = 1 (tee) + E[approach ESHO at landing]
            total_mean = 1.0 + float(np.mean(esho_vals))
            total_var  = float(np.var(esho_vals))

            result = {
                "club":       club,
                "aim_offset": float(aim),
                "mean":       total_mean,
                "var":        total_var,
                "n_samples":  len(esho_vals),
            }
            all_results.append(result)

            if best is None or total_mean < best["mean"]:
                best = result

    return best, all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tee_shot_overlay(
    best_tee: dict,
    all_tee: list[dict],
    hole: HoleData,
    output_path: Path,
) -> None:
    """Plot hole layout with all tee-shot landing distributions and best shot."""
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(14, 16))
    plot_hole_layout(hole, title="Optimal Tee Shot", plot_strategy_points=False, ax=ax)

    tee = hole.tee_point
    target = hole.hole

    # Draw all tee-shot mean landings, coloured by total expected strokes
    means  = [r["mean"] for r in all_tee]
    norm   = mpl.colors.Normalize(vmin=min(means), vmax=max(means))
    cmap   = plt.get_cmap("RdYlGn_r")

    for r in all_tee:
        club, aim = r["club"], r["aim_offset"]
        mu = hole.club_distributions[club]["mean"]
        total_dist = float(np.linalg.norm(np.array(target) - np.array(tee)))
        angle_deg = float(np.degrees(np.arctan(aim / total_dist))) if total_dist > 0 else 0.0
        lp = rotation_translator(float(mu[0]), float(mu[1]), angle_deg, tee, target)
        color = cmap(norm(r["mean"]))
        ax.scatter(lp[0], lp[1], color=color, s=18, alpha=0.6, zorder=15)

    # Highlight the best tee shot
    best_club = best_tee["club"]
    best_aim  = best_tee["aim_offset"]
    mu_best   = hole.club_distributions[best_club]["mean"]
    total_dist = float(np.linalg.norm(np.array(target) - np.array(tee)))
    angle_deg  = float(np.degrees(np.arctan(best_aim / total_dist))) if total_dist > 0 else 0.0
    lp_best    = rotation_translator(float(mu_best[0]), float(mu_best[1]), angle_deg, tee, target)

    ax.scatter(*lp_best, color="gold", s=200, zorder=30, edgecolors="black",
               linewidths=2, label=f"Best: {best_club}, aim {best_aim:+.0f} yds")
    ax.annotate(
        f"{best_club}\naim {best_aim:+.0f} yd\nE[strokes]={best_tee['mean']:.3f}",
        xy=lp_best, xytext=(lp_best[0] + 10, lp_best[1] + 15),
        fontsize=9, color="black", zorder=40,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("E[total strokes from tee]")

    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved tee shot plot → %s", output_path)


# ---------------------------------------------------------------------------
# Load results from existing CSV
# ---------------------------------------------------------------------------

def _csv_to_results(csv_path: Path) -> list[dict]:
    """Convert a seed####_N####.csv back into optimal_results list[dict]."""
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        results.append({
            "start":      (float(row["x"]), float(row["y"])),
            "club":       row["club"],
            "aim_offset": float(row["aim_offset"]),
            "mean":       float(row["esho_mean"]),
            "var":        float(row["esho_var"]),
            "n_total":    int(row.get("n_total", row["N"])),
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full hole simulation: approach shots + tee shot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--load-csv", type=Path, default=None,
        help="Load existing approach-shot CSV (e.g. outputs/seed0000/seed0000_N0300.csv) "
             "and skip re-simulation.",
    )
    src.add_argument(
        "--n-shots", type=int, default=300,
        help="Total approach shots to simulate per (grid-point, club, aim). Ignored if --load-csv.",
    )
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--gp-iter",        type=int,   default=100,
                   help="Training iterations for both GPRs.")
    p.add_argument("--tee-n-samples",  type=int,   default=50,
                   help="Monte Carlo samples for tee shot evaluation.")
    p.add_argument("--tee-aim-range",  type=float, nargs=2, default=[-30.0, 30.0])
    p.add_argument("--tee-aim-step",   type=float, default=2.0)
    p.add_argument("--data-dir",       type=Path,
                   default=Path(__file__).parent.parent / "data")
    p.add_argument("--output-dir",     type=Path,  default=Path("outputs/full_hole"))
    p.add_argument("--log-level",      default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Build hole (always needed — geometry + putting GPR)
    # ------------------------------------------------------------------
    logger.info("Building hole geometry and training putting GPR...")
    hole = build_hole(args.data_dir, gp_training_iter=args.gp_iter)
    logger.info("Hole ready.  %d strategy points.", len(hole.strategy_points))

    # ------------------------------------------------------------------
    # 2. Approach shots — simulate or load
    # ------------------------------------------------------------------
    if args.load_csv is not None:
        logger.info("Loading approach results from %s", args.load_csv)
        optimal_results = _csv_to_results(args.load_csv)
        n_label = args.load_csv.stem.split("_N")[-1] if "_N" in args.load_csv.stem else "loaded"
    else:
        logger.info("Simulating %d approach shots per strategy ...", args.n_shots)
        optimal_results, _ = simulate_approach_shots(
            hole=hole,
            n_new=args.n_shots,
            accumulator=None,
        )
        n_label = str(args.n_shots)
        # Save approach CSV
        from core import results_to_dataframe
        df = results_to_dataframe(optimal_results, seed=args.seed, N=args.n_shots)
        csv_path = args.output_dir / f"approach_N{args.n_shots:04d}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved approach CSV → %s", csv_path)

    logger.info("Approach results: %d grid points.", len(optimal_results))

    # Save approach PNG
    approach_png = args.output_dir / f"approach_N{n_label}.png"
    plot_optimal_approaches(
        optimal_results, hole,
        title=f"Optimal Approach Strategy  (N={n_label})",
        output_path=approach_png,
    )

    # ------------------------------------------------------------------
    # 3. Fit approach GPR over the strategy grid
    # ------------------------------------------------------------------
    logger.info("Fitting approach GPR over %d strategy points ...", len(optimal_results))
    approach_gpr, approach_lik = fit_approach_gpr(optimal_results, gp_training_iter=args.gp_iter)

    # ------------------------------------------------------------------
    # 4. Evaluate tee shot
    # ------------------------------------------------------------------
    logger.info("Evaluating tee shot (aim range %s, step %.1f, %d samples) ...",
                args.tee_aim_range, args.tee_aim_step, args.tee_n_samples)
    best_tee, all_tee = evaluate_tee_shot(
        hole=hole,
        approach_model=approach_gpr,
        approach_likelihood=approach_lik,
        aim_range=tuple(args.tee_aim_range),
        aim_step=args.tee_aim_step,
        n_samples=args.tee_n_samples,
    )

    # ------------------------------------------------------------------
    # 5. Print and save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  OPTIMAL TEE SHOT")
    print(f"  Club       : {best_tee['club']}")
    print(f"  Aim offset : {best_tee['aim_offset']:+.0f} yards")
    print(f"  E[strokes] : {best_tee['mean']:.4f}  (SE={best_tee['var']**0.5:.4f})")
    print("=" * 55)

    # Top-5 tee shots by expected strokes
    top5 = sorted(all_tee, key=lambda r: r["mean"])[:5]
    print("\n  Top-5 tee shot options:")
    print(f"  {'Club':<14} {'Aim':>6}  {'E[strokes]':>12}  {'Var':>10}")
    print("  " + "-" * 48)
    for r in top5:
        print(f"  {r['club']:<14} {r['aim_offset']:>+6.0f}  "
              f"{r['mean']:>12.4f}  {r['var']:>10.5f}")
    print()

    # Save tee shot results CSV
    tee_df = pd.DataFrame(all_tee).sort_values("mean")
    tee_csv = args.output_dir / "tee_shot_results.csv"
    tee_df.to_csv(tee_csv, index=False)
    logger.info("Saved tee results → %s", tee_csv)

    # Save tee shot plot
    tee_png = args.output_dir / f"tee_shot_N{n_label}.png"
    plot_tee_shot_overlay(best_tee, all_tee, hole, tee_png)

    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
