"""
par4_birdie_standalone.py – Standalone Par-4 birdie probability analysis.

Mirrors par4birdiemodel.ipynb but uses the statistically correct model:
  - Green: VariationalGP + BernoulliLikelihood (probit link) on binary 0/1 labels
    instead of GaussianLikelihood + np.clip (which is wrong for binary data).
  - Approach: simulate shots from every grid point, off-green → P=0.
  - Surface GPR: ExactGP fit over optimal-per-grid-point birdie probabilities.
  - Tee shot: sweep clubs/aim-offsets, maximise expected birdie probability.

Usage:
    cd Parallelisation/convergence_birdie
    python par4_birdie_standalone.py
    python par4_birdie_standalone.py --n-approach 100 --gp-iter 200 --output-dir out/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── local imports ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from core_birdie import (
    BirdieHoleData,
    build_hole_birdie,
    evaluate_birdie_prob,
    get_lie_category,
    plot_optimal_approaches_birdie,
    results_to_dataframe,
    rotation_translator,
    simulate_approach_shots_birdie,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── second-stage GPR: smooth the optimal-birdie-prob surface ──────────────────

class _SurfaceGPR(gpytorch.models.ExactGP):
    """ExactGP fit over the optimal P(birdie) per approach grid point."""

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = 15.0

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _fit_surface_gpr(
    optimal_results: list[dict],
    n_iter: int = 100,
) -> tuple[_SurfaceGPR, gpytorch.likelihoods.GaussianLikelihood]:
    """Fit a smooth GPR surface over per-grid-point optimal birdie probabilities."""
    X = torch.tensor(
        [[r["start"][0], r["start"][1]] for r in optimal_results],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [r["mean_birdie_prob"] for r in optimal_results],
        dtype=torch.float32,
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model      = _SurfaceGPR(X, y, likelihood)

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll       = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(X), y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            logger.info("  Surface GPR iter %d/%d  loss=%.4f", i + 1, n_iter, loss.item())

    model.eval(); likelihood.eval()
    return model, likelihood


# ── tee-shot evaluation ───────────────────────────────────────────────────────

def evaluate_tee_shot(
    hole: BirdieHoleData,
    surface_model: _SurfaceGPR,
    surface_likelihood: gpytorch.likelihoods.GaussianLikelihood,
    aim_range: tuple[float, float] = (-30.0, 30.0),
    aim_step: float = 2.0,
    n_samples: int = 100,
) -> tuple[dict, list[dict]]:
    """Sweep clubs × aim offsets from the tee, return best and all results."""
    aim_offsets = list(np.arange(aim_range[0], aim_range[1] + aim_step, aim_step))
    results: list[dict] = []
    best: dict | None = None

    for club, stats in hole.club_distributions.items():
        mu, cov = stats["mean"], stats["cov"]
        for aim in aim_offsets:
            samples = np.random.multivariate_normal(mu, cov, size=n_samples)
            birdie_preds: list[float] = []

            for shot in samples:
                angle_deg = float(np.degrees(np.arctan(
                    aim / np.linalg.norm(np.array(hole.hole) - np.array(hole.tee_point))
                )))
                lp = rotation_translator(
                    float(shot[0]), float(shot[1]),
                    angle_deg, hole.tee_point, hole.hole,
                )
                test_x = torch.tensor([[lp[0], lp[1]]], dtype=torch.float32)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred = surface_likelihood(surface_model(test_x))
                    p = float(pred.mean.item())
                birdie_preds.append(p)

            mean_p = float(np.mean(birdie_preds))
            var_p  = float(np.var(birdie_preds))
            entry  = {"club": club, "aim_offset": aim, "mean_birdie_prob": mean_p, "var_birdie_prob": var_p}
            results.append(entry)

            if best is None or mean_p > best["mean_birdie_prob"]:
                best = entry

    return best, results


# ── surface-GPR visualisation ─────────────────────────────────────────────────

def _plot_surface(
    optimal_results: list[dict],
    surface_model: _SurfaceGPR,
    surface_likelihood: gpytorch.likelihoods.GaussianLikelihood,
    output_path: Path,
) -> None:
    xs = [r["start"][0] for r in optimal_results]
    ys = [r["start"][1] for r in optimal_results]
    xg = np.linspace(min(xs), max(xs), 100)
    yg = np.linspace(min(ys), max(ys), 100)
    grid = torch.tensor([[x, y] for y in yg for x in xg], dtype=torch.float32)

    with torch.no_grad():
        pred_mean = surface_likelihood(surface_model(grid)).mean.numpy().reshape(len(yg), len(xg))

    fig, ax = plt.subplots(figsize=(10, 14))
    im = ax.imshow(
        pred_mean, extent=[xg.min(), xg.max(), yg.min(), yg.max()],
        origin="lower", cmap="viridis", aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="P(birdie) — approach surface")
    ax.scatter(xs, ys, c=[r["mean_birdie_prob"] for r in optimal_results],
               s=12, cmap="viridis", edgecolors="k", linewidths=0.5,
               label="Optimal per grid point")
    ax.set_xlabel("x (yards)"); ax.set_ylabel("y (yards)")
    ax.set_title("GPR Birdie Probability Surface — Approach")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved surface plot → %s", output_path)


# ── tee-shot visualisation ────────────────────────────────────────────────────

def _plot_tee_shot(
    best: dict,
    hole: BirdieHoleData,
    output_path: Path,
) -> None:
    club, aim = best["club"], best["aim_offset"]
    mu = hole.club_distributions[club]["mean"]
    angle_deg = float(np.degrees(np.arctan(
        aim / np.linalg.norm(np.array(hole.hole) - np.array(hole.tee_point))
    )))
    land_x, land_y = rotation_translator(
        float(mu[0]), float(mu[1]), angle_deg, hole.tee_point, hole.hole
    )

    fig, ax = plt.subplots(figsize=(10, 14))

    # Hole background
    from core_birdie import _plot_hole_layout
    _plot_hole_layout(hole, "Optimal Tee Shot — Maximise P(Birdie)", ax)

    ax.plot([hole.tee_point[0], land_x], [hole.tee_point[1], land_y],
            "k--", linewidth=1.5, zorder=10)
    ax.scatter([land_x], [land_y], color="gold", edgecolors="black",
               s=400, marker="*", zorder=20)
    pct = best["mean_birdie_prob"] * 100
    ax.annotate(
        f"Optimal Tee Shot\n{club}\nAim: {aim:+.0f} yds\nP(birdie) = {pct:.1f}%",
        xy=(land_x, land_y),
        xytext=(land_x - 40, land_y - 80),
        textcoords="data", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        zorder=30,
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved tee-shot plot → %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Par-4 birdie model (BernoulliLikelihood GPR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    type=Path, default=None,
                   help="Path to data directory (default: Parallelisation/data/).")
    p.add_argument("--output-dir",  type=Path, default=Path("birdie_outputs"),
                   help="Directory for plots and CSV output.")
    p.add_argument("--n-approach",  type=int, default=50,
                   help="Approach shots sampled per (grid point, club, aim).")
    p.add_argument("--n-tee",       type=int, default=100,
                   help="Tee shots sampled per (club, aim).")
    p.add_argument("--gp-iter",     type=int, default=200,
                   help="Training iterations for the birdie GPR (green model).")
    p.add_argument("--surface-iter", type=int, default=100,
                   help="Training iterations for the surface GPR.")
    p.add_argument("--aim-range",   type=float, nargs=2, default=[-20.0, 20.0],
                   metavar=("MIN", "MAX"),
                   help="Aim offset range in yards.")
    p.add_argument("--aim-step",    type=float, default=2.0,
                   help="Aim offset step size in yards.")
    p.add_argument("--seed",        type=int, default=42,
                   help="NumPy random seed.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir or (Path(__file__).parent.parent / "data")

    # ── 1. Build hole (trains BernoulliLikelihood GPR on green) ──────────────
    logger.info("Building hole and training birdie GPR (%d iter) ...", args.gp_iter)
    hole = build_hole_birdie(data_dir=data_dir, gp_training_iter=args.gp_iter)
    logger.info(
        "Hole ready.  Tee=%.1f,%.1f  Pin=%.1f,%.1f  Grid=%d points",
        *hole.tee_point, *hole.hole, len(hole.strategy_points),
    )

    # ── 2. Approach simulation ────────────────────────────────────────────────
    aim_range = tuple(args.aim_range)  # type: ignore[arg-type]
    logger.info(
        "Simulating %d approach shots per strategy (aim %+.0f…%+.0f yds, step %.0f) ...",
        args.n_approach, aim_range[0], aim_range[1], args.aim_step,
    )
    optimal_results, _ = simulate_approach_shots_birdie(
        hole=hole,
        n_new=args.n_approach,
        aim_range=aim_range,
        aim_step=args.aim_step,
    )
    logger.info("Approach simulation done. %d grid points.", len(optimal_results))

    # ── 3. Plot approach strategy map ─────────────────────────────────────────
    approach_plot = args.output_dir / "approach_birdie_strategy.png"
    plot_optimal_approaches_birdie(
        optimal_results=optimal_results,
        hole=hole,
        title="Optimal Approach Strategy — Maximise P(Birdie)",
        output_path=approach_plot,
    )

    # ── 4. Fit surface GPR over optimal probabilities ─────────────────────────
    logger.info("Fitting surface GPR (%d iter) ...", args.surface_iter)
    surface_model, surface_likelihood = _fit_surface_gpr(
        optimal_results, n_iter=args.surface_iter
    )
    _plot_surface(optimal_results, surface_model, surface_likelihood,
                  args.output_dir / "birdie_surface_gpr.png")

    # ── 5. Tee-shot evaluation ────────────────────────────────────────────────
    logger.info("Evaluating tee shots (%d samples each) ...", args.n_tee)
    best, all_tee = evaluate_tee_shot(
        hole=hole,
        surface_model=surface_model,
        surface_likelihood=surface_likelihood,
        aim_range=aim_range,
        aim_step=args.aim_step,
        n_samples=args.n_tee,
    )

    # ── 6. Print results ──────────────────────────────────────────────────────
    top5 = sorted(all_tee, key=lambda r: r["mean_birdie_prob"], reverse=True)[:5]
    print("\n" + "=" * 55)
    print("  TOP-5 TEE SHOTS — P(Birdie)")
    print("=" * 55)
    print(f"  {'Club':<14}  {'Aim (yds)':>9}  {'P(birdie)':>10}  {'StdDev':>8}")
    print("  " + "-" * 50)
    for r in top5:
        std = float(np.sqrt(r["var_birdie_prob"]))
        print(f"  {r['club']:<14}  {r['aim_offset']:>+9.0f}  "
              f"{r['mean_birdie_prob'] * 100:>9.2f}%  {std * 100:>7.2f}%")
    print("=" * 55)
    print(f"\n  Best: {best['club']}  aim={best['aim_offset']:+.0f} yds  "
          f"P(birdie)={best['mean_birdie_prob'] * 100:.2f}%\n")

    # ── 7. Save tee-shot plot ─────────────────────────────────────────────────
    _plot_tee_shot(best, hole, args.output_dir / "tee_shot_birdie.png")

    # ── 8. Save CSVs ─────────────────────────────────────────────────────────
    approach_csv = args.output_dir / "approach_results.csv"
    results_to_dataframe(optimal_results, seed=args.seed, N=args.n_approach).to_csv(
        approach_csv, index=False
    )
    logger.info("Saved approach results → %s", approach_csv)

    tee_csv = args.output_dir / "tee_shot_results.csv"
    pd.DataFrame(all_tee).to_csv(tee_csv, index=False)
    logger.info("Saved tee-shot results → %s", tee_csv)

    logger.info("Done.  Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
