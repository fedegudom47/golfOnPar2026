"""
gen_birdie_single_run.py -- one-off, single-seed, N=300 full-grid sweep +
second-stage GP + tee-shot maximisation for the birdie (aggressive) scheme,
mirroring multifidelity_strategic_blanket.py's ESHO pipeline but maximising
birdie probability instead of minimising ESHO, and no multi-fidelity fusion
(per the paper's own text: f_approach_b is a plain ExactGP, no real
per-location birdie ground truth is available to fuse against).

Uses the CURRENT core_birdie.py, with the fixed +/-40deg / 5deg aim-angle
grid (see MyScripts/convergenceNEW.qmd, "Proposed fix") to match the ESHO
scheme for consistency (the birdie scheme's own default is still the old
fixed-yard +/-20/step2 grid).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core_birdie import (
    build_hole_birdie, simulate_approach_shots_birdie, results_to_dataframe,
    plot_optimal_approaches_birdie, rotation_translator,
)

SEED = 9
N = 1000
AIM_ANGLE_RANGE = (-40.0, 40.0)
AIM_ANGLE_STEP = 5.0
OUT = Path(__file__).parent / "outputs_single_run"
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

print("Building birdie hole (gp_training_iter=200, matches core_birdie default)...")
hole = build_hole_birdie(gp_training_iter=200)

print(f"Running single-batch birdie sweep, n_new={N}, full grid, "
      f"fixed-angle aim={AIM_ANGLE_RANGE} step {AIM_ANGLE_STEP}deg...")
optimal_results, _ = simulate_approach_shots_birdie(
    hole, n_new=N, aim_angle_range=AIM_ANGLE_RANGE, aim_angle_step=AIM_ANGLE_STEP,
)

df = results_to_dataframe(optimal_results, seed=SEED, N=N)
csv_path = OUT / f"seed{SEED:04d}_N{N:04d}_birdie.csv"
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}  ({len(df)} rows)")

png_path = OUT / f"seed{SEED:04d}_N{N:04d}_birdie_approach.png"
plot_optimal_approaches_birdie(optimal_results, hole, title=f"Aggressive (Birdie) approach strategy, N={N}", output_path=png_path)
print(f"Saved {png_path}")

# ---------------------------------------------------------------------
# Second-stage GP: f_approach_b, standard ExactGP on mean_birdie_prob
# (matches par4birdiemodel.ipynb GPModelApproachBirdie: lengthscale warm
# start not specified there beyond default init; keep consistent with the
# ESHO f_sim GP's own hyperparams: lr=0.1, 100 iter, lengthscale=15 init)
# ---------------------------------------------------------------------
class _ApproachBirdieGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = 15.0

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

X = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
y = torch.tensor(df["mean_birdie_prob"].values, dtype=torch.float32)
lik = gpytorch.likelihoods.GaussianLikelihood()
gp = _ApproachBirdieGP(X, y, lik)
gp.train(); lik.train()
opt = torch.optim.Adam(gp.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)
print("\nTraining f_approach_b (100 iter, lr=0.1)...")
for i in range(1, 101):
    opt.zero_grad()
    loss = -mll(gp(X), y)
    loss.backward(); opt.step()
    if i % 25 == 0 or i == 1:
        print(f"  {i:3d}: loss={float(loss):.4f}")
gp.eval(); lik.eval()

def f_approach_b(pts: np.ndarray) -> np.ndarray:
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        t = torch.tensor(pts, dtype=torch.float32)
        return lik(gp(t)).mean.numpy()

# ---------------------------------------------------------------------
# Tee-shot Monte Carlo: maximise mean birdie probability
# ---------------------------------------------------------------------
tee = hole.tee_point
pin = hole.hole
total_dist = float(np.linalg.norm(np.array(pin) - np.array(tee)))
clubs_avg_carry = {c: s["mean"][1] for c, s in hole.club_distributions.items()}
long_clubs = [c for c, carry in clubs_avg_carry.items() if carry >= 150.0]
aimpoints = list(np.arange(-40, 44, 4))
N_MC = 2000

best = None
rows = []
print(f"\nTee-shot MC over {len(long_clubs)} long clubs x {len(aimpoints)} aims, N={N_MC}...")
for club in long_clubs:
    mu, cov = hole.club_distributions[club]["mean"], hole.club_distributions[club]["cov"]
    for aim in aimpoints:
        angle = float(np.degrees(np.arctan(aim / total_dist))) if total_dist > 0 else 0.0
        samples = np.random.multivariate_normal(mu, cov, size=N_MC)
        lps = np.array([rotation_translator(float(s[0]), float(s[1]), angle, tee, pin) for s in samples])
        probs = f_approach_b(lps)
        mean_prob = float(np.mean(probs))
        rows.append({"club": club, "aim_yards": float(aim), "mean_birdie_prob": mean_prob})
        if best is None or mean_prob > best["mean_birdie_prob"]:
            best = {"club": club, "aim_yards": float(aim), "mean_birdie_prob": mean_prob}

print(f"\nBest tee shot (birdie-maximising): {best}")

import json
with open(OUT / "birdie_tee_shot_result.json", "w") as f:
    json.dump({"best_tee_shot": best, "N_grid": N, "N_tee_mc": N_MC,
               "aim_angle_range": AIM_ANGLE_RANGE, "aim_angle_step": AIM_ANGLE_STEP}, f, indent=2)

# ---------------------------------------------------------------------
# Final figure: approach-grid plot + best tee shot star overlay
# ---------------------------------------------------------------------
plot_optimal_approaches_birdie(optimal_results, hole,
    title=f"Aggressive (Birdie-Maximising) Strategy, N={N} | tee: {best['club']} {best['aim_yards']:+.0f}y",
    output_path=OUT / f"BirdieOddsOptimal_regenerated_N{N}.png")

# overlay the tee shot star on top of the same figure
fig, ax = plt.subplots(figsize=(14, 16))
from core_birdie import _plot_hole_layout, CLUB_STYLES
_plot_hole_layout(hole, f"Aggressive (Birdie-Maximising) Strategy, N={N}", ax)
xs = [r["start"][0] for r in optimal_results]
ys = [r["start"][1] for r in optimal_results]
probs = [r["mean_birdie_prob"] for r in optimal_results]
face_colors = [CLUB_STYLES.get(r["club"], {"color": "#999999"})["color"] for r in optimal_results]
sc = ax.scatter(xs, ys, c=face_colors, s=25, alpha=0.85, zorder=20)
angle = float(np.degrees(np.arctan(best["aim_yards"] / total_dist))) if total_dist > 0 else 0.0
mu = hole.club_distributions[best["club"]]["mean"]
land = rotation_translator(float(mu[0]), float(mu[1]), angle, tee, pin)
ax.scatter([land[0]], [land[1]], color="red", s=200, marker="*", zorder=100, label="Best Tee Shot")
ax.text(land[0] + 2, land[1] + 2, f"{best['club']}, {best['aim_yards']:+.0f}y", fontsize=10, color="red", zorder=101)
ax.legend(loc="upper right")
fig.savefig(OUT / f"BirdieOddsOptimal_with_tee_N{N}.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}/BirdieOddsOptimal_with_tee_N{N}.png")
print("DONE")
