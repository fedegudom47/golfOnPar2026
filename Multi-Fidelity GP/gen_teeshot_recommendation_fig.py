"""Optimal tee-shot recommendation, N=1000, for two objectives, on one plot:
  - ESHO under the multi-fidelity (KOH) fused surface f_fused = rho*f_sim + delta
    (reusing multifidelity_strategic_blanket.py's optimise_tee_shot unmodified)
  - P(birdie), maximised against _BirdieApproachGPR
    (reusing run_hpc_sensitivity_birdie.py's _evaluate_tee_shot_birdie unmodified)

Each recommendation is drawn as a simple arrow from the tee: direction from
the recommended aimpoint, length = the recommended club's mean carry (not
the simulated landing spot) — i.e. where that club "typically" ends up.
Annotated with objective, club, and aimpoint. Standard translucent course.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gpytorch
from shapely import wkt as shapely_wkt
from shapely.affinity import affine_transform

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Parallelisation" / "convergence"))
sys.path.insert(0, str(ROOT / "Parallelisation" / "convergence_birdie"))
sys.path.insert(0, str(ROOT / "Parallelisation" / "sensitivity"))

import core  # noqa: E402
import core_birdie as cb  # noqa: E402
from run_hpc_sensitivity_birdie import (  # noqa: E402
    _BirdieApproachGPR, _fit_birdie_approach_gpr, _evaluate_tee_shot_birdie,
)
from multifidelity_strategic_blanket import (  # noqa: E402
    _LowFidelityGPR, fit_discrepancy_gp, make_sim_surface, make_fused_surface,
    optimise_tee_shot, GP_LF_LR, GP_LF_ITER, OBSERVED_DATA_FILE,
)

ROT = [0, 1, -1, 0, 0, 0]  # 90 deg clockwise: (x, y) -> (y, -x)
N_SIMULATIONS = 1000
AIMPOINTS = list(range(-40, 44, 4))


def rot_pt(pt):
    return (pt[1], -pt[0])


# ---------------------------------------------------------------------------
# 1. ESHO, multi-fidelity fused surface
# ---------------------------------------------------------------------------
hole = core.build_hole(gp_training_iter=100)

low_csv = ROOT / "Parallelisation" / "convergence" / "outputs" / "full_hole_paper" / "approach_N1000.csv"
low = pd.read_csv(low_csv)
X_low = torch.tensor(low[["x", "y"]].values, dtype=torch.float32)
y_low = torch.tensor(low["esho_mean"].values, dtype=torch.float32)
lik_low = gpytorch.likelihoods.GaussianLikelihood()
model_low = _LowFidelityGPR(X_low, y_low, lik_low)
model_low.train(); lik_low.train()
opt = torch.optim.Adam(model_low.parameters(), lr=GP_LF_LR)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik_low, model_low)
print(f"Fitting f_sim on {low_csv.name} ({len(low)} pts)...")
for _ in range(GP_LF_ITER):
    opt.zero_grad()
    loss = -mll(model_low(X_low), y_low)
    loss.backward()
    opt.step()
model_low.eval(); lik_low.eval()
f_sim_fn = make_sim_surface(model_low, lik_low)

print(f"Fitting KOH discrepancy GP against {OBSERVED_DATA_FILE.name}...")
delta_model, delta_lik = fit_discrepancy_gp(f_sim_fn)
f_fused_fn = make_fused_surface(delta_model, delta_lik)

print(f"Optimising ESHO tee shot (multi-fidelity), N={N_SIMULATIONS}...")
np.random.seed(0)
best_esho, _ = optimise_tee_shot(
    hole=hole, evaluation_surface=f_fused_fn, shot_profiles=hole.club_distributions,
    clubs=list(hole.club_distributions.keys()), aimpoints=AIMPOINTS, N=N_SIMULATIONS,
)
print("  ESHO best:", best_esho)

# ---------------------------------------------------------------------------
# 2. Birdie probability, approach GPR
# ---------------------------------------------------------------------------
hole_b = cb.build_hole_birdie(gp_training_iter=10)
grid_csv = ROOT / "Parallelisation" / "convergence_birdie" / "outputs_single_run" / "seed0009_N0300_birdie.csv"
grid_b = pd.read_csv(grid_csv)
optimal_results = [{"start": (r.x, r.y), "mean_birdie_prob": r.mean_birdie_prob}
                    for r in grid_b.itertuples()]
model_b, lik_b = _fit_birdie_approach_gpr(optimal_results, gp_training_iter=100)

print(f"Optimising birdie tee shot, N={N_SIMULATIONS}...")
np.random.seed(0)
best_birdie, _ = _evaluate_tee_shot_birdie(
    hole=hole_b, approach_model=model_b, approach_likelihood=lik_b,
    aim_range=(min(AIMPOINTS), max(AIMPOINTS)), aim_step=4.0, n_samples=N_SIMULATIONS,
)
print("  Birdie best:", best_birdie)


# ---------------------------------------------------------------------------
# 3. Plot: one figure, both recommendations as simple arrows from the tee
# ---------------------------------------------------------------------------
def arrow_endpoint(tee, pin, club, aim_offset, club_distributions):
    total_dist = float(np.linalg.norm(np.array(pin) - np.array(tee)))
    angle_deg = float(np.degrees(np.arctan(aim_offset / total_dist)))
    mean_carry = float(club_distributions[club]["mean"][1])
    return core.rotation_translator(0.0, mean_carry, angle_deg, tee, pin)


tee, pin = hole.tee_point, hole.hole
esho_end = arrow_endpoint(tee, pin, best_esho["optimal_club"],
                           best_esho["optimal_aimpoint_yards"], hole.club_distributions)
birdie_end = arrow_endpoint(tee, pin, best_birdie["club"],
                             best_birdie["aim_offset"], hole_b.club_distributions)

fig, ax = plt.subplots(figsize=(11, 6))

for _, row in hole.hole_9.iterrows():
    geom = affine_transform(shapely_wkt.loads(row["WKT"]), ROT)
    color = core._LIE_COLORS.get(row["lie"], "lightgrey")
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.18, fc=color, ec="black", linewidth=0.5, zorder=0)
for _, row in hole.new_fairway.iterrows():
    x, y = affine_transform(row["geometry"], ROT).exterior.xy
    ax.fill(x, y, alpha=0.18, fc=core._LIE_COLORS["new_fairway"], ec="black", linewidth=0.5, zorder=0)
for _, row in hole.new_hazard3.iterrows():
    x, y = affine_transform(row["geometry"], ROT).exterior.xy
    ax.fill(x, y, alpha=0.18, fc=core._LIE_COLORS["new_hazard3"], ec="black", linewidth=0.5, zorder=0)

ob_lo, ob_hi = sorted([-hole.ob_x_left, -hole.ob_x_right])
ob_far = hole.ob_y_far
for v in (ob_lo, ob_hi):
    ax.axhline(v, color="firebrick", linestyle="--", linewidth=1, zorder=2)
ax.axvline(ob_far, color="firebrick", linestyle="--", linewidth=1, zorder=2)
ax.axhspan(ob_hi, ob_hi + 20, color="lightcoral", alpha=0.25, zorder=1)
ax.axhspan(ob_lo - 20, ob_lo, color="lightcoral", alpha=0.25, zorder=1)
ax.axvspan(ob_far, ob_far + 20, color="lightcoral", alpha=0.25, zorder=1)

tx, ty = rot_pt(tee)
px, py = rot_pt(pin)

recs = [
    ("ESHO (multi-fidelity)", esho_end, "#0072B2", "-",
     best_esho["optimal_club"], best_esho["optimal_aimpoint_yards"],
     f"E[strokes] = {best_esho['risk_at_optimum']:.2f}"),
    ("P(birdie)", birdie_end, "#E69F00", "--",
     best_birdie["club"], best_birdie["aim_offset"],
     f"P(birdie) = {best_birdie['mean_birdie_prob']:.2f}"),
]
label_offsets = [(0, 18), (0, -28)]  # ESHO label above its arrow, birdie below
for (label, end, color, linestyle, club, aim, metric), (ox, oy) in zip(recs, label_offsets):
    ex, ey = rot_pt(end)
    ax.annotate("", xy=(ex, ey), xytext=(tx, ty),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.5,
                                 linestyle=linestyle, mutation_scale=22), zorder=20)
    mx, my = (tx + ex) / 2, (ty + ey) / 2
    ax.annotate(f"{label}\n{club}, aim {aim:+.0f}y\n{metric}", (mx, my), fontsize=9, fontweight="bold",
                color=color, ha="center", va="center",
                xytext=(ox, oy), textcoords="offset points", zorder=21)

ax.plot(tx, ty, marker="s", color="black", markersize=10, linestyle="None", zorder=30, label="Tee")
ax.plot(px, py, "o", markersize=10, zorder=30, label="Hole",
        markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4)

ax.set_xlabel("$x_2$ (yards)")
ax.set_ylabel("$x_1$ (yards)")
ax.set_aspect("equal")
ax.grid(True, linestyle=":")
ax.set_title(f"Optimal tee shot, N={N_SIMULATIONS}: ESHO (multi-fidelity) vs. birdie probability")
ax.set_xlim(tx - 20, ob_far + 20)
ax.set_ylim(ob_lo - 20, ob_hi + 20)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

out = ROOT / "TowardsEnd" / "On_Par" / "images" / "teeshot_recommendation_esho_vs_birdie.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
