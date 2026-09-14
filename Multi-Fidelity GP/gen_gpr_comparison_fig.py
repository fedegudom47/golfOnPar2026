"""Two GPR fits, stacked: (1) a regular single-fidelity GPR fit to the
low-fidelity simulated ESHO grid (approach_N0300.csv, same data as
fig1_low_fidelity), and (2) the Kennedy-O'Hagan multi-fidelity fusion of
that same low-fidelity data with the high-fidelity observed strokes
(observed_esho_data.csv) — reusing the exact model classes and fitting
functions from multifidelity_strategic_blanket.py, just pointed at this
session's low-fidelity CSV instead of its own bundled one.

Course/tee/pin drawn with this session's standing conventions: translucent
fill, black-square tee, white/black-edge pin (drawn last), x_1/x_2 axes,
whole scene rotated 90 deg clockwise. Colour scales are independent between
panels (different quantities: ESHO vs. observed strokes). Training-point
dots differentiate low-fidelity vs. high-fidelity data by marker, not by
the value colour scale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import gpytorch
from shapely import wkt as shapely_wkt
from shapely.affinity import affine_transform

from multifidelity_strategic_blanket import (
    _LowFidelityGPR, _DiscrepancyGP,
    make_sim_surface, make_fused_surface, fit_discrepancy_gp,
    GP_LF_LS, GP_LF_LR, GP_LF_ITER,
    OBSERVED_DATA_FILE, _DATA,
)
from core import build_hole, _LIE_COLORS  # noqa: E402

HERE = Path(__file__).parent
LOW_FIDELITY_CSV = (
    HERE.parent / "Parallelisation" / "convergence" / "outputs"
    / "full_hole_paper" / "approach_N0300.csv"
)

ROT = [0, 1, -1, 0, 0, 0]  # 90 deg clockwise: (x, y) -> (y, -x)


def rot_pt(pt):
    return (pt[1], -pt[0])


# ---------------------------------------------------------------------------
# 1. Fit f_sim to THIS session's low-fidelity grid (not the module's own CSV)
# ---------------------------------------------------------------------------
low = pd.read_csv(LOW_FIDELITY_CSV)
X_low = torch.tensor(low[["x", "y"]].values, dtype=torch.float32)
y_low = torch.tensor(low["esho_mean"].values, dtype=torch.float32)

lik_low = gpytorch.likelihoods.GaussianLikelihood()
model_low = _LowFidelityGPR(X_low, y_low, lik_low)
model_low.train(); lik_low.train()
opt = torch.optim.Adam(model_low.parameters(), lr=GP_LF_LR)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik_low, model_low)
print(f"Training f_sim on {LOW_FIDELITY_CSV.name} ({len(low)} pts, {GP_LF_ITER} iter)...")
for i in range(GP_LF_ITER):
    opt.zero_grad()
    loss = -mll(model_low(X_low), y_low)
    loss.backward()
    opt.step()
model_low.eval(); lik_low.eval()
f_sim_fn = make_sim_surface(model_low, lik_low)

# ---------------------------------------------------------------------------
# 2. Fuse with the high-fidelity observed data (module's own function —
#    reads OBSERVED_DATA_FILE = observed_esho_data.csv unmodified)
# ---------------------------------------------------------------------------
print(f"\nFitting KOH discrepancy GP against {OBSERVED_DATA_FILE.name}...")
delta_model, delta_lik = fit_discrepancy_gp(f_sim_fn)
f_fused_fn = make_fused_surface(delta_model, delta_lik)

high = pd.read_csv(OBSERVED_DATA_FILE)

# ---------------------------------------------------------------------------
# 3. Course geometry (for translucent underlay)
# ---------------------------------------------------------------------------
hole = build_hole(_DATA, gp_training_iter=10)

# ---------------------------------------------------------------------------
# 4. Evaluate both surfaces on a shared grid spanning both datasets' extent
# ---------------------------------------------------------------------------
x_range = np.linspace(-60, 90, 150)
y_range = np.linspace(50, 400, 350)
X, Y = np.meshgrid(x_range, y_range)
grid_t = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

Z_sim = f_sim_fn(grid_t).reshape(X.shape)
Z_fused = f_fused_fn(grid_t).reshape(X.shape)

# Rotated meshgrid for plotting: (x, y) -> (y, -x)
X_rot, Y_rot = Y, -X

# ---------------------------------------------------------------------------
# 5. Plot: stacked, rotated 90 deg clockwise, translucent course, x1/x2 axes
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
fig.subplots_adjust(hspace=0.35)


def draw_course(ax):
    for _, row in hole.hole_9.iterrows():
        geom = affine_transform(shapely_wkt.loads(row["WKT"]), ROT)
        color = _LIE_COLORS.get(row["lie"], "lightgrey")
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.18, fc=color, ec="black", linewidth=0.5, zorder=1)
    for _, row in hole.new_fairway.iterrows():
        x, y = affine_transform(row["geometry"], ROT).exterior.xy
        ax.fill(x, y, alpha=0.18, fc=_LIE_COLORS["new_fairway"], ec="black", linewidth=0.5, zorder=1)
    for _, row in hole.new_hazard3.iterrows():
        x, y = affine_transform(row["geometry"], ROT).exterior.xy
        ax.fill(x, y, alpha=0.18, fc=_LIE_COLORS["new_hazard3"], ec="black", linewidth=0.5, zorder=1)


def draw_markers(ax):
    ox, oy = rot_pt(hole.tee_point)
    px, py = rot_pt(hole.hole)
    ax.plot(ox, oy, marker="s", color="black", markersize=9, linestyle="None",
            label="Tee", zorder=30)
    ax.plot(px, py, "o", markersize=8, label="Pin", zorder=30,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.3)


# ── Panel 1: single-fidelity GPR on the low-fidelity grid alone ────────────
draw_course(ax1)
cm1 = ax1.contourf(X_rot, Y_rot, Z_sim, levels=30, cmap="viridis_r", alpha=0.72, zorder=2)
fig.colorbar(cm1, ax=ax1, fraction=0.035, pad=0.02, label="$f_{sim}$ — ESHO (strokes)")

low_rot = np.array([rot_pt(p) for p in low[["x", "y"]].values])
ax1.scatter(low_rot[:, 0], low_rot[:, 1], s=14, facecolor="white", edgecolor="black",
            linewidths=0.4, alpha=0.85, zorder=10, label="Low-fidelity training pts")
draw_markers(ax1)
ax1.set_aspect("equal")
ax1.set_title(r"Single-fidelity GPR — $f_{sim}$ fit to $\mathcal{D}_L$ alone")
ax1.set_xlabel("$x_1$ (yards)"); ax1.set_ylabel("$x_2$ (yards)")
ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)

# ── Panel 2: multi-fidelity fused GPR (KOH), trained on D_L + D_H ──────────
draw_course(ax2)
cm2 = ax2.contourf(X_rot, Y_rot, Z_fused, levels=30, cmap="viridis_r", alpha=0.72, zorder=2)
fig.colorbar(cm2, ax=ax2, fraction=0.035, pad=0.02, label=r"$f_{fused}=\rho f_{sim}+\delta$ (strokes)")

high_rot = np.array([rot_pt(p) for p in high[["x1", "x2"]].values])
ax2.scatter(low_rot[:, 0], low_rot[:, 1], s=14, facecolor="white", edgecolor="black",
            linewidths=0.4, alpha=0.7, zorder=10, label="Low-fidelity ($\\mathcal{D}_L$)")
ax2.scatter(high_rot[:, 0], high_rot[:, 1], s=16, marker="^", facecolor="black",
            edgecolor="white", linewidths=0.4, alpha=0.9, zorder=11,
            label="High-fidelity ($\\mathcal{D}_H$)")
draw_markers(ax2)
ax2.set_aspect("equal")
ax2.set_title(r"Multi-fidelity GPR (KOH) — fused on $\mathcal{D}_L$ + $\mathcal{D}_H$")
ax2.set_xlabel("$x_1$ (yards)"); ax2.set_ylabel("$x_2$ (yards)")
ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)

out = HERE / "gpr_comparison_stacked.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
