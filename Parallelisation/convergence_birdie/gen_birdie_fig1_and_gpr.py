"""Birdie-probability analogue of fig1_low_fidelity_course_translucent.png,
plus the corresponding fitted GPR (reusing _BirdieApproachGPR /
_fit_birdie_approach_gpr from run_hpc_sensitivity_birdie.py — the actual
pipeline used elsewhere for this exact (x,y) -> mean_birdie_prob fit, not a
new one invented for this figure).

Two colourways, each producing its own pair of separate files (simulated
grid, and fitted GPR + white training-point dots matching GPRPar4.png's
convention):
  - viridis (regular direction — high P(birdie) = yellow = good)
  - plasma — a genuinely different palette, not a viridis variant

Standing conventions: translucent course, black-square tee, white/black-edge
hole marker drawn last, x_2/x_1 axes, OB boundary flush to the edges,
rotated 90 deg clockwise. N=300, OB-aware (core_birdie.py's build_hole
already includes the OB fix).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import torch
from shapely import wkt as shapely_wkt
from shapely.affinity import affine_transform

import core_birdie as cb

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "sensitivity"))
from run_hpc_sensitivity_birdie import _BirdieApproachGPR, _fit_birdie_approach_gpr  # noqa: E402

ROT = [0, 1, -1, 0, 0, 0]  # 90 deg clockwise: (x, y) -> (y, -x)
N_SIMULATIONS = 1000
GRID_CSV = HERE / "outputs_single_run" / "seed0009_N1000_birdie.csv"


def rot_pt(pt):
    return (pt[1], -pt[0])


hole = cb.build_hole_birdie(gp_training_iter=10)
grid = pd.read_csv(GRID_CSV)
print(f"{len(grid)} grid points, N={grid['N'].iloc[0]}")

# Same idea as fig1_low_fidelity's plot_low_fidelity: the downrange (y) rows
# are closely spaced relative to the horizontal axis, so labels collide
# unless we thin them for the text-grid panel. Every other row still left
# adjacent labels touching, so keep every 3rd instead.
y_values = np.sort(grid["y"].unique())
keep_y = y_values[::3]
grid_text = grid[grid["y"].isin(keep_y)].reset_index(drop=True)

# Reuse the actual pipeline's GPR fit, not a new one.
optimal_results = [{"start": (r.x, r.y), "mean_birdie_prob": r.mean_birdie_prob}
                    for r in grid.itertuples()]
model, likelihood = _fit_birdie_approach_gpr(optimal_results, gp_training_iter=100)

x_range = np.linspace(-55, 75, 150)
y_range = np.linspace(45, 300, 260)
Xg, Yg = np.meshgrid(x_range, y_range)
query = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32)
with torch.no_grad():
    Zg = likelihood(model(query)).mean.numpy().reshape(Xg.shape)
X_rot, Y_rot = Yg, -Xg


def draw_course(ax):
    for _, row in hole.hole_9.iterrows():
        geom = affine_transform(shapely_wkt.loads(row["WKT"]), ROT)
        color = cb._LIE_COLORS.get(row["lie"], "lightgrey")
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.18, fc=color, ec="black", linewidth=0.5, zorder=0)
    for _, row in hole.new_fairway.iterrows():
        x, y = affine_transform(row["geometry"], ROT).exterior.xy
        ax.fill(x, y, alpha=0.18, fc=cb._LIE_COLORS["new_fairway"], ec="black", linewidth=0.5, zorder=0)
    for _, row in hole.new_hazard3.iterrows():
        x, y = affine_transform(row["geometry"], ROT).exterior.xy
        ax.fill(x, y, alpha=0.18, fc=cb._LIE_COLORS["new_hazard3"], ec="black", linewidth=0.5, zorder=0)


def draw_ob(ax):
    ob_lo, ob_hi = sorted([-hole.ob_x_left, -hole.ob_x_right])
    ob_far = hole.ob_y_far
    for v in (ob_lo, ob_hi):
        ax.axhline(v, color="firebrick", linestyle="--", linewidth=1, zorder=2)
    ax.axvline(ob_far, color="firebrick", linestyle="--", linewidth=1, zorder=2)
    ax.axhspan(ob_hi, ob_hi + 20, color="lightcoral", alpha=0.25, zorder=1)
    ax.axhspan(ob_lo - 20, ob_lo, color="lightcoral", alpha=0.25, zorder=1)
    ax.axvspan(ob_far, ob_far + 20, color="lightcoral", alpha=0.25, zorder=1)
    return ob_lo, ob_hi, ob_far


def draw_markers(ax):
    tx, ty = rot_pt(hole.tee_point)
    px, py = rot_pt(hole.hole)
    ax.plot(tx, ty, marker="s", color="black", markersize=9, linestyle="None", zorder=30, label="Tee")
    ax.plot(px, py, "o", markersize=8, zorder=30, label="Hole",
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.3)
    return tx


def _fmt(p: float) -> str:
    """2 sig figs, but an exact/rounds-to-zero probability just shows "0"."""
    return "0" if round(p, 2) == 0 else f"{p:.2f}"


def _finish_ax(ax, tx, ob_lo, ob_hi, ob_far):
    ax.set_xlabel("$x_2$ (yards)")
    ax.set_ylabel("$x_1$ (yards)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":")
    ax.set_xlim(tx - 20, ob_far + 20)
    ax.set_ylim(ob_lo - 20, ob_hi + 20)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def make_grid_figure(cmap_name: str, out_name: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = grid["mean_birdie_prob"].min(), grid["mean_birdie_prob"].max()
    norm = plt.Normalize(vmin, vmax)

    # plasma's darker end reads poorly without help; give it a black outline.
    text_effects = [pe.withStroke(linewidth=0.9, foreground="black")] if cmap_name == "plasma" else None

    draw_course(ax)
    for r in grid_text.itertuples():
        rx, ry = rot_pt((r.x, r.y))
        ax.text(rx, ry, _fmt(r.mean_birdie_prob), ha="center", va="center",
                 fontsize=11, fontweight="bold", color=cmap(norm(r.mean_birdie_prob)), zorder=10,
                 path_effects=text_effects)
    ob_lo, ob_hi, ob_far = draw_ob(ax)
    tx = draw_markers(ax)
    ax.set_title("Simulated P(birdie)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="P(birdie)")
    _finish_ax(ax, tx, ob_lo, ob_hi, ob_far)

    out = HERE.parent.parent / "TowardsEnd" / "On_Par" / "images" / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


def make_gpr_figure(cmap_name: str, out_name: str):
    fig, ax = plt.subplots(figsize=(11, 6))

    draw_course(ax)
    cm = ax.contourf(X_rot, Y_rot, Zg, levels=30, cmap=cmap_name, alpha=0.85, zorder=2)
    grid_rot = np.array([rot_pt(p) for p in grid[["x", "y"]].values])
    ax.scatter(grid_rot[:, 0], grid_rot[:, 1], facecolor="white", edgecolor="black",
               linewidths=0.5, s=22, zorder=9, label="Grid points")
    ob_lo, ob_hi, ob_far = draw_ob(ax)
    tx = draw_markers(ax)
    ax.set_title("GPR fit")
    fig.colorbar(cm, ax=ax, fraction=0.03, pad=0.02, label="Predicted P(birdie)")
    _finish_ax(ax, tx, ob_lo, ob_hi, ob_far)

    out = HERE.parent.parent / "TowardsEnd" / "On_Par" / "images" / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


def make_combined_figure(cmap_name: str, out_name: str):
    """Grid (top) + GPR (bottom) stacked in one figure, sharing one colour
    scale spanning both the raw simulated values and the fitted surface, so
    a colour means the same thing in both panels."""
    cmap = plt.get_cmap(cmap_name)
    vmin = min(grid["mean_birdie_prob"].min(), Zg.min())
    vmax = max(grid["mean_birdie_prob"].max(), Zg.max())
    norm = plt.Normalize(vmin, vmax)
    text_effects = [pe.withStroke(linewidth=0.9, foreground="black")] if cmap_name == "plasma" else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.5))
    fig.subplots_adjust(hspace=0.12)

    # -- top: raw simulated grid ---------------------------------------------
    draw_course(ax1)
    for r in grid_text.itertuples():
        rx, ry = rot_pt((r.x, r.y))
        ax1.text(rx, ry, _fmt(r.mean_birdie_prob), ha="center", va="center",
                  fontsize=11, fontweight="bold", color=cmap(norm(r.mean_birdie_prob)), zorder=10,
                  path_effects=text_effects)
    ob_lo, ob_hi, ob_far = draw_ob(ax1)
    tx = draw_markers(ax1)
    ax1.set_title("Simulated P(birdie)")

    # -- bottom: GPR fit + white training-point dots -------------------------
    draw_course(ax2)
    cm = ax2.contourf(X_rot, Y_rot, Zg, levels=30, cmap=cmap_name, norm=norm, alpha=0.85, zorder=2)
    grid_rot = np.array([rot_pt(p) for p in grid[["x", "y"]].values])
    ax2.scatter(grid_rot[:, 0], grid_rot[:, 1], facecolor="white", edgecolor="black",
                linewidths=0.5, s=22, zorder=9, label="Grid points")
    draw_ob(ax2)
    draw_markers(ax2)
    ax2.set_title("GPR fit")

    for ax in (ax1, ax2):
        _finish_ax(ax, tx, ob_lo, ob_hi, ob_far)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=(ax1, ax2), fraction=0.03, pad=0.02, label="P(birdie)")

    out = HERE.parent.parent / "TowardsEnd" / "On_Par" / "images" / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


for cmap_name in ("viridis", "plasma"):
    make_grid_figure(cmap_name, f"birdie_grid_{cmap_name}.png")
    make_gpr_figure(cmap_name, f"birdie_gpr_{cmap_name}.png")

make_combined_figure("plasma", "birdie_grid_and_gpr_plasma_stacked.png")
