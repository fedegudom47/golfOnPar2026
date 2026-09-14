"""Course-view shot-outcome figure for the paper (replaces shot_outcomes1/2).

Same settings as the original par-3 notebook version (Hybrid, aim +10y / -10y
from the tee, N=300, mean ESHO printed per panel), but:
  - both panels stacked (aim +10 on top, -10 below), sharing one colour scale
  - colour scale direction reversed
  - whole scene rotated 90 deg clockwise: (x, y) -> (y, -x)
  - axes labelled x_1 / x_2 (yards)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shapely import wkt as shapely_wkt
from shapely.affinity import affine_transform

import core

CLUB = "Hybrid"
AIMS = [10.0, -10.0]
N = 300
np.random.seed(0)

# 90-deg clockwise rotation about the origin: (x, y) -> (y, -x)
ROT = [0, 1, -1, 0, 0, 0]  # shapely affine_transform matrix [a,b,d,e,xoff,yoff]


def rot_pt(pt):
    x, y = pt
    return (y, -x)


hole = core.build_hole(gp_training_iter=100)
origin = hole.tee_point
target = hole.hole
total_distance = float(np.linalg.norm(np.array(target) - np.array(origin)))

starting_lie = core.get_lie_category(origin, hole)
ob_lie = {"bunker": "sand"}.get(starting_lie, starting_lie)
if ob_lie not in hole.broadie_interpolators:
    ob_lie = "rough"
ob_value = core.evaluate_broadie(origin, target, ob_lie, hole.broadie_interpolators) + 1.0

mu = hole.club_distributions[CLUB]["mean"]
cov = hole.club_distributions[CLUB]["cov"]


def simulate(aim_offset):
    angle_deg = float(np.degrees(np.arctan(aim_offset / total_distance)))
    samples = np.random.multivariate_normal(mu, cov, size=N)
    landings, esho = [], []
    for shot in samples:
        lp = core.rotation_translator(float(shot[0]), float(shot[1]), angle_deg, origin, target)
        landings.append(lp)
        if core.is_out_of_bounds(lp, hole):
            cost = ob_value + 1.0
        else:
            es = core.evaluate_shot(lp, origin, target, hole)
            cost = es + 1.0 if not np.isnan(es) else np.nan
        esho.append(cost)
    return np.array(landings), np.array(esho), angle_deg


results = [(aim, *simulate(aim)) for aim in AIMS]
all_esho = np.concatenate([r[2] for r in results])
vmin, vmax = np.nanmin(all_esho), np.nanmax(all_esho)

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.35)
cmap = "viridis_r"

for ax, (aim, landings, esho, angle_deg) in zip(axes, results):
    # -- course geometry, rotated ------------------------------------------
    for _, row in hole.hole_9.iterrows():
        geom = affine_transform(shapely_wkt.loads(row["WKT"]), ROT)
        color = core._LIE_COLORS.get(row["lie"], "lightgrey")
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, ec="black", linewidth=0.5)
    for _, row in hole.new_fairway.iterrows():
        x, y = affine_transform(row["geometry"], ROT).exterior.xy
        ax.fill(x, y, alpha=0.5, fc=core._LIE_COLORS["fairway"], ec="black", linewidth=0.5)
    for _, row in hole.new_hazard3.iterrows():
        x, y = affine_transform(row["geometry"], ROT).exterior.xy
        ax.fill(x, y, alpha=0.5, fc=core._LIE_COLORS["water_hazard"], ec="black", linewidth=0.5)

    # -- OB shading, rotated (vertical bands -> horizontal; far band -> right) --
    # constant old-x lines (lateral OB) -> horizontal lines at new_y = -old_x
    rx_lo, rx_hi = -hole.ob_x_left, -hole.ob_x_right
    # constant old-y line (far OB) -> vertical line at new_x = old_y
    r_far = hole.ob_y_far
    ax.axhline(rx_lo, color="firebrick", linestyle="--", linewidth=1, zorder=2)
    ax.axhline(rx_hi, color="firebrick", linestyle="--", linewidth=1, zorder=2)
    ax.axvline(r_far, color="firebrick", linestyle="--", linewidth=1, zorder=2)

    # -- tee, pin, aim line, rotated ------------------------------------------
    ox, oy = rot_pt(origin)
    tx, ty = rot_pt(target)
    ax.plot(ox, oy, "rx", markersize=8, label="Tee")
    ax.plot(tx, ty, "ko", markersize=6, label="Hole")

    angle_rad = np.radians(angle_deg)
    direction = np.array(target) - np.array(origin)
    unit_dir = direction / np.linalg.norm(direction)
    aim_dir = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)],
    ]) @ np.array([0, 1])
    aim_vec_global = np.array([
        unit_dir[0] * aim_dir[1] - unit_dir[1] * aim_dir[0],
        unit_dir[1] * aim_dir[1] + unit_dir[0] * aim_dir[0],
    ])
    aim_end = np.array(origin) + aim_vec_global * total_distance * 1.15
    ax_e, ay_e = rot_pt(tuple(aim_end))
    ax.plot([ox, ax_e], [oy, ay_e], "r--", linewidth=1.5, label="Aim Line", zorder=3)

    # -- landings, coloured by ESHO -------------------------------------------
    rl = np.array([rot_pt(p) for p in landings])
    sc = ax.scatter(rl[:, 0], rl[:, 1], c=esho, cmap=cmap, vmin=vmin, vmax=vmax,
                     s=28, edgecolors="k", linewidths=0.3, zorder=10)

    mean_esho = float(np.nanmean(esho))
    ax.set_title(f"{CLUB} | Aim {aim:+.0f} yd ({angle_deg:+.1f}°) | Mean ESHO: {mean_esho:.2f}")
    ax.set_ylabel("$x_2$ (yards)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("$x_1$ (yards)")
fig.colorbar(sc, ax=axes, label="Expected Strokes to Hole Out", fraction=0.04, pad=0.02)

out = "../../untitled folder/On_Par/images/shot_outcomes_stacked.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
