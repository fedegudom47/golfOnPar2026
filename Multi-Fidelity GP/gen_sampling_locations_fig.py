"""Very simple annotated figure: WHERE the high-fidelity data's sampling
locations are (the named hotspot zones from generate_observed_data.py),
marked with their name and spatial extent (dimensions) — no observations
plotted, just the zones themselves on the course.

Standing conventions: translucent course, black-square tee, white/black-edge
hole marker drawn last, x_2/x_1 axes (horizontal/vertical, per this
session's fix), OB boundary outlined, rotated 90 deg clockwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "Parallelisation" / "convergence"))
from core import build_hole, _LIE_COLORS  # noqa: E402
from shapely import wkt as shapely_wkt
from shapely.affinity import affine_transform

from generate_observed_data import HOTSPOTS, PIN  # noqa: E402

ROT = [0, 1, -1, 0, 0, 0]  # 90 deg clockwise: (x, y) -> (y, -x)
DATA_DIR = HERE.parent / "Parallelisation" / "data"


def rot_pt(pt):
    return (pt[1], -pt[0])


hole = build_hole(DATA_DIR, gp_training_iter=10)

fig, ax = plt.subplots(figsize=(11, 6))

# Translucent course.
for _, row in hole.hole_9.iterrows():
    geom = affine_transform(shapely_wkt.loads(row["WKT"]), ROT)
    color = _LIE_COLORS.get(row["lie"], "lightgrey")
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.18, fc=color, ec="black", linewidth=0.5, zorder=0)
for _, row in hole.new_fairway.iterrows():
    x, y = affine_transform(row["geometry"], ROT).exterior.xy
    ax.fill(x, y, alpha=0.18, fc=_LIE_COLORS["new_fairway"], ec="black", linewidth=0.5, zorder=0)
for _, row in hole.new_hazard3.iterrows():
    x, y = affine_transform(row["geometry"], ROT).exterior.xy
    ax.fill(x, y, alpha=0.18, fc=_LIE_COLORS["new_hazard3"], ec="black", linewidth=0.5, zorder=0)

# OB boundary, flush to the edges.
ob_lo, ob_hi = sorted([-hole.ob_x_left, -hole.ob_x_right])
ob_far = hole.ob_y_far
for v in (ob_lo, ob_hi):
    ax.axhline(v, color="firebrick", linestyle="--", linewidth=1, zorder=2)
ax.axvline(ob_far, color="firebrick", linestyle="--", linewidth=1, zorder=2)
ax.axhspan(ob_hi, ob_hi + 20, color="lightcoral", alpha=0.25, zorder=1)
ax.axhspan(ob_lo - 20, ob_lo, color="lightcoral", alpha=0.25, zorder=1)
ax.axvspan(ob_far, ob_far + 20, color="lightcoral", alpha=0.25, zorder=1)

# Sampling-location zones: a 1-sigma ellipse (diameter = 2*sigma) at each
# named hotspot's centre, dimensions only, no observation dots. Label offsets
# are hand-placed (not a generic fan) so each leader line points unambiguously
# at its own zone in this tight cluster.
LABEL_OFFSETS = {
    "ditch":           (-45, 22),
    "left_rough_scar": (48, 16),
    "fairway_hollow":  (-58, 6),
    "heavy_rough":     (-16, -38),
    "hazard_fringe":   (38, -38),
}

for hs in HOTSPOTS:
    cx, cy = rot_pt((hs["cx"], hs["cy"]))
    width, height = 2 * hs["sigma_y"], 2 * hs["sigma_x"]  # rotated: x<->y swap
    ell = mpatches.Ellipse((cx, cy), width=width, height=height,
                            fc="darkorange", alpha=0.15, ec="darkorange", linewidth=1.8, zorder=5)
    ax.add_patch(ell)
    ax.plot(cx, cy, "+", color="darkorange", markersize=8, markeredgewidth=1.5, zorder=6)

    dx, dy = LABEL_OFFSETS[hs["name"]]
    ax.annotate(hs["name"], (cx, cy), ha="center", va="center", fontsize=10,
                fontweight="bold", color="darkorange", zorder=6,
                xytext=(dx, dy), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="darkorange", linewidth=1.1, alpha=0.9))

tee_r, ty = rot_pt(hole.tee_point)
px, py = rot_pt(hole.hole)
ax.plot(tee_r, ty, marker="s", color="black", markersize=10, linestyle="None", zorder=30, label="Tee")
ax.plot(px, py, "o", markersize=10, zorder=30, label="Hole",
        markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4)

ax.set_xlabel("$x_2$ (yards)")
ax.set_ylabel("$x_1$ (yards)")
ax.set_aspect("equal")
ax.grid(True, linestyle=":")
ax.legend(loc="upper right", fontsize=9)
ax.set_title("High-fidelity ($\\mathcal{D}_H$) sampling locations")

ax.set_xlim(tee_r - 20, ob_far + 20)
ax.set_ylim(ob_lo - 20, ob_hi + 20)

out = "../TowardsEnd/On_Par/images/sampling_locations.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
