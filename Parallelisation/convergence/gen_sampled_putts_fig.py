"""Regenerate sampledPutts.png: the actual putt-training data
(Parallelisation/data/gpr_green_dataset.csv — the same 156 samples core.py's
build_hole trains the putt GPR on), rotated 90 deg clockwise, translucent
course fill, and this session's standing tee/pin marker + discrete-colour
conventions. Only 3 putt colours (no 4-putt category exists in the data).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.affinity import affine_transform

import core

ROT = [0, 1, -1, 0, 0, 0]  # 90 deg clockwise: (x, y) -> (y, -x)


def rot_pt(pt):
    return (pt[1], -pt[0])


# Exact hex the user supplied, matching STROKE_COLORS' first three entries
# in final images/course_overlay_variants.py — no "4 putts" here since none
# exist in this dataset.
PUTT_COLORS = {1: "#349d34", 2: "#92d3e3", 3: "#ffa503"}

data_dir = core._DEFAULT_DATA_DIR
putts = pd.read_csv(data_dir / "gpr_green_dataset.csv")

hole_9_raw = pd.read_csv(data_dir / "hole_9_data.csv")
hole_9_raw["geometry"] = hole_9_raw["WKT"].apply(shapely_wkt.loads)
pin = (5.0, 174.0)  # matches the raw/unshifted frame this dataset is already in

fig, ax = plt.subplots(figsize=(8, 7))

for _, row in hole_9_raw.iterrows():
    geom = affine_transform(row["geometry"], ROT)
    color = core._LIE_COLORS.get(row["lie"], "lightgrey")
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.18, fc=color, ec="black", linewidth=0.5, zorder=0)

pts = putts[["x", "y"]].values
rot_xy = [rot_pt(p) for p in pts]
rx = [p[0] for p in rot_xy]
ry = [p[1] for p in rot_xy]
colors = [PUTT_COLORS[int(n)] for n in putts["simulated_strokes"]]
ax.scatter(rx, ry, c=colors, s=45, edgecolors="black", linewidths=0.6, alpha=0.85, zorder=10)

px, py = rot_pt(pin)
ax.plot(px, py, "o", markersize=9, zorder=30,
        markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4)

present = sorted(putts["simulated_strokes"].unique())
legend_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor=PUTT_COLORS[n], markeredgecolor="black",
               label=f"{n} putt" + ("" if n == 1 else "s"))
    for n in present
]
legend_handles.append(
    plt.Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor="white", markeredgecolor="black", label="Hole")
)
ax.legend(handles=legend_handles, title="Putts", loc="best", framealpha=0.9)

ax.set_title("Simulated Putts")
ax.set_xlabel("$x_1$ (yards)")
ax.set_ylabel("$x_2$ (yards)")
ax.set_aspect("equal")
ax.grid(True, linestyle=":")

margin = 8
ax.set_xlim(min(rx) - margin, max(rx) + margin)
ax.set_ylim(min(ry) - margin, max(ry) + margin)

out = "../../untitled folder/On_Par/images/sampledPutts.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
