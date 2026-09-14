"""Course-view shot-outcome figure, on the RAW (unshifted, un-extended) hole
geometry — matching the original 'simulating on a par 3' notebook's scale,
not core.py's shifted/extended Par-4 pipeline. Reuses core.py's already-
trained Broadie interpolators and putt GPR (distance/rotation invariant, or
frame-shift-compensated for the green) rather than retraining anything.

Settings: Hybrid, aim +10y / -10y, N=300, mean ESHO printed per panel.
Both panels stacked, sharing one (reversed) colour scale, rotated 90 deg
clockwise. Tee marked with a black square (no red-on-green). Yards only,
no degree annotation.
"""
from __future__ import annotations

import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.affinity import affine_transform
from shapely.geometry import Point

import core

CLUB = "Hybrid"
AIMS = [10.0, -10.0]
N = 300
Y_SHIFT = 160.0  # core.py's putt GPR was trained on hole_9 coords shifted +160y
np.random.seed(0)

ROT = [0, 1, -1, 0, 0, 0]  # 90 deg clockwise: (x, y) -> (y, -x)


def rot_pt(pt):
    x, y = pt
    return (y, -x)


# ---------------------------------------------------------------------------
# 1. Reusable pieces from core.py's already-trained pipeline (frame-agnostic:
#    Broadie interpolators are distance-based; the putt GPR needs a +160y
#    shift to match its training frame, handled in evaluate_shot_raw below).
# ---------------------------------------------------------------------------
trained = core.build_hole(gp_training_iter=100)

# ---------------------------------------------------------------------------
# 2. Raw (unshifted) hole geometry + tee/pin, exactly as the original notebook.
# ---------------------------------------------------------------------------
data_dir = core._DEFAULT_DATA_DIR
hole_9_raw = pd.read_csv(data_dir / "hole_9_data.csv")
hole_9_raw["geometry"] = hole_9_raw["WKT"].apply(shapely_wkt.loads)

teeboxes = hole_9_raw[hole_9_raw["lie"].str.contains("tee", case=False)].copy()
green_row = hole_9_raw[hole_9_raw["lie"] == "green"].iloc[0]
green_polygon = shapely_wkt.loads(green_row["WKT"])
green_centre = green_polygon.centroid.coords[0]

teeboxes["centroid"] = teeboxes["geometry"].apply(lambda g: g.centroid.coords[0])
teeboxes["dist_to_green"] = teeboxes["centroid"].apply(
    lambda pt: np.linalg.norm(np.array(pt) - np.array(green_centre)))
tee_point = teeboxes.loc[teeboxes["dist_to_green"].idxmax()]["centroid"]
pin = (5.0, 174.0)  # matches the original notebook's hardcoded pin

ns = types.SimpleNamespace(
    green_polygon=green_polygon,
    bunker_polygons=[shapely_wkt.loads(r.WKT) for r in hole_9_raw[hole_9_raw.lie == "bunker"].itertuples()],
    water_polygons=[shapely_wkt.loads(r.WKT) for r in hole_9_raw[hole_9_raw.lie == "water_hazard"].itertuples()],
    fairway_polygons=[shapely_wkt.loads(r.WKT) for r in hole_9_raw[hole_9_raw.lie == "fairway"].itertuples()],
)


def get_lie_category_raw(point):
    pt = Point(point)
    if ns.green_polygon.contains(pt):
        return "green"
    if any(p.contains(pt) for p in ns.water_polygons):
        return "water"
    if any(p.contains(pt) for p in ns.bunker_polygons):
        return "bunker"
    if any(p.contains(pt) for p in ns.fairway_polygons):
        return "fairway"
    return "rough"


def evaluate_shot_raw(point, starting_point, target):
    lie = get_lie_category_raw(point)
    if lie == "green":
        shifted = (point[0], point[1] + Y_SHIFT)
        return core.evaluate_on_green(shifted, trained.putt_model, trained.putt_likelihood)
    elif lie == "water":
        drop = core._get_water_drop(starting_point, point, ns.water_polygons)
        if drop is not None:
            return 1.0 + core.evaluate_broadie(drop, target, "rough", trained.broadie_interpolators)
        return float("nan")
    else:
        broadie_lie = {"bunker": "sand"}.get(lie, lie)
        return core.evaluate_broadie(point, target, broadie_lie, trained.broadie_interpolators)


# ---------------------------------------------------------------------------
# 3. Simulate
# ---------------------------------------------------------------------------
total_distance = float(np.linalg.norm(np.array(pin) - np.array(tee_point)))
mu = trained.club_distributions[CLUB]["mean"]
cov = trained.club_distributions[CLUB]["cov"]


def simulate(aim_offset):
    angle_deg = float(np.degrees(np.arctan(aim_offset / total_distance)))
    samples = np.random.multivariate_normal(mu, cov, size=N)
    landings, esho = [], []
    for shot in samples:
        lp = core.rotation_translator(float(shot[0]), float(shot[1]), angle_deg, tee_point, pin)
        landings.append(lp)
        es = evaluate_shot_raw(lp, tee_point, pin)
        esho.append(es + 1.0 if not np.isnan(es) else np.nan)
    return np.array(landings), np.array(esho)


results = [(aim, *simulate(aim)) for aim in AIMS]
all_esho = np.concatenate([r[2] for r in results])
vmin, vmax = np.nanmin(all_esho), np.nanmax(all_esho)

# ---------------------------------------------------------------------------
# 4. Plot: stacked, shared colour scale, rotated 90 deg clockwise
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3)
cmap = "viridis_r"

for ax, (aim, landings, esho) in zip(axes, results):
    for _, row in hole_9_raw.iterrows():
        geom = affine_transform(row["geometry"], ROT)
        color = core._LIE_COLORS.get(row["lie"], "lightgrey")
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, ec="black", linewidth=0.5)

    ox, oy = rot_pt(tee_point)
    px, py = rot_pt(pin)
    ax.plot(ox, oy, marker="s", color="black", markersize=8, linestyle="None", label="Tee")
    ax.plot(px, py, "ko", markersize=5, label="Hole", markerfacecolor="white", markeredgecolor="black")

    angle_deg = float(np.degrees(np.arctan(aim / total_distance)))
    angle_rad = np.radians(angle_deg)
    direction = np.array(pin) - np.array(tee_point)
    unit_dir = direction / np.linalg.norm(direction)
    aim_dir = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)],
    ]) @ np.array([0, 1])
    aim_vec_global = np.array([
        unit_dir[0] * aim_dir[1] - unit_dir[1] * aim_dir[0],
        unit_dir[1] * aim_dir[1] + unit_dir[0] * aim_dir[0],
    ])
    aim_end = np.array(tee_point) + aim_vec_global * total_distance * 1.15
    ax_e, ay_e = rot_pt(tuple(aim_end))
    ax.plot([ox, ax_e], [oy, ay_e], "r--", linewidth=1.5, label="Aim Line", zorder=3)

    rl = np.array([rot_pt(p) for p in landings])
    sc = ax.scatter(rl[:, 0], rl[:, 1], c=esho, cmap=cmap, vmin=vmin, vmax=vmax,
                     s=28, edgecolors="k", linewidths=0.3, zorder=10)

    mean_esho = float(np.nanmean(esho))
    ax.set_title(f"{CLUB} | Aim {aim:+.0f} yd | Mean ESHO: {mean_esho:.2f}")
    ax.set_ylabel("$x_2$ (yards)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("$x_1$ (yards)")
fig.colorbar(sc, ax=axes, label="Expected Strokes to Hole Out", fraction=0.045, pad=0.02)

out = "../../untitled folder/On_Par/images/shot_outcomes_stacked.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
