"""Curated D_L vs D_H comparison figure for the paper (replaces the 'dl vs dh.jpeg' sketch).

Reuses the exact panel-drawing logic from generate_observed_data.py's ax1 (true ESHO
surface, the low-fidelity simulation D_L) and ax3 (observed strokes, the high-fidelity
proxy D_H) — no new simulation, just a clean 2-panel figure built from data already on disk.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

from generate_observed_data import (
    PIN, HOTSPOTS, load_polygons, _draw_polygons, _lie_type,
    baseline_esho, difficulty_multiplier, OUTPUT_DIR,
)

# Same fixed integer-stroke palette as final images/course_overlay_variants.py —
# kept in sync by hand since that module lives in a space-containing path.
STROKE_COLORS = {
    2: "#349d34", 3: "#92d3e3", 4: "#ffa503", 5: "#ff0000", 6: "#a52a2a",
    7: "#800080", 8: "#4b0082", 9: "#2f4f4f", 10: "#000000",
}
_EXTRA_CMAP = plt.get_cmap("cool")


def stroke_color(n: int) -> str:
    if n in STROKE_COLORS:
        return STROKE_COLORS[n]
    return _EXTRA_CMAP((n - max(STROKE_COLORS)) / 5 % 1.0)


def draw_pin(ax):
    """Standing convention: white circle, black edge, drawn last (on top)."""
    ax.plot(*PIN, "o", markersize=8, label="Pin", zorder=30,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.3)


polygons = load_polygons()
df = pd.read_csv(OUTPUT_DIR / "observed_esho_data.csv")

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 6.5))

# ── D_L: true_esho simulated surface ────────────────────────────────────────
x_range = np.linspace(-60, 80, 160)
y_range = np.linspace(60, 390, 320)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
for i in range(Y.shape[0]):
    for j in range(X.shape[1]):
        pt = Point(X[i, j], Y[i, j])
        lie = _lie_type(pt, polygons)
        if lie == "Green":
            Z[i, j] = np.nan
        else:
            dist = float(np.linalg.norm(np.array([X[i, j], Y[i, j]]) - np.array(PIN)))
            Z[i, j] = baseline_esho(dist, lie) * difficulty_multiplier(X[i, j], Y[i, j])

cm = ax1.contourf(X, Y, Z, levels=30, cmap="viridis_r", alpha=0.85)
fig.colorbar(cm, ax=ax1, fraction=0.045, label="true ESHO (strokes)")
_draw_polygons(ax1, polygons, alpha=0.2)
for hs in HOTSPOTS:
    ax1.plot(hs["cx"], hs["cy"], "x", color="white", markersize=8, markeredgewidth=2, zorder=5)
draw_pin(ax1)
ax1.set_aspect("equal")
ax1.set_title(r"$\mathcal{D}_L$ — Simulated ESHO surface")
ax1.set_xlabel("$x_1$ (yards)"); ax1.set_ylabel("$x_2$ (yards)")
ax1.legend(fontsize=8)

# ── D_H: sparse observed-stroke proxy (discrete per-stroke colours, same
#    fixed table as fig2_high_fidelity_course_translucent.png) ──────────────
_draw_polygons(ax3, polygons, alpha=0.25)
colors = [stroke_color(int(n)) for n in df["observed_strokes"]]
ax3.scatter(df["x1"], df["x2"], c=colors, s=26, alpha=0.9,
            edgecolors="black", linewidths=0.5)
draw_pin(ax3)
ax3.set_aspect("equal")
ax3.set_title(r"$\mathcal{D}_H$ — Observed strokes-to-hole-out ($n$=%d)" % len(df))
ax3.set_xlabel("$x_1$ (yards)"); ax3.set_ylabel("$x_2$ (yards)")

present = sorted(int(v) for v in df["observed_strokes"].unique())
stroke_handles = [plt.Line2D([0], [0], marker="o", linestyle="None",
                              color=stroke_color(n), markersize=8, label=f"{n} strokes")
                  for n in present]
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles=handles + stroke_handles, fontsize=8, loc="best")

plt.tight_layout()
out = "../untitled folder/On_Par/images/dl_vs_dh_curated.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
