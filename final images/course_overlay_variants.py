"""Generate low/high-fidelity ESHO figures with course-layout context, in
three overlay styles: (1) as previously used, (2) more translucent,
(3) outlines only. Course polygons are rotated 90deg to match the landscape
label plots (distance-to-pin horizontal, crossrange vertical)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import shapely.wkt as shapely_wkt

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Parallelisation" / "convergence"))
from core import build_hole, _LIE_COLORS  # noqa: E402

DATA_DIR = ROOT / "Parallelisation" / "data"
CONV_OUT = ROOT / "Parallelisation" / "convergence" / "outputs" / "full_hole_paper"
MFGP_DIR = ROOT / "Multi-Fidelity GP"
PIN = (5.0, 334.0)
N_SIMULATIONS = 300

# Fixed integer-stroke -> colour mapping. First three values are the exact
# hex the sampledPutts.png reference uses (best/lowest = green, through to
# worst/highest); 5 and 6 are simulatingpar3script.py's putt_color_map
# ("red", "brown") converted to hex, continuing the same reference. Nothing
# beyond that was defined anywhere, so 7+ get new, mutually distinct colours
# rather than reusing "purple" for everything past 5 the way that script did.
# A given stroke count always gets the same colour across every figure,
# regardless of which subset of values that particular figure happens to show.
STROKE_COLORS = {
    2: "#349d34",
    3: "#92d3e3",
    4: "#ffa503",
    5: "#ff0000",   # matplotlib "red"
    6: "#a52a2a",   # matplotlib "brown"
    7: "#800080",   # matplotlib "purple"
    8: "#4b0082",   # indigo
    9: "#2f4f4f",   # dark slate
    10: "#000000",  # black
}
_EXTRA_CMAP = plt.get_cmap("cool")


def stroke_color(n: int) -> str:
    if n in STROKE_COLORS:
        return STROKE_COLORS[n]
    # Distinct colour for anything outside the fixed table (e.g. 10+ strokes),
    # never reusing a colour already assigned above.
    return _EXTRA_CMAP((n - max(STROKE_COLORS)) / 5 % 1.0)


STYLES = {
    "original":      dict(alpha=0.5, ec="black", lw=0.5, fill=True),
    "translucent": dict(alpha=0.18, ec="black", lw=0.5, fill=True),
    "outline":   dict(alpha=1.0, ec=None, lw=1.6, fill=False),
}


def draw_course(ax, hole, style: str) -> None:
    """Draw course polygons rotated: horizontal=y (downrange), vertical=x (crossrange)."""
    cfg = STYLES[style]

    def _fill(xs, ys, color):
        # rotate: plot (y, x)
        if cfg["fill"]:
            ax.fill(ys, xs, alpha=cfg["alpha"], fc=color, ec=cfg["ec"],
                     linewidth=cfg["lw"], zorder=0)
        else:
            ax.fill(ys, xs, alpha=cfg["alpha"], fc="none", ec=color,
                     linewidth=cfg["lw"], zorder=0)

    for _, row in hole.hole_9.iterrows():
        geom = shapely_wkt.loads(row["WKT"])
        color = _LIE_COLORS.get(row["lie"], "lightgrey")
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            _fill(xs, ys, color)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                xs, ys = poly.exterior.xy
                _fill(xs, ys, color)

    for _, row in hole.new_fairway.iterrows():
        xs, ys = row["geometry"].exterior.xy
        _fill(xs, ys, _LIE_COLORS["new_fairway"])

    for _, row in hole.new_hazard3.iterrows():
        xs, ys = row["geometry"].exterior.xy
        _fill(xs, ys, _LIE_COLORS["new_hazard3"])


def draw_markers(ax, hole) -> None:
    """Tee + pin, drawn LAST so they always sit on top of everything else.

    Standing convention (matches shot_outcomes_stacked.png): tee is a black
    square (a red cross is illegible on the green fill), pin is a white
    circle with a black edge.
    """
    ax.plot(hole.tee_point[1], hole.tee_point[0], marker="s", color="black",
            markersize=9, linestyle="None", zorder=30, label="Tee")
    ax.plot(hole.hole[1], hole.hole[0], "o", markersize=8, zorder=30, label="Pin",
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.3)


def course_bounds(hole):
    """Full (x_min, x_max, y_min, y_max) of the hole geometry, physical coords."""
    geoms = []
    for _, row in hole.hole_9.iterrows():
        geoms.append(shapely_wkt.loads(row["WKT"]))
    for _, row in hole.new_fairway.iterrows():
        geoms.append(row["geometry"])
    for _, row in hole.new_hazard3.iterrows():
        geoms.append(row["geometry"])

    xmins, ymins, xmaxs, ymaxs = zip(*(g.bounds for g in geoms))
    x_min, y_min = min(xmins), min(ymins)
    x_max, y_max = max(xmaxs), max(ymaxs)

    # Include tee and pin points too
    x_min = min(x_min, hole.tee_point[0], hole.hole[0])
    x_max = max(x_max, hole.tee_point[0], hole.hole[0])
    y_min = min(y_min, hole.tee_point[1], hole.hole[1])
    y_max = max(y_max, hole.tee_point[1], hole.hole[1])
    return x_min, x_max, y_min, y_max


def plot_low_fidelity(hole, style: str) -> Path:
    low = pd.read_csv(CONV_OUT / f"approach_N{N_SIMULATIONS:04d}.csv").sort_values("y").reset_index(drop=True)

    # Grid is 10 crossrange (x) values x 28 downrange (y) values. The y values
    # are closely spaced (~8.7yd apart over a wide horizontal axis), so labels
    # collide. Keep every other downrange column to give labels room to breathe.
    y_values = np.sort(low["y"].unique())
    keep_y = y_values[::2]
    low = low[low["y"].isin(keep_y)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(20, 7))
    draw_course(ax, hole, style)

    # viridis_r: low ESHO (good) = yellow, high ESHO (bad) = purple — same
    # direction as shot_outcomes_stacked.png, kept consistent across the paper.
    vmin, vmax = low["esho_mean"].min(), low["esho_mean"].max()
    cmap = plt.get_cmap("viridis_r")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for _, r in low.iterrows():
        ax.text(r["y"], r["x"], f"{r['esho_mean']:.1f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=cmap(norm(r["esho_mean"])),
                zorder=10,
                path_effects=[pe.withStroke(linewidth=0.8, foreground="black", alpha=0.85)])

    draw_markers(ax, hole)
    ax.axvline(PIN[1], color="black", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    x_min, x_max, y_min, y_max = course_bounds(hole)
    ax.set_xlim(y_min - 15, y_max + 15)
    ax.set_ylim(x_min - 12, x_max + 12)
    ax.set_xlabel("$x_1$ (yards)", fontsize=11)
    ax.set_ylabel("$x_2$ (yards)", fontsize=11)
    ax.set_title(f"Low-Fidelity Simulated ESHO Grid  (N={N_SIMULATIONS}, aim -40..40y step 4y)"
                 f"  —  {style}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4, zorder=1)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Expected Strokes to Hole Out (ESHO)", fontsize=10)

    out = HERE / f"fig1_low_fidelity_course_{style}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_high_fidelity(hole, style: str) -> Path:
    obs = pd.read_csv(MFGP_DIR / "observed_esho_data.csv")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(obs), size=60, replace=False)
    obs_sub = obs.iloc[idx].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(20, 7))
    draw_course(ax, hole, style)

    # Discrete per-stroke-count colours (fixed table, see STROKE_COLORS) rather
    # than a continuous colormap — these are whole numbers, not a continuum.
    for _, r in obs_sub.iterrows():
        n = int(r["observed_strokes"])
        ax.text(r["x2"], r["x1"], f"{n}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=stroke_color(n),
                zorder=10,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="black", alpha=0.85)])

    draw_markers(ax, hole)
    ax.axvline(PIN[1], color="black", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    x_min, x_max, y_min, y_max = course_bounds(hole)
    ax.set_xlim(y_min - 15, y_max + 15)
    ax.set_ylim(x_min - 12, x_max + 12)
    ax.set_xlabel("$x_1$ (yards)", fontsize=11)
    ax.set_ylabel("$x_2$ (yards)", fontsize=11)
    ax.set_title(f"High-Fidelity Observed Strokes  (n={len(obs_sub)} of {len(obs)} shown)"
                 f"  —  {style}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4, zorder=1)

    present = sorted(int(v) for v in obs_sub["observed_strokes"].unique())
    handles, _ = ax.get_legend_handles_labels()
    stroke_handles = [mpl.lines.Line2D([0], [0], marker="o", linestyle="None",
                                        color=stroke_color(n), markersize=8,
                                        label=f"{n} strokes") for n in present]
    ax.legend(handles=handles + stroke_handles, loc="upper right", fontsize=8, framealpha=0.8)

    out = HERE / f"fig2_high_fidelity_course_{style}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    hole = build_hole(DATA_DIR, gp_training_iter=10)
    outputs = []
    for style in STYLES:
        outputs.append(plot_low_fidelity(hole, style))
        outputs.append(plot_high_fidelity(hole, style))
    for p in outputs:
        print("saved", p.name)


if __name__ == "__main__":
    main()
