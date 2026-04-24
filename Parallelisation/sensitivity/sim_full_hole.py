"""
sim_full_hole.py – Full hole trajectory simulator for sensitivity analysis.

Simulates complete Par-4 holes (tee → fairway → green → hole-out) under a
given (carry_shift, variance_scale) configuration, recording the (x, y)
landing position after every shot.

Key classes / functions:
  ClubSelector    – wraps club selection; enforces Driver-only-on-tee constraint.
  simulate_hole   – simulate one full hole, returning a trajectory DataFrame.
  run_sensitivity – run n_sims holes for one config, save to CSV.
  get_tee_summary – summarise tee shot choices from the output DataFrame.
  generate_sensitivity_plots – produce per-run PNG outputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

# Reuse hole geometry, distributions, and GPR from the convergence pipeline.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "convergence"))

from core import (
    HoleData,
    build_hole,
    evaluate_on_green,
    get_lie_category,
    rotation_translator,
    _get_water_drop,
)

logger = logging.getLogger(__name__)

# Clubs that are only valid from the tee box.
_TEE_ONLY_CLUBS: frozenset[str] = frozenset({"Driver"})
# Maximum shots per hole before we declare "lost ball" (safety limit).
_MAX_SHOTS_PER_HOLE: int = 20

_LIE_COLORS = {
    "bunker":       "tan",
    "fairway":      "forestgreen",
    "green":        "lightgreen",
    "OB":           "lightcoral",
    "rough":        "mediumseagreen",
    "tee":          "darkgreen",
    "water_hazard": "skyblue",
    "hole":         "black",
}


# ---------------------------------------------------------------------------
# ClubSelector
# ---------------------------------------------------------------------------

class ClubSelector:
    """Select candidate clubs for a shot based on remaining distance.

    Enforces a conditional constraint: Driver is only available for the
    initial tee shot.  All subsequent shots (approach, layup, recovery)
    may only use 3-woods, irons, and wedges.
    """

    def __init__(self, club_distributions: dict, top_n: int = 5) -> None:
        self.club_distributions = club_distributions
        self.top_n = top_n
        self._avg_carry: dict[str, float] = {
            club: stats["mean"][1]
            for club, stats in club_distributions.items()
        }

    def select(
        self,
        distance: float,
        is_tee_shot: bool = False,
    ) -> list[str]:
        available = {
            club: carry
            for club, carry in self._avg_carry.items()
            if is_tee_shot or club not in _TEE_ONLY_CLUBS
        }
        ranked = sorted(available.items(), key=lambda kv: abs(kv[1] - distance))
        return [club for club, _ in ranked[: self.top_n]]


# ---------------------------------------------------------------------------
# Shot outcome helpers
# ---------------------------------------------------------------------------

def _sample_shot(
    mu: np.ndarray,
    cov: np.ndarray,
    aim_offset: float,
    starting_point: tuple[float, float],
    target: tuple[float, float],
) -> tuple[float, float]:
    total_dist = float(np.linalg.norm(np.array(target) - np.array(starting_point)))
    angle_deg  = (
        float(np.degrees(np.arctan(aim_offset / total_dist)))
        if total_dist > 0 else 0.0
    )
    draw = np.random.multivariate_normal(mu, cov)
    return rotation_translator(float(draw[0]), float(draw[1]), angle_deg, starting_point, target)


def _pick_best_club(
    selector: ClubSelector,
    distance: float,
    is_tee_shot: bool,
    lie: str,
    hole: HoleData,
) -> str:
    candidates = selector.select(distance, is_tee_shot=is_tee_shot)
    if lie == "rough":
        return candidates[min(1, len(candidates) - 1)]
    return candidates[0]


# ---------------------------------------------------------------------------
# Single hole simulation
# ---------------------------------------------------------------------------

@dataclass
class ShotRecord:
    sim_id:     int
    shot_num:   int
    x:          float
    y:          float
    club:       str
    lie:        str
    is_tee:     bool
    aim_offset: float


def simulate_hole(
    hole: HoleData,
    selector: ClubSelector,
    sim_id: int = 0,
    aim_offset: float = 0.0,
    max_shots: int = _MAX_SHOTS_PER_HOLE,
) -> list[ShotRecord]:
    """Simulate one complete Par-4 hole, returning shot-by-shot trajectory."""
    records: list[ShotRecord] = []
    current_pos = hole.tee_point
    target      = hole.hole
    is_tee_shot = True

    for shot_num in range(1, max_shots + 1):
        lie = get_lie_category(current_pos, hole)

        # ---- On green: putt out ----------------------------------------
        if lie == "green":
            mean_putts = evaluate_on_green(
                current_pos, hole.putt_model, hole.putt_likelihood
            )
            n_putts = max(1, int(round(mean_putts)))
            for p in range(n_putts):
                records.append(ShotRecord(
                    sim_id=sim_id,
                    shot_num=shot_num + p,
                    x=float(target[0]),
                    y=float(target[1]),
                    club="Putter",
                    lie="hole",
                    is_tee=False,
                    aim_offset=0.0,
                ))
            break

        # ---- Water: drop with penalty, don't record extra position ------
        if lie == "water":
            drop = _get_water_drop(hole.tee_point, current_pos, hole.water_polygons)
            current_pos = drop if drop is not None else current_pos
            lie = "rough"

        # ---- Select club and distributions ------------------------------
        distance = float(np.linalg.norm(np.array(target) - np.array(current_pos)))
        club     = _pick_best_club(selector, distance, is_tee_shot, lie, hole)

        if lie == "rough":
            mu  = hole.rough_distributions[club]["mean"]
            cov = hole.rough_distributions[club]["cov"]
        else:
            mu  = hole.club_distributions[club]["mean"]
            cov = hole.club_distributions[club]["cov"]

        # ---- Simulate shot ---------------------------------------------
        new_pos = _sample_shot(mu, cov, aim_offset, current_pos, target)

        records.append(ShotRecord(
            sim_id=sim_id,
            shot_num=shot_num,
            x=float(new_pos[0]),
            y=float(new_pos[1]),
            club=club,
            lie=get_lie_category(new_pos, hole),
            is_tee=is_tee_shot,
            aim_offset=aim_offset,
        ))

        current_pos = new_pos
        is_tee_shot = False  # Driver excluded from all subsequent shots

    return records


# ---------------------------------------------------------------------------
# Tee shot summary
# ---------------------------------------------------------------------------

def get_tee_summary(df: pd.DataFrame) -> dict:
    """Summarise tee shot choices and landing statistics from a sim DataFrame."""
    tee = df[df["is_tee"] == True].copy()
    if tee.empty:
        return {}

    club_counts = tee["club"].value_counts()
    top_club    = club_counts.index[0]
    club_dist   = club_counts.to_dict()

    return {
        "n_tee_shots":     int(len(tee)),
        "top_club":        top_club,
        "club_distribution": {k: int(v) for k, v in club_dist.items()},
        "landing_mean_x":  float(tee["x"].mean()),
        "landing_mean_y":  float(tee["y"].mean()),
        "landing_std_x":   float(tee["x"].std()),
        "landing_std_y":   float(tee["y"].std()),
        "landing_lie_distribution": tee["lie"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# Hole geometry background plotter
# ---------------------------------------------------------------------------

def _draw_hole_background(hole: HoleData, ax: plt.Axes) -> None:
    """Draw hole polygons (fairway, green, water, bunkers) onto ax."""
    for _, row in hole.hole_9.iterrows():
        geom  = shapely_wkt.loads(row["WKT"])
        color = _LIE_COLORS.get(row["lie"], "lightgrey")
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.45, fc=color, ec="black", lw=0.4)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.45, fc=color, ec="black", lw=0.4)

    for _, row in hole.new_fairway.iterrows():
        x, y = row["geometry"].exterior.xy
        ax.fill(x, y, alpha=0.45, fc=_LIE_COLORS["fairway"], ec="black", lw=0.4)
    for _, row in hole.new_hazard3.iterrows():
        x, y = row["geometry"].exterior.xy
        ax.fill(x, y, alpha=0.45, fc=_LIE_COLORS["water_hazard"], ec="black", lw=0.4)

    ax.plot(*hole.tee_point, "rx", ms=8, label="Tee")
    ax.plot(*hole.hole,      "ko", ms=6, label="Pin")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.4)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

_LIE_ORDER = ["fairway", "rough", "green", "water_hazard", "bunker", "hole"]
_LIE_PLOT_COLORS = {
    "fairway":      "#2ecc71",
    "rough":        "#27ae60",
    "green":        "#a8e6a1",
    "water_hazard": "#3498db",
    "bunker":       "#d4a96a",
    "hole":         "#2c3e50",
    "other":        "#95a5a6",
}


def generate_sensitivity_plots(
    df: pd.DataFrame,
    hole: HoleData,
    carry_shift: float,
    variance_scale: float,
    output_dir: Path,
    fname_base: str,
) -> dict[str, Path]:
    """Generate three PNGs for a sensitivity run and return their paths.

    Outputs
    -------
    landing_map   : all shot landings on the hole geometry, coloured by lie.
    score_dist    : histogram of total strokes per simulated hole.
    tee_shot      : tee-shot landings only, with landing ellipse and top club.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    title_suffix = f"carry +{carry_shift:.1f} yd  |  var ×{variance_scale:.2f}"

    # ── 1. Landing map ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 14))
    _draw_hole_background(hole, ax)

    non_putter = df[df["club"] != "Putter"]
    for lie in _LIE_ORDER:
        sub = non_putter[non_putter["lie"] == lie]
        if sub.empty:
            continue
        color = _LIE_PLOT_COLORS.get(lie, _LIE_PLOT_COLORS["other"])
        ax.scatter(sub["x"], sub["y"], c=color, s=4, alpha=0.25,
                   label=lie, zorder=10)

    # Mark mean tee landing
    tee = df[df["is_tee"] == True]
    if not tee.empty:
        ax.scatter(tee["x"].mean(), tee["y"].mean(),
                   color="red", marker="*", s=250, zorder=30, label="Mean tee landing")

    ax.set_title(f"Shot Landing Map\n{title_suffix}", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, markerscale=2)
    plt.tight_layout()
    p = output_dir / f"{fname_base}_landing_map.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["landing_map"] = p

    # ── 2. Score distribution ─────────────────────────────────────────────────
    strokes_per_hole = df.groupby("sim_id")["shot_num"].max()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(int(strokes_per_hole.min()), int(strokes_per_hole.max()) + 2)
    ax.hist(strokes_per_hole, bins=bins, color="#2980b9", edgecolor="white",
            linewidth=0.7, align="left")
    ax.axvline(strokes_per_hole.mean(), color="red", linestyle="--",
               label=f"Mean = {strokes_per_hole.mean():.2f}")
    ax.set_xlabel("Strokes per hole")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution\n{title_suffix}", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = output_dir / f"{fname_base}_score_dist.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["score_dist"] = p

    # ── 3. Tee shot detail ────────────────────────────────────────────────────
    tee = df[df["is_tee"] == True].copy()
    fig, ax = plt.subplots(figsize=(10, 14))
    _draw_hole_background(hole, ax)

    if not tee.empty:
        top_club = tee["club"].value_counts().index[0]
        club_colors = {c: plt.cm.tab10(i) for i, c in enumerate(tee["club"].unique())}
        for club, grp in tee.groupby("club"):
            ax.scatter(grp["x"], grp["y"], c=[club_colors[club]], s=12, alpha=0.4,
                       label=f"{club} (n={len(grp)})", zorder=10)

        mx, my = tee["x"].mean(), tee["y"].mean()
        sx, sy = tee["x"].std(), tee["y"].std()
        ellipse = matplotlib.patches.Ellipse(
            (mx, my), width=2 * sx, height=2 * sy,
            edgecolor="red", facecolor="none", linewidth=2, linestyle="--", zorder=20,
        )
        ax.add_patch(ellipse)
        ax.scatter(mx, my, color="red", marker="*", s=300, zorder=30, label="Mean landing")
        ax.annotate(
            f"Top club: {top_club}\nMean: ({mx:.1f}, {my:.1f})\n±({sx:.1f}, {sy:.1f}) yds",
            xy=(mx, my), xytext=(mx + 15, my - 20),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    ax.set_title(f"Tee Shot Landings\n{title_suffix}", fontsize=11)
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    p = output_dir / f"{fname_base}_tee_shot.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["tee_shot"] = p

    logger.info("Plots saved: %s", ", ".join(str(v) for v in paths.values()))
    return paths


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_sensitivity(
    carry_shift: float,
    variance_scale: float,
    n_sims: int = 1_000,
    seed: int = 0,
    data_dir: Optional[Path] = None,
    gp_training_iter: int = 100,
    aim_offset: float = 0.0,
    log_interval: int = 100,
) -> tuple[pd.DataFrame, HoleData]:
    """Simulate `n_sims` holes under a (carry_shift, variance_scale) config.

    Returns
    -------
    (df, hole)
        df   : DataFrame with columns sim_id, shot_num, x, y, club, lie,
               is_tee, aim_offset, carry_shift, variance_scale.
        hole : HoleData (reuse for plotting without rebuilding).
    """
    np.random.seed(seed)
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    logger.info(
        "Building hole (carry_shift=%.2f, variance_scale=%.4f) ...",
        carry_shift, variance_scale,
    )
    hole = build_hole(
        data_dir=data_dir,
        gp_training_iter=gp_training_iter,
        carry_shift_yards=carry_shift,
        variance_scale=variance_scale,
    )

    selector = ClubSelector(hole.club_distributions, top_n=5)
    logger.info("Simulating %d holes ...", n_sims)

    all_records: list[ShotRecord] = []
    t0 = time.time()

    for sim_id in range(n_sims):
        records = simulate_hole(hole, selector, sim_id=sim_id, aim_offset=aim_offset)
        all_records.extend(records)

        if (sim_id + 1) % log_interval == 0 or sim_id == n_sims - 1:
            elapsed  = time.time() - t0
            pct      = (sim_id + 1) / n_sims * 100
            rate     = (sim_id + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_sims - sim_id - 1) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.0f%%)  elapsed %.1fs  est. remaining %.1fs",
                sim_id + 1, n_sims, pct, elapsed, remaining,
            )

    df = pd.DataFrame([r.__dict__ for r in all_records])
    df["carry_shift"]    = carry_shift
    df["variance_scale"] = variance_scale

    total_elapsed = time.time() - t0
    logger.info(
        "Done. %d holes | %d shot records | %.1f shots/hole avg | %.1fs total",
        n_sims, len(df), len(df) / n_sims if n_sims else 0, total_elapsed,
    )
    return df, hole


def output_filename(carry_shift: float, variance_scale: float) -> str:
    """Canonical CSV filename stem for a given config."""
    return f"sim_output_dist{carry_shift:.2f}_disp{variance_scale:.4f}"
