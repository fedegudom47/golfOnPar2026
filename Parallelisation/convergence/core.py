"""
core.py – Hole geometry setup, club distributions, and shot simulation.

Faithfully ported from the working scriptpar4.py / par4model.ipynb code.
All state is bundled into a `HoleData` dataclass returned by `build_hole()`,
so every worker process can initialise independently (pickle-safe for
multiprocessing and Slurm).

Data files expected in `data_dir` (default: Parallelisation/data/):
    hole_9_data.csv
    newshapes.geojson
    gpr_green_dataset.csv
    strokes_by_lie_yards_broadie.csv
    simulated_lpga_shot_data2.csv
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from shapely import wkt as shapely_wkt
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPyTorch model definitions (module-level so they are pickleable)
# ---------------------------------------------------------------------------

class _PuttGPModel(gpytorch.models.ExactGP):
    """ExactGP for expected putts from (x, y) on the green."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ---------------------------------------------------------------------------
# HoleData container
# ---------------------------------------------------------------------------

@dataclass
class HoleData:
    """All geometry, distributions, and trained models for one hole.

    Everything a convergence worker needs is here.  Workers call
    `build_hole(data_dir)` themselves so nothing is pickled across process
    boundaries.
    """
    hole_9: pd.DataFrame                   # original hole CSV (geometry shifted)
    tee_point: tuple[float, float]
    hole: tuple[float, float]              # pin location in game coords
    strategy_points: list[tuple[float, float]]
    green_polygon: object                  # shapely Polygon
    fairway_polygons: list
    water_polygons: list
    bunker_polygons: list
    rough_polygons: list
    new_fairway: gpd.GeoDataFrame
    new_hazard3: gpd.GeoDataFrame
    club_distributions: dict[str, dict]   # club → {"mean": ndarray, "cov": ndarray}
    rough_distributions: dict[str, dict]
    broadie_interpolators: dict[str, object]
    putt_model: _PuttGPModel
    putt_likelihood: gpytorch.likelihoods.GaussianLikelihood


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def build_hole(data_dir: Optional[Path] = None, gp_training_iter: int = 100) -> HoleData:
    """Load data and set up the full Par-4 hole layout.

    Returns a HoleData that is fully self-contained and ready for simulation.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    logger.info("Building hole from data_dir=%s", data_dir)

    # ------------------------------------------------------------------
    # 1. Load hole_9 CSV and parse WKT
    # ------------------------------------------------------------------
    hole_9 = pd.read_csv(data_dir / "hole_9_data.csv")
    hole_9["geometry"] = hole_9["WKT"].apply(shapely_wkt.loads)

    # ------------------------------------------------------------------
    # 2. Find tee point (centroid of tee box furthest from green)
    # ------------------------------------------------------------------
    def _centroid(row):
        return shapely_wkt.loads(row["WKT"]).centroid.coords[0]

    teeboxes = hole_9[hole_9["lie"].str.contains("tee", case=False)].copy()
    green_row = hole_9[hole_9["lie"] == "green"].iloc[0]
    green_centre = _centroid(green_row)

    teeboxes["centroid"] = teeboxes.apply(_centroid, axis=1)
    teeboxes["dist_to_green"] = teeboxes["centroid"].apply(
        lambda pt: np.linalg.norm(np.array(pt) - np.array(green_centre))
    )
    tee_point: tuple[float, float] = teeboxes.loc[teeboxes["dist_to_green"].idxmax()]["centroid"]

    # ------------------------------------------------------------------
    # 3. Load GeoJSON hazards / extended fairway
    # ------------------------------------------------------------------
    gdf = gpd.read_file(data_dir / "newshapes.geojson")

    new_fairway = gdf[gdf["lie"] == "fairway"].to_crs(epsg=32611).copy()
    new_hazard3 = gdf[gdf["lie"] == "water_hazard_3"].to_crs(epsg=32611).copy()

    # ------------------------------------------------------------------
    # 4. First alignment: shift all new geometries so fairway centroid
    #    sits at (0, 100), then rotate by the fairway's own long-axis angle
    # ------------------------------------------------------------------
    ref_centroid = new_fairway.iloc[0].geometry.centroid
    x_shift0 = 0.0 - ref_centroid.x
    y_shift0 = 100.0 - ref_centroid.y

    for col in [new_fairway, new_hazard3]:
        col["geometry"] = col["geometry"].apply(
            lambda g: shp_translate(g, xoff=x_shift0, yoff=y_shift0)
        )

    fp = new_fairway.iloc[0].geometry
    vec = np.array([fp.centroid.x, fp.bounds[3]]) - np.array([fp.centroid.x, fp.bounds[1]])
    rot_angle = float(np.degrees(np.arctan2(vec[0], vec[1])))

    for col in [new_fairway, new_hazard3]:
        col["geometry"] = col["geometry"].apply(
            lambda g: shp_rotate(g, angle=-rot_angle, origin="centroid", use_radians=False)
        )

    # ------------------------------------------------------------------
    # 5. Shift the original hole_9 shapes (green, bunker, water, fairway)
    #    up by 160 yards and update geometry
    # ------------------------------------------------------------------
    y_shift_base = 160
    hole_pin: tuple[float, float] = (5, 174 + y_shift_base)

    shift_idx = hole_9[hole_9["lie"].isin(["green", "bunker", "water_hazard", "fairway"])].index

    def _shift_wkt(wkt_str: str) -> str:
        shape = shapely_wkt.loads(wkt_str)
        return shp_translate(shape, yoff=y_shift_base).wkt

    hole_9.loc[shift_idx, "WKT"] = hole_9.loc[shift_idx, "WKT"].apply(_shift_wkt)
    # Re-parse geometry so containment checks use the shifted coordinates
    hole_9["geometry"] = hole_9["WKT"].apply(shapely_wkt.loads)

    # ------------------------------------------------------------------
    # 6. Final positioning of new_fairway: centroid → (20, 175), rotate -68°
    # ------------------------------------------------------------------
    fc = new_fairway.iloc[0].geometry.centroid
    new_fairway["geometry"] = new_fairway["geometry"].apply(
        lambda g: shp_translate(g, xoff=20.0 - fc.x, yoff=175.0 - fc.y)
    )
    new_fairway["geometry"] = new_fairway["geometry"].apply(
        lambda g: shp_rotate(g, angle=-68, origin="centroid", use_radians=False)
    )

    # Final positioning of new_hazard3: centroid → (0, 210), rotate 110°
    h3c = new_hazard3.iloc[0].geometry.centroid
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(
        lambda g: shp_translate(g, xoff=0.0 - h3c.x, yoff=210.0 - h3c.y)
    )
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(
        lambda g: shp_rotate(g, angle=110, origin="centroid", use_radians=False)
    )

    # ------------------------------------------------------------------
    # 7. Build polygon lookup lists
    # ------------------------------------------------------------------
    def _get_polygons(df: pd.DataFrame, lie_type: str) -> list:
        return [geom for geom in df[df["lie"] == lie_type]["geometry"]]

    green_polygon = _get_polygons(hole_9, "green")[0]
    bunker_polygons = _get_polygons(hole_9, "bunker")
    rough_polygons = _get_polygons(hole_9, "rough")
    fairway_polygons = _get_polygons(hole_9, "fairway") + list(new_fairway["geometry"])
    water_polygons = _get_polygons(hole_9, "water_hazard") + list(new_hazard3["geometry"])

    # ------------------------------------------------------------------
    # 8. Strategy grid (approach-shot evaluation points)
    # ------------------------------------------------------------------
    hole_vec = np.array(hole_pin)
    tee_vec = np.array(tee_point)
    ht_length = float(np.linalg.norm(hole_vec - tee_vec))

    x_vals = np.linspace(-40, 60, int(100 / 10))
    y_vals = np.linspace(50, ht_length - 50, int((ht_length - 50) / 10))
    strategy_points = [(float(x), float(y)) for y in y_vals for x in x_vals]

    logger.info("Strategy grid: %d points (ht_length=%.1f)", len(strategy_points), ht_length)

    # ------------------------------------------------------------------
    # 9. Broadie expected-strokes interpolators
    # ------------------------------------------------------------------
    broadie_data = pd.read_csv(data_dir / "strokes_by_lie_yards_broadie.csv")
    broadie_data = broadie_data.rename(columns={"Distance (yards)": "distance"})
    broadie_long = broadie_data.melt(id_vars="distance", var_name="lie", value_name="strokes")
    broadie_long["lie"] = broadie_long["lie"].str.lower()

    broadie_interpolators: dict = {}
    for lie, group in broadie_long.groupby("lie"):
        group = group.sort_values("distance")
        broadie_interpolators[lie] = interp1d(
            group["distance"], group["strokes"],
            kind="linear", fill_value="extrapolate",
        )

    # ------------------------------------------------------------------
    # 10. Club distributions (mean, covariance) for LPGA data
    # ------------------------------------------------------------------
    lpga_clubs = pd.read_csv(data_dir / "simulated_lpga_shot_data2.csv")
    club_distributions: dict = {}
    for club, group in lpga_clubs.groupby("Club"):
        club_distributions[club] = {
            "mean": group[["Side", "Carry"]].mean().to_numpy(),
            "cov":  np.cov(group[["Side", "Carry"]].T),
        }

    # Rough distributions: shorter carry + larger variance, scaled by club length
    club_names_sorted = sorted(
        club_distributions.keys(),
        key=lambda c: club_distributions[c]["mean"][1],
    )
    n_clubs = len(club_names_sorted)
    rough_distributions: dict = {}
    for i, club in enumerate(club_names_sorted):
        t = i / (n_clubs - 1) if n_clubs > 1 else 0.0
        carry_loss = 0.05 + t * (0.17 - 0.05)
        var_increase = 0.10 + t * (0.40 - 0.10)
        mean = club_distributions[club]["mean"].copy()
        cov = club_distributions[club]["cov"].copy()
        mean[1] *= (1.0 - carry_loss)
        cov *= (1.0 + var_increase)
        rough_distributions[club] = {"mean": mean, "cov": cov}

    # ------------------------------------------------------------------
    # 11. Putting GPR: ExactGP → E[putts | (x, y) on green]
    # ------------------------------------------------------------------
    putts_df = pd.read_csv(data_dir / "gpr_green_dataset.csv").copy()
    putts_df["y"] += 160  # shift to match game coords

    X_train = torch.tensor(putts_df[["x", "y"]].values, dtype=torch.float32)
    y_train = torch.tensor(putts_df["simulated_strokes"].values, dtype=torch.float32)

    putt_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    putt_model = _PuttGPModel(X_train, y_train, putt_likelihood)
    putt_model.train()
    putt_likelihood.train()

    optimizer = torch.optim.Adam(putt_model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(putt_likelihood, putt_model)
    for i in range(gp_training_iter):
        optimizer.zero_grad()
        loss = -mll(putt_model(X_train), y_train)
        loss.backward()
        optimizer.step()

    putt_model.eval()
    putt_likelihood.eval()
    logger.info("Putt GPR trained (%d iterations).", gp_training_iter)

    return HoleData(
        hole_9=hole_9,
        tee_point=tee_point,
        hole=hole_pin,
        strategy_points=strategy_points,
        green_polygon=green_polygon,
        fairway_polygons=fairway_polygons,
        water_polygons=water_polygons,
        bunker_polygons=bunker_polygons,
        rough_polygons=rough_polygons,
        new_fairway=new_fairway,
        new_hazard3=new_hazard3,
        club_distributions=club_distributions,
        rough_distributions=rough_distributions,
        broadie_interpolators=broadie_interpolators,
        putt_model=putt_model,
        putt_likelihood=putt_likelihood,
    )


# ---------------------------------------------------------------------------
# Lie detection
# ---------------------------------------------------------------------------

def get_lie_category(point: tuple[float, float], hole: HoleData) -> str:
    pt = Point(point)
    if hole.green_polygon.contains(pt):
        return "green"
    if any(poly.contains(pt) for poly in hole.water_polygons):
        return "water"
    if any(poly.contains(pt) for poly in hole.bunker_polygons):
        return "bunker"
    if any(poly.contains(pt) for poly in hole.fairway_polygons):
        return "fairway"
    return "rough"


# ---------------------------------------------------------------------------
# Shot evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_on_green(
    point: tuple[float, float],
    model: _PuttGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
) -> float:
    test_x = torch.tensor([[point[0], point[1]]], dtype=torch.float32)
    with torch.no_grad():
        pred = likelihood(model(test_x))
    return float(pred.mean.item())


def evaluate_broadie(
    point: tuple[float, float],
    target: tuple[float, float],
    lie: str,
    interpolators: dict,
) -> float:
    dist = float(np.linalg.norm(np.array(point) - np.array(target)))
    lie = lie.lower()
    if lie not in interpolators:
        raise ValueError(f"No Broadie interpolator for lie '{lie}'")
    return float(interpolators[lie](dist))


def _get_water_drop(
    starting_point: tuple[float, float],
    ball_in_water: tuple[float, float],
    water_polygons: list,
) -> Optional[tuple[float, float]]:
    """Return the entry point where the shot line first crosses water."""
    shot_line = LineString([starting_point, ball_in_water])
    closest: Optional[Point] = None
    min_dist = float("inf")

    for poly in water_polygons:
        intersection = shot_line.intersection(poly.boundary)
        if intersection.is_empty:
            continue
        if isinstance(intersection, Point):
            d = Point(starting_point).distance(intersection)
            if d < min_dist:
                min_dist = d
                closest = intersection
        else:
            for pt in getattr(intersection, "geoms", [intersection]):
                if isinstance(pt, Point):
                    d = Point(starting_point).distance(pt)
                    if d < min_dist:
                        min_dist = d
                        closest = pt

    return (closest.x, closest.y) if closest is not None else None


def evaluate_shot(
    point: tuple[float, float],
    starting_point: tuple[float, float],
    target: tuple[float, float],
    hole: HoleData,
) -> float:
    """Return expected strokes to hole out from `point` given `target`."""
    lie = get_lie_category(point, hole)

    if lie == "green":
        return evaluate_on_green(point, hole.putt_model, hole.putt_likelihood)
    elif lie == "fairway":
        return evaluate_broadie(point, target, "fairway", hole.broadie_interpolators)
    elif lie == "rough":
        return evaluate_broadie(point, target, "rough", hole.broadie_interpolators)
    elif lie == "bunker":
        return evaluate_broadie(point, target, "sand", hole.broadie_interpolators)
    elif lie == "water":
        drop = _get_water_drop(starting_point, point, hole.water_polygons)
        if drop is not None:
            return 1.0 + evaluate_broadie(drop, target, "rough", hole.broadie_interpolators)
        return float("nan")
    else:
        return evaluate_broadie(point, target, "rough", hole.broadie_interpolators)


# ---------------------------------------------------------------------------
# Shot rotation / geometry
# ---------------------------------------------------------------------------

def rotation_translator(
    x_side: float,
    y_carry: float,
    angle_deg: float,
    starting_point: tuple[float, float],
    target: tuple[float, float],
) -> tuple[float, float]:
    """Rotate a (side, carry) shot into global (x, y) coordinates."""
    direction = np.array(target) - np.array(starting_point)
    unit_dir = direction / np.linalg.norm(direction)
    angle_rad = np.radians(angle_deg)
    rot = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)],
    ])
    local_shot = np.array([x_side, y_carry])
    rotated = rot @ local_shot
    global_vec = np.array([
        unit_dir[0] * rotated[1] - unit_dir[1] * rotated[0],
        unit_dir[1] * rotated[1] + unit_dir[0] * rotated[0],
    ])
    result = np.array(starting_point) + global_vec
    return (float(result[0]), float(result[1]))


# ---------------------------------------------------------------------------
# Core approach-shot simulation
# ---------------------------------------------------------------------------

def simulate_approach_shots(
    hole: HoleData,
    n_samples: int,
    strategy_points: Optional[list] = None,
    aim_range: tuple[float, float] = (-20.0, 20.0),
    aim_step: float = 2.0,
) -> list[dict]:
    """Simulate approach shots for every strategy point with `n_samples` draws.

    Returns a list of dicts:
        {start, club, aim_offset, mean, var}

    `mean` already includes the +1 stroke overhead (or +2 for water lies).
    """
    if strategy_points is None:
        strategy_points = hole.strategy_points

    target = hole.hole
    aim_points = list(np.arange(aim_range[0], aim_range[1] + aim_step, aim_step))

    clubs_avg_carry = {
        club: stats["mean"][1]
        for club, stats in hole.club_distributions.items()
    }

    optimal_results: list[dict] = []

    for starting_point in strategy_points:
        starting_lie = get_lie_category(starting_point, hole)

        if starting_lie == "water":
            drop = _get_water_drop(hole.tee_point, starting_point, hole.water_polygons)
            playing_location = drop if drop is not None else starting_point
            penalty = 2.0
        else:
            playing_location = starting_point
            penalty = 1.0

        total_distance = float(np.linalg.norm(
            np.array(target) - np.array(playing_location)
        ))

        # Top-5 clubs by carry proximity to remaining distance
        top_clubs = [
            club for club, _ in sorted(
                ((c, abs(carry - total_distance)) for c, carry in clubs_avg_carry.items()),
                key=lambda x: x[1],
            )[:5]
        ]

        best_res: Optional[dict] = None

        for club in top_clubs:
            if starting_lie == "rough":
                mu = hole.rough_distributions[club]["mean"]
                cov = hole.rough_distributions[club]["cov"]
            else:
                mu = hole.club_distributions[club]["mean"]
                cov = hole.club_distributions[club]["cov"]

            for aim_offset in aim_points:
                angle_deg = float(np.degrees(np.arctan(aim_offset / total_distance))) if total_distance > 0 else 0.0
                samples = np.random.multivariate_normal(mu, cov, size=n_samples)

                strokes: list[float] = []
                for shot in samples:
                    lp = rotation_translator(
                        float(shot[0]), float(shot[1]),
                        angle_deg, playing_location, target,
                    )
                    es = evaluate_shot(lp, playing_location, target, hole)
                    if not np.isnan(es):
                        strokes.append(es)

                if not strokes:
                    continue

                mean_val = float(np.mean(strokes)) + penalty
                var_val = float(np.var(strokes))

                if best_res is None or mean_val < best_res["mean"]:
                    best_res = {
                        "start":      starting_point,
                        "club":       club,
                        "aim_offset": float(aim_offset),
                        "mean":       mean_val,
                        "var":        var_val,
                    }

        if best_res is not None:
            optimal_results.append(best_res)

    return optimal_results


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

CLUB_STYLES: dict[str, dict] = {
    "Driver":  {"short": "D",   "color": "#000000"},
    "3-wood":  {"short": "3w",  "color": "#FFAE00"},
    "5-wood":  {"short": "5w",  "color": "#CC79A7"},
    "Hybrid":  {"short": "Hy",  "color": "#009E73"},
    "4 Iron":  {"short": "4i",  "color": "royalblue"},
    "5 Iron":  {"short": "5i",  "color": "#56B4E9"},
    "6 Iron":  {"short": "6i",  "color": "#4DAF4A"},
    "7 Iron":  {"short": "7i",  "color": "#D55E00"},
    "8 Iron":  {"short": "8i",  "color": "#984EA3"},
    "9 Iron":  {"short": "9i",  "color": "#FFEE00"},
    "PW":      {"short": "Pw",  "color": "#0C034F"},
    "50 deg":  {"short": "50",  "color": "#E41A1C"},
    "54 deg":  {"short": "54",  "color": "#999999"},
    "60 deg":  {"short": "60",  "color": "#3E1F1F"},
}

_LIE_COLORS = {
    "bunker":        "tan",
    "fairway":       "forestgreen",
    "new_fairway":   "forestgreen",
    "green":         "lightgreen",
    "OB":            "lightcoral",
    "rough":         "mediumseagreen",
    "tee":           "darkgreen",
    "water_hazard":  "skyblue",
    "new_hazard3":   "skyblue",
}


def plot_hole_layout(
    hole: HoleData,
    title: str = "Hole Layout",
    figsize: tuple = (10, 10),
    plot_strategy_points: bool = True,
    ax: Optional[object] = None,
) -> object:
    """Draw the hole geometry and (optionally) the strategy grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for _, row in hole.hole_9.iterrows():
        geom = shapely_wkt.loads(row["WKT"])
        color = _LIE_COLORS.get(row["lie"], "lightgrey")
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, ec="black", linewidth=0.5)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc=color, ec="black", linewidth=0.5)

    for _, row in hole.new_fairway.iterrows():
        x, y = row["geometry"].exterior.xy
        ax.fill(x, y, alpha=0.5, fc=_LIE_COLORS["new_fairway"], ec="black", linewidth=0.5)

    for _, row in hole.new_hazard3.iterrows():
        x, y = row["geometry"].exterior.xy
        ax.fill(x, y, alpha=0.5, fc=_LIE_COLORS["new_hazard3"], ec="black", linewidth=0.5)

    ax.plot(hole.tee_point[0], hole.tee_point[1], "rx", markersize=7, label="Tee")
    ax.plot(hole.hole[0], hole.hole[1], "ko", markersize=5, label="Hole")

    if plot_strategy_points:
        xs, ys = zip(*hole.strategy_points)
        ax.scatter(xs, ys, color="black", s=15, alpha=0.4, zorder=10, label="Grid")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, linestyle=":")
    return ax


def plot_optimal_approaches(
    optimal_results: list[dict],
    hole: HoleData,
    title: str = "Optimal Approach Strategy",
    figsize: tuple = (14, 16),
    output_path: Optional[Path] = None,
) -> None:
    """Scatter the optimal strategy grid coloured by ESHO and club."""
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=figsize)
    plot_hole_layout(hole, title=title, plot_strategy_points=False, ax=ax)

    xs    = [r["start"][0] for r in optimal_results]
    ys    = [r["start"][1] for r in optimal_results]
    means = [r["mean"]     for r in optimal_results]

    face_colors = [
        CLUB_STYLES.get(r["club"], {"color": "#999999"})["color"]
        for r in optimal_results
    ]
    norm = mpl.colors.Normalize(vmin=min(means), vmax=max(means))
    edge_colors = [plt.get_cmap("viridis")(norm(m)) for m in means]

    ax.scatter(
        xs, ys,
        c=face_colors,
        s=25, alpha=0.85, zorder=20,
        edgecolors=edge_colors, linewidths=1.5,
    )

    for r, x, y in zip(optimal_results, xs, ys):
        short = CLUB_STYLES.get(r["club"], {"short": r["club"]})["short"]
        ax.text(x - 2, y + 2.5, f'{short},{int(r["aim_offset"]):+}', fontsize=5, color="black", zorder=21)

    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("ESHO (Expected Strokes to Hole Out)")

    legend_patches = [
        mpatches.Patch(facecolor=v["color"], edgecolor="k", label=v["short"])
        for v in CLUB_STYLES.values()
    ]
    ax.legend(handles=legend_patches, title="Club", loc="upper left", fontsize=7)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("Saved approach plot → %s", output_path)
    plt.close(fig)
