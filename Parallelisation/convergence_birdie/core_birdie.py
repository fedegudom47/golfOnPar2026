"""
core_birdie.py – Hole geometry, birdie GPR, and birdie probability simulation.

Mirrors convergence/core.py but targets P(birdie) instead of ESHO:
  - Green model: VariationalGP + BernoulliLikelihood (probit/logit link)
    trained on binary 1-putt labels (1 = 1-putt = birdie, 0 = 2+ putts).
  - Shot simulation: for each sampled landing position,
      * on green  → query birdie GPR for P(1-putt | x, y)
      * off green → P(birdie) = 0
  - Optimisation: maximise mean birdie probability per grid point.

Data files expected in data_dir (default Parallelisation/data/):
    hole_9_data.csv
    newshapes.geojson
    gpr_green_dataset.csv        (columns: x, y, simulated_strokes)
    simulated_lpga_shot_data2.csv
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
from shapely import wkt as shapely_wkt
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import Point

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPyTorch model: VariationalGP + BernoulliLikelihood for birdie on green
# ---------------------------------------------------------------------------

class _BirdieGreenModel(gpytorch.models.ApproximateGP):
    """Approximate GP classifier: P(1-putt | x, y on green)."""

    def __init__(self, inducing_points: torch.Tensor) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
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
class BirdieHoleData:
    """All geometry, distributions, and trained birdie model for one hole.

    Analogous to HoleData in core.py but without Broadie tables (off-green
    shots contribute zero birdie probability — no stroke lookup needed).
    """
    hole_9: pd.DataFrame
    tee_point: tuple[float, float]
    hole: tuple[float, float]
    strategy_points: list[tuple[float, float]]
    green_polygon: object
    fairway_polygons: list
    water_polygons: list
    bunker_polygons: list
    rough_polygons: list
    new_fairway: gpd.GeoDataFrame
    new_hazard3: gpd.GeoDataFrame
    club_distributions: dict[str, dict]
    rough_distributions: dict[str, dict]
    birdie_model: _BirdieGreenModel
    birdie_likelihood: gpytorch.likelihoods.BernoulliLikelihood


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def build_hole_birdie(
    data_dir: Optional[Path] = None,
    gp_training_iter: int = 200,
    carry_shift_yards: float = 0.0,
    variance_scale: float = 1.0,
) -> BirdieHoleData:
    """Load data and set up the Par-4 hole for birdie simulation.

    Parameters
    ----------
    carry_shift_yards : float
        Yards added to the mean carry of every club (positive = farther).
    variance_scale : float
        Multiplier applied to every club's covariance matrix.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    logger.info("Building birdie hole from data_dir=%s", data_dir)

    # ------------------------------------------------------------------
    # 1. Load hole_9 CSV and parse WKT
    # ------------------------------------------------------------------
    hole_9 = pd.read_csv(data_dir / "hole_9_data.csv")
    hole_9["geometry"] = hole_9["WKT"].apply(shapely_wkt.loads)

    # ------------------------------------------------------------------
    # 2. Find tee point
    # ------------------------------------------------------------------
    def _centroid(row: pd.Series) -> tuple[float, float]:
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
    # 4. First alignment: shift fairway centroid to (0, 100), rotate
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
    # 5. Shift hole_9 shapes +160 yards
    # ------------------------------------------------------------------
    y_shift_base = 160
    hole_pin: tuple[float, float] = (5, 174 + y_shift_base)

    shift_idx = hole_9[hole_9["lie"].isin(["green", "bunker", "water_hazard", "fairway"])].index

    def _shift_wkt(wkt_str: str) -> str:
        shape = shapely_wkt.loads(wkt_str)
        return shp_translate(shape, yoff=y_shift_base).wkt

    hole_9.loc[shift_idx, "WKT"] = hole_9.loc[shift_idx, "WKT"].apply(_shift_wkt)
    hole_9["geometry"] = hole_9["WKT"].apply(shapely_wkt.loads)

    # ------------------------------------------------------------------
    # 6. Final positioning of new_fairway and new_hazard3
    # ------------------------------------------------------------------
    fc = new_fairway.iloc[0].geometry.centroid
    new_fairway["geometry"] = new_fairway["geometry"].apply(
        lambda g: shp_translate(g, xoff=20.0 - fc.x, yoff=175.0 - fc.y)
    )
    new_fairway["geometry"] = new_fairway["geometry"].apply(
        lambda g: shp_rotate(g, angle=-68, origin="centroid", use_radians=False)
    )

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

    green_polygon  = _get_polygons(hole_9, "green")[0]
    bunker_polygons = _get_polygons(hole_9, "bunker")
    rough_polygons  = _get_polygons(hole_9, "rough")
    fairway_polygons = (
        _get_polygons(hole_9, "fairway") + list(new_fairway["geometry"])
    )
    water_polygons = (
        _get_polygons(hole_9, "water_hazard") + list(new_hazard3["geometry"])
    )

    # ------------------------------------------------------------------
    # 8. Strategy grid
    # ------------------------------------------------------------------
    hole_vec    = np.array(hole_pin)
    tee_vec     = np.array(tee_point)
    ht_length   = float(np.linalg.norm(hole_vec - tee_vec))

    x_vals = np.linspace(-40, 60, int(100 / 10))
    y_vals = np.linspace(50, ht_length - 50, int((ht_length - 50) / 10))
    strategy_points = [(float(x), float(y)) for y in y_vals for x in x_vals]

    logger.info("Strategy grid: %d points (ht_length=%.1f)", len(strategy_points), ht_length)

    # ------------------------------------------------------------------
    # 9. Club distributions with optional carry shift and variance scale
    # ------------------------------------------------------------------
    lpga_clubs = pd.read_csv(data_dir / "simulated_lpga_shot_data2.csv")
    club_distributions: dict = {}
    for club, group in lpga_clubs.groupby("Club"):
        mu  = group[["Side", "Carry"]].mean().to_numpy()
        cov = np.cov(group[["Side", "Carry"]].T)
        # Fix Side-Carry correlation sign (see core.py for explanation).
        cov[0, 1] = abs(cov[0, 1])
        cov[1, 0] = abs(cov[1, 0])
        mu[1]  += carry_shift_yards
        cov    *= variance_scale
        club_distributions[club] = {"mean": mu, "cov": cov}

    # Rough distributions (applied on top of already-shifted/scaled base)
    club_names_sorted = sorted(
        club_distributions.keys(),
        key=lambda c: club_distributions[c]["mean"][1],
    )
    n_clubs = len(club_names_sorted)
    rough_distributions: dict = {}
    for i, club in enumerate(club_names_sorted):
        t = i / (n_clubs - 1) if n_clubs > 1 else 0.0
        carry_loss   = 0.05 + t * (0.17 - 0.05)
        var_increase = 0.10 + t * (0.40 - 0.10)
        mean = club_distributions[club]["mean"].copy()
        cov  = club_distributions[club]["cov"].copy()
        mean[1] *= (1.0 - carry_loss)
        cov      *= (1.0 + var_increase)
        rough_distributions[club] = {"mean": mean, "cov": cov}

    # ------------------------------------------------------------------
    # 10. Birdie GPR: VariationalGP + BernoulliLikelihood
    #     Target: P(1-putt) i.e. y=1 if simulated_strokes==1, else 0
    # ------------------------------------------------------------------
    putts_df = pd.read_csv(data_dir / "gpr_green_dataset.csv").copy()
    putts_df["y"] += 160

    X_train = torch.tensor(putts_df[["x", "y"]].values, dtype=torch.float32)
    y_train = torch.tensor(
        (putts_df["simulated_strokes"] == 1).astype(float).values,
        dtype=torch.float32,
    )

    birdie_likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    birdie_model = _BirdieGreenModel(inducing_points=X_train.clone())

    birdie_model.train()
    birdie_likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": birdie_model.parameters()},
            {"params": birdie_likelihood.parameters()},
        ],
        lr=0.1,
    )
    mll = gpytorch.mlls.VariationalELBO(birdie_likelihood, birdie_model, num_data=len(y_train))

    for _ in range(gp_training_iter):
        optimizer.zero_grad()
        output = birdie_model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    birdie_model.eval()
    birdie_likelihood.eval()
    logger.info("Birdie GPR trained (%d iterations).", gp_training_iter)

    return BirdieHoleData(
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
        birdie_model=birdie_model,
        birdie_likelihood=birdie_likelihood,
    )


# ---------------------------------------------------------------------------
# Lie detection (identical to core.py)
# ---------------------------------------------------------------------------

def get_lie_category(point: tuple[float, float], hole: BirdieHoleData) -> str:
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
# Birdie probability helpers
# ---------------------------------------------------------------------------

def evaluate_birdie_prob(
    point: tuple[float, float],
    model: _BirdieGreenModel,
    likelihood: gpytorch.likelihoods.BernoulliLikelihood,
) -> float:
    """Return P(1-putt | landing position on green)."""
    test_x = torch.tensor([[point[0], point[1]]], dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
    return float(pred.probs.item())


# ---------------------------------------------------------------------------
# Shot rotation / geometry (identical to core.py)
# ---------------------------------------------------------------------------

def rotation_translator(
    x_side: float,
    y_carry: float,
    angle_deg: float,
    starting_point: tuple[float, float],
    target: tuple[float, float],
) -> tuple[float, float]:
    direction = np.array(target) - np.array(starting_point)
    unit_dir  = direction / np.linalg.norm(direction)
    angle_rad = np.radians(angle_deg)
    rot = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)],
    ])
    local_shot = np.array([x_side, y_carry])
    rotated    = rot @ local_shot
    global_vec = np.array([
        unit_dir[0] * rotated[1] - unit_dir[1] * rotated[0],
        unit_dir[1] * rotated[1] + unit_dir[0] * rotated[0],
    ])
    result = np.array(starting_point) + global_vec
    return (float(result[0]), float(result[1]))


# ---------------------------------------------------------------------------
# Birdie approach simulation
# ---------------------------------------------------------------------------

BirdieAccumulator = dict  # (start_x, start_y, club, aim_offset) → np.ndarray of probs


def simulate_approach_shots_birdie(
    hole: BirdieHoleData,
    n_new: int,
    accumulator: Optional[BirdieAccumulator] = None,
    strategy_points: Optional[list] = None,
    aim_range: tuple[float, float] = (-20.0, 20.0),
    aim_step: float = 2.0,
) -> tuple[list[dict], BirdieAccumulator]:
    """Simulate n_new approach shots per (grid-point, club, aim) and track
    birdie probability via incremental accumulation.

    Returns
    -------
    optimal_results : list[dict]
        One entry per grid point:
        {start, club, aim_offset, mean_birdie_prob, var_birdie_prob, n_total}
    new_accumulator : BirdieAccumulator
        Pass back into the next call.
    """
    if strategy_points is None:
        strategy_points = hole.strategy_points
    if accumulator is None:
        accumulator = {}

    target     = hole.hole
    aim_points = list(np.arange(aim_range[0], aim_range[1] + aim_step, aim_step))

    clubs_avg_carry = {
        club: stats["mean"][1]
        for club, stats in hole.club_distributions.items()
    }

    new_accumulator: BirdieAccumulator = {}
    optimal_results: list[dict] = []

    for starting_point in strategy_points:
        starting_lie = get_lie_category(starting_point, hole)

        # Water starting positions: no chance of birdie on this hole
        if starting_lie == "water":
            optimal_results.append({
                "start":            starting_point,
                "club":             "Driver",
                "aim_offset":       0.0,
                "mean_birdie_prob": 0.0,
                "var_birdie_prob":  0.0,
                "n_total":          0,
            })
            continue

        total_distance = float(np.linalg.norm(
            np.array(target) - np.array(starting_point)
        ))

        # Top-5 clubs by carry proximity; Driver excluded — approach shots only
        top_clubs = [
            club for club, _ in sorted(
                ((c, abs(carry - total_distance)) for c, carry in clubs_avg_carry.items()
                 if c != "Driver"),
                key=lambda item: item[1],
            )[:5]
        ]

        best_res: Optional[dict] = None

        for club in top_clubs:
            if starting_lie == "rough":
                mu  = hole.rough_distributions[club]["mean"]
                cov = hole.rough_distributions[club]["cov"]
            else:
                mu  = hole.club_distributions[club]["mean"]
                cov = hole.club_distributions[club]["cov"]

            for aim_offset in aim_points:
                key = (starting_point[0], starting_point[1], club, aim_offset)
                angle_deg = (
                    float(np.degrees(np.arctan(aim_offset / total_distance)))
                    if total_distance > 0 else 0.0
                )

                new_samples = np.random.multivariate_normal(mu, cov, size=n_new)
                new_probs: list[float] = []

                for shot in new_samples:
                    lp = rotation_translator(
                        float(shot[0]), float(shot[1]),
                        angle_deg, starting_point, target,
                    )
                    if get_lie_category(lp, hole) == "green":
                        p = evaluate_birdie_prob(lp, hole.birdie_model, hole.birdie_likelihood)
                    else:
                        p = 0.0
                    new_probs.append(p)

                prior    = accumulator.get(key, np.array([], dtype=np.float32))
                combined = np.concatenate([prior, np.array(new_probs, dtype=np.float32)])
                new_accumulator[key] = combined

                if len(combined) == 0:
                    continue

                mean_val = float(np.mean(combined))
                var_val  = float(np.var(combined))

                if best_res is None or mean_val > best_res["mean_birdie_prob"]:
                    best_res = {
                        "start":            starting_point,
                        "club":             club,
                        "aim_offset":       float(aim_offset),
                        "mean_birdie_prob": mean_val,
                        "var_birdie_prob":  var_val,
                        "n_total":          int(len(combined)),
                    }

        if best_res is not None:
            optimal_results.append(best_res)

    return optimal_results, new_accumulator


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def results_to_dataframe(
    optimal_results: list[dict],
    seed: int,
    N: int,
) -> pd.DataFrame:
    rows = []
    for r in optimal_results:
        rows.append({
            "x":               float(r["start"][0]),
            "y":               float(r["start"][1]),
            "club":            r["club"],
            "aim_offset":      float(r["aim_offset"]),
            "mean_birdie_prob": float(r["mean_birdie_prob"]),
            "var_birdie_prob":  float(r["var_birdie_prob"]),
            "n_total":         int(r.get("n_total", N)),
            "seed":            seed,
            "N":               N,
        })
    return pd.DataFrame(rows)


CLUB_STYLES: dict[str, dict] = {
    "Driver":    {"short": "D",    "color": "#000000"},
    "3-wood":    {"short": "3w",   "color": "#FFAE00"},
    "5-wood":    {"short": "5w",   "color": "#CC79A7"},
    "Hybrid":    {"short": "Hy",   "color": "#009E73"},
    "4 Iron":    {"short": "4i",   "color": "royalblue"},
    "5 Iron":    {"short": "5i",   "color": "#56B4E9"},
    "6 Iron":    {"short": "6i",   "color": "#4DAF4A"},
    "7 Iron":    {"short": "7i",   "color": "#D55E00"},
    "8 Iron":    {"short": "8i",   "color": "#984EA3"},
    "9 Iron":    {"short": "9i",   "color": "#FFEE00"},
    "PW":        {"short": "Pw",   "color": "#0C034F"},
    "50 deg":    {"short": "50",   "color": "#E41A1C"},
    "54 deg":    {"short": "54",   "color": "#999999"},
    "60 deg":    {"short": "60",   "color": "#3E1F1F"},
    "H 50 deg":  {"short": "H50",  "color": "#FF6B6B"},
    "H 54 deg":  {"short": "H54",  "color": "#C0C0C0"},
    "H 60 deg":  {"short": "H60",  "color": "#8B6565"},
    "3Q 60 deg": {"short": "3Q60", "color": "#A0522D"},
    "1Q 60 deg": {"short": "1Q60", "color": "#D2B48C"},
    "1E 60 deg": {"short": "1E60", "color": "#F5DEB3"},
}

_LIE_COLORS = {
    "bunker":       "tan",
    "fairway":      "forestgreen",
    "new_fairway":  "forestgreen",
    "green":        "lightgreen",
    "OB":           "lightcoral",
    "rough":        "mediumseagreen",
    "tee":          "darkgreen",
    "water_hazard": "skyblue",
    "new_hazard3":  "skyblue",
}


def _plot_hole_layout(hole: BirdieHoleData, title: str, ax: object) -> None:
    for _, row in hole.hole_9.iterrows():
        geom  = shapely_wkt.loads(row["WKT"])
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
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, linestyle=":")


def plot_optimal_approaches_birdie(
    optimal_results: list[dict],
    hole: BirdieHoleData,
    title: str = "Optimal Birdie Strategy",
    figsize: tuple = (14, 16),
    output_path: Optional[Path] = None,
    match_rate: Optional[float] = None,
) -> None:
    import matplotlib as mpl

    if match_rate is not None:
        title = f"{title}  |  Match rate vs prev: {match_rate * 100:.1f}%"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_hole_layout(hole, title, ax)

    xs   = [r["start"][0]            for r in optimal_results]
    ys   = [r["start"][1]            for r in optimal_results]
    prbs = [r["mean_birdie_prob"]    for r in optimal_results]

    face_colors = [
        CLUB_STYLES.get(r["club"], {"color": "#999999"})["color"]
        for r in optimal_results
    ]
    norm        = mpl.colors.Normalize(vmin=min(prbs), vmax=max(prbs))
    edge_colors = [plt.get_cmap("plasma")(norm(p)) for p in prbs]

    ax.scatter(
        xs, ys,
        c=face_colors, s=25, alpha=0.85, zorder=20,
        edgecolors=edge_colors, linewidths=1.5,
    )

    for r, x, y in zip(optimal_results, xs, ys):
        short = CLUB_STYLES.get(r["club"], {"short": r["club"]})["short"]
        ax.text(x - 2, y + 2.5,
                f'{short},{int(r["aim_offset"]):+}',
                fontsize=5, color="black", zorder=21)

    sm = mpl.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("P(birdie) — expected birdie probability")

    legend_patches = [
        mpatches.Patch(facecolor=v["color"], edgecolor="k", label=v["short"])
        for v in CLUB_STYLES.values()
    ]
    ax.legend(handles=legend_patches, title="Club", loc="upper left", fontsize=7)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("Saved birdie approach plot → %s", output_path)
    plt.close(fig)
