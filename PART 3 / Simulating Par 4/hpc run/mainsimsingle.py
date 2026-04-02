import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from shapely.affinity import translate, rotate
import torch
import gpytorch
from scipy.interpolate import interp1d
from shapely.geometry import LineString
import matplotlib.patches as mpatches
import sys, os
from datetime import datetime

# DATA FILES
HOLE_9 = "hole_9_data.csv"
NEW_HAZARDS = "newshapes.geojson"
PUTTS_DATASETOG = "gpr_green_dataset.csv"
BROADIE_ESTIMATES = "strokes_by_lie_yards_broadie.csv"
LPGA_CLUB_DATA = "simulated_lpga_shot_data2.csv"

# --- LOAD DATA ---
putts_og_loc = pd.read_csv(PUTTS_DATASETOG)
lpga_clubs = pd.read_csv(LPGA_CLUB_DATA)
broadie_data = pd.read_csv(BROADIE_ESTIMATES)
hole_9 = pd.read_csv(HOLE_9)
gdf = gpd.read_file(NEW_HAZARDS)

# --- UTILS / MODELS ---
def hole_layout(target_x3 = 0, target_y3 = 230):
    hole = (5, 174)
    hole_9_copy = hole_9.copy()
    hole_9_copy["geometry"] = hole_9_copy["WKT"].apply(wkt.loads)
    gdf_copy = gdf.copy()

    def get_centroid(row):
        shape = wkt.loads(row["WKT"])
        return shape.centroid.coords[0]
    teeboxes = hole_9_copy[hole_9_copy["lie"].str.contains("tee", case=False)]
    green = hole_9_copy[hole_9_copy["lie"] == "green"].iloc[0]
    green_shape = wkt.loads(green["WKT"])
    green_centre = get_centroid(green)
    teeboxes["centroid"] = teeboxes.apply(get_centroid, axis=1)
    teeboxes["dist_to_green"] = teeboxes["centroid"].apply(
        lambda pt: np.linalg.norm(np.array(pt) - np.array(green_centre))
    )
    longest_teebox = teeboxes.loc[teeboxes["dist_to_green"].idxmax()]
    tee_point = longest_teebox["centroid"]

    new_fairway = gdf_copy[gdf_copy["lie"] == "fairway"]
    new_hazard3 = gdf_copy[gdf_copy["lie"] == "water_hazard_3"]

    new_fairway = new_fairway.to_crs(epsg=32611)
    new_hazard3 = new_hazard3.to_crs(epsg=32611)

    ref_centroid = new_fairway.iloc[0].geometry.centroid
    target_x, target_y = 0, 100
    x_shift = target_x - ref_centroid.x
    y_shift = target_y - ref_centroid.y

    def shift_geometry(geom):
        return translate(geom, xoff=x_shift, yoff=y_shift)

    new_fairway["geometry"] = new_fairway["geometry"].apply(shift_geometry)
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(shift_geometry)

    fairway_poly = new_fairway.iloc[0].geometry
    minx, miny, maxx, maxy = fairway_poly.bounds
    start = np.array([fairway_poly.centroid.x, miny])
    end = np.array([fairway_poly.centroid.x, maxy])
    vec = end - start
    angle_rad = np.arctan2(vec[0], vec[1])
    angle_deg = np.degrees(angle_rad)

    def rotate_geometry(geom, angle, origin="centroid"):
        return rotate(geom, angle=-angle, origin=origin, use_radians=False)

    rotation_angle = angle_deg
    new_fairway["geometry"] = new_fairway["geometry"].apply(rotate_geometry, args=(rotation_angle,))
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(rotate_geometry, args=(rotation_angle,))

    shift_targets = hole_9_copy[hole_9_copy["lie"].isin(["green", "bunker", "water_hazard", "fairway"])]
    y_shift = 160
    hole = (5, 174 + y_shift)

    def shift_wkt(wkt_str):
        shape = wkt.loads(wkt_str)
        return translate(shape, yoff=y_shift)

    hole_9_copy.loc[shift_targets.index, "WKT"] = shift_targets["WKT"].apply(shift_wkt).apply(lambda g: g.wkt)

    fairway_centroid = new_fairway.iloc[0].geometry.centroid
    target_x, target_y = 20, 175
    x_shift = target_x - fairway_centroid.x
    y_shift = target_y - fairway_centroid.y

    new_fairway["geometry"] = new_fairway["geometry"].apply(
        lambda geom: translate(geom, xoff=x_shift, yoff=y_shift)
    )
    new_fairway["geometry"] = new_fairway["geometry"].apply(
        lambda geom: rotate(geom, angle=-68, origin="centroid", use_radians=False)
    )

    hazard3_centroid = new_hazard3.iloc[0].geometry.centroid
    x3_shift = target_x3 - hazard3_centroid.x
    y3_shift = target_y3 - hazard3_centroid.y

    new_hazard3["geometry"] = new_hazard3["geometry"].apply(
        lambda geom: translate(geom, xoff=x3_shift, yoff=y3_shift)
    )
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(
        lambda geom: rotate(geom, angle=110, origin="centroid", use_radians=False)
    )

    hole_vec = np.array(hole)
    tee_vec= np.array(tee_point)
    hole_to_tee_vec = hole_vec - tee_vec
    ht_length = np.linalg.norm(hole_to_tee_vec)

    x_vals = np.linspace(-40, 60, int(100 / 10))
    y_vals = np.linspace(50, ht_length - 50, int((ht_length - 50) / 10))
    strategy_points = [(x, y) for y in y_vals for x in x_vals]

    print("finished hole generation")

    return {
        'hole_data': hole_9_copy,
        'new_fairway': new_fairway,
        'new_hazard3': new_hazard3,
        'hole_position': hole,
        'tee_point': tee_point,
        'strategy_points': strategy_points,
        'hole_to_tee_length': ht_length,
        'green_shape': green_shape,
        "green_centre": green_centre,
        "water_loc" : target_y3
    }

# shift putts
shifted_putts = putts_og_loc.copy()
shifted_putts["y"] = shifted_putts["y"] + 160

# CLUB DISTRIBUTIONS
def generate_fairway_dist():
    club_distributions = {}
    for club, group in lpga_clubs.groupby("Club"):
        mean = group[["Side", "Carry"]].mean().to_numpy()
        cov = np.cov(group[["Side", "Carry"]].T)
        club_distributions[club] = {"mean": mean, "cov": cov}
    return club_distributions

club_distributions = generate_fairway_dist()

def generate_rough_dist():
    club_names_list = list(club_distributions.keys())
    sorted_clubs = sorted(club_names_list, key=lambda c: club_distributions[c]["mean"][1])
    min_loss, max_loss = 0.05, 0.17
    min_var_increase, max_var_increase = 0.10, 0.40
    rough_distributions = {}
    n = len(sorted_clubs)
    for i, club in enumerate(sorted_clubs):
        t = i / (n - 1) if n > 1 else 0
        carry_loss = min_loss + t * (max_loss - min_loss)
        var_increase = min_var_increase + t * (max_var_increase - min_var_increase)
        mean, cov = club_distributions[club]["mean"].copy(), club_distributions[club]["cov"].copy()
        mean[1] *= (1 - carry_loss)
        cov *= (1 + var_increase)
        rough_distributions[club] = {"mean": mean, "cov": cov}
    return rough_distributions

rough_distributions = generate_rough_dist()

def get_club_distribution(club):
    return club_distributions[club]["mean"], club_distributions[club]["cov"]

def get_rough_distribution(club):
    return rough_distributions[club]["mean"], rough_distributions[club]["cov"]

# GP for putts
X_train = torch.tensor(shifted_putts[["x", "y"]].values, dtype=torch.float32)
y_train = torch.tensor(shifted_putts["simulated_strokes"].values, dtype=torch.float32)

class PuttModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood_putt = gpytorch.likelihoods.GaussianLikelihood()
model_putt = PuttModel(X_train, y_train, likelihood_putt)
model_putt.train()
likelihood_putt.train()
optimizer = torch.optim.Adam(model_putt.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_putt, model_putt)
training_iter = 100
for i in range(training_iter):
    optimizer.zero_grad()
    output = model_putt(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
model_putt.eval()
likelihood_putt.eval()

def evaluate_on_green(point, model=model_putt, likelihood=likelihood_putt):
    model.eval()
    likelihood.eval()
    test_x = torch.tensor([[point[0], point[1]]], dtype=torch.float)
    with torch.no_grad():
        pred = likelihood(model(test_x))
        mean = pred.mean.item()
        std = pred.stddev.item()
    return mean, std

# Broadie interpolation
broadie_interpolators = {}
broadie_data = broadie_data.rename(columns={"Distance (yards)": "distance"})
broadie_long = broadie_data.melt(id_vars="distance", var_name="lie", value_name="strokes")
broadie_long["lie"] = broadie_long["lie"].str.lower()
for lie, group in broadie_long.groupby("lie"):
    group = group.sort_values("distance")
    f = interp1d(
        group["distance"], group["strokes"],
        kind="linear", fill_value="extrapolate"
    )
    broadie_interpolators[lie.lower()] = f

def interpolate_broadie(lie, distance):
    lie = lie.lower()
    if lie not in broadie_interpolators:
        raise ValueError(f"No interpolator found for lie: {lie}")
    return broadie_interpolators[lie](distance)

def evaluate_broadie(point, target, lie):
    dist = np.linalg.norm(np.array(point) - np.array(target))
    return interpolate_broadie(lie, dist)

# Polygons / lie detection
def get_polygons(df, lie_type):
    return [geom for geom in df[df["lie"] == lie_type]["geometry"]]

def define_polygons(hole_dic):
    return {
        "bunker": get_polygons(hole_dic["hole_data"], "bunker"),
        "rough": get_polygons(hole_dic["hole_data"], "rough"),
        "green": get_polygons(hole_dic["hole_data"], "green")[0],
        "fairway": get_polygons(hole_dic["hole_data"], "fairway") + list(hole_dic["new_fairway"]["geometry"]),
        "water": get_polygons(hole_dic["hole_data"], "water") + list(hole_dic["new_hazard3"]["geometry"])
    }

def get_lie_category(point, hole_dic):
    polygons = define_polygons(hole_dic)
    pt = Point(point)
    if polygons["green"].contains(pt):
        return "green"
    if any(poly.contains(pt) for poly in polygons["water"]):
        return "water"
    if any(poly.contains(pt) for poly in polygons["bunker"]):
        return "bunker"
    if any(poly.contains(pt) for poly in polygons["fairway"]):
        return "fairway"
    if any(poly.contains(pt) for poly in polygons["rough"]):
        return "rough"
    return "rough"

def get_water_intersection(starting_point, ball_in_water, hole_dic):
    water_polygons = define_polygons(hole_dic)["water"]
    tee_point = hole_dic["tee_point"]
    shot_line = LineString([starting_point, ball_in_water])
    closest_intersection = None
    min_dist = float("inf")
    for poly in water_polygons:
        intersection = shot_line.intersection(poly.boundary)
        if intersection.is_empty:
            continue
        if isinstance(intersection, Point):
            dist_from_start = Point(starting_point).distance(intersection)
            if dist_from_start < min_dist:
                min_dist = dist_from_start
                closest_intersection = intersection
        else:
            for pt in intersection.geoms:
                if isinstance(pt, Point):
                    dist_from_start = Point(tee_point).distance(pt)
                    if dist_from_start < min_dist:
                        min_dist = dist_from_start
                        closest_intersection = pt
    if closest_intersection:
        return (closest_intersection.x, closest_intersection.y)
    else:
        return None

def evaluate_water_hazard(starting_point, point, target, hole_dic, model=model_putt, likelihood=likelihood_putt):
    drop_location = get_water_intersection(starting_point, point, hole_dic)
    green_polygon = define_polygons(hole_dic)["green"]
    if drop_location:
        if green_polygon.contains(Point(drop_location)):
            return 1 + evaluate_on_green(drop_location, model, likelihood)[0]
        else:
            return 1 + evaluate_broadie(drop_location, target, "rough")
    return np.nan

def evaluate_shot(point, starting_point, target, polygons, hole_dic, model, likelihood):
    lie = get_lie_category(point, hole_dic)
    if lie == "green":
        return evaluate_on_green(point, model, likelihood)[0]
    elif lie == "water":
        return evaluate_water_hazard(starting_point, point, target, hole_dic, model, likelihood)
    elif lie == "fairway":
        return evaluate_broadie(point, target, "fairway")
    elif lie == "rough":
        return evaluate_broadie(point, target, "rough")
    elif lie == "bunker":
        return evaluate_broadie(point, target, "sand")
    else:
        return evaluate_broadie(point, target, "rough")

def rotation_translator(x_side, y_carry, angle, starting_point, target):
    line_to_target = np.array(target) - np.array(starting_point)
    base_angle = np.arctan2(line_to_target[1], line_to_target[0])
    theta = -np.radians(angle) + base_angle
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    original_vec = np.array([x_side, -y_carry])
    rotated_vec = rot_matrix @ original_vec
    global_rot = np.array([
        [np.cos(base_angle), -np.sin(base_angle)],
        [np.sin(base_angle), np.cos(base_angle)]
    ])
    final_vec = global_rot @ rotated_vec
    return tuple(final_vec + np.array(starting_point))

def rotation_translator2(x_side, y_carry, angle_deg, starting_point, target):
    direction = np.array(target) - np.array(starting_point)
    unit_direction = direction / np.linalg.norm(direction)
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    local_shot = np.array([x_side, y_carry])
    rotated_shot = rot_matrix @ local_shot
    global_direction = np.array([
        unit_direction[0] * rotated_shot[1] - unit_direction[1] * rotated_shot[0],
        unit_direction[1] * rotated_shot[1] + unit_direction[0] * rotated_shot[0]
    ])
    return tuple(np.array(starting_point) + global_direction)

def yard_offset_to_angle(yards_offset, starting_point, target):
    starting_point = np.array(starting_point)
    target = np.array(target)
    dis_target = np.linalg.norm(target - starting_point)
    angle_rad = np.arctan(yards_offset / dis_target)
    return np.degrees(angle_rad)

def angle_offset_to_yard(angle_offset, starting_point, target):
    starting_point = np.array(starting_point)
    target = np.array(target)
    dis_target = np.linalg.norm(target - starting_point)
    angle_rad = np.radians(angle_offset)
    offset = np.tan(angle_rad) * dis_target
    return offset

def simulate_and_evaluate(starting_point, target, club, hole_dic, model, likelihood, aim_offset=0, n_samples=100):
    polygons = define_polygons(hole_dic)
    starting_lie = get_lie_category(starting_point, hole_dic)
    if starting_lie == "rough":
        mu, cov = get_rough_distribution(club)
    else:
        mu, cov = get_club_distribution(club)

    raw_samples = np.random.multivariate_normal(mu, cov, size=n_samples)
    total_distance = np.linalg.norm(np.array(target) - np.array(starting_point))
    angle_rad = np.arctan(aim_offset / total_distance)
    angle_deg = np.degrees(angle_rad)

    evaluated = []
    for shot in raw_samples:
        x_rot, y_rot = rotation_translator2(shot[0], shot[1], angle_deg, starting_point, target)
        es = evaluate_shot(
            (x_rot, y_rot),
            starting_point,
            target,
            polygons,
            hole_dic,
            model,
            likelihood
        )
        evaluated.append({
            "original_shot": (shot[0], shot[1]),
            "landing_point": (x_rot, y_rot),
            "expected_strokes": es
        })
    all_strokes = [d["expected_strokes"] for d in evaluated if not np.isnan(d["expected_strokes"])]
    mean_es = np.mean(all_strokes) if all_strokes else np.nan
    var_es = np.var(all_strokes) if all_strokes else np.nan
    return {"evaluated": evaluated, "mean": mean_es, "variance": var_es}

def simulate_approach_shots(starting_points, clubs, hole_dic, model, likelihood,
                            target=None, aim_range=(-20, 20), aim_step=2, n_samples=100):
    if target is None:
        target = hole_dic["hole_position"]
    tee_point = hole_dic["tee_point"]
    polygons = define_polygons(hole_dic)
    aim_points = range(aim_range[0], aim_range[1] + 1, aim_step)
    clubs_avg_carry = {club: stats["mean"][1] for club, stats in club_distributions.items()}
    optimal_results = []

    for starting_point in starting_points:
        starting_lie = get_lie_category(starting_point, hole_dic)
        if starting_lie == "water":
            playing_location = get_water_intersection(tee_point, starting_point, hole_dic)
            total_distance = np.linalg.norm(np.array(target) - np.array(playing_location))
        else:
            playing_location = starting_point
            total_distance = np.linalg.norm(np.array(target) - np.array(playing_location))
        club_diffs = [(club, abs(avg_carry - total_distance)) for club, avg_carry in clubs_avg_carry.items()]
        top_5_clubs = [club for club, _ in sorted(club_diffs, key=lambda x: x[1])[:5]]
        best_res = None
        for club in top_5_clubs:
            for aim_offset in aim_points:
                result = simulate_and_evaluate(
                    starting_point=playing_location,
                    target=target,
                    club=club,
                    aim_offset=aim_offset,
                    n_samples=n_samples,
                    hole_dic=hole_dic,
                    model=model,
                    likelihood=likelihood
                )
                if np.isnan(result["mean"]):
                    continue
                mean_adjusted = (result["mean"] + 2) if starting_lie == "water" else (result["mean"] + 1)
                if (best_res is None) or (mean_adjusted < best_res["mean"]):
                    best_res = {
                        "start": starting_point,
                        "club": club,
                        "aim_offset": aim_offset,
                        "mean": mean_adjusted,
                        "variance": result["variance"]
                    }
        if best_res:
            optimal_results.append(best_res)
    return optimal_results

club_names = {
    'Driver':    {"short": "D",    "color": "#000000"},
    '3-wood':    {"short": "3w",   "color": "#FFAE00"},
    '5-wood':    {"short": "5w",   "color": "#CC79A7"},
    'Hybrid':    {"short": "Hy",   "color": "#009E73"},
    '4 Iron':    {"short": "4i",   "color": "royalblue"},
    '5 Iron':    {"short": "5i",   "color": "#56B4E9"},
    '6 Iron':    {"short": "6i",   "color": "#4DAF4A"},
    '7 Iron':    {"short": "7i",   "color": "#D55E00"},
    '8 Iron':    {"short": "8i",   "color": "#984EA3"},
    '9 Iron':    {"short": "9i",   "color": "#FFEE00"},
    'PW':        {"short": "Pw",   "color": "#0C034F"},
    '50 deg':    {"short": "50",   "color": "#E41A1C"},
    '54 deg':    {"short": "54",   "color": "#999999"},
    '60 deg':    {"short": "60",   "color": "#D8B3B3"},
    'H 50 deg':  {"short": "H50",   "color": "#3B45D4"},
    '3Q 60 deg': {"short": "3Q60",   "color": "#3E1F1F"},
    'H 54 deg':   {"short": "H54",   "color": "#FF37DE"},
    'H 60 deg':    {"short": "H60",   "color": "#908C0D"},
    '1Q 60 deg':   {"short": "1Q60",   "color": "#F4193A"},
    '1E 60 deg' :  {"short": "1E60",   "color": "#BFC5D2"},
}

def plot_hole_layout(hole_geom_df, new_fairway=None, new_hazards=None,
                     tee_point=None, hole_point=None, fairway_grid=None,
                     title="Hole Layout", figsize=(10, 10), lie_colors=None, plot_approach=True):

    if lie_colors is None:
        lie_colors = {
            "bunker": "tan",
            "fairway": "forestgreen",
            "new_fairway": "forestgreen",
            "green": "lightgreen",
            "OB": "lightcoral",
            "rough": "mediumseagreen",
            "tee": "darkgreen",
            "water_hazard": "skyblue",
            "new_hazard1": "skyblue",
            "new_hazard2": "skyblue",
            "new_hazard3": "skyblue"
        }

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in hole_geom_df.iterrows():
        geom = wkt.loads(row["WKT"])
        color = lie_colors.get(row["lie"], "lightgrey")
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, edgecolor="black", linewidth=0.5)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc=color, edgecolor="black", linewidth=0.5)

    if new_fairway is not None:
        for _, row in new_fairway.iterrows():
            x, y = row["geometry"].exterior.xy
            ax.fill(x, y, alpha=0.5, fc=lie_colors["new_fairway"], edgecolor="black", linewidth=0.5)

    if new_hazards:
        for i, hazard_df in enumerate(new_hazards, start=1):
            key = f"new_hazard{i}"
            if key in lie_colors:
                for _, row in hazard_df.iterrows():
                    x, y = row["geometry"].exterior.xy
                    ax.fill(x, y, alpha=0.5, fc=lie_colors[key], edgecolor="black", linewidth=0.5)

    if tee_point:
        ax.plot(tee_point[0], tee_point[1], marker='x', color='red', markersize=7, label="Tee")
    if hole_point:
        ax.plot(hole_point[0], hole_point[1], marker='o', color='black', markersize=5, label="Hole")

    if plot_approach:
        ax = plt.gca()
        # strategy_points assumed global in original; ensure passed context uses correct variable
        # This function doesn't have strategy_points; it's expected caller uses global or captures.
        try:
            xs, ys = zip(*strategy_points)
            ax.scatter(xs, ys, color='black', s=20, label="Strategy Points", zorder=10, alpha=.4)
            for x, y in strategy_points[::20]:
                ax.text(x, y, f"{int(y)}", fontsize=6, color='red', ha='center')
        except NameError:
            pass

        ax.legend()
        plt.draw()

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    return ax

def plot_optimal_approaches(optimal_results, hole_dic, cmap="plasma", alpha=0.8, zorder=20,
                           title="Optimal Shot Strategy", save_path=None):
    xs = [res["start"][0] for res in optimal_results]
    ys = [res["start"][1] for res in optimal_results]
    means = [res["mean"] for res in optimal_results]

    labels = []
    for res in optimal_results:
        club_raw = res["club"]
        club_info = club_names.get(club_raw, {})
        short = club_info.get("short", club_raw)
        aim = int(res["aim_offset"]) if abs(res["aim_offset"] - int(res["aim_offset"])) < 1e-2 else res["aim_offset"]
        labels.append(f"{short}, {aim}")

    plt.figure(figsize=(20, 18))

    plot_hole_layout(
        hole_geom_df=hole_dic["hole_data"],
        new_fairway=hole_dic["new_fairway"],
        new_hazards=[hole_dic["new_hazard3"]],
        tee_point=hole_dic["tee_point"],
        hole_point=hole_dic["hole_position"],
        title=title,
        plot_approach=False
    )

    ax = plt.gca()
    scatter = ax.scatter(xs, ys, c=means, marker=(5, 2), cmap=cmap, s=10, alpha=alpha, zorder=zorder)

    for x, y, label in zip(xs, ys, labels):
        ax.text(x - 5, y + 2.5, label, fontsize=5.5, color="black", zorder=zorder + 1)

    cbar = plt.colorbar(scatter, ax=ax, label="Expected Strokes to Hole Out")

    ax.legend(loc="upper right", fontsize="small")
    ax.set_aspect("equal")
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# --- GPModelApp & tee evaluation omitted here for brevity if not used in single-run strategy ---

# --- Logging ---
LOG_FILE = "simulation_progress.log"

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

# --- MAIN SINGLE WATER LOCATION RUN ---
if __name__ == "__main__":
    # build layout for one water location (adjust as desired)
    layout = hole_layout(target_x3=0, target_y3=230)
    log(f"=== SINGLE RUN STARTED with water location {layout['water_loc']} ===")

    # simulate optimal approach shots (lighter for laptop)
    optimal_points = simulate_approach_shots(
        starting_points=layout["strategy_points"],
        clubs=list(club_distributions.keys()),
        target=layout["hole_position"],
        aim_range=(-20, 20),
        aim_step=2,
        n_samples=50,
        hole_dic=layout,
        model=model_putt,
        likelihood=likelihood_putt
    )
    log("Optimal approach simulation completed.")

    # save only the optimal strategy plot
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "optimal_strategy.png")
    plot_optimal_approaches(
        optimal_results=optimal_points,
        hole_dic=layout,
        title="Optimal Shot Strategy",
        save_path=output_path
    )
    log(f"Saved optimal strategy to {output_path}")
    log("=== RUN COMPLETE ===")
