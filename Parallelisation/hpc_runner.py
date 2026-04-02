#!/usr/bin/env python3
"""
hpc_runner.py  –  Parallel golf simulation sweep across a parameter grid.

Each worker is fully self-contained (loads data, trains GP, runs sim, saves
outputs) so that no GPyTorch models are ever passed across process boundaries.

LOCAL quick test  (~2-3 min on a laptop):
    python hpc_runner.py --mode local --mechanism optimization --data-dir /path/to/data

HPC (Slurm) full sweep:
    python hpc_runner.py --mode hpc --mechanism optimization --data-dir /path/to/data --workers 16

Required files in --data-dir:
    hole_9_data.csv
    newshapes.geojson
    gpr_green_dataset.csv
    strokes_by_lie_yards_broadie.csv
    simulated_lpga_shot_data.csv   (or simulated_lpga_shot_data2.csv)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive; safe on HPC nodes with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- golf / geometry imports ---
import geopandas as gpd
import torch
import gpytorch
from scipy.interpolate import interp1d
from shapely import wkt
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import LineString, Point

# --- our simulator module ---
sys.path.insert(0, str(Path(__file__).parent))
from src.core.simulator import (
    BirdieMechanism,
    Config,
    OptimizationMechanism,
    Player,
    SimulationResult,
    Simulator,
)

# ---------------------------------------------------------------------------
# Task / result containers
# ---------------------------------------------------------------------------

@dataclass
class RunTask:
    """One cell in the parameter grid — fully pickleable (plain Python types)."""
    dispersion_multiplier: float
    power_multiplier: float
    water_loc_y: float
    mechanism_type: Literal["optimization", "birdie"]
    n_shots: int
    n_strategy_pts: int          # subsample for local; 0 = use all
    gp_training_iter: int
    data_dir: str
    output_dir: str
    job_id: str                  # unique string for file naming


@dataclass
class RunResult:
    job_id: str
    n_results: int
    output_plot: str
    output_data: str
    status: str
    error: str = ""


# ---------------------------------------------------------------------------
# Golf helper functions  (module-level → pickleable)
# ---------------------------------------------------------------------------

def _load_data(data_dir: str) -> dict:
    d = Path(data_dir)
    lpga_name = (
        "simulated_lpga_shot_data2.csv"
        if (d / "simulated_lpga_shot_data2.csv").exists()
        else "simulated_lpga_shot_data.csv"
    )
    return {
        "putts":   pd.read_csv(d / "gpr_green_dataset.csv"),
        "lpga":    pd.read_csv(d / lpga_name),
        "broadie": pd.read_csv(d / "strokes_by_lie_yards_broadie.csv"),
        "hole9":   pd.read_csv(d / "hole_9_data.csv"),
        "hazards": gpd.read_file(d / "newshapes.geojson"),
    }


def _build_hole_layout(data: dict, water_loc_y: float) -> dict:
    hole_9 = data["hole9"].copy()
    gdf    = data["hazards"].copy()
    hole_9["geometry"] = hole_9["WKT"].apply(wkt.loads)

    def centroid(row):
        return wkt.loads(row["WKT"]).centroid.coords[0]

    teeboxes = hole_9[hole_9["lie"].str.contains("tee", case=False)].copy()
    green_row = hole_9[hole_9["lie"] == "green"].iloc[0]
    green_shape  = wkt.loads(green_row["WKT"])
    green_centre = centroid(green_row)

    teeboxes["centroid"] = teeboxes.apply(centroid, axis=1)
    teeboxes["dist_to_green"] = teeboxes["centroid"].apply(
        lambda pt: np.linalg.norm(np.array(pt) - np.array(green_centre))
    )
    tee_point = teeboxes.loc[teeboxes["dist_to_green"].idxmax()]["centroid"]

    new_fairway  = gdf[gdf["lie"] == "fairway"].to_crs(epsg=32611).copy()
    new_hazard3  = gdf[gdf["lie"] == "water_hazard_3"].to_crs(epsg=32611).copy()

    # --- align new fairway to grid origin, then rotate ---
    ref = new_fairway.iloc[0].geometry.centroid
    def _shift(geom, dx, dy): return shp_translate(geom, xoff=dx, yoff=dy)
    def _rot(geom, angle, origin="centroid"): return shp_rotate(geom, angle=-angle, origin=origin, use_radians=False)

    new_fairway["geometry"] = new_fairway["geometry"].apply(_shift, args=(0 - ref.x, 100 - ref.y))
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(_shift, args=(0 - ref.x, 100 - ref.y))

    fp = new_fairway.iloc[0].geometry
    vec = np.array([fp.centroid.x, fp.bounds[3]]) - np.array([fp.centroid.x, fp.bounds[1]])
    rot_angle = np.degrees(np.arctan2(vec[0], vec[1]))
    new_fairway["geometry"] = new_fairway["geometry"].apply(_rot, args=(rot_angle,))
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(_rot, args=(rot_angle,))

    # --- shift original shapes up ---
    shift_targets = hole_9[hole_9["lie"].isin(["green", "bunker", "water_hazard", "fairway"])]
    y_shift = 160
    hole = (5, 174 + y_shift)

    def _shift_wkt(w):
        return shp_translate(wkt.loads(w), yoff=y_shift).wkt
    hole_9.loc[shift_targets.index, "WKT"] = shift_targets["WKT"].apply(_shift_wkt)

    # --- final fairway position ---
    fc = new_fairway.iloc[0].geometry.centroid
    new_fairway["geometry"] = new_fairway["geometry"].apply(_shift, args=(20 - fc.x, 175 - fc.y))
    new_fairway["geometry"] = new_fairway["geometry"].apply(
        lambda g: shp_rotate(g, angle=-68, origin="centroid", use_radians=False)
    )

    # --- hazard position (parameterised by water_loc_y) ---
    hc = new_hazard3.iloc[0].geometry.centroid
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(_shift, args=(0 - hc.x, water_loc_y - hc.y))
    new_hazard3["geometry"] = new_hazard3["geometry"].apply(
        lambda g: shp_rotate(g, angle=110, origin="centroid", use_radians=False)
    )

    hole_vec = np.array(hole)
    tee_vec  = np.array(tee_point)
    ht_len   = float(np.linalg.norm(hole_vec - tee_vec))

    x_vals = np.linspace(-40, 60, int(100 / 10))
    y_vals = np.linspace(50, ht_len - 50, int((ht_len - 50) / 10))
    strategy_points = [(float(x), float(y)) for y in y_vals for x in x_vals]

    return {
        "hole_data":        hole_9,
        "new_fairway":      new_fairway,
        "new_hazard3":      new_hazard3,
        "hole_position":    hole,
        "tee_point":        tee_point,
        "strategy_points":  strategy_points,
        "green_shape":      green_shape,
        "green_centre":     green_centre,
        "ht_length":        ht_len,
    }


def _build_polygons(layout: dict) -> dict:
    hd = layout["hole_data"]
    def _get(lie): return [wkt.loads(r["WKT"]) for _, r in hd[hd["lie"] == lie].iterrows()]
    return {
        "bunker":  _get("bunker"),
        "rough":   _get("rough"),
        "green":   _get("green")[0],
        "fairway": _get("fairway") + list(layout["new_fairway"]["geometry"]),
        "water":   _get("water") + list(layout["new_hazard3"]["geometry"]),
    }


def _get_lie(point: tuple, polys: dict) -> str:
    pt = Point(point)
    if polys["green"].contains(pt):   return "green"
    if any(p.contains(pt) for p in polys["water"]):   return "water"
    if any(p.contains(pt) for p in polys["bunker"]):  return "bunker"
    if any(p.contains(pt) for p in polys["fairway"]): return "fairway"
    return "rough"


def _build_club_distributions(lpga_df: pd.DataFrame, player: Player) -> dict[str, dict]:
    dists = {}
    for club, grp in lpga_df.groupby("Club"):
        mean = grp[["Side", "Carry"]].mean().to_numpy(dtype=np.float32)
        cov  = np.cov(grp[["Side", "Carry"]].T).astype(np.float32)
        mean, cov = player.apply_to_distribution(mean, cov)
        dists[club] = {"mean": mean, "cov": cov}
    return dists


def _build_broadie_interps(broadie_df: pd.DataFrame) -> dict:
    broadie_df = broadie_df.rename(columns={"Distance (yards)": "distance"})
    long = broadie_df.melt(id_vars="distance", var_name="lie", value_name="strokes")
    long["lie"] = long["lie"].str.lower()
    interps = {}
    for lie, grp in long.groupby("lie"):
        grp = grp.sort_values("distance")
        interps[lie] = interp1d(grp["distance"], grp["strokes"],
                                kind="linear", fill_value="extrapolate")
    return interps


class _PuttGPModel(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lk):
        super().__init__(tx, ty, lk)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _train_putt_gpr(putts_df: pd.DataFrame, y_offset: float, n_iter: int) -> tuple:
    df = putts_df.copy()
    df["y"] = df["y"] + y_offset
    tx = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
    ty = torch.tensor(df["simulated_strokes"].values, dtype=torch.float32)
    lk = gpytorch.likelihoods.GaussianLikelihood()
    m  = _PuttGPModel(tx, ty, lk)
    m.train(); lk.train()
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lk, m)
    for _ in range(n_iter):
        opt.zero_grad()
        (-mll(m(tx), ty)).backward()
        opt.step()
    m.eval(); lk.eval()
    return m, lk


def _eval_putt(point: tuple, model, likelihood) -> float:
    tx = torch.tensor([[point[0], point[1]]], dtype=torch.float32)
    with torch.no_grad():
        pred = likelihood(model(tx))
    return pred.mean.item()


def _water_drop(start, ball, polys) -> Optional[tuple]:
    line = LineString([start, ball])
    best, best_d = None, float("inf")
    for poly in polys["water"]:
        inter = line.intersection(poly.boundary)
        if inter.is_empty: continue
        pts = [inter] if isinstance(inter, Point) else list(inter.geoms)
        for pt in pts:
            if isinstance(pt, Point):
                d = Point(start).distance(pt)
                if d < best_d:
                    best_d, best = d, (pt.x, pt.y)
    return best


# ---------------------------------------------------------------------------
# Worker function  (module-level → pickleable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def worker_fn(task: RunTask) -> RunResult:
    log_path = Path(task.output_dir) / f"{task.job_id}.log"
    logging.basicConfig(
        filename=str(log_path), level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(task.job_id)
    log.info("Worker started  job_id=%s", task.job_id)

    try:
        Path(task.output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Load data
        data   = _load_data(task.data_dir)
        layout = _build_hole_layout(data, task.water_loc_y)
        polys  = _build_polygons(layout)
        log.info("Hole layout built.")

        # 2. Player + club distributions
        player = Player(
            dispersion=task.dispersion_multiplier,
            power=task.power_multiplier,
        )
        club_dists   = _build_club_distributions(data["lpga"], player)
        broadie_intp = _build_broadie_interps(data["broadie"])

        # 3. Train putt GPR (always needed; used directly by OptimizationMechanism
        #    or as a fallback evaluator for BirdieMechanism)
        putt_model, putt_lk = _train_putt_gpr(
            data["putts"], y_offset=160.0, n_iter=task.gp_training_iter
        )
        log.info("Putt GPR trained.")

        # 4. Build evaluate_shot closure (captures local models; never pickled)
        def evaluate_shot(landing_pt, start_pt, target_pt) -> float:
            lie = _get_lie(landing_pt, polys)
            if lie == "green":
                return _eval_putt(landing_pt, putt_model, putt_lk)
            if lie == "water":
                drop = _water_drop(start_pt, landing_pt, polys)
                if drop is None:
                    return float("nan")
                drop_lie = _get_lie(drop, polys)
                if drop_lie == "green":
                    return _eval_putt(drop, putt_model, putt_lk)
                dist = np.linalg.norm(np.array(drop) - np.array(target_pt))
                return broadie_intp.get("rough", broadie_intp[next(iter(broadie_intp))])(dist)
            dist = np.linalg.norm(np.array(landing_pt) - np.array(target_pt))
            key  = {"fairway": "fairway", "bunker": "sand", "rough": "rough"}.get(lie, "rough")
            interp = broadie_intp.get(key, broadie_intp.get("rough"))
            return float(interp(dist))

        def get_penalty(sp) -> float:
            lie = _get_lie(sp, polys)
            return 2.0 if lie == "water" else 1.0

        # 5. Build mechanism
        cfg = Config(
            n_shots=task.n_shots,
            dispersion_multiplier=task.dispersion_multiplier,
            power_multiplier=task.power_multiplier,
            gp_training_iter=task.gp_training_iter,
        )

        if task.mechanism_type == "optimization":
            # Train on putt dataset: X=(x,y), y=strokes
            df = data["putts"].copy()
            df["y_coord"] = df["y"] + 160.0
            X_tr = df[["x", "y_coord"]].to_numpy(dtype=np.float32)
            y_tr = df["simulated_strokes"].to_numpy(dtype=np.float32)
            mech = OptimizationMechanism(cfg)
            mech.train(X_tr, y_tr)

        else:  # "birdie"
            # Synthetic birdie labels: positions near the green with low expected
            # strokes are labelled birdie=1.  Replace with real labels when available.
            df = data["putts"].copy()
            df["y_coord"] = df["y"] + 160.0
            X_tr = df[["x", "y_coord"]].to_numpy(dtype=np.float32)
            y_tr = (df["simulated_strokes"] <= 1.5).astype(np.float32).to_numpy()
            mech = BirdieMechanism(cfg)
            mech.train(X_tr, y_tr)

        log.info("Mechanism trained  type=%s", task.mechanism_type)

        # 6. Strategy points (optionally subsampled for local mode)
        pts = layout["strategy_points"]
        if task.n_strategy_pts > 0:
            step = max(1, len(pts) // task.n_strategy_pts)
            pts  = pts[::step]
        log.info("Running simulation on %d strategy points.", len(pts))

        # 7. Run simulation
        sim     = Simulator(player=player, mechanism=mech, config=cfg)
        results = sim.simulate_all(
            strategy_points=pts,
            target=layout["hole_position"],
            club_distributions=club_dists,
            evaluate_shot_fn=evaluate_shot,
            get_penalty_fn=get_penalty,
        )
        log.info("Simulation complete: %d results.", len(results))

        # 8. Save raw data (.npz — preserves full shot arrays)
        data_path = str(Path(task.output_dir) / f"{task.job_id}_data.npz")
        np.savez(
            data_path,
            starts=np.array([r.start for r in results], dtype=np.float32),
            means=np.array([r.total_strokes_mean for r in results], dtype=np.float32),
            variances=np.array([r.total_strokes_variance for r in results], dtype=np.float32),
            clubs=np.array([r.club for r in results]),
            aim_offsets=np.array([r.aim_offset for r in results], dtype=np.float32),
            # full distributions stored as a ragged list via object array
            strokes_arrays=np.array([r.strokes_array for r in results], dtype=object),
        )

        # 9. Plot
        plot_path = str(Path(task.output_dir) / f"{task.job_id}_plot.png")
        _plot_results(results, layout, task, plot_path)
        log.info("Saved plot → %s", plot_path)

        return RunResult(
            job_id=task.job_id,
            n_results=len(results),
            output_plot=plot_path,
            output_data=data_path,
            status="ok",
        )

    except Exception as exc:
        logging.getLogger(task.job_id).exception("Worker failed")
        return RunResult(
            job_id=task.job_id, n_results=0,
            output_plot="", output_data="",
            status="error", error=str(exc),
        )


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_results(
    results: list[SimulationResult],
    layout: dict,
    task: RunTask,
    save_path: str,
) -> None:
    if not results:
        return

    xs    = [r.start[0] for r in results]
    ys    = [r.start[1] for r in results]
    means = [r.total_strokes_mean for r in results]
    stds  = [np.sqrt(r.total_strokes_variance) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, values, label, cmap in zip(
        axes,
        [means, stds],
        ["Mean expected strokes", "Std-dev of strokes"],
        ["plasma", "viridis"],
    ):
        _draw_hole(ax, layout)
        sc = ax.scatter(xs, ys, c=values, cmap=cmap, s=25, zorder=10, alpha=0.85)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.grid(True, linestyle=":")

    fig.suptitle(
        f"disp={task.dispersion_multiplier:.2f}  power={task.power_multiplier:.2f}"
        f"  water_y={task.water_loc_y}  mech={task.mechanism_type}"
        f"  n_shots={task.n_shots}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_hole(ax, layout: dict) -> None:
    """Minimal hole outline: fairway, hazards, tee, pin."""
    colors = {
        "fairway": "forestgreen", "green": "lightgreen",
        "bunker": "tan", "rough": "mediumseagreen",
        "water": "skyblue", "water_hazard": "skyblue",
    }
    for _, row in layout["hole_data"].iterrows():
        geom  = wkt.loads(row["WKT"])
        color = colors.get(row["lie"], "lightgrey")
        if geom.geom_type == "Polygon":
            ax.fill(*geom.exterior.xy, alpha=0.4, fc=color, ec="black", lw=0.4)
    for _, row in layout["new_fairway"].iterrows():
        ax.fill(*row["geometry"].exterior.xy, alpha=0.4, fc="forestgreen", ec="black", lw=0.4)
    for _, row in layout["new_hazard3"].iterrows():
        ax.fill(*row["geometry"].exterior.xy, alpha=0.4, fc="skyblue", ec="black", lw=0.4)
    tp = layout["tee_point"]
    hp = layout["hole_position"]
    ax.plot(tp[0], tp[1], "rx", markersize=8, label="Tee")
    ax.plot(hp[0], hp[1], "ko", markersize=6, label="Pin")


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(args) -> list[RunTask]:
    if args.mode == "local":
        dispersion_vals = [0.8, 1.0, 1.2]
        water_locs      = [230.0]
        n_shots         = 20
        n_strat         = 15     # subsample to 15 strategy points
        gp_iter         = 40
    else:  # hpc
        dispersion_vals = [round(v, 2) for v in np.arange(0.5, 1.55, 0.25)]
        water_locs      = [210.0, 230.0, 250.0]
        n_shots         = 200
        n_strat         = 0      # 0 = use all points
        gp_iter         = 100

    tasks = []
    for disp in dispersion_vals:
        for wloc in water_locs:
            jid = f"{args.mechanism}_d{disp:.2f}_w{int(wloc)}"
            tasks.append(RunTask(
                dispersion_multiplier=disp,
                power_multiplier=1.0,
                water_loc_y=wloc,
                mechanism_type=args.mechanism,
                n_shots=n_shots,
                n_strategy_pts=n_strat,
                gp_training_iter=gp_iter,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                job_id=jid,
            ))
    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Golf simulation parameter sweep.")
    parser.add_argument("--mode",       choices=["local", "hpc"], default="local")
    parser.add_argument("--mechanism",  choices=["optimization", "birdie"], default="optimization")
    parser.add_argument("--data-dir",   required=True, help="Folder containing the 5 data files.")
    parser.add_argument("--output-dir", default="outputs", help="Folder for plots and .npz files.")
    parser.add_argument("--workers",    type=int, default=None,
                        help="Max parallel workers. Defaults to 2 (local) or os.cpu_count() (hpc).")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MAIN] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output_dir) / "runner.log"),
        ],
    )
    log = logging.getLogger("main")

    tasks = build_grid(args)
    n_workers = args.workers or (2 if args.mode == "local" else os.cpu_count())
    log.info("Grid: %d tasks  |  workers: %d  |  mode: %s", len(tasks), n_workers, args.mode)

    results: list[RunResult] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(worker_fn, t): t.job_id for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            status_str = f"[{res.status.upper()}]" if res.status == "ok" else f"[ERROR: {res.error[:60]}]"
            log.info("  %-40s  n=%d  %s", res.job_id, res.n_results, status_str)

    ok    = sum(1 for r in results if r.status == "ok")
    total = len(results)
    log.info("Done: %d / %d tasks succeeded.", ok, total)

    # Summary CSV
    summary_path = Path(args.output_dir) / "summary.csv"
    pd.DataFrame([
        {"job_id": r.job_id, "n_results": r.n_results,
         "status": r.status, "error": r.error,
         "plot": r.output_plot, "data": r.output_data}
        for r in results
    ]).to_csv(summary_path, index=False)
    log.info("Summary written → %s", summary_path)


if __name__ == "__main__":
    main()
