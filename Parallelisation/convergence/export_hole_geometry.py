"""
export_hole_geometry.py – dump the hole layout as WKT for plotting in R.

Uses the exact same `core.build_hole` geometry (game coordinates) that the
strategy grid and the simulation use, so the polygons line up 1:1 with the
grid-point (x, y) values in the equivalence-set CSVs.

Outputs (default: MyScripts/data/):
  hole_geometry.csv   – columns [lie, wkt]; one row per polygon, plus POINT
                        rows lie='tee' and lie='pin'. Water includes the
                        out-of-bounds extension flush to x=ob_x_right.
  strategy_grid.csv   – columns [x, y]; the 280 grid points
  ob_bounds.csv        – columns [name, value]; ob_x_left, ob_x_right, ob_y_far

Usage:
    python export_hole_geometry.py                     # -> MyScripts/data/
    python export_hole_geometry.py --out-dir some/dir
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from shapely.geometry import Point

from core import build_hole


def _rows(hole):
    def poly_rows(lie, polys):
        for g in polys:
            if g is not None and not g.is_empty:
                yield {"lie": lie, "wkt": g.wkt}

    yield {"lie": "green", "wkt": hole.green_polygon.wkt}
    yield from poly_rows("fairway", hole.fairway_polygons)
    yield from poly_rows("water", hole.water_polygons)
    yield from poly_rows("bunker", hole.bunker_polygons)
    yield from poly_rows("rough", hole.rough_polygons)
    yield {"lie": "tee", "wkt": Point(hole.tee_point).wkt}
    yield {"lie": "pin", "wkt": Point(hole.hole).wkt}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Directory with hole_9_data.csv etc (default: Parallelisation/data/).")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).parent / "MyScripts" / "data",
                   help="Where to write hole_geometry.csv / strategy_grid.csv.")
    p.add_argument("--gp-iter", type=int, default=1,
                   help="Putt-GPR training iters — geometry doesn't need real training.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hole = build_hole(args.data_dir, gp_training_iter=args.gp_iter)

    geo_path = args.out_dir / "hole_geometry.csv"
    with open(geo_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lie", "wkt"])
        w.writeheader()
        w.writerows(_rows(hole))

    grid_path = args.out_dir / "strategy_grid.csv"
    with open(grid_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        w.writerows(hole.strategy_points)

    ob_path = args.out_dir / "ob_bounds.csv"
    with open(ob_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "value"])
        w.writerow(["ob_x_left", hole.ob_x_left])
        w.writerow(["ob_x_right", hole.ob_x_right])
        w.writerow(["ob_y_far", hole.ob_y_far])

    print(f"Wrote {geo_path}")
    print(f"Wrote {grid_path}  ({len(hole.strategy_points)} points)")
    print(f"Wrote {ob_path}")


if __name__ == "__main__":
    main()
