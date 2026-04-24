"""export_hole_geometry.py
Export the Par-4 hole layout geometry to flat CSVs for use in R / ggplot2.

Run once from this directory:
    python export_hole_geometry.py

Outputs (written to this same folder):
    hole_polygons.csv   – one row per vertex; columns: lie, polygon_id, x, y
    hole_points.csv     – tee and pin coordinates
    strategy_grid.csv   – approach-shot evaluation grid points
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt
from shapely.affinity import translate as shp_translate, rotate as shp_rotate

HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
OUT_DIR  = HERE

# ---------------------------------------------------------------------------
# 1. Load hole_9 and shift green / bunker / water / fairway by 160 yd
# ---------------------------------------------------------------------------
hole_9 = pd.read_csv(DATA_DIR / "hole_9_data.csv")
hole_9["geometry"] = hole_9["WKT"].apply(shapely_wkt.loads)

Y_SHIFT = 160
hole_pin: tuple[float, float] = (5.0, 174.0 + Y_SHIFT)

for idx, row in hole_9.iterrows():
    if row["lie"] in {"green", "bunker", "water_hazard", "fairway"}:
        hole_9.at[idx, "geometry"] = shp_translate(row["geometry"], yoff=Y_SHIFT)

# ---------------------------------------------------------------------------
# 2. Load GeoJSON, reproject, and position new_fairway + new_hazard3
# ---------------------------------------------------------------------------
gdf = gpd.read_file(DATA_DIR / "newshapes.geojson")

new_fairway = gdf[gdf["lie"] == "fairway"].to_crs(epsg=32611).copy()
new_hazard3 = gdf[gdf["lie"] == "water_hazard_3"].to_crs(epsg=32611).copy()

# First alignment: centroid → (0, 100), then rotate by fairway long-axis angle
ref_centroid = new_fairway.iloc[0].geometry.centroid
x0 = 0.0 - ref_centroid.x
y0 = 100.0 - ref_centroid.y

for col in [new_fairway, new_hazard3]:
    col["geometry"] = col["geometry"].apply(lambda g: shp_translate(g, xoff=x0, yoff=y0))

fp = new_fairway.iloc[0].geometry
vec = np.array([fp.centroid.x, fp.bounds[3]]) - np.array([fp.centroid.x, fp.bounds[1]])
rot_angle = float(np.degrees(np.arctan2(vec[0], vec[1])))

for col in [new_fairway, new_hazard3]:
    col["geometry"] = col["geometry"].apply(
        lambda g: shp_rotate(g, angle=-rot_angle, origin="centroid", use_radians=False)
    )

# Final position: new_fairway centroid → (20, 175), rotate -68°
fc = new_fairway.iloc[0].geometry.centroid
new_fairway["geometry"] = new_fairway["geometry"].apply(
    lambda g: shp_translate(g, xoff=20.0 - fc.x, yoff=175.0 - fc.y)
)
new_fairway["geometry"] = new_fairway["geometry"].apply(
    lambda g: shp_rotate(g, angle=-68, origin="centroid", use_radians=False)
)

# Final position: new_hazard3 centroid → (0, 210), rotate 110°
h3c = new_hazard3.iloc[0].geometry.centroid
new_hazard3["geometry"] = new_hazard3["geometry"].apply(
    lambda g: shp_translate(g, xoff=0.0 - h3c.x, yoff=210.0 - h3c.y)
)
new_hazard3["geometry"] = new_hazard3["geometry"].apply(
    lambda g: shp_rotate(g, angle=110, origin="centroid", use_radians=False)
)

# ---------------------------------------------------------------------------
# 3. Tee point: centroid of tee box furthest from green
# ---------------------------------------------------------------------------
def _centroid(geom):
    return geom.centroid.coords[0]

teeboxes = hole_9[hole_9["lie"].str.contains("tee", case=False)].copy()
green_centre = _centroid(hole_9[hole_9["lie"] == "green"].iloc[0]["geometry"])

teeboxes["centroid"] = teeboxes["geometry"].apply(_centroid)
teeboxes["dist_to_green"] = teeboxes["centroid"].apply(
    lambda pt: np.linalg.norm(np.array(pt) - np.array(green_centre))
)
tee_point: tuple[float, float] = teeboxes.loc[teeboxes["dist_to_green"].idxmax()]["centroid"]

# ---------------------------------------------------------------------------
# 4. Extract polygon vertices → flat CSV
# ---------------------------------------------------------------------------
rows = []
poly_id = 0

def _extract(geom, lie: str, pid: int) -> list[dict]:
    coords = list(geom.exterior.coords)
    return [{"lie": lie, "polygon_id": pid, "x": c[0], "y": c[1]} for c in coords]

for _, row in hole_9.iterrows():
    geom, lie = row["geometry"], row["lie"]
    if geom.geom_type == "Polygon":
        rows.extend(_extract(geom, lie, poly_id)); poly_id += 1
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            rows.extend(_extract(poly, lie, poly_id)); poly_id += 1

for _, row in new_fairway.iterrows():
    geom = row["geometry"]
    if geom.geom_type == "Polygon":
        rows.extend(_extract(geom, "new_fairway", poly_id)); poly_id += 1
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            rows.extend(_extract(poly, "new_fairway", poly_id)); poly_id += 1

for _, row in new_hazard3.iterrows():
    geom = row["geometry"]
    if geom.geom_type == "Polygon":
        rows.extend(_extract(geom, "new_hazard3", poly_id)); poly_id += 1
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            rows.extend(_extract(poly, "new_hazard3", poly_id)); poly_id += 1

polygons_df = pd.DataFrame(rows)
polygons_df.to_csv(OUT_DIR / "hole_polygons.csv", index=False)
print(f"Saved hole_polygons.csv  ({len(polygons_df):,} rows, {poly_id} polygons)")

# ---------------------------------------------------------------------------
# 5. Special points
# ---------------------------------------------------------------------------
points_df = pd.DataFrame([
    {"label": "Tee", "x": tee_point[0], "y": tee_point[1]},
    {"label": "Pin", "x": hole_pin[0],  "y": hole_pin[1]},
])
points_df.to_csv(OUT_DIR / "hole_points.csv", index=False)
print("Saved hole_points.csv")

# ---------------------------------------------------------------------------
# 6. Strategy grid
# ---------------------------------------------------------------------------
ht_length = float(np.linalg.norm(np.array(hole_pin) - np.array(tee_point)))
x_vals = np.linspace(-40, 60, int(100 / 10))
y_vals = np.linspace(50, ht_length - 50, int((ht_length - 50) / 10))
grid_df = pd.DataFrame(
    [(float(x), float(y)) for y in y_vals for x in x_vals],
    columns=["x", "y"],
)
grid_df.to_csv(OUT_DIR / "strategy_grid.csv", index=False)
print(f"Saved strategy_grid.csv   ({len(grid_df)} points)")
