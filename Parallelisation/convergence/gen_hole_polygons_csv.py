"""Export hole geometry (incl. OB boundary) to r_hole_layout/hole_polygons.csv,
consumed by sensitivity/rankedOutputs/hole_outline_heatmap.py + hole_under_heatmap.py
to draw lie-boundary outlines over the sensitivity heatmaps.
"""
from pathlib import Path
from shapely import wkt as shapely_wkt
import pandas as pd
import core

OUT = Path(__file__).parent.parent / "r_hole_layout" / "hole_polygons.csv"
OUT.parent.mkdir(exist_ok=True)

hole = core.build_hole(gp_training_iter=1)
rows = []

for _, row in hole.hole_9.iterrows():
    geom = shapely_wkt.loads(row["WKT"])
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for pid, poly in enumerate(polys):
        for x, y in poly.exterior.coords:
            rows.append({"lie": row["lie"], "polygon_id": f"{row['lie']}_{pid}", "x": x, "y": y})

for label, gdf in [("new_fairway", hole.new_fairway), ("new_hazard3", hole.new_hazard3)]:
    for pid, geom in enumerate(gdf["geometry"]):
        for x, y in geom.exterior.coords:
            rows.append({"lie": label, "polygon_id": f"{label}_{pid}", "x": x, "y": y})

# OB boundary lines (not closed polygons — just the two edges the sensitivity
# grid can actually reach: the lateral limits. The far y-limit sits beyond the
# strategy grid's own extent so it wouldn't be visible on these heatmaps.)
y_lo, y_hi = hole.tee_point[1] - 20, hole.ob_y_far + 20
for side, xval in [("OB_left", hole.ob_x_left), ("OB_right", hole.ob_x_right)]:
    rows.append({"lie": "OB", "polygon_id": side, "x": xval, "y": y_lo})
    rows.append({"lie": "OB", "polygon_id": side, "x": xval, "y": y_hi})

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} rows, {df['lie'].nunique()} lie types -> {OUT}")
