import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import wkt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import os

# Load Green Polygon Data (for shape)
df = pd.read_csv("PART 1/Map Digitisation/Mountain Meadows/MountainMeadows_Separated/hole_9/hole_9_data.csv")
green_info = df[df["lie"] == "green"].iloc[0]
green_polygon = wkt.loads(green_info["WKT"])

# Handle both Polygon and MultiPolygon types
if isinstance(green_polygon, Polygon):
    green_shape = green_polygon
elif isinstance(green_polygon, MultiPolygon):
    green_shape = max(green_polygon.geoms, key=lambda p: p.area)  # take largest part
else:
    raise TypeError("Unexpected geometry type")

# Define elevation surface of the green
def green_contour(x, y):
    # Define a tiered green using tanh + sine for curvature
    curve_center = 185 + 3 * np.sin(0.1 * x)
    tier_height = 0.3
    tier_width = 3
    curved_tier = (tier_height / 2) * (np.tanh((y - curve_center) / tier_width) + 1)

    # Cosine bumps (upper-left = ridge, lower-right = depression)
    def cosine_bump(xc, yc, amp, rad):
        r2 = (x - xc)**2 + (y - yc)**2
        return np.where(r2 < rad**2, amp * 0.5 * (1 + np.cos(np.pi * np.sqrt(r2) / rad)), 0)

    upper_left = cosine_bump(-10, 195, 0.15, 10)
    lower_right = -cosine_bump(5, 170, 0.27, 15)

    # Global tilt (l to r and front to back)
    tilt = 0.015 * x + 0.00003 * y

    return curved_tier + upper_left + lower_right + tilt

#  coordinate grid over the green polygon to evaluate elevation, slope etc
minx, miny, maxx, maxy = green_shape.bounds
x_vals = np.linspace(minx, maxx, 300)
y_vals = np.linspace(miny, maxy, 300)
X, Y = np.meshgrid(x_vals, y_vals)

# mask to restrict calculations to the green area
points = np.column_stack((X.ravel(), Y.ravel()))
mask = np.array([green_shape.contains(Point(x, y)) for x, y in points]).reshape(X.shape)

# Compute green elevation
Z = green_contour(X, Y)
Z[~mask] = np.nan  # mask out non-green area

# Pin location
pin_x, pin_y = -3.6, 177

# 3D Plot Surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    X, Y, Z,
    color='lightgreen',
    edgecolor='black',
    linewidth=0.1,
    alpha=0.95,
    antialiased=True
)
ax.set_title("3D Green Surface")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Elevation")
ax.set_zlim(0, 1.75)
plt.plot(pin_x, pin_y, marker='*', color='red', markersize=12, alpha=0.9)
plt.tight_layout()

# Slope percentage
dy, dx = np.gradient(Z, y_vals, x_vals)
slope_percent = np.sqrt(dx**2 + dy**2) * 100
slope_percent[~mask] = np.nan

# Putt view colours
puttview_colors = [
    "#666666", "#2c7bb6", "#00a884", "#d9ef8b",
    "#fdae61", "#f46d43", "#d73027", "#7f3b08"
]
boundaries = [0, 1, 2, 3, 4, 5, 6, 7, 100]
cmap = mcolors.ListedColormap(puttview_colors)
norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, slope_percent, levels=boundaries, cmap=cmap, norm=norm)
plt.colorbar(cp, ticks=boundaries, label='Slope (%)')
plt.title("PuttView-Style Slope Zones")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.plot(pin_x, pin_y, marker='*', color='red', markersize=12, alpha=0.9)
plt.tight_layout()

# Continuous putt view
puttview_gradient = [
    (0/7, "#666666"), (1/7, "#2c7bb6"), (2/7, "#00a884"), (3/7, "#d9ef8b"),
    (4/7, "#fdae61"), (5/7, "#f46d43"), (6/7, "#d73027"), (1.0, "#7f3b08")
]
cmap = mcolors.LinearSegmentedColormap.from_list("puttview_smooth", puttview_gradient)
norm = mcolors.Normalize(vmin=0, vmax=7)

plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, slope_percent, levels=100, cmap=cmap, norm=norm)
plt.colorbar(cp, label='Slope (%)')
plt.title("Smooth PuttView-Style Slope Heatmap")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.plot(pin_x, pin_y, marker='*', color='red', markersize=12, alpha=0.9)
plt.tight_layout()

# Continuous putt view
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, slope_percent, levels=[0, 1.5, 2.5, 3.5, 6], colors=['lightgreen', 'gold', 'coral', 'crimson'])
plt.colorbar(label='Pins: Easy 0–1.5, Med 1.5–2.5, Hard 2.5–3.5, Imp 3.5+')
plt.title("Green Slope Zones (for Pin Placement / AimPoint)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.plot(pin_x, pin_y, marker='*', color='red', markersize=12, alpha=0.9)
plt.tight_layout()

# Getting arrows
yd_per_grid = (maxx - minx) / 300
step_1yd = int(round(1 / yd_per_grid))     # spacing for arrows
step_3yd = int(round(2.5 / yd_per_grid))   # spacing for text

# Subsample slope and gradient vectors
X_sub = X[::step_1yd, ::step_1yd]
Y_sub = Y[::step_1yd, ::step_1yd]
dx_sub = dx[::step_1yd, ::step_1yd]
dy_sub = dy[::step_1yd, ::step_1yd]
slope_sub = slope_percent[::step_1yd, ::step_1yd]

# Normalize gradient vectors (to unit arrows)
mag = np.sqrt(dx_sub**2 + dy_sub**2)
dx_norm = -dx_sub / (mag + 1e-6)
dy_norm = -dy_sub / (mag + 1e-6)

#  Annotated slope map
plt.figure(figsize=(10, 8))
cp = plt.contourf(X, Y, slope_percent, levels=boundaries, cmap=cmap, norm=norm)
plt.colorbar(cp, ticks=boundaries, label='Slope (%)')

# Arrows showing downhill direction
plt.quiver(
    X_sub, Y_sub, dx_norm, dy_norm,
    scale=30, width=0.002,
    headwidth=2, headlength=3,
    color='white', alpha=0.6
)

# Annotate slope % every 3 yards
for i in range(0, X_sub.shape[0], step_3yd // step_1yd):
    for j in range(0, X_sub.shape[1], step_3yd // step_1yd):
        val = slope_sub[i, j]
        if not np.isnan(val):
            plt.text(
                X_sub[i, j], Y_sub[i, j],
                f"{val:.1f}%", color='black',
                fontsize=7, ha='center', va='center', alpha=0.9
            )

plt.title("PuttView-Style Slope Map with Arrows and Labels")
plt.xlabel("x (left-right)")
plt.ylabel("y (front-back)")
plt.axis('equal')
plt.plot(pin_x, pin_y, marker='*', color='red', markersize=12, alpha=0.9)
plt.tight_layout()
plt.show()

# Flatten and export slope percent to CSV
slope_df = pd.DataFrame({
    "x": X.ravel(),
    "y": Y.ravel(),
    "slope_percent": slope_percent.ravel()
})

# Drop any NaNs (outside the green mask)
slope_df = slope_df.dropna()

# Ensure output folder exists
output_path = "PART 2/Green simulation/results"
os.makedirs(output_path, exist_ok=True)

# Save to CSV
slope_df.to_csv(os.path.join(output_path, "green_slope_percent.csv"), index=False)
print("Saved: green_slope_percent.csv")