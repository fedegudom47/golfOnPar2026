import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from shapely import affinity, wkt
from shapely.geometry import Point
import geopandas as gpd

# --- 1. ROTATION HELPERS (-90 Degrees) ---
def rotate_xy(x, y):
    # -90 degrees (Clockwise): (x, y) -> (y, -x)
    return y, -x

def rotate_geom(geom_input):
    if geom_input is None: return None

    # Extract geometry if input is a DataFrame/Series
    if hasattr(geom_input, 'geometry'):
        # If GeoDataFrame, grab the first geometry
        if hasattr(geom_input.geometry, 'iloc'):
            g = geom_input.geometry.iloc[0]
        else:
            g = geom_input.geometry
    elif hasattr(geom_input, 'columns') and 'WKT' in geom_input.columns:
        g = geom_input['WKT'].iloc[0]
    else:
        g = geom_input

    # Ensure it's a Shapely object
    g_obj = wkt.loads(g) if isinstance(g, str) else g
    
    # Rotate -90 deg around origin (0,0)
    return affinity.rotate(g_obj, -90, origin=(0, 0))

# --- 2. GENERATE SYNTHETIC "HISTORICAL" DATA ---
def generate_historical_shots(n_samples, fairway_geom, hazard_geoms, hole_pos):
    """
    Generates random points and assigns a 'Strokes to Hole Out' score
    based on distance and lie difficulty.
    """
    data = []
    
    # Define bounding box for generation (approximate based on your previous grid)
    min_x, max_x = -60, 80
    min_y, max_y = 0, 450
    
    for _ in range(n_samples):
        # 1. Random Location
        rx = np.random.uniform(min_x, max_x)
        ry = np.random.uniform(min_y, max_y)
        pt = Point(rx, ry)
        
        # 2. Distance to Hole (Euclidean)
        dist_yards = np.linalg.norm(np.array([rx, ry]) - np.array(hole_pos))
        
        # 3. Determine Lie & Penalty
        # Base formula: approx 2 putts + 1 shot per ~140 yards
        expected_score = 2.0 + (dist_yards / 140.0) 
        
        # Check interactions (simplified for visual proxy)
        in_fairway = False
        if fairway_geom is not None:
            # Handle if fairway is df or geom
            f_poly = fairway_geom.geometry.iloc[0] if hasattr(fairway_geom, 'geometry') else fairway_geom
            if f_poly.contains(pt):
                in_fairway = True
                expected_score -= 0.2 # Slight bonus for fairway
        
        in_hazard = False
        for haz in hazard_geoms:
            h_poly = haz.geometry.iloc[0] if hasattr(haz, 'geometry') else haz
            if h_poly.contains(pt):
                in_hazard = True
                expected_score += 1.5 # Big penalty for water/bunker
                
        if not in_fairway and not in_hazard:
            expected_score += 0.4 # Rough penalty
            
        # 4. Add Noise and Discretize
        # Add random variance (some players are better than others)
        noise = np.random.normal(0, 0.6)
        final_score = int(np.round(expected_score + noise))
        
        # Cap min score at 1 (hole in one)
        final_score = max(1, final_score)
        
        data.append({
            "x": rx, "y": ry, "score": final_score
        })
        
    return pd.DataFrame(data)

# --- 3. GENERATE THE DATA ---
# (Using the variables from your environment: new_fairway, new_hazard3, hole)
# Generate 300 historical observations
df_history = generate_historical_shots(
    n_samples=300, 
    fairway_geom=new_fairway, 
    hazard_geoms=[new_hazard3], # Add your other hazards here if needed
    hole_pos=hole
)

# --- 4. ROTATED PLOTTING FUNCTION ---
def plot_historical_data_rotated(hole_geom_df, shot_data, new_fairway=None, new_hazards=None,
                                 tee_point=None, hole_point=None, 
                                 title="Historical Shot Outcomes", figsize=(14, 8)):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # A. Rotate and Plot Geometry
    # ---------------------------
    
    # 1. Base Geometry (Original hole_9)
    # We assume hole_geom_df has 'WKT' and 'lie' columns
    lie_colors = {
        "fairway": "forestgreen", "green": "lightgreen", 
        "rough": "mediumseagreen", "bunker": "tan", 
        "water": "skyblue", "water_hazard": "skyblue"
    }
    
    for _, row in hole_geom_df.iterrows():
        rot_poly = rotate_geom(row['WKT'])
        color = lie_colors.get(row.get('lie'), 'lightgrey')
        if rot_poly.geom_type in ['Polygon', 'MultiPolygon']:
            if rot_poly.geom_type == 'Polygon': polys = [rot_poly]
            else: polys = rot_poly.geoms
            
            for p in polys:
                x, y = p.exterior.xy
                ax.fill(x, y, alpha=0.5, fc=color, ec="black", lw=0.5)

    # 2. New Fairway
    if new_fairway is not None:
        rot_fw = rotate_geom(new_fairway)
        x, y = rot_fw.exterior.xy
        ax.fill(x, y, alpha=0.4, fc="forestgreen", ec="black", lw=0.5, label="Fairway")

    # 3. New Hazards
    if new_hazards:
        for haz in new_hazards:
            rot_haz = rotate_geom(haz)
            x, y = rot_haz.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="skyblue", ec="black", lw=0.5, hatch='///', label="Water Hazard")

    # 4. Tee & Hole
    tx, ty = rotate_xy(tee_point[0], tee_point[1])
    hx, hy = rotate_xy(hole_point[0], hole_point[1])
    
    ax.scatter(tx, ty, marker='D', color='black', s=100, zorder=20, label="Tee")
    ax.scatter(hx, hy, marker='X', color='red', s=100, zorder=20, label="Hole")

    # B. Rotate and Plot Historical Data
    # ----------------------------------
    # Rotate the points
    rot_xs, rot_ys = [], []
    for _, row in shot_data.iterrows():
        rx, ry = rotate_xy(row['x'], row['y'])
        rot_xs.append(rx)
        rot_ys.append(ry)
        
    # Plot Scatter
    # Color map: Green (low score) -> Red (high score)
    sc = ax.scatter(rot_xs, rot_ys, c=shot_data['score'], 
                    cmap='RdYlGn_r', # Reversed so Green is low (good), Red is high (bad)
                    s=50, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=15)
    
    # Add labels for specific points to illustrate the concept
    # (Just grab a few random ones to show the discrete integers)
    subset = shot_data.sample(15) 
    for _, row in subset.iterrows():
        rx, ry = rotate_xy(row['x'], row['y'])
        ax.text(rx + 2, ry + 2, str(int(row['score'])), fontsize=9, fontweight='bold', zorder=25)

    # C. Formatting
    # -------------
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, pad=15)
    
    # Create a custom legend for the scores
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Strokes to Hole Out (Historical)", rotation=270, labelpad=20)
    
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlabel("Y Coordinate (Rotated)")
    ax.set_ylabel("X Coordinate (Rotated)")
    
    plt.tight_layout()
    plt.show()

# --- 5. EXECUTE ---
plot_historical_data_rotated(
    hole_geom_df=hole_9, 
    shot_data=df_history, 
    new_fairway=new_fairway, 
    new_hazards=[new_hazard3],
    tee_point=tee_point, 
    hole_point=hole,
    title="Figure 2: Location-Specific Outcome Data\n(Discrete Strokes to Hole Out)"
)