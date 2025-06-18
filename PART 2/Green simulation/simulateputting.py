import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.interpolate import interp1d, griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Load data
slope_df = pd.read_csv("PART 2/Green simulation/green_slope_percent.csv")
truth_df = pd.read_csv("PART 1/broadiedata/strokes_on_green_feet_broadie.csv")
truth_df.columns = ["distance_ft", "true_strokes"]

# Create interpolators
truth_interp = interp1d(truth_df["distance_ft"], truth_df["true_strokes"], kind="cubic", fill_value="extrapolate")

# Build GPR from your fakeputtingdata.py
# (Load your `sim_df` from that script here and refit GPR if not saved already)

# Fit GPR model (reproduce if not saved)
X = sim_df["distance_ft"].values.reshape(-1, 1)
y = sim_df["strokes_to_hole_out"].values
kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=0.01)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10).fit(X, y)

# Slope interpolator
def get_slope(x, y):
    return griddata(
        (slope_df["x"], slope_df["y"]),
        slope_df["slope_percent"],
        (x, y),
        method="linear",
        fill_value=np.nan
    )

# Elevation function from `greensimtwotier.py`
def green_elevation(x, y):
    curve_center = 185 + 3 * np.sin(0.1 * x)
    tier_height = 0.3
    tier_width = 3
    curved_tier = (tier_height / 2) * (np.tanh((y - curve_center) / tier_width) + 1)
    def cosine_bump(xc, yc, amp, rad):
        r2 = (x - xc)**2 + (y - yc)**2
        return np.where(r2 < rad**2, amp * 0.5 * (1 + np.cos(np.pi * np.sqrt(r2) / rad)), 0)
    upper_left = cosine_bump(-10, 195, 0.15, 10)
    lower_right = -cosine_bump(5, 170, 0.27, 15)
    tilt = 0.015 * x + 0.00003 * y
    return curved_tier + upper_left + lower_right + tilt

# Main function
def simulate_putt(ball_x, ball_y, pin_x, pin_y):
    dist = np.linalg.norm([ball_x - pin_x, ball_y - pin_y])
    dist_ft = dist * 3

    # GPR Prediction
    pred, std = gpr.predict([[dist_ft]], return_std=True)
    strokes = np.random.normal(pred[0], std[0])

    # Tier difference penalty
    ball_z = green_elevation(ball_x, ball_y)
    pin_z = green_elevation(pin_x, pin_y)
    elevation_diff = pin_z - ball_z
    if elevation_diff > 0:  # Uphill
        tier_penalty = 0.02 * elevation_diff
    else:  # Downhill
        tier_penalty = 0.08 * abs(elevation_diff)

    # AimPoint penalty: simulate "breakiness"
    slope = get_slope(ball_x, ball_y)
    aim_penalty = 0.015 * slope if not np.isnan(slope) else 0

    total_penalty = tier_penalty + aim_penalty
    return strokes + total_penalty

# Example
ball_x, ball_y = 3.0, 185
pin_x, pin_y = -3.6, 177
simulated_score = simulate_putt(ball_x, ball_y, pin_x, pin_y)
print(f"Simulated strokes: {simulated_score:.3f}")
