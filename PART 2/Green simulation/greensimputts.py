import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.interpolate import interp1d

# ------------------------
# Step 1: Load true strokes gained data
# ------------------------
truth_df = pd.read_csv("PART 1/broadiedata/strokes_on_green_feet_broadie.csv")
truth_df.columns = ["distance_ft", "true_strokes"]

# Create a smooth interpolator for the ground truth
truth_interp = interp1d(
    truth_df["distance_ft"],
    truth_df["true_strokes"],
    kind="cubic",
    fill_value="extrapolate"
)

# ------------------------
# Step 2: Simulate tournament-averaged data
# ------------------------
np.random.seed(42)
n_short = 500
n_long = 100  # Boost long putts

# Exponential bias for short putts
short_putts = np.random.exponential(scale=15, size=n_short)
short_putts = np.clip(short_putts, truth_df["distance_ft"].min(), 60)

# Uniform long putts to better cover sparse tail
long_putts = np.random.uniform(60, truth_df["distance_ft"].max(), size=n_long)

putt_distances = np.concatenate([short_putts, long_putts])
true_vals = truth_interp(putt_distances)

# Noise increases with distance, simulating tournament-level averages
noise_sd = 0.01 + 0.0025 * putt_distances
avg_vals = np.random.normal(loc=true_vals, scale=noise_sd)

sim_df = pd.DataFrame({
    "distance_ft": putt_distances,
    "strokes_to_hole_out": avg_vals
})

# ------------------------
# Step 3: Fit Gaussian Process Regression with tuned hyperparameters
# ------------------------
X = sim_df["distance_ft"].values.reshape(-1, 1)
y = sim_df["strokes_to_hole_out"].values

kernel = RBF(length_scale=10.0, length_scale_bounds=(1e-1, 1e2)) + \
         WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1))

gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10)
gpr.fit(X, y)

print("Learned kernel after optimisation:")
print(gpr.kernel_)

# Predict over grid
x_true = np.linspace(truth_df["distance_ft"].min(), truth_df["distance_ft"].max(), 300)
X_pred = x_true.reshape(-1, 1)
y_pred, y_std = gpr.predict(X_pred, return_std=True)

# ------------------------
# Step 4: Plot everything together
# ------------------------
plt.figure(figsize=(12, 7))
plt.plot(x_true, truth_interp(x_true), "k-", lw=2, label="True (Interpolated)")
plt.plot(X_pred, y_pred, "b-", lw=2, label="GPR Prediction")
plt.fill_between(X_pred.ravel(), y_pred - y_std, y_pred + y_std, color="blue", alpha=0.2, label="±1 std dev")
plt.scatter(sim_df["distance_ft"], sim_df["strokes_to_hole_out"], alpha=0.3, s=15, color="gray", label="Simulated Averages")

plt.xlabel("Distance (feet)")
plt.ylabel("Strokes to hole out")
plt.title("GPR Fit on Simulated Tournament-Level Averages")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
