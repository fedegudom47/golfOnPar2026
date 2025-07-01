import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy

# Load true strokes-gained data
truth_df = pd.read_csv("PART 1/broadiedata/strokes_on_green_feet_broadie.csv")
truth_df.columns = ["distance_ft", "true_strokes"]

# Interpolator for truth
from scipy.interpolate import interp1d
truth_interp = interp1d(
    truth_df["distance_ft"],
    truth_df["true_strokes"],
    kind="cubic",
    fill_value="extrapolate"
)

# Simulate data
np.random.seed(42)
n_short, n_long = 500, 100

short_putts = np.random.exponential(scale=15, size=n_short)
short_putts = np.clip(short_putts, truth_df["distance_ft"].min(), 60)
long_putts = np.random.uniform(60, truth_df["distance_ft"].max(), size=n_long)

putt_distances = np.concatenate([short_putts, long_putts])
true_vals = truth_interp(putt_distances)
noise_sd = 0.01 + 0.0025 * putt_distances
avg_vals = np.random.normal(loc=true_vals, scale=noise_sd)

X = putt_distances.reshape(-1, 1)
y = avg_vals.reshape(-1, 1)

# Define kernel (RBF + noise)
kernel = GPy.kern.RBF(input_dim=1,  lengthscale=10.0) #variance=1.0,

# Bayesian GP model with Gaussian likelihood
model = GPy.models.GPRegression(X, y, kernel)
model.Gaussian_noise.variance = 0.01  # initial guess
model.Gaussian_noise.variance.fix()   # fix noise if known, or remove this line to estimate

# Fit model (optimise kernel hyperparameters)
model.optimize(messages=True)

# Predict over grid
x_pred = np.linspace(truth_df["distance_ft"].min(), truth_df["distance_ft"].max(), 300).reshape(-1, 1)
mean_pred, var_pred = model.predict(x_pred)

# Plot mean + uncertainty
plt.figure(figsize=(10, 6))
plt.plot(x_pred, truth_interp(x_pred.ravel()), "k--", lw=2, label="True (Interpolated)")
plt.plot(x_pred, mean_pred, "b-", lw=2, label="GP Mean")
plt.fill_between(
    x_pred.ravel(),
    (mean_pred - 2 * np.sqrt(var_pred)).ravel(),
    (mean_pred + 2 * np.sqrt(var_pred)).ravel(),
    color="blue", alpha=0.2, label="95% CI"
)
plt.scatter(X, y, alpha=0.3, s=10, color="gray", label="Simulated Averages")
plt.xlabel("Distance (feet)")
plt.ylabel("Strokes to hole out")
plt.title("Bayesian GPR with GPy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Draw 20 posterior predictive samples (functions)
posterior_draws = model.posterior_samples_f(x_pred, size=100)  # shape: (300, 20)

plt.figure(figsize=(10, 6))
for i in range(posterior_draws.shape[1]):
    plt.plot(x_pred, posterior_draws[:, i], alpha=0.3)

plt.plot(x_pred, mean_pred, "b-", lw=2, label="GP Mean")
plt.plot(x_pred, truth_interp(x_pred.ravel()), "k--", lw=2, label="Underlying Truth")
plt.xlabel("Distance (feet)")
plt.ylabel("Strokes to hole out")
plt.title("Posterior Predictive Samples (Bayesian GPR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save posterior draws to CSV
posterior_df = pd.DataFrame(posterior_draws, columns=[f"sample_{i}" for i in range(posterior_draws.shape[1])])
posterior_df["distance_ft"] = x_pred.ravel()
posterior_df.to_csv("rPART 2/Green simulation/results/gpy_posterior_samples.csv", index=False)

# Save model
model.save_model("PART 2/Green simulation/results/models/gpy_bayesian_gpr_model.zip")
