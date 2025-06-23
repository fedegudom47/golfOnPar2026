import numpy as np
import torch
import gpytorch
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import random
from scipy.stats import norm

# Load data
putts = pd.read_csv("/Users/federicadomecq/Documents/golfModeL47-1/PART 1/broadiedata/strokes_on_green_feet_broadie.csv")
dist, avg_putts = putts["Distance (feet)"], putts["Green"]

# Interpolation of true function
expected_putts = interp1d(dist, avg_putts, kind="cubic", fill_value="extrapolate")
x_true = np.linspace(min(dist), max(dist) + 10, 500)
y_true = expected_putts(x_true)
plt.plot(x_true, y_true, label="true function", linewidth=1)
plt.xlabel("Distance")
plt.ylabel("Expected Putts")
plt.show()

# Simulate noisy data
random.seed(47)
def samples_at(dist):
    decay_rate = 0.04
    p = 1 / (1 + np.exp(decay_rate * (dist - 20)))
    return np.random.binomial(n=15, p=p)

def noise_sd(d):
    return min(0.0028 * d + 0.02, 1.2)

simulated_averages = []
for distance in range(3, 100):
    current_expected = expected_putts(distance)
    current_variance = noise_sd(distance)
    n_samples = samples_at(distance)
    for _ in range(n_samples):
        simulated_averages.append((distance, np.random.normal(current_expected, current_variance)))

sim_df = pd.DataFrame(simulated_averages, columns=["distance", "avg_putts"])
xs_np = sim_df["distance"].values
ys_np = sim_df["avg_putts"].values

# Compute true empirical variances per distance
grouped = sim_df.groupby("distance")
empirical_vars_dict = {
    k: max(v, 0.03) if len(grouped.get_group(k)) >= 3 else 0.03
    for k, v in grouped["avg_putts"].var().fillna(0.03).items()
}
empirical_vars = [empirical_vars_dict.get(int(x), 0.01) for x in xs_np]

# GP model with heteroscedastic noise
train_x = torch.tensor(xs_np).unsqueeze(1).float()
train_y = torch.tensor(ys_np).float()
noise_tensor = torch.tensor(empirical_vars).float() + 0.03

likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise_tensor)

class HeteroGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        rbf = gpytorch.kernels.RBFKernel()
        rbf.raw_lengthscale.requires_grad = True
        rbf.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(10.0, 50.0))
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )
    


model = HeteroGP(train_x, train_y, likelihood)
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(75):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

# Prediction
model.eval()
likelihood.eval()
test_x = torch.linspace(train_x.min(), train_x.max(), 500).unsqueeze(1)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(test_x))
    mean = preds.mean
    lower, upper = preds.confidence_region()

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(xs_np, ys_np, alpha=0.5, label="Simulated data")
plt.plot(x_true, y_true, color="red", label="Underlying truth")
plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Posterior Mean')
plt.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(),
                 alpha=0.3, color="orange", label="95% CI")
plt.xlabel("Distance (feet)")
plt.ylabel("Average putts to hole out")
plt.title("Heteroscedastic GP Regression (True Empirical Variance)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Posterior predictive distribution
with torch.no_grad():
    posterior = likelihood(model(test_x))
    mean = posterior.mean
    lower, upper = posterior.confidence_region()
    posterior_samples = posterior.sample(torch.Size([3]))

# Save predictions
predictions_df = pd.DataFrame({
    "distance_ft": test_x.squeeze().numpy(),
    "posterior_mean": mean.numpy(),
    "lower_95ci": lower.numpy(),
    "upper_95ci": upper.numpy()
})
predictions_df.to_csv("PART 2/Green simulation/results/GPR_predictions_posterior.csv", index=False)

# Plot posterior samples
plt.figure(figsize=(10, 6))
plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observations')
plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Posterior Mean')
plt.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(),
                 alpha=0.3, color="orange", label="95% CI")
for i in range(3):
    plt.plot(test_x.numpy(), posterior_samples[i].numpy(), alpha=0.4, label=f'Posterior Sample {i+1}')
plt.plot(x_true, y_true, label="Underlying truth", color="red")
plt.xlabel("Distance (ft)")
plt.ylabel("Average Putts to Hole Out")
plt.legend()
plt.title("Posterior Predictive Distribution with Heteroscedastic GP")
plt.grid(True)
plt.show()

# Static bell curve at specific distance
value_to_check = 4
x_query = torch.tensor([[value_to_check]])
with torch.no_grad():
    pred_dist = likelihood(model(x_query))
    mu = pred_dist.mean.item()
    sigma = pred_dist.stddev.item()

y_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
pdf_vals = norm.pdf(y_vals, loc=mu, scale=sigma)
plt.plot(y_vals, pdf_vals, label=f'Posterior at {x_query.item()} ft')
plt.axvline(mu, color='red', linestyle='--', label='Posterior Mean')
plt.fill_between(y_vals, pdf_vals,
                 where=(y_vals > mu - 2 * sigma) & (y_vals < mu + 2 * sigma),
                 color='skyblue', alpha=0.5, label='~95% CI')
plt.title(f"Posterior Predictive Distribution at {x_query.item()} ft")
plt.xlabel("Predicted Average Putts")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()
