import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from scipy import interpolate
import random
from scipy.stats import norm

putts = pd.read_csv("/Users/federicadomecq/Documents/golfModeL47-1/PART 1/broadiedata/strokes_on_green_feet_broadie.csv")
columns = putts.columns

#Interpolating expected putts
dist, avg_putts = putts["Distance (feet)"], putts["Green"]
expected_putts = interpolate.interp1d(dist, avg_putts, kind = "cubic", fill_value = "extrapolate")
x_true = np.linspace(min(dist), max(dist) + 10, 500)
y_true = expected_putts(x_true)
plt.plot(x_true, y_true, label = "true function", linewidth = 1 )
plt.xlabel("Distance")
plt.ylabel("Expected Putts")
plt.show()

#Simulating Fake Data
random.seed(47)
def samples_at(dist):
        decay_rate = 0.04  # tweak this!
        p = 1 / (1 + np.exp(decay_rate * (dist - 20))) 
        return np.random.binomial(n=5, p=p)
    
def noise_sd(d):
    return min(0.0028 * d + 0.02, 1.2)

simulated_averages = []
for distance in range(3,100):
    current_expected = expected_putts(distance)
    current_variance = noise_sd(d = distance)
    n_samples = samples_at(distance)
    for samples in range(n_samples):
        simulated_averages.append((distance, np.random.normal(current_expected, current_variance)))

xs_sim_distance = []
ys_sim_average = []

for pair in simulated_averages:
    distance, average = pair
    xs_sim_distance.append(distance)
    ys_sim_average.append(average)

plt.scatter(xs_sim_distance, ys_sim_average, alpha = .5)
plt.plot(x_true, y_true, color = "red")
plt.xlabel("Distance (feet)")
plt.ylabel("Average putts to hole out")

# GPR with GpyTorch
train_x = torch.tensor(xs_sim_distance).unsqueeze(1).float()
train_y = torch.tensor(ys_sim_average).float()

class MyGaussianModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        # Parent constructor
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        rbf = gpytorch.kernels.RBFKernel()
        rbf.lengthscale = 20.0  # <-- set the initial smoothness      
        self.covar_module =  gpytorch.kernels.ScaleKernel(rbf)
  
    
    # Method telling the GP model how ot compute the predictive distribution
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

#Instantiating Likelihood and Model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MyGaussianModel(train_x, train_y, likelihood)

optimiser = torch.optim.Adam(model.parameters(), lr = .1)
# takes steps of .1
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

model.train()
likelihood.train()

# Forward pass through the model
output = model(train_x)

loss = -mll(output, train_y)

for i in range(75):
    optimiser.zero_grad() # clear old gradients to not accumulate
    output = model(train_x) # forward pass - predicted posterior at training points
    loss = -mll(output, train_y) # compute how well the GP explains data
    loss.backward() # compute the gradients of loss wrt all model params
    optimiser.step() # update the params by taking step in gradient direction

# Visualising Work
# Where we are going to evaluate these
x_test = torch.linspace(train_x.min(), train_x.max(), 500).unsqueeze(1)

# Set model in eval mode
model.eval()
likelihood.eval()

# Prior samples, before training
with torch.no_grad():
    prior_dist = model(x_test)
    prior_samples = prior_dist.sample(torch.Size([3]))

# Posterior samples after training
# Posterior samples after training
with torch.no_grad():
    posterior = likelihood(model(x_test))
    mean = posterior.mean
    lower, upper = posterior.confidence_region()
    posterior_samples = posterior.sample(torch.Size([3]))

# Save posterior predictions to CSV
predictions_df = pd.DataFrame({
    "distance_ft": x_test.squeeze().numpy(),
    "posterior_mean": mean.numpy(),
    "lower_95ci": lower.numpy(),
    "upper_95ci": upper.numpy()
})
predictions_df.to_csv("PART 2/Green simulation/results/GPR_predictions_posterior.csv", index=False)

# Plot
plt.figure(figsize=(10, 6))

# Training points
plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observations')

# Posterior mean
plt.plot(x_test.numpy(), mean.numpy(), 'b', label='Posterior Mean')

# Confidence interval (±2 std dev)
plt.fill_between(
    x_test.squeeze().numpy(),
    lower.numpy(),
    upper.numpy(),
    alpha=0.3,
    color = "orange",
    label='95% Confidence Region'
)

# Posterior samples
for i in range(3):
    plt.plot(x_test.numpy(), posterior_samples[i].numpy(), alpha=0.4, label=f'Posterior Sample {i+1}')
plt.plot(x_true, y_true, label = "Underlying truth")
plt.legend()
plt.title("Posterior Predictive Distribution with GP")
plt.xlabel("Distance (ft)")
plt.ylabel("Average Putts to Hole Out")
plt.grid(True)
plt.show()

# Slicing to evaluate GPR at different points

# Distance at which to evaluate the posterior
value_to_check = 4
x_query = torch.tensor([[value_to_check]])  # GP requires input shape [n, d]

# Set model and likelihood to evaluation mode 
model.eval()
likelihood.eval()

# Compute posterior predictive distribution at the given input
with torch.no_grad():
    pred_dist = likelihood(model(x_query))  # Returns a MultivariateNormal object
    mu = pred_dist.mean.item()              # Extract the predicted mean
    sigma = pred_dist.stddev.item()         # Extract the predicted standard deviation

# Generate y-values across standard deviations for bell curve plotting
y_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf_vals = norm.pdf(y_vals, loc=mu, scale=sigma)  # Evaluate normal density at each point

# Plot the posterior predictive distribution (bell curve)
plt.plot(y_vals, pdf_vals, label=f'Posterior at {x_query.item()} ft')

# Plot vertical line at the mean prediction
plt.axvline(mu, color='red', linestyle='--', label='Posterior Mean')

# Shade the 95% confidence interval (±2σ)
plt.fill_between(
    y_vals,
    pdf_vals,
    where=(y_vals > mu - 2*sigma) & (y_vals < mu + 2*sigma),
    color='skyblue',
    alpha=0.5,
    label='~95% CI'
)

# Final plot formatting
plt.title(f"Posterior Predictive Distribution at {x_query.item()} ft")
plt.xlabel("Predicted Average Putts")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()

#---- Interactive ---

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm

# Set up figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

# Slider setup (discrete values only)
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "Distance (ft)", valmin=3.0, valmax=95.0, valinit=10.0, valstep=1.0)

# Update function
def update(distance):
    x_query = torch.tensor([[distance]])
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(x_query))
        mu = pred.mean.item()
        sigma = pred.stddev.item()

    y_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    pdf_vals = norm.pdf(y_vals, loc=mu, scale=sigma)

    ax.clear()

    # Bell curve
    ax.plot(y_vals, pdf_vals, label=f'Posterior at {distance:.1f} ft')

    # Mean line
    ax.axvline(mu, color='red', linestyle='--', label='Posterior Mean')

    # Confidence interval shading
    ax.fill_between(
        y_vals,
        pdf_vals,
        where=(y_vals > mu - 2 * sigma) & (y_vals < mu + 2 * sigma),
        color='skyblue',
        alpha=0.5,
        label='~95% CI'
    )

    # ❗ Fixed axes for consistent zoom and motion feel
    ax.set_xlim(0.5, 4)   # Adjust as needed based on your putt ranges
    ax.set_ylim(0, 4)

    # Labels and formatting
    ax.set_title(f"Posterior Predictive Distribution at {distance:.1f} ft")
    ax.set_xlabel("Predicted Average Putts")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()

    # Text annotation (top-left)
    annotation = f"Posterior Mean: {mu:.3f}\nStd Dev: {sigma:.3f}"
    ax.text(0.02, 0.95, annotation, transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

    fig.canvas.draw_idle()

# Initial draw
update(slider.val)

# Hook slider
slider.on_changed(update)

# Show interactive window
plt.show()

