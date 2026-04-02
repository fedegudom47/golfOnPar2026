# GEMINI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# 1. Load the data
# Assuming the file is named 'simulated_lgpa_shot_data2.csv'
df = pd.read_csv('PART 1/Trackman Fake Data/simulated_lpga_shot_data2.csv')

# 2. Define Helper function for Confidence Ellipses
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # 2D dataset covariance matrix
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# 3. Setup the Plot
fig, ax = plt.subplots(figsize=(12, 5)) # Wide format for rotated view

# 4. Sort Clubs by Distance (so the legend orders correctly from short to long or vice versa)
# We calculate mean carry for each club to determine order
club_order = df.groupby('Club')['Carry'].mean().sort_values(ascending=True).index
# Create a color map
colors = plt.cm.tab20(np.linspace(0, 1, len(club_order)))

# 5. Iterate and Plot
for i, club in enumerate(club_order):
    subset = df[df['Club'] == club]
    
    # ROTATION HAPPENS HERE:
    # X = Carry (Distance), Y = Side (Dispersion)
    x = subset['Carry'].values
    y = subset['Side'].values
    color = colors[i]
    
    # Plot Scatter points
    ax.scatter(x, y, s=30, alpha=0.5, label=club, color=color, edgecolors='none')
    
    # Plot Covariance Ellipse (2 Standard Deviations)
    confidence_ellipse(x, y, ax, n_std=2, edgecolor=color, linestyle='--', linewidth=2, zorder=10)

# 6. Formatting
ax.axhline(0, color='black', linewidth=1) # Center line for "Side" is now horizontal
ax.set_title("Simulated: LPGA Player Shot Dispersion", fontsize=16)
ax.set_xlabel("Carry Distance (yards)", fontsize=12)
ax.set_ylabel("Side Dispersion (yards)", fontsize=12)
ax.grid(True, linestyle='-', alpha=0.6)

# Force equal aspect ratio so 10 yards left looks the same as 10 yards long
# However, since carry is huge compared to side, 'equal' might make it too thin.
# Often 'auto' is better for this specific rotated view, but let's try to keep it proportional.
# ax.set_aspect('equal') 

# Legend formatting
handles, labels = ax.get_legend_handles_labels()
# Reverse legend to have Driver at top if desired, or keep as is (Shortest to Longest)
ax.legend(handles[::-1], labels[::-1], title="Shot", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.show()