import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# LPGA average carry distances (yards)
club_data = {
    'Driver': 223,
    '3-wood': 200,
    '5-wood': 189,
    'Hybrid': 178,
    '4 Iron': 175,
    '5 Iron': 166,
    '6 Iron': 155,
    '7 Iron': 143,
    '8 Iron': 133,
    '9 Iron': 123,
    'PW': 111,
    '50 deg': 107,
    '54 deg': 96,
    '60 deg': 78,
    'H 50 deg': 85,
    '3Q 60 deg': 65,
    'H 54 deg': 54,
    'H 60 deg': 39,
    '1Q 60 deg': 25,
    '1E 60 deg' : 15
}

# Returns (side sd, distance sd) based on carry
def get_dispersion(avg_carry):
    return 0.08 * avg_carry, 0.04 * avg_carry

n_shots = 50
all_shots = []

# Simulate shot data
for club, carry_avg in club_data.items():
    side_sd, dist_sd = get_dispersion(carry_avg)
    side = np.random.normal(0, side_sd, size=n_shots)

    bias = np.where(
        side < 0,
        np.random.normal(5, 1, size=n_shots),   # pulls
        np.random.normal(-2, 1, size=n_shots)   # pushes
    )

    carry = carry_avg + bias + np.random.normal(0, dist_sd, size=n_shots)

    df_club = pd.DataFrame({
        'Side': side,
        'Carry': carry,
        'Club': club
    })
    all_shots.append(df_club)

df = pd.concat(all_shots, ignore_index=True)

# Color map
unique_clubs = list(club_data.keys())
colors = plt.cm.get_cmap('tab20', len(unique_clubs))

# Start plot
fig, ax = plt.subplots(figsize=(12, 12))

for i, club in enumerate(unique_clubs):
    club_df = df[df['Club'] == club]
    x = club_df['Side'].values
    y = club_df['Carry'].values
    col = colors(i)

    # Plot shots
    ax.scatter(x, y, label=club, color=col, alpha=0.6, s=40)

    # Plot covariance ellipse (95% confidence ~ 2 std devs)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(eigvals)  # 2 std devs = ~95%

    ellipse = Ellipse((x_mean, y_mean), width, height, angle=angle,
                      edgecolor=col, facecolor='none', linestyle='--', lw=2)
    ax.add_patch(ellipse)

# Plot formatting
ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=1)
ax.set_xlabel("Carry Flat - Side (yards)")
ax.set_ylabel("Carry Flat - Distance (yards)")
ax.set_title("LPGA Shot Dispersion by Club with Covariance Ellipses")
ax.legend()
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
plt.tight_layout()
plt.show()
