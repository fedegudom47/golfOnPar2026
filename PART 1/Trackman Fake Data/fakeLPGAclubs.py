import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


# Scale dispersions based on distance (rough approximation)
# More carry = more spread (both side and long)
def get_dispersion(avg_carry):
    side = 0.08 * avg_carry    # ~8% of distance
    dist = 0.04 * avg_carry    # ~4% of distance
    return side, dist

n_shots = 30
all_shots = []

# Simulate each club’s pattern
for club, carry_avg in club_data.items():
    side_sd, dist_sd = get_dispersion(carry_avg)

    # Left/right offset
    side = np.random.normal(0, side_sd, size=n_shots)

    # Pulls go longer, fades shorter
    bias = np.where(
        side < 0,
        np.random.normal(5, 1, size=n_shots),   # pulls
        np.random.normal(-2, 1, size=n_shots)   # fades/pushes
    )

    # Final carry distance
    carry = carry_avg + bias + np.random.normal(0, dist_sd, size=n_shots)

    club_df = pd.DataFrame({
        'Side': side,
        'Carry': carry,
        'Club': club
    })

    all_shots.append(club_df)

# Combine all shots into one DataFrame
df = pd.concat(all_shots, ignore_index=True)


# Plot
plt.figure(figsize=(10, 10))
for club in df['Club'].unique():
    club_data = df[df['Club'] == club]
    plt.scatter(
        club_data['Side'],
        club_data['Carry'],
        label=club,
        alpha=0.6,
        s=40
    )

# Add baseline cross
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.xlabel("Carry Flat - Side (yards)")
plt.ylabel("Carry Flat - Distance (yards)")
plt.title("LPGA Shot Dispersion by Club (Simulated)")
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

df.to_csv("PART 1/Trackman Fake Data/simulated_lpga_shot_data2.csv", index=False)
