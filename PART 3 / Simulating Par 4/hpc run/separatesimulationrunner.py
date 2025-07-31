# separatesimulationrunner.py

import pandas as pd
from mainsimulationrunner import hole_layout, run_simulation_from_layout

def run_simulation_wrapper(y_coord):
    # 1. Generate layout
    layout = hole_layout(target_y3=y_coord)

    # 2. Run full simulation
    best_tee_shot, optimal_points, model, likelihood = run_simulation_from_layout(
        layout_dict=layout,
        n_samples=1,  # adjust this if you want
        save_prefix=str(y_coord),
        plot_gpr=True,
        plot_tee=True,
        plot_overlay=True
    )

    # 3. Return summary result
    result = {
        "water_y": y_coord,
        "best_club": best_tee_shot["club"],
        "aim_offset": best_tee_shot["aim_offset"],
        "mean_esho": best_tee_shot["mean"],
        "variance": best_tee_shot["variance"]
    }

    return result
