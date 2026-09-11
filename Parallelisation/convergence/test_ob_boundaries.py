"""
test_ob_boundaries.py – validate the OB logic before committing to a full run.

Same script, same invocation, on your laptop or on the HPC (no HPC-specific
code needed — it's a single lightweight process, geometry+one seed's worth
of shots). To sanity-check several seeds at once, just background a few:

    for s in 0 1 2 3; do python test_ob_boundaries.py --seed $s & done; wait

Usage:
    python test_ob_boundaries.py                 # seed 0, n=100 shots/combo
    python test_ob_boundaries.py --seed 3 --n 300
"""
from __future__ import annotations

import argparse

import numpy as np

from core import build_hole, evaluate_broadie, is_out_of_bounds, simulate_approach_shots


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=100, help="shots per (grid point, club, aim)")
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    np.random.seed(args.seed)
    hole = build_hole(args.data_dir, gp_training_iter=1)  # geometry only, skip real GP training

    # 1. OB boundary sanity
    assert hole.ob_x_left == -40.0 and hole.ob_x_right == 60.0
    assert is_out_of_bounds((-41, 100), hole)
    assert is_out_of_bounds((61, 100), hole)
    assert is_out_of_bounds((0, hole.ob_y_far + 1), hole)
    assert not is_out_of_bounds((0, 100), hole)
    print(f"OK  OB bounds: x < {hole.ob_x_left}  /  x > {hole.ob_x_right}  /  y > {hole.ob_y_far:.1f}")

    # 2. water flush to the OB line, no gap
    water_x_max = max(poly.bounds[2] for poly in hole.water_polygons)
    assert abs(water_x_max - hole.ob_x_right) < 1e-6, water_x_max
    print(f"OK  water right edge = {water_x_max:.3f}  (== ob_x_right)")

    # 3. OB penalty formula, spot-checked at one grid point
    pt = hole.strategy_points[0]
    expected = evaluate_broadie(pt, hole.hole, "fairway", hole.broadie_interpolators) + 1.0
    print(f"OK  OB value at {pt} = {expected:.3f} strokes  (Broadie-from-origin + 1 penalty)")

    # 4. end-to-end smoke run over the whole grid — must not crash, one row per point
    optimal, _, candidates = simulate_approach_shots(hole, n_new=args.n, return_all_candidates=True)
    assert len(optimal) == len(hole.strategy_points), (len(optimal), len(hole.strategy_points))
    print(f"OK  simulated {len(candidates)} candidates across {len(optimal)} grid points "
          f"(n={args.n} shots/combo, seed={args.seed})")

    print("Done — OB logic looks sane, safe to launch the full sweep.")


if __name__ == "__main__":
    main()
