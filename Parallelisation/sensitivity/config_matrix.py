"""
config_matrix.py – 24-row (carry_shift, variance_scale) configuration matrix.

A 6 × 4 grid:
  carry_shifts   : 0, 3, 6, 9, 12, 15 yards    (max +15 yds above baseline)
  variance_scales: 1.0, 0.97, 0.94, 0.90        (baseline → 10 % tighter)

Each row is labelled with the trend it primarily illustrates:
  Trend 1 – fixed dispersion (variance_scale = 1.0), increasing distance
  Trend 2 – fixed distance (carry_shift = 0),  decreasing dispersion
  Trend 3 – both vary simultaneously

Usage:
    python config_matrix.py               # print table + save param_configs.csv
    python config_matrix.py --task-id 7   # print one row
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Grid definition — edit here to change the sweep range
# ---------------------------------------------------------------------------

CARRY_SHIFTS:    list[float] = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]   # yards
VARIANCE_SCALES: list[float] = [1.0, 0.97, 0.94, 0.90]             # multiplier


def _trend(carry_shift: float, variance_scale: float) -> int:
    if variance_scale == 1.0:
        return 1   # dispersion fixed, distance varies
    if carry_shift == 0.0:
        return 2   # distance fixed, dispersion varies
    return 3       # both vary


def build_config_matrix() -> pd.DataFrame:
    """Return a 24-row DataFrame with columns: task_id, trend, carry_shift, variance_scale."""
    rows = []
    for task_id, (cs, vs) in enumerate(itertools.product(CARRY_SHIFTS, VARIANCE_SCALES)):
        rows.append({
            "task_id":        task_id,
            "trend":          _trend(cs, vs),
            "carry_shift":    cs,
            "variance_scale": vs,
        })
    df = pd.DataFrame(rows)
    assert len(df) == len(CARRY_SHIFTS) * len(VARIANCE_SCALES)
    return df


def get_config(task_id: int, df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = build_config_matrix()
    row = df[df["task_id"] == task_id]
    if row.empty:
        raise ValueError(f"task_id={task_id} not found (valid: 0–{len(df)-1})")
    return row.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--output",  type=Path, default=Path(__file__).parent / "param_configs.csv")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = build_config_matrix()

    if args.task_id is not None:
        cfg = get_config(args.task_id, df)
        print(f"\n  task_id={cfg['task_id']}  trend={cfg['trend']}  "
              f"carry_shift={cfg['carry_shift']:+.1f} yd  "
              f"variance_scale={cfg['variance_scale']:.2f}\n")
        return

    print(f"\n{'task':>5}  {'trend':>5}  {'carry_shift':>12}  {'variance_scale':>14}")
    print("  " + "-" * 42)
    for _, row in df.iterrows():
        print(f"  {int(row.task_id):>3}    {int(row.trend):>3}    "
              f"{row.carry_shift:>+8.1f} yd    {row.variance_scale:>10.2f}")
    print(f"\n  {len(df)} configurations  "
          f"({len(CARRY_SHIFTS)} carry × {len(VARIANCE_SCALES)} variance)\n")

    if not args.no_save:
        df.to_csv(args.output, index=False)
        print(f"  Saved → {args.output}\n")


if __name__ == "__main__":
    main()
