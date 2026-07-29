"""
equivalence.py – Equivalence-set convergence testing.

Replaces the old single-arg-min convergence definition. At many grid points
several (club, aim) combinations are statistically tied in expected outcome,
so tracking a single arg-min oscillates forever even once the model has, in
every meaningful sense, converged. Instead we track the *set* of candidates
that are statistically indistinguishable from the best one, and ask whether
that set stabilises.

This module is pure post-processing: it consumes the per-candidate
R(s,theta) = mean and n_total (giving SE(s,theta) = sqrt(var/n_total)) that
`core.simulate_approach_shots(..., return_all_candidates=True)` produces and
`convergence_worker.py` persists per (seed, N) as a parquet file. It does
not touch the GP fitting, Monte Carlo simulation, or R/SE computation.

Definitions
-----------
Equivalence set, for grid point g, seed k, sample size N, multiplier e:

    E*(g, k, N, e) = { (s, theta) : R(s,theta) <= R_min + e * SE_min }

where R_min, SE_min come from the arg-min (s*, theta*) combination at
(g, k, N) (not each candidate's own SE). Before membership is evaluated,
theta is binned to a tolerance so near-identical aimpoints count as the same
set element rather than being spuriously distinct: +/-2 deg within a seed
(sequential N-to-N comparisons), +/-3 deg across seeds (cross-seed pooling).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GridPoint = tuple[float, float]
SetElement = tuple[str, int]          # (club, binned theta)
EquivSet = frozenset[SetElement]


@dataclass
class EquivalenceConfig:
    """Tunable parameters for the equivalence-set convergence tests."""
    e_values: tuple[float, ...] = (1.0, 1.645, 2.0)          # ~68% / 90% / 95% one-sided bands
    aim_tol_within: float = 2.0     # deg — theta binning for sequential N-to-N (within-seed) tests
    aim_tol_cross: float = 3.0      # deg — theta binning for cross-seed pooling
    jaccard_threshold: float = 1.0  # J >= threshold required to call two snapshots "matched"
    k_consecutive: int = 3          # consecutive matched N-steps required to declare convergence
    n_start: int = 10
    n_step: int = 10
    n_max: int = 500   # sweep runs to this N; points/seeds not stable by here are reported as non-converged


def bin_theta(aim_offset: pd.Series | np.ndarray, tol: float) -> np.ndarray:
    """Bin aim offsets to the given tolerance so near-identical aimpoints coincide."""
    return np.round(np.asarray(aim_offset, dtype=float) / tol).astype(int)


# ---------------------------------------------------------------------------
# Per-(seed, N) equivalence sets, one grid point at a time (vectorized)
# ---------------------------------------------------------------------------

def compute_equivalence_sets(
    candidates_df: pd.DataFrame,
    e_values: Iterable[float],
    aim_tol: float,
) -> dict[GridPoint, dict[float, dict]]:
    """Compute E*(g, e) for every grid point in a single (seed, N) candidates frame.

    Parameters
    ----------
    candidates_df : columns [x, y, club, aim_offset, mean, var, n_total]
        All (grid-point, club, aim) candidates for one (seed, N).
    e_values : multipliers to sweep.
    aim_tol : theta-binning tolerance (deg) to apply before computing set
        membership — pass aim_tol_within for sequential tests, aim_tol_cross
        for cross-seed pooling.

    Returns
    -------
    dict[(x, y)] -> dict[e] -> {"set": EquivSet, "size": int, "R_min": float, "SE_min": float}
    """
    df = candidates_df.copy()
    df["theta_bin"] = bin_theta(df["aim_offset"], aim_tol)
    df["se"] = np.sqrt(df["var"] / df["n_total"].clip(lower=1))

    out: dict[GridPoint, dict[float, dict]] = {}
    for (x, y), g in df.groupby(["x", "y"], sort=False):
        gp: GridPoint = (float(x), float(y))
        argmin_idx = g["mean"].idxmin()
        r_min = float(g.loc[argmin_idx, "mean"])
        se_min = float(g.loc[argmin_idx, "se"])

        per_e: dict[float, dict] = {}
        for e in e_values:
            threshold = r_min + e * se_min
            mask = g["mean"] <= threshold
            members = frozenset(zip(g.loc[mask, "club"], g.loc[mask, "theta_bin"]))
            per_e[e] = {
                "set": members,
                "size": len(members),
                "R_min": r_min,
                "SE_min": se_min,
            }
        out[gp] = per_e

    return out


def load_candidates(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Test 1: within-seed sequential set-stability (Jaccard)
# ---------------------------------------------------------------------------

def jaccard(a: EquivSet, b: EquivSet) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@dataclass
class SequentialState:
    """Rolling state for one (grid point, seed, e) sequential-stability test."""
    history_J: deque = field(default_factory=lambda: deque(maxlen=0))
    run_length: int = 0
    converged_N: int | None = None
    prev_set: EquivSet | None = None
    last_size: int = 0
    last_N: int | None = None

    def update(self, N: int, current_set: EquivSet, k_consecutive: int, threshold: float) -> None:
        self.last_size = len(current_set)
        self.last_N = N
        if self.prev_set is not None and self.converged_N is None:
            J = jaccard(self.prev_set, current_set)
            if J >= threshold:
                self.run_length += 1
            else:
                self.run_length = 0
            if self.run_length >= k_consecutive - 1:
                self.converged_N = N
        self.prev_set = current_set


def summarize_sequential(
    states: dict[tuple[GridPoint, int, float], SequentialState],
) -> pd.DataFrame:
    """One row per (grid point, seed, e).

    `equiv_set_size_final` is |E*| at convergence if converged, otherwise |E*|
    at the last N processed (i.e. at n_max, if the sweep ran that far without
    the set stabilising). `stop_reason` records which of those happened:
    "converged" or "reached_n_max_no_stability" — the latter is exactly the
    set of (grid point, seed) pairs that did NOT reach convergence by n_max.
    """
    rows = []
    for (gp, seed, e), st in states.items():
        converged = st.converged_N is not None
        rows.append({
            "x": gp[0], "y": gp[1], "seed": seed, "e": e,
            "converged_N": st.converged_N,
            "converged": converged,
            "equiv_set_size_final": st.last_size,
            "final_N": st.last_N,
            "stop_reason": "converged" if converged else "reached_n_max_no_stability",
        })
    return pd.DataFrame(rows)


def non_converged_report(sequential_df: pd.DataFrame) -> pd.DataFrame:
    """Which grid points drove non-convergence, and how badly, per e.

    One row per (grid point, e): how many/what fraction of seeds never
    stabilised by n_max, and the mean/max |E*| those non-converged seeds were
    stuck at (a large stuck-set size means genuine, persistent strategic
    ambiguity at that point on the course — not just slow convergence).
    """
    rows = []
    for (x, y, e), g in sequential_df.groupby(["x", "y", "e"]):
        not_conv = g[~g["converged"]]
        if len(not_conv) == 0:
            continue
        rows.append({
            "x": x, "y": y, "e": e,
            "n_seeds_total": len(g),
            "n_seeds_not_converged": len(not_conv),
            "frac_seeds_not_converged": len(not_conv) / len(g),
            "mean_equiv_set_size_stuck": float(not_conv["equiv_set_size_final"].mean()),
            "max_equiv_set_size_stuck": int(not_conv["equiv_set_size_final"].max()),
        })
    return pd.DataFrame(rows).sort_values(
        "frac_seeds_not_converged", ascending=False
    ) if rows else pd.DataFrame(
        columns=["x", "y", "e", "n_seeds_total", "n_seeds_not_converged",
                 "frac_seeds_not_converged", "mean_equiv_set_size_stuck", "max_equiv_set_size_stuck"]
    )


# ---------------------------------------------------------------------------
# Test 2: cross-seed core-set agreement
# ---------------------------------------------------------------------------

def cross_seed_core_stats(seed_sets: list[EquivSet]) -> dict:
    """Given E*(g, k, N, e) pooled (theta-binned) across all seeds at fixed (g, N, e)."""
    if not seed_sets:
        return {"core_size": 0, "union_size": 0, "full_agreement_frac": 0.0}
    core = seed_sets[0]
    union: set[SetElement] = set(seed_sets[0])
    for s in seed_sets[1:]:
        core = core & s
        union |= s
    n_full_agree = sum(1 for s in seed_sets if s == core)
    return {
        "core_size": len(core),
        "union_size": len(union),
        "full_agreement_frac": n_full_agree / len(seed_sets),
    }


# ---------------------------------------------------------------------------
# Test 3: grid-level summary
# ---------------------------------------------------------------------------

def grid_level_summary(sequential_df: pd.DataFrame, n_grid_points: int) -> dict:
    """% of grid points converged by n_max (majority-of-seeds rule) + |E*| distribution.

    A grid point counts as "converged by N=n_max" if at least half its seeds
    reached convergence; |E*| at convergence is averaged across converged seeds
    for that grid point.
    """
    per_gp = sequential_df.groupby(["x", "y"])
    converged_flags = per_gp["converged"].mean() >= 0.5
    pct_converged = 100.0 * converged_flags.mean()

    converged_rows = sequential_df[sequential_df["converged"]]
    size_dist = converged_rows["equiv_set_size_final"]

    return {
        "pct_grid_points_converged": float(pct_converged),
        "pct_grid_points_not_converged": float(100.0 - pct_converged),
        "mean_equiv_set_size_at_convergence": float(size_dist.mean()) if len(size_dist) else float("nan"),
        "median_equiv_set_size_at_convergence": float(size_dist.median()) if len(size_dist) else float("nan"),
        "n_grid_points": n_grid_points,
    }
