"""
equivalence.py – Equivalence-set convergence testing.

Replaces the old single-arg-min convergence definition. At many grid points
several (club, aim) combinations are statistically tied in expected outcome,
so tracking a single arg-min oscillates forever even once the model has, in
every meaningful sense, converged. Instead we track the *set* of candidates
that are statistically indistinguishable from the best one, and ask whether
that set stabilises.

Equivalence set
---------------
For grid point g, seed k, sample size N (multiplier e is fixed at 1):

    E*(g, k, N) = { (club, aim) : R(club,aim) <= R_min + e * SE_min }

where R(club,aim) is the mean ESHO (expected strokes to hole out, lie penalty
included) for that combination, and R_min / SE_min come from the arg-min
combination (club*, aim*) at (g, k, N). SE_min = sqrt(var_min / n_total_min),
the standard error of that combination's mean ESHO.

There is NO aim tolerance / binning: with the coarse aim grid (step 5 yd over
[-40, 40]) every distinct (club, aim) pair is its own set element. Aim offsets
are rounded to 1 dp only to defuse float noise.

Test 1 – within-seed stabilisation (computed live in the worker)
    A grid point is "stabilised" once its equivalence set is unchanged
    (Jaccard == 1.0 vs the previous N) for `k_consecutive` snapshots in a row.
    The worker keeps sweeping to n_max regardless; it just records the N at
    which each point first stabilised and whether 100 % of points ever did.

Test 2 – cross-seed agreement (post-processing, see cross_seed_jaccard.py)
    Pairwise Jaccard between the per-seed equivalence sets at fixed (g, N).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GridPoint = tuple[float, float]
SetElement = tuple[str, float]        # (club, aim_offset rounded to 1 dp)
EquivSet = frozenset[SetElement]

AIM_ROUND_DP = 1                      # decimals to round aim_offset before set membership


@dataclass
class EquivalenceConfig:
    """Tunable parameters for the equivalence-set convergence tests."""
    e: float = 1.0                  # SE multiplier for the equivalence band
    jaccard_threshold: float = 1.0  # J >= threshold required to call two snapshots "matched"
    k_consecutive: int = 3          # consecutive matched N-steps required to declare a point stabilised
    n_start: int = 10
    n_step: int = 10
    n_max: int = 500                # sweep runs to this N; points not stable by here are reported as such


# ---------------------------------------------------------------------------
# Per-(seed, N) equivalence sets, one grid point at a time
# ---------------------------------------------------------------------------

def compute_equivalence_sets(
    candidates_df: pd.DataFrame,
    e: float = 1.0,
) -> dict[GridPoint, dict]:
    """Compute E*(g) for every grid point in a single (seed, N) candidates frame.

    Parameters
    ----------
    candidates_df : columns [x, y, club, aim_offset, mean, var, n_total]
        Every (grid-point, club, aim) candidate for one (seed, N). `mean` is
        R(club,aim) = expected strokes to hole out (lie penalty already added).
    e : SE multiplier (fixed at 1.0 for this study).

    Returns
    -------
    dict[(x, y)] -> {
        "set":       frozenset[(club, aim_offset)]  – the equivalence set,
        "size":      int,
        "R_min":     float,                          – arg-min mean ESHO,
        "SE_min":    float,                          – sqrt(var/n_total) of the arg-min,
        "argmin":    (club, aim_offset),
        "members_df": DataFrame[club, aim_offset, mean, se, n_total, is_argmin],
    }
    """
    df = candidates_df.copy()
    df["aim_offset"] = df["aim_offset"].astype(float).round(AIM_ROUND_DP)
    df["se"] = np.sqrt(df["var"] / df["n_total"].clip(lower=1))

    out: dict[GridPoint, dict] = {}
    for (x, y), g in df.groupby(["x", "y"], sort=False):
        gp: GridPoint = (float(x), float(y))
        argmin_idx = g["mean"].idxmin()
        r_min = float(g.loc[argmin_idx, "mean"])
        se_min = float(g.loc[argmin_idx, "se"])

        threshold = r_min + e * se_min
        members = g[g["mean"] <= threshold].copy()
        members["is_argmin"] = members.index == argmin_idx

        out[gp] = {
            "set": frozenset(zip(members["club"], members["aim_offset"])),
            "size": int(len(members)),
            "R_min": r_min,
            "SE_min": se_min,
            "argmin": (str(g.loc[argmin_idx, "club"]), float(g.loc[argmin_idx, "aim_offset"])),
            "members_df": members[["club", "aim_offset", "mean", "se", "n_total", "is_argmin"]]
            .reset_index(drop=True),
        }

    return out


def load_candidates(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_equivset(path: Path) -> dict[GridPoint, EquivSet]:
    """Reconstruct per-grid-point equivalence sets from a stored equivset CSV."""
    df = pd.read_csv(path)
    out: dict[GridPoint, EquivSet] = {}
    for (x, y), g in df.groupby(["x", "y"], sort=False):
        out[(float(x), float(y))] = frozenset(
            zip(g["club"], g["aim_offset"].astype(float).round(AIM_ROUND_DP))
        )
    return out


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------

def jaccard(a: EquivSet, b: EquivSet) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Test 1: within-seed sequential set-stability
# ---------------------------------------------------------------------------

@dataclass
class SequentialState:
    """Rolling state for one (grid point, seed) sequential-stability test."""
    run_length: int = 0
    converged_N: int | None = None
    prev_set: EquivSet | None = None
    last_size: int = 0
    last_N: int | None = None

    def update(self, N: int, current_set: EquivSet, k_consecutive: int, threshold: float) -> None:
        self.last_size = len(current_set)
        self.last_N = N
        if self.prev_set is not None:
            J = jaccard(self.prev_set, current_set)
            if J >= threshold:
                self.run_length += 1
            else:
                self.run_length = 0
            if self.run_length >= k_consecutive - 1 and self.converged_N is None:
                self.converged_N = N
        self.prev_set = current_set


# ---------------------------------------------------------------------------
# Live per-seed tracker (used by convergence_worker.py during the HPC run)
# ---------------------------------------------------------------------------

@dataclass
class SeedStabilityTracker:
    """Accumulates within-seed stabilisation state across the N-sweep.

    Call `update(N, candidates_df)` once per iteration. Returns a dict of
    per-N summary stats (also what the worker appends to
    seed{SEED}_stabilisation.tsv) plus the equivalence sets for that N.
    """
    e: float = 1.0
    k_consecutive: int = 3
    jaccard_threshold: float = 1.0
    states: dict[GridPoint, SequentialState] = field(default_factory=dict)
    first_N_all_stable: int | None = None

    def update(self, N: int, candidates_df: pd.DataFrame) -> tuple[dict, dict[GridPoint, dict]]:
        sets = compute_equivalence_sets(candidates_df, self.e)

        jac_vals: list[float] = []
        for gp, info in sets.items():
            st = self.states.setdefault(gp, SequentialState())
            if st.prev_set is not None:
                jac_vals.append(jaccard(st.prev_set, info["set"]))
            st.update(N, info["set"], self.k_consecutive, self.jaccard_threshold)

        n_points = len(self.states)
        n_stable_ever = sum(1 for s in self.states.values() if s.converged_N is not None)
        n_stable_now = sum(
            1 for s in self.states.values()
            if s.run_length >= self.k_consecutive - 1
        )
        n_size1 = sum(1 for s in self.states.values() if s.last_size == 1)
        pct_ever = n_stable_ever / n_points if n_points else 0.0

        if pct_ever >= 1.0 and self.first_N_all_stable is None:
            self.first_N_all_stable = N

        row = {
            "N": N,
            "n_points": n_points,
            "n_stable_ever": n_stable_ever,
            "pct_stable_ever": pct_ever,
            "n_stable_now": n_stable_now,
            "pct_stable_now": n_stable_now / n_points if n_points else 0.0,
            "n_equiv_size1": n_size1,
            "pct_equiv_size1": n_size1 / n_points if n_points else 0.0,
            "mean_jaccard_vs_prev": float(np.mean(jac_vals)) if jac_vals else float("nan"),
            "all_points_stable": bool(pct_ever >= 1.0),
        }
        return row, sets

    def summary(self) -> dict:
        n_points = len(self.states)
        n_stable_ever = sum(1 for s in self.states.values() if s.converged_N is not None)
        return {
            "n_grid_points": n_points,
            "n_points_stabilised": n_stable_ever,
            "final_pct_stable": n_stable_ever / n_points if n_points else 0.0,
            "reached_100pct_stable": bool(n_points and n_stable_ever == n_points),
            "first_N_100pct_stable": self.first_N_all_stable,
        }


# ---------------------------------------------------------------------------
# Post-processing summaries (used by run_equivalence_analysis.py)
# ---------------------------------------------------------------------------

def summarize_sequential(
    states: dict[tuple[GridPoint, int], SequentialState],
) -> pd.DataFrame:
    """One row per (grid point, seed)."""
    rows = []
    for (gp, seed), st in states.items():
        converged = st.converged_N is not None
        rows.append({
            "x": gp[0], "y": gp[1], "seed": seed,
            "converged_N": st.converged_N,
            "converged": converged,
            "equiv_set_size_final": st.last_size,
            "final_N": st.last_N,
            "stop_reason": "converged" if converged else "reached_n_max_no_stability",
        })
    return pd.DataFrame(rows)


def non_converged_report(sequential_df: pd.DataFrame) -> pd.DataFrame:
    """Which grid points drove non-convergence, and how badly."""
    rows = []
    for (x, y), g in sequential_df.groupby(["x", "y"]):
        not_conv = g[~g["converged"]]
        if len(not_conv) == 0:
            continue
        rows.append({
            "x": x, "y": y,
            "n_seeds_total": len(g),
            "n_seeds_not_converged": len(not_conv),
            "frac_seeds_not_converged": len(not_conv) / len(g),
            "mean_equiv_set_size_stuck": float(not_conv["equiv_set_size_final"].mean()),
            "max_equiv_set_size_stuck": int(not_conv["equiv_set_size_final"].max()),
        })
    cols = ["x", "y", "n_seeds_total", "n_seeds_not_converged", "frac_seeds_not_converged",
            "mean_equiv_set_size_stuck", "max_equiv_set_size_stuck"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("frac_seeds_not_converged", ascending=False)


def cross_seed_core_stats(seed_sets: list[EquivSet]) -> dict:
    """Intersection / union / exact-agreement across seeds at fixed (g, N)."""
    if not seed_sets:
        return {"core_size": 0, "union_size": 0, "full_agreement_frac": 0.0, "n_seeds": 0}
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
        "n_seeds": len(seed_sets),
    }


def pairwise_jaccard_matrix(seed_sets: dict[int, EquivSet]) -> tuple[list[int], np.ndarray]:
    """S x S pairwise Jaccard matrix for the equivalence sets of one grid point.

    Returns (ordered seed ids, matrix). Diagonal is 1.0.
    """
    seeds = sorted(seed_sets)
    m = len(seeds)
    mat = np.ones((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            v = jaccard(seed_sets[seeds[i]], seed_sets[seeds[j]])
            mat[i, j] = mat[j, i] = v
    return seeds, mat


def grid_level_summary(sequential_df: pd.DataFrame, n_grid_points: int) -> dict:
    """% of grid points stabilised by n_max (majority-of-seeds rule) + |E*| distribution."""
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
