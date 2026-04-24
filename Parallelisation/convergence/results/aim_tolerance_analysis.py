#!/usr/bin/env python3
"""
aim_tolerance_analysis.py
==========================
Re-analyses the already-simulated strategy CSVs to see how the convergence
picture changes when we allow ±1, ±2, or ±3 yard error in aim offset.

Two complementary metrics are computed for each tolerance:

  1. Sequential match rate  — for each seed, the fraction of grid points
     whose strategy (club + aim_offset) did NOT change between consecutive
     N snapshots, where aim is considered "unchanged" if |Δaim| ≤ tolerance.
     This mirrors what convergence_worker.py computes on the fly, but lets
     us sweep tolerances post-hoc.

  2. Cross-seed aim agreement — for each (x, y) grid point at each N,
     the fraction of seeds whose aim_offset is within ±tolerance of the
     cross-seed median.  Tells us how much the seeds agree on where to aim,
     under each tolerance assumption.

Outputs (all in results/)
--------------------------
  aim_tolerance_match_rates.csv     — long-form: seed × N × tolerance × match_rate
  aim_tolerance_agreement.csv       — long-form: x × y × N × tolerance × agreement
  plots/aim_tolerance_match_rate.png
  plots/aim_tolerance_agreement_heatmaps.png
  plots/aim_tolerance_by_distance.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ---------------------------------------------------------------------------
HERE      = Path(__file__).parent
PLOTS     = HERE / "plots"
PLOTS.mkdir(exist_ok=True)

TOLERANCES   = [1, 2, 3]           # yards — what we're sweeping
TOL_LABELS   = {t: f"±{t} yd" for t in TOLERANCES}
TOL_COLOURS  = {1: "#E91E63", 2: "#FF9800", 3: "#4CAF50"}

DISTANCE_BANDS = [(50, 80), (80, 120), (120, 180), (180, 250), (250, 320)]
BAND_LABELS    = ["50-80 yd", "80-120 yd", "120-180 yd", "180-250 yd", "250+ yd"]

def band_label(y: float) -> str:
    for (lo, hi), lbl in zip(DISTANCE_BANDS, BAND_LABELS):
        if lo <= y < hi:
            return lbl
    return "250+ yd"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_strategies() -> pd.DataFrame:
    path = HERE / "strategies_key_N.csv"
    print(f"Loading {path.name} …")
    df = pd.read_csv(path)
    print(f"  {len(df):,} rows  |  seeds: {df['seed'].nunique()}  |  "
          f"N values: {sorted(df['N'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Metric 1 — Sequential match rate at each tolerance
# ---------------------------------------------------------------------------

def compute_sequential_match_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each seed, compare consecutive N snapshots.
    A point 'matches' if club is identical AND |Δaim| ≤ tolerance.
    Returns long-form DataFrame: seed, N_from, N_to, tolerance, match_rate.
    """
    print("\nComputing sequential match rates …")
    n_vals   = sorted(df["N"].unique())
    n_pairs  = list(zip(n_vals[:-1], n_vals[1:]))   # (10→20), (20→30), …
    seeds    = sorted(df["seed"].unique())

    records = []
    for seed in seeds:
        seed_df = df[df["seed"] == seed].set_index(["x", "y"])
        for n_from, n_to in n_pairs:
            prev = seed_df[seed_df["N"] == n_from][["club", "aim_offset"]]
            curr = seed_df[seed_df["N"] == n_to  ][["club", "aim_offset"]]
            # align on index (x, y)
            both = prev.join(curr, lsuffix="_prev", rsuffix="_curr", how="inner")
            if both.empty:
                continue
            for tol in TOLERANCES:
                same_club = both["club_prev"] == both["club_curr"]
                same_aim  = (both["aim_offset_prev"] - both["aim_offset_curr"]).abs() <= tol
                match_rate = (same_club & same_aim).mean() * 100
                records.append(dict(
                    seed=seed, N_from=n_from, N_to=n_to,
                    N=n_to,              # label on X-axis = the "arrival" snapshot
                    tolerance=tol,
                    match_rate=match_rate,
                ))
        if seed % 25 == 0:
            print(f"  … seed {seed}")

    out = pd.DataFrame(records)
    out.to_csv(HERE / "aim_tolerance_match_rates.csv", index=False)
    print(f"  Saved aim_tolerance_match_rates.csv  ({len(out):,} rows)")
    return out


# ---------------------------------------------------------------------------
# Metric 2 — Cross-seed aim agreement at each tolerance
# ---------------------------------------------------------------------------

def compute_aim_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (x, y, N) group, measure what fraction of seeds have
    aim_offset within ±tolerance of the cross-seed median aim.
    Also compute club agreement (unchanged from original) for reference.
    """
    print("\nComputing cross-seed aim agreement …")
    records = []
    grouped = df.groupby(["x", "y", "N"])
    total   = len(grouped)
    for i, ((x, y, n), grp) in enumerate(grouped):
        median_aim = grp["aim_offset"].median()
        mode_club  = grp["club"].mode().iloc[0]
        club_agree = (grp["club"] == mode_club).mean()

        for tol in TOLERANCES:
            aim_agree = (grp["aim_offset"] - median_aim).abs().le(tol).mean()
            records.append(dict(
                x=x, y=y, N=n,
                tolerance=tol,
                club_agreement=club_agree,
                aim_agreement=aim_agree,
                median_aim=median_aim,
                n_seeds=len(grp),
            ))
        if i % 5000 == 0:
            print(f"  … {i}/{total} groups")

    out = pd.DataFrame(records)
    out.to_csv(HERE / "aim_tolerance_agreement.csv", index=False)
    print(f"  Saved aim_tolerance_agreement.csv  ({len(out):,} rows)")
    return out


# ---------------------------------------------------------------------------
# Plot 1 — Sequential match rate curves, one panel per tolerance, overlaid
# ---------------------------------------------------------------------------

def plot_match_rate_comparison(mr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: mean ± std per tolerance ---
    ax = axes[0]
    for tol in TOLERANCES:
        sub = mr[mr["tolerance"] == tol]
        grp = sub.groupby("N")["match_rate"].agg(["mean", "std"]).reset_index()
        ax.plot(grp["N"], grp["mean"], label=TOL_LABELS[tol],
                color=TOL_COLOURS[tol], linewidth=2.5, marker="o", markersize=4)
        ax.fill_between(grp["N"],
                        grp["mean"] - grp["std"],
                        grp["mean"] + grp["std"],
                        alpha=0.12, color=TOL_COLOURS[tol])

    ax.axhline(80, color="grey", linestyle="--", linewidth=1, label="80% threshold")
    ax.set_xlabel("N (shots per grid point)")
    ax.set_ylabel("Match rate vs previous snapshot (%)")
    ax.set_title("Sequential Match Rate by Aim Tolerance\n(mean ± 1 SD across seeds)")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # --- Right: distribution at final N ---
    final_n = mr["N"].max()
    ax2 = axes[1]
    for tol in TOLERANCES:
        vals = mr[(mr["tolerance"] == tol) & (mr["N"] == final_n)]["match_rate"].dropna()
        ax2.hist(vals, bins=18, alpha=0.55, color=TOL_COLOURS[tol],
                 edgecolor="white", label=f"{TOL_LABELS[tol]} (mean={vals.mean():.1f}%)")

    ax2.axvline(80, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel(f"Match rate at N={final_n} (%)")
    ax2.set_ylabel("Number of seeds")
    ax2.set_title(f"Distribution of Final Match Rate by Tolerance")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS / "aim_tolerance_match_rate.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved → plots/{out.name}")


# ---------------------------------------------------------------------------
# Plot 2 — Side-by-side aim agreement heatmaps at N=300 for each tolerance
# ---------------------------------------------------------------------------

def plot_aim_agreement_heatmaps(ag: pd.DataFrame, n_val: int = 300) -> None:
    at_n = ag[ag["N"] == n_val]
    if at_n.empty:
        print(f"  No data at N={n_val} — skipping heatmaps.")
        return

    fig, axes = plt.subplots(1, len(TOLERANCES), figsize=(6 * len(TOLERANCES), 7),
                             sharey=True)

    for ax, tol in zip(axes, TOLERANCES):
        sub  = at_n[at_n["tolerance"] == tol]
        piv  = sub.pivot_table(index="y", columns="x", values="aim_agreement", aggfunc="mean")
        piv  = piv.sort_index(ascending=False)

        im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=1,
                       cmap="RdYlGn", origin="upper")
        plt.colorbar(im, ax=ax, label="Aim agreement fraction")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.0f}" for v in piv.columns],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:.0f}" for v in piv.index], fontsize=7)
        ax.set_xlabel("x  (lateral yards)")
        ax.set_title(f"Aim Agreement  {TOL_LABELS[tol]}\nat N={n_val}")
        if ax == axes[0]:
            ax.set_ylabel("y  (yards from tee)")

    plt.suptitle(f"Cross-Seed Aim Offset Agreement at N={n_val}\n"
                 "Fraction of seeds within ±tol yards of the median aim",
                 fontsize=11)
    plt.tight_layout()
    out = PLOTS / f"aim_tolerance_agreement_heatmaps_N{n_val:04d}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → plots/{out.name}")


# ---------------------------------------------------------------------------
# Plot 3 — Aim agreement vs N, by distance band, one line per tolerance
# ---------------------------------------------------------------------------

def plot_aim_by_distance(ag: pd.DataFrame) -> None:
    df = ag.copy()
    df["band"] = df["y"].apply(band_label)
    df = df[df["band"].isin(BAND_LABELS)]

    grp = (df.groupby(["N", "band", "tolerance"])["aim_agreement"]
             .mean()
             .reset_index(name="mean_aim_agree"))

    # One subplot per distance band
    n_bands = len(BAND_LABELS)
    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4.5), sharey=True)

    for ax, band in zip(axes, BAND_LABELS):
        sub_band = grp[grp["band"] == band]
        for tol in TOLERANCES:
            sub = sub_band[sub_band["tolerance"] == tol].sort_values("N")
            ax.plot(sub["N"], sub["mean_aim_agree"] * 100,
                    label=TOL_LABELS[tol],
                    color=TOL_COLOURS[tol], linewidth=2, marker="o", markersize=3)
        ax.axhline(80, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(band, fontsize=9)
        ax.set_xlabel("N")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Mean aim agreement (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(TOLERANCES),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cross-Seed Aim Agreement vs N by Distance Band", fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = PLOTS / "aim_tolerance_by_distance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → plots/{out.name}")


# ---------------------------------------------------------------------------
# Plot 4 — How many points are "converged" under each tolerance?
# ---------------------------------------------------------------------------

def plot_convergence_rate_comparison(ag: pd.DataFrame,
                                     mr: pd.DataFrame,
                                     threshold: float = 0.80) -> None:
    """Two panels:
    Left  — fraction of grid points with aim agreement ≥ threshold, vs N
    Right — fraction of seeds with match rate ≥ 80%, vs N
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left — grid point aim convergence rate
    ax = axes[0]
    n_vals_ag = sorted(ag["N"].unique())
    for tol in TOLERANCES:
        sub = ag[ag["tolerance"] == tol]
        conv_frac = (sub.groupby("N")["aim_agreement"]
                       .apply(lambda s: (s >= threshold).mean() * 100)
                       .reset_index(name="pct_converged"))
        conv_frac = conv_frac.sort_values("N")
        ax.plot(conv_frac["N"], conv_frac["pct_converged"],
                label=TOL_LABELS[tol], color=TOL_COLOURS[tol],
                linewidth=2.5, marker="o", markersize=4)

    ax.axhline(80, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("N (shots per grid point)")
    ax.set_ylabel(f"% of grid points with aim agreement ≥ {threshold:.0%}")
    ax.set_title("Fraction of Grid Points Aim-Converged vs N")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Right — seed-level match rate convergence
    ax2 = axes[1]
    for tol in TOLERANCES:
        sub = mr[mr["tolerance"] == tol]
        seed_conv = (sub.groupby("N")["match_rate"]
                       .apply(lambda s: (s >= 80).mean() * 100)
                       .reset_index(name="pct_seeds"))
        seed_conv = seed_conv.sort_values("N")
        ax2.plot(seed_conv["N"], seed_conv["pct_seeds"],
                 label=TOL_LABELS[tol], color=TOL_COLOURS[tol],
                 linewidth=2.5, marker="o", markersize=4)

    ax2.axhline(80, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel("N (shots per grid point)")
    ax2.set_ylabel("% of seeds with match rate ≥ 80%")
    ax2.set_title("Fraction of Seeds Above 80% Match Rate vs N")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS / "aim_tolerance_convergence_rates.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → plots/{out.name}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(mr: pd.DataFrame, ag: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("AIM TOLERANCE SENSITIVITY — SUMMARY")
    print("=" * 65)

    final_n = mr["N"].max()
    key_n   = [n for n in sorted(mr["N"].unique()) if n in [50, 100, 150, 200, 250, 300]]

    print(f"\n{'N':>5}  " +
          "  ".join(f"{'match rate ' + TOL_LABELS[t]:>18}" for t in TOLERANCES))
    print("-" * (5 + 20 * len(TOLERANCES)))
    for n in key_n:
        row = f"{n:>5}  "
        for tol in TOLERANCES:
            vals = mr[(mr["tolerance"] == tol) & (mr["N"] == n)]["match_rate"].dropna()
            row += f"  {vals.mean():6.1f}% ± {vals.std():4.1f}%     "
        print(row)

    print(f"\n  Aim agreement at N={final_n} (fraction of grid points ≥ 80%):")
    at_final = ag[ag["N"] == final_n]
    for tol in TOLERANCES:
        sub = at_final[at_final["tolerance"] == tol]
        pct = (sub["aim_agreement"] >= 0.80).mean() * 100
        mean_agree = sub["aim_agreement"].mean() * 100
        print(f"    {TOL_LABELS[tol]}:  {pct:.1f}% of points converged   "
              f"(mean agreement = {mean_agree:.1f}%)")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Aim Tolerance Sensitivity Analysis")
    print("====================================")

    df = load_strategies()

    match_rates = compute_sequential_match_rates(df)
    agreement   = compute_aim_agreement(df)

    plot_match_rate_comparison(match_rates)
    plot_aim_agreement_heatmaps(agreement, n_val=300)
    plot_aim_by_distance(agreement)
    plot_convergence_rate_comparison(agreement, match_rates)

    print_summary(match_rates, agreement)
    print(f"\nAll outputs written to: {HERE}")


if __name__ == "__main__":
    main()
