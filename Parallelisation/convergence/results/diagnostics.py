#!/usr/bin/env python3
"""
convergence/results/diagnostics.py
===================================
Diagnostic pipeline for the golf convergence study.

Outputs (all written to the same results/ directory):
  Data
  ----
  match_rates_all_seeds.csv       – long-form match-rate table (seed × N)
  strategies_key_N.parquet        – merged strategy table at key N checkpoints
  gridpoint_agreement.csv         – per-grid-point club agreement rate (x, y, N, agreement)
  non_converging_gridpoints.csv   – points with agreement < threshold at N=300
  convergence_summary_stats.csv   – per-seed stats (final match rate, reached_80, ...)

  Plots
  -----
  plots/match_rate_curves.png           – all-seed curves + mean ± std band
  plots/gridpoint_agreement_N300.png    – spatial heatmap of agreement at N=300
  plots/agreement_by_distance.png       – agreement vs N grouped by distance band
  plots/non_converging_map.png          – which grid points never converge
  plots/club_stability_by_distance.png  – violin of final agreement per distance band
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent          # …/convergence/results/
OUTPUTS = HERE.parent / "outputs"     # …/convergence/outputs/
PLOTS = HERE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# N checkpoints to analyse per-grid-point agreement (subset keeps runtime sane)
KEY_N = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300]

# Convergence threshold for "converged" label
AGREEMENT_THRESHOLD = 0.80   # 80 % of seeds agree on same club

# Distance bands (y values in yards from pin)
DISTANCE_BANDS = [(50, 80), (80, 120), (120, 180), (180, 250), (250, 320)]
BAND_LABELS    = ["50-80 yd", "80-120 yd", "120-180 yd", "180-250 yd", "250+ yd"]

# ---------------------------------------------------------------------------
# Helper: discover all seed directories
# ---------------------------------------------------------------------------
def discover_seeds() -> list[int]:
    seeds = sorted(
        int(p.name.replace("seed", ""))
        for p in OUTPUTS.iterdir()
        if p.is_dir() and re.fullmatch(r"seed\d{4}", p.name)
    )
    print(f"Found {len(seeds)} seed directories.")
    return seeds


# ===========================================================================
# STEP 1 – Aggregate match-rate TSV files
# ===========================================================================
def aggregate_match_rates(seeds: list[int]) -> pd.DataFrame:
    print("\n[1/4] Aggregating match-rate TSV files …")
    frames = []
    for seed in seeds:
        tsv_path = OUTPUTS / f"seed{seed:04d}" / f"seed{seed:04d}_match_rate.tsv"
        if not tsv_path.exists():
            print(f"  WARNING: {tsv_path.name} not found – skipping seed {seed}")
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        df["seed"] = seed
        # "N/A" → NaN
        df["match_rate_pct"] = pd.to_numeric(df["match_rate_pct"], errors="coerce")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    out = HERE / "match_rates_all_seeds.csv"
    combined.to_csv(out, index=False)
    print(f"  Saved → {out.name}  ({len(combined):,} rows)")

    # Per-seed summary stats
    final = (
        combined.groupby("seed")["match_rate_pct"]
        .agg(
            final_match_rate=lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan,
            max_match_rate="max",
            mean_match_rate="mean",
        )
        .reset_index()
    )
    # First N where match rate >= 80
    def first_n_above(group, threshold=80):
        above = group[group["match_rate_pct"] >= threshold]
        if above.empty:
            return np.nan
        return above["N"].iloc[0]

    reached_80 = combined.groupby("seed").apply(
        lambda g: first_n_above(g, 80), include_groups=False
    ).rename("N_reached_80pct").reset_index()

    summary = final.merge(reached_80, on="seed")
    out_s = HERE / "convergence_summary_stats.csv"
    summary.to_csv(out_s, index=False)
    print(f"  Saved → {out_s.name}")

    return combined


# ===========================================================================
# STEP 2 – Load strategy CSVs at key N checkpoints
# ===========================================================================
def load_strategies(seeds: list[int]) -> pd.DataFrame:
    cache_path = HERE / "strategies_key_N.csv"
    if cache_path.exists():
        print("\n[2/4] Loading cached strategy CSV …")
        df = pd.read_csv(cache_path)
        print(f"  Loaded {len(df):,} rows from cache.")
        return df

    print(f"\n[2/4] Loading strategy CSVs for N ∈ {KEY_N} …")
    frames = []
    total = len(seeds) * len(KEY_N)
    done = 0
    for seed in seeds:
        for n in KEY_N:
            csv_path = OUTPUTS / f"seed{seed:04d}" / f"seed{seed:04d}_N{n:04d}.csv"
            if not csv_path.exists():
                done += 1
                continue
            df = pd.read_csv(csv_path, usecols=["x", "y", "club", "aim_offset", "esho_mean", "esho_var", "seed", "N"])
            frames.append(df)
            done += 1
        if seed % 10 == 0:
            print(f"  … {done}/{total} files loaded")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(cache_path, index=False)
    print(f"  Saved → {cache_path.name}  ({len(combined):,} rows)")
    return combined


# ===========================================================================
# STEP 3 – Per-grid-point club agreement
# ===========================================================================
def compute_gridpoint_agreement(strategies: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/4] Computing per-grid-point club agreement …")

    def modal_fraction(series):
        """Fraction of values equal to the mode."""
        if series.empty:
            return np.nan
        mode_val = series.mode().iloc[0]
        return (series == mode_val).mean()

    records = []
    for n_val, grp_n in strategies.groupby("N"):
        for (x, y), grp_pt in grp_n.groupby(["x", "y"]):
            club_agree = modal_fraction(grp_pt["club"])
            aim_agree  = modal_fraction(grp_pt["aim_offset"])
            # Joint: same club AND same aim offset
            joint = grp_pt[["club", "aim_offset"]].apply(tuple, axis=1)
            joint_agree = modal_fraction(joint)
            records.append(
                dict(x=x, y=y, N=n_val,
                     club_agreement=club_agree,
                     aim_agreement=aim_agree,
                     joint_agreement=joint_agree,
                     n_seeds=len(grp_pt))
            )

    agreement = pd.DataFrame(records)
    out = HERE / "gridpoint_agreement.csv"
    agreement.to_csv(out, index=False)
    print(f"  Saved → {out.name}  ({len(agreement):,} rows)")

    # Non-converging points at N=300
    at_300 = agreement[agreement["N"] == 300].copy()
    non_conv = at_300[at_300["club_agreement"] < AGREEMENT_THRESHOLD].sort_values(
        "club_agreement"
    )
    out_nc = HERE / "non_converging_gridpoints.csv"
    non_conv.to_csv(out_nc, index=False)
    print(
        f"  Non-converging points (club agreement < {AGREEMENT_THRESHOLD:.0%} at N=300): "
        f"{len(non_conv)} / {len(at_300)}"
    )
    print(f"  Saved → {out_nc.name}")
    return agreement


# ===========================================================================
# STEP 4 – Visualisations
# ===========================================================================
def plot_match_rate_curves(match_rates: pd.DataFrame) -> None:
    print("\n[4/4] Plotting …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # --- Left: all seeds ---
    ax = axes[0]
    pivoted = match_rates.pivot_table(index="N", columns="seed", values="match_rate_pct", aggfunc="mean")
    # individual seed lines
    ax.plot(pivoted.index, pivoted.values, color="#2196F3", alpha=0.12, linewidth=0.7)
    # mean ± std band
    mu  = pivoted.mean(axis=1)
    std = pivoted.std(axis=1)
    ax.fill_between(pivoted.index, mu - std, mu + std, alpha=0.25, color="#E91E63", label="±1 SD")
    ax.plot(pivoted.index, mu, color="#E91E63", linewidth=2.5, label="Mean")
    ax.axhline(80, color="grey", linestyle="--", linewidth=1, label="80 % threshold")
    ax.set_xlabel("N (shots per grid point)")
    ax.set_ylabel("Match rate vs previous N  (%)")
    ax.set_title("Match Rate Convergence – All Seeds")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # --- Right: distribution at final N ---
    ax2 = axes[1]
    final_n = match_rates["N"].max()
    final_vals = match_rates[match_rates["N"] == final_n]["match_rate_pct"].dropna()
    ax2.hist(final_vals, bins=20, color="#2196F3", edgecolor="white", alpha=0.85)
    ax2.axvline(final_vals.mean(), color="#E91E63", linewidth=2, label=f"Mean = {final_vals.mean():.1f}%")
    ax2.axvline(80, color="grey", linestyle="--", linewidth=1, label="80 % threshold")
    ax2.set_xlabel(f"Match rate at N={final_n}  (%)")
    ax2.set_ylabel("Number of seeds")
    ax2.set_title(f"Distribution of Final Match Rate (N={final_n})")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS / "match_rate_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → plots/{out.name}")


def plot_gridpoint_agreement_heatmap(agreement: pd.DataFrame, n_val: int = 300) -> None:
    at_n = agreement[agreement["N"] == n_val].copy()
    if at_n.empty:
        print(f"  No data for N={n_val} – skipping heatmap.")
        return

    pivot = at_n.pivot_table(index="y", columns="x", values="club_agreement", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)   # closest shot at bottom

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, col, title, cmap in zip(
        axes,
        ["club_agreement", "joint_agreement"],
        [f"Club Agreement (N={n_val})", f"Joint Club+Aim Agreement (N={n_val})"],
        ["RdYlGn", "RdYlGn"],
    ):
        piv = at_n.pivot_table(index="y", columns="x", values=col, aggfunc="mean")
        piv = piv.sort_index(ascending=False)

        im = ax.imshow(
            piv.values,
            aspect="auto",
            vmin=0, vmax=1,
            cmap=cmap,
            origin="upper",
        )
        plt.colorbar(im, ax=ax, label="Fraction of seeds agreeing")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.0f}" for v in piv.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in piv.index], fontsize=7)
        ax.set_xlabel("x  (lateral yards)")
        ax.set_ylabel("y  (yards from tee)")
        ax.set_title(title)

    plt.tight_layout()
    out = PLOTS / f"gridpoint_agreement_N{n_val:04d}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → plots/{out.name}")


def plot_agreement_by_distance(agreement: pd.DataFrame) -> None:
    """Mean club agreement vs N, faceted by distance band."""
    def band_label(y):
        for (lo, hi), lbl in zip(DISTANCE_BANDS, BAND_LABELS):
            if lo <= y < hi:
                return lbl
        return f"{y:.0f}+ yd"

    df = agreement.copy()
    df["distance_band"] = df["y"].apply(band_label)
    # Keep only the defined bands
    df = df[df["distance_band"].isin(BAND_LABELS)]

    grp = (
        df.groupby(["N", "distance_band"])["club_agreement"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    palette = sns.color_palette("viridis", len(BAND_LABELS))
    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl, color in zip(BAND_LABELS, palette):
        sub = grp[grp["distance_band"] == lbl].sort_values("N")
        ax.plot(sub["N"], sub["mean"] * 100, label=lbl, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(
            sub["N"],
            (sub["mean"] - sub["std"]) * 100,
            (sub["mean"] + sub["std"]) * 100,
            alpha=0.12, color=color,
        )

    ax.axhline(80, color="grey", linestyle="--", linewidth=1, label="80 % threshold")
    ax.set_xlabel("N (shots per grid point)")
    ax.set_ylabel("Mean club agreement across seeds  (%)")
    ax.set_title("Club Strategy Convergence by Distance Band")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS / "agreement_by_distance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → plots/{out.name}")


def plot_non_converging_map(agreement: pd.DataFrame) -> None:
    """Scatter of grid points coloured by final agreement; non-converging highlighted."""
    at_300 = agreement[agreement["N"] == 300].copy()
    if at_300.empty:
        print("  No N=300 data – skipping non-converging map.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: spatial scatter ---
    ax = axes[0]
    sc = ax.scatter(
        at_300["x"], at_300["y"],
        c=at_300["club_agreement"],
        cmap="RdYlGn", vmin=0, vmax=1,
        s=80, edgecolors="k", linewidths=0.3, alpha=0.9,
    )
    plt.colorbar(sc, ax=ax, label="Club agreement fraction")
    # circle the non-converging ones
    nc = at_300[at_300["club_agreement"] < AGREEMENT_THRESHOLD]
    ax.scatter(nc["x"], nc["y"], s=200, facecolors="none",
               edgecolors="red", linewidths=1.5, zorder=5, label=f"< {AGREEMENT_THRESHOLD:.0%}")
    ax.set_xlabel("x  (lateral yards)")
    ax.set_ylabel("y  (distance from tee, yards)")
    ax.set_title(f"Grid-point Club Agreement at N=300\n(red = < {AGREEMENT_THRESHOLD:.0%}, n={len(nc)})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # --- Right: agreement distribution by distance band ---
    ax2 = axes[1]
    def band_label(y):
        for (lo, hi), lbl in zip(DISTANCE_BANDS, BAND_LABELS):
            if lo <= y < hi:
                return lbl
        return "250+ yd"

    at_300["band"] = at_300["y"].apply(band_label)
    band_order = [b for b in BAND_LABELS if b in at_300["band"].unique()]
    palette = dict(zip(BAND_LABELS, sns.color_palette("viridis", len(BAND_LABELS))))

    sns.boxplot(
        data=at_300, x="band", y="club_agreement", hue="band",
        order=band_order, palette=palette, ax=ax2,
        width=0.5, legend=False,
    )
    ax2.axhline(AGREEMENT_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
                label=f"{AGREEMENT_THRESHOLD:.0%} threshold")
    ax2.set_xlabel("Distance band")
    ax2.set_ylabel("Club agreement fraction at N=300")
    ax2.set_title("Club Agreement Distribution by Distance")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = PLOTS / "non_converging_map.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → plots/{out.name}")


def plot_club_stability_violin(agreement: pd.DataFrame) -> None:
    """Violin plot: distribution of club agreement per distance band across all N values."""
    def band_label(y):
        for (lo, hi), lbl in zip(DISTANCE_BANDS, BAND_LABELS):
            if lo <= y < hi:
                return lbl
        return "250+ yd"

    df = agreement.copy()
    df["band"] = df["y"].apply(band_label)
    df = df[df["band"].isin(BAND_LABELS)]

    # Focus on N >= 50 to avoid noise at tiny N
    df = df[df["N"] >= 50]

    palette = dict(zip(BAND_LABELS, sns.color_palette("viridis", len(BAND_LABELS))))
    band_order = [b for b in BAND_LABELS if b in df["band"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Violin of club agreement
    sns.violinplot(
        data=df, x="band", y="club_agreement", hue="band",
        order=band_order, palette=palette, ax=axes[0],
        inner="box", cut=0, legend=False,
    )
    axes[0].axhline(AGREEMENT_THRESHOLD, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title("Club Agreement Distribution (N ≥ 50)")
    axes[0].set_xlabel("Distance band")
    axes[0].set_ylabel("Club agreement fraction")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Fraction of points converged at each N, by band
    conv_rate = (
        df.groupby(["N", "band"])["club_agreement"]
        .apply(lambda s: (s >= AGREEMENT_THRESHOLD).mean())
        .reset_index(name="pct_converged")
    )
    palette_list = sns.color_palette("viridis", len(BAND_LABELS))
    for lbl, color in zip(BAND_LABELS, palette_list):
        sub = conv_rate[conv_rate["band"] == lbl].sort_values("N")
        if sub.empty:
            continue
        axes[1].plot(sub["N"], sub["pct_converged"] * 100, label=lbl, color=color,
                     linewidth=2, marker="o", markersize=4)

    axes[1].axhline(80, color="grey", linestyle="--", linewidth=1)
    axes[1].set_xlabel("N (shots per grid point)")
    axes[1].set_ylabel(f"% of grid points with club agreement ≥ {AGREEMENT_THRESHOLD:.0%}")
    axes[1].set_title("Fraction of Points Converged, by Distance Band")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS / "club_stability_by_distance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → plots/{out.name}")


def print_summary(agreement: pd.DataFrame, match_rates: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("CONVERGENCE SUMMARY")
    print("=" * 60)

    # Match rate stats at key N
    for n_val in [50, 100, 200, 300]:
        sub = match_rates[match_rates["N"] == n_val]["match_rate_pct"].dropna()
        if sub.empty:
            continue
        print(f"  N={n_val:3d} | match rate: mean={sub.mean():.1f}%  "
              f"median={sub.median():.1f}%  "
              f"p10={sub.quantile(0.1):.1f}%  "
              f"p90={sub.quantile(0.9):.1f}%")

    # Grid point agreement at N=300
    at_300 = agreement[agreement["N"] == 300]
    if not at_300.empty:
        print(f"\n  Grid-point club agreement at N=300:")
        print(f"    Mean:   {at_300['club_agreement'].mean():.1%}")
        print(f"    Median: {at_300['club_agreement'].median():.1%}")
        nc = (at_300["club_agreement"] < AGREEMENT_THRESHOLD).sum()
        print(f"    Points with agreement < {AGREEMENT_THRESHOLD:.0%}: {nc} / {len(at_300)}")

        def band_label(y):
            for (lo, hi), lbl in zip(DISTANCE_BANDS, BAND_LABELS):
                if lo <= y < hi:
                    return lbl
            return "250+ yd"

        at_300 = at_300.copy()
        at_300["band"] = at_300["y"].apply(band_label)
        print(f"\n  Mean club agreement at N=300 by distance band:")
        for band in BAND_LABELS:
            sub = at_300[at_300["band"] == band]
            if sub.empty:
                continue
            print(f"    {band:12s}: {sub['club_agreement'].mean():.1%}  "
                  f"(n_points={len(sub)}, "
                  f"not converged={( sub['club_agreement'] < AGREEMENT_THRESHOLD).sum()})")

    print("=" * 60)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("Golf Convergence Diagnostic Pipeline")
    print("=====================================")

    seeds = discover_seeds()

    match_rates = aggregate_match_rates(seeds)
    strategies  = load_strategies(seeds)
    agreement   = compute_gridpoint_agreement(strategies)

    plot_match_rate_curves(match_rates)
    plot_gridpoint_agreement_heatmap(agreement, n_val=300)
    plot_agreement_by_distance(agreement)
    plot_non_converging_map(agreement)
    plot_club_stability_violin(agreement)

    print_summary(agreement, match_rates)
    print(f"\nAll outputs written to: {HERE}")


if __name__ == "__main__":
    main()
