"""
top2_esho_analysis.py – Compare top-2 recommended strategies per grid point.

For each (x, y, N) group across 100 seeds, find the two most frequently
recommended (club, aim_offset) strategies and test whether their ESHO
difference is smaller than the SE of the top strategy.  When it is, the
two strategies are *functionally equivalent* — we cannot statistically
distinguish which is better, suggesting N is still too low.

Usage (from results/ directory):
    python top2_esho_analysis.py
    python top2_esho_analysis.py --input strategies_key_N.csv --min-freq 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_analysis(
    input_csv: Path,
    output_csv: Path,
    min_freq: int = 1,
) -> pd.DataFrame:
    """Load strategy CSV and compute top-2 comparison for every (x, y, N).

    Parameters
    ----------
    input_csv : Path
        Path to strategies_key_N.csv (columns: x, y, club, aim_offset,
        esho_mean, esho_var, seed, N).
    output_csv : Path
        Where to write the per-(x, y, N) results.
    min_freq : int
        Minimum number of seeds that must recommend a strategy for it to
        be considered a candidate top-2 entry.  Default 1 (include all).

    Returns
    -------
    pd.DataFrame with one row per (x, y, N) where a top-2 pair exists.
    """
    logger.info("Loading %s …", input_csv)
    df = pd.read_csv(input_csv)

    # Round aim_offset to nearest yard to avoid float noise in grouping
    df["aim_offset_r"] = df["aim_offset"].round(0).astype(int)
    df["strategy"] = list(zip(df["club"], df["aim_offset_r"]))

    records: list[dict] = []

    for (x, y, N), group in df.groupby(["x", "y", "N"]):
        freq = group["strategy"].value_counts()

        # Need at least 2 distinct strategies to compare
        candidates = freq[freq >= min_freq]
        if len(candidates) < 2:
            continue

        top1_key = candidates.index[0]
        top2_key = candidates.index[1]

        g1 = group[group["strategy"] == top1_key]
        g2 = group[group["strategy"] == top2_key]

        esho1 = float(g1["esho_mean"].mean())
        esho2 = float(g2["esho_mean"].mean())

        # SE of ESHO estimate: sqrt(sample_variance / n_shots)
        # esho_var is variance of shot outcomes; n_shots ≈ N
        mean_var1 = float(g1["esho_var"].mean())
        se1 = float(np.sqrt(mean_var1 / N)) if N > 0 else float("inf")

        delta = abs(esho1 - esho2)
        func_equiv = delta < se1

        records.append({
            "x":         float(x),
            "y":         float(y),
            "N":         int(N),
            # Top-1 strategy
            "club_1":    top1_key[0],
            "aim_1":     int(top1_key[1]),
            "esho_1":    esho1,
            "se_1":      se1,
            "freq_1":    int(candidates.iloc[0]),
            # Top-2 strategy
            "club_2":    top2_key[0],
            "aim_2":     int(top2_key[1]),
            "esho_2":    esho2,
            "freq_2":    int(candidates.iloc[1]),
            # Comparison
            "delta_esho":            delta,
            "functionally_equivalent": func_equiv,
        })

    out = pd.DataFrame(records)
    out.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows → %s", len(out), output_csv)
    return out


def print_summary(df: pd.DataFrame) -> None:
    """Print per-N fraction of functionally equivalent grid points."""
    print("\n" + "=" * 65)
    print("  Fraction of grid points where top-2 strategies are")
    print("  functionally equivalent (|ΔESHO| < SE of top strategy)")
    print("  → these points need higher N to distinguish strategies")
    print("-" * 65)
    print(f"  {'N':>6}  {'Flagged / Total':>18}  {'Fraction':>10}")
    print("-" * 65)

    summary = (
        df.groupby("N")["functionally_equivalent"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "flagged", "count": "total"})
    )
    summary["fraction"] = summary["flagged"] / summary["total"]

    for N, row in summary.iterrows():
        flag = "  ← needs more N" if row["fraction"] > 0.20 else ""
        print(f"  {int(N):>6}  {int(row['flagged']):>7} / {int(row['total']):<9}  "
              f"{row['fraction']:>9.1%}{flag}")

    best_N = summary[summary["fraction"] <= 0.10].index.min()
    if pd.notna(best_N):
        print(f"\n  Suggested minimum N: {int(best_N)}  (≤10 % of points flagged)")
    else:
        print("\n  N=300 may still be insufficient — >10 % of points flagged at all N.")
    print("=" * 65 + "\n")


def plot_flagged_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Save one heatmap per N showing which grid points are flagged."""
    for N, group in df.groupby("N"):
        fig, ax = plt.subplots(figsize=(8, 10))
        flagged   = group[group["functionally_equivalent"]]
        unflagged = group[~group["functionally_equivalent"]]

        ax.scatter(unflagged["x"], unflagged["y"],
                   c="steelblue", s=40, alpha=0.7, label="Distinguishable")
        ax.scatter(flagged["x"], flagged["y"],
                   c="tomato", s=60, alpha=0.9, marker="X", label="Equiv (need ↑N)")

        frac = group["functionally_equivalent"].mean()
        ax.set_title(f"N={N}  |  {frac:.1%} of points functionally equivalent")
        ax.set_xlabel("x (lateral yards)")
        ax.set_ylabel("y (distance from tee, yards)")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":")

        out = output_dir / f"top2_flagged_N{int(N):04d}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved heatmap → %s", out)


def plot_delta_vs_se(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter |ΔESHO| vs SE_1 coloured by N to visualise separation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    N_vals = sorted(df["N"].unique())
    cmap   = plt.get_cmap("viridis", len(N_vals))

    for i, N in enumerate(N_vals):
        g = df[df["N"] == N]
        ax.scatter(g["se_1"], g["delta_esho"],
                   s=12, alpha=0.5, color=cmap(i), label=f"N={N}")

    # Diagonal: |Δ| = SE (boundary of equivalence)
    lim = max(df["se_1"].max(), df["delta_esho"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="|Δ| = SE")
    ax.fill_between([0, lim], 0, [0, lim], alpha=0.07, color="tomato",
                    label="Equiv region")

    ax.set_xlabel("SE of top strategy (√(esho_var / N))")
    ax.set_ylabel("|ESHO_1 − ESHO_2|")
    ax.set_title("ESHO difference vs. SE — top-2 strategy comparison")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved delta-vs-SE plot → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Compare top-2 ESHO strategies per grid point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",    type=Path, default=here / "strategies_key_N.csv")
    p.add_argument("--output",   type=Path, default=here / "top2_analysis.csv")
    p.add_argument("--plots",    type=Path, default=here,
                   help="Directory for output plots.")
    p.add_argument("--min-freq", type=int,  default=1,
                   help="Min seeds recommending a strategy to count as top-2 candidate.")
    p.add_argument("--heatmaps", action="store_true",
                   help="Save per-N heatmap PNGs of flagged grid points.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    df = run_analysis(args.input, args.output, min_freq=args.min_freq)
    print_summary(df)

    args.plots.mkdir(parents=True, exist_ok=True)
    plot_delta_vs_se(df, args.plots / "top2_delta_vs_se.png")
    if args.heatmaps:
        plot_flagged_heatmap(df, args.plots)


if __name__ == "__main__":
    main()
