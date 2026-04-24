"""
validate_sensitivity_output.py – Post-run output checker for sensitivity tasks.

Run after a Slurm job (test or full) to verify every expected output file
exists, has sensible content, and passes the Driver-exclusion check.

Usage:
    python validate_sensitivity_output.py --output-dir outputs/
    python validate_sensitivity_output.py --output-dir test_outputs/ --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("validate")

EXPECTED_CSV_COLS = {
    "sim_id", "shot_num", "x", "y", "club",
    "lie", "is_tee", "aim_offset", "carry_shift", "variance_scale",
}
EXPECTED_PNG_SUFFIXES = ["_landing_map.png", "_score_dist.png", "_tee_shot.png"]
PLAUSIBLE_STROKES     = (2, 15)


def _ok(label: str) -> None:
    logger.info("  PASS  %s", label)


def _fail(label: str) -> None:
    logger.error("  FAIL  %s", label)


def validate_one(csv_path: Path, verbose: bool = False) -> dict:
    """Validate one CSV file and its sidecar outputs. Returns a result dict."""
    stem    = csv_path.stem                      # e.g. sim_output_dist0.00_disp1.0000
    out_dir = csv_path.parent
    results = {"file": csv_path.name, "checks": [], "passed": 0, "failed": 0}

    def check(cond: bool, label: str, detail: str = "") -> None:
        if cond:
            results["checks"].append(("PASS", label))
            results["passed"] += 1
            if verbose:
                _ok(label)
        else:
            results["checks"].append(("FAIL", label + (f" — {detail}" if detail else "")))
            results["failed"] += 1
            _fail(label + (f" — {detail}" if detail else ""))

    # 1. CSV readable
    try:
        df = pd.read_csv(csv_path)
        check(True, "CSV readable")
    except Exception as e:
        check(False, "CSV readable", str(e))
        return results  # can't continue

    # 2. Columns
    missing = EXPECTED_CSV_COLS - set(df.columns)
    check(not missing, "CSV columns complete", f"missing {missing}")

    # 3. Non-empty
    check(len(df) > 0, "CSV non-empty", f"{len(df)} rows")

    # 4. Driver never in non-tee shots
    if "is_tee" in df.columns and "club" in df.columns:
        bad = df[(df["is_tee"] == False) & (df["club"] == "Driver")]
        check(len(bad) == 0, "Driver excluded from non-tee shots",
              f"{len(bad)} violations")

    # 5. All shot_num==1 are tee shots
    if "shot_num" in df.columns and "is_tee" in df.columns:
        shot1 = df[df["shot_num"] == 1]
        check((shot1["is_tee"] == True).all(), "All shot_num=1 have is_tee=True")
        late_tee = df[(df["shot_num"] > 1) & (df["is_tee"] == True)]
        check(len(late_tee) == 0, "No is_tee=True after shot_num=1",
              f"{len(late_tee)} rows")

    # 6. Score range
    if "sim_id" in df.columns and "shot_num" in df.columns:
        strokes = df.groupby("sim_id")["shot_num"].max()
        lo, hi  = PLAUSIBLE_STROKES
        out_rng = strokes[(strokes < lo) | (strokes > hi)]
        check(len(out_rng) == 0, f"Scores in [{lo}, {hi}]",
              f"{len(out_rng)} holes outside range: {out_rng.tolist()[:5]}")
        if verbose:
            logger.info("    Strokes: mean=%.2f std=%.2f min=%d max=%d",
                        strokes.mean(), strokes.std(), strokes.min(), strokes.max())

    # 7. Meta JSON
    meta_path = out_dir / f"{stem}_meta.json"
    check(meta_path.exists(), f"Meta JSON exists ({stem}_meta.json)")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            required_keys = {"task_id", "carry_shift", "variance_scale",
                             "n_sims", "mean_strokes", "tee_top_club"}
            missing_keys = required_keys - set(meta)
            check(not missing_keys, "Meta JSON has required keys",
                  f"missing {missing_keys}")
            if verbose and "mean_strokes" in meta:
                logger.info("    mean_strokes=%.2f  tee_club=%s",
                            meta["mean_strokes"], meta.get("tee_top_club", "?"))
        except Exception as e:
            check(False, "Meta JSON parseable", str(e))

    # 8. Tee summary JSON
    tee_path = out_dir / f"{stem}_tee.json"
    check(tee_path.exists(), f"Tee summary JSON exists ({stem}_tee.json)")
    if tee_path.exists():
        try:
            tee = json.loads(tee_path.read_text())
            tee_keys = {"n_tee_shots", "top_club", "landing_mean_x", "landing_mean_y"}
            check(tee_keys.issubset(tee), "Tee JSON has required keys",
                  f"missing {tee_keys - set(tee)}")
            if verbose:
                logger.info("    Tee: top_club=%s  mean=(%.1f, %.1f)",
                            tee.get("top_club"), tee.get("landing_mean_x"),
                            tee.get("landing_mean_y"))
        except Exception as e:
            check(False, "Tee JSON parseable", str(e))

    # 9. PNGs
    for suffix in EXPECTED_PNG_SUFFIXES:
        png = out_dir / f"{stem}{suffix}"
        check(png.exists(), f"Plot exists ({png.name})")

    return results


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Directory containing sensitivity outputs.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-check details even for passes.")
    args = p.parse_args()

    csv_files = sorted(args.output_dir.glob("sim_output_dist*.csv"))
    if not csv_files:
        logger.error("No sim_output_dist*.csv files found in %s", args.output_dir)
        sys.exit(1)

    logger.info("Validating %d output file(s) in %s", len(csv_files), args.output_dir)
    logger.info("")

    all_passed = 0
    all_failed = 0

    for csv_path in csv_files:
        logger.info("─── %s ───", csv_path.name)
        result = validate_one(csv_path, verbose=args.verbose)
        all_passed += result["passed"]
        all_failed += result["failed"]
        status = "OK" if result["failed"] == 0 else f"{result['failed']} FAIL"
        logger.info("    → %s  (%d/%d checks)", status,
                    result["passed"], result["passed"] + result["failed"])
        logger.info("")

    logger.info("=" * 60)
    logger.info("Total: %d passed  %d failed  across %d files",
                all_passed, all_failed, len(csv_files))

    if all_failed > 0:
        logger.error("Validation FAILED — do not proceed with full run.")
        sys.exit(1)
    else:
        logger.info("All checks passed. Safe to submit full 24-task array.")
        sys.exit(0)


if __name__ == "__main__":
    main()
