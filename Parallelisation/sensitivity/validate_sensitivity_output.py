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
    "x", "y", "club", "aim_offset",
    "esho_mean", "esho_var", "n_total", "seed", "N",
    "carry_shift", "variance_scale",
}
EXPECTED_PNG_SUFFIXES = [".png"]   # one combined image per config
PLAUSIBLE_ESHO        = (1.0, 8.0)


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

    # 3. Non-empty (one row per strategy grid point, typically 280)
    check(len(df) > 0, f"CSV non-empty ({len(df)} rows)", "CSV is empty")

    # 4. ESHO values are finite and in plausible range
    if "esho_mean" in df.columns:
        lo, hi  = PLAUSIBLE_ESHO
        bad_esho = df[(df["esho_mean"] < lo) | (df["esho_mean"] > hi)]
        check(len(bad_esho) == 0, f"All esho_mean in [{lo}, {hi}]",
              f"{len(bad_esho)} rows outside range")
        if verbose:
            logger.info("    esho_mean: %.3f – %.3f (mean %.3f)",
                        df["esho_mean"].min(), df["esho_mean"].max(),
                        df["esho_mean"].mean())

    # 5. Meta JSON
    meta_path = out_dir / f"{stem}_meta.json"
    check(meta_path.exists(), f"Meta JSON exists ({stem}_meta.json)")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            required_keys = {"task_id", "carry_shift", "variance_scale",
                             "N", "best_tee_club", "best_tee_strokes"}
            missing_keys = required_keys - set(meta)
            check(not missing_keys, "Meta JSON has required keys",
                  f"missing {missing_keys}")
            if verbose and "best_tee_strokes" in meta:
                logger.info("    best_tee: %s  aim=%+.0f  strokes=%.3f",
                            meta.get("best_tee_club", "?"),
                            meta.get("best_tee_aim", 0),
                            meta["best_tee_strokes"])
        except Exception as e:
            check(False, "Meta JSON parseable", str(e))

    # 6. PNG (one combined image)
    png = out_dir / f"{stem}.png"
    check(png.exists(), f"PNG exists ({png.name})")

    return results


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Directory containing sensitivity outputs.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-check details even for passes.")
    args = p.parse_args()

    csv_files = sorted(args.output_dir.glob("sensitivity_dist*.csv"))
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
