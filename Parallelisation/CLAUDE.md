# Golf Simulation & Probabilistic Modeling Framework

## Project Goals
1. **Binary Birdie Model:** Use GPyTorch with `BernoulliLikelihood` (Logit/Probit link) for birdie classification.
2. **OOP Simulation:** Modular `Player`, `Course`, and `Simulator` objects for sensitivity analysis (e.g., varying dispersion, distance, or hazard placement).
3. **Data Retention Fix:** Stop discarding individual shot outcomes. Keep the full distribution of simulated shots for each "optimal" coordinate to allow for confidence bound estimation in the next recursive step when another GPR is fit to these optimal points in Par 4s.
4. **HPC Scaling:** Parallelise across parameters and simulation types (Birdie vs. Regular) using `ProcessPoolExecutor` for Slurm clusters.

## Architecture Guidelines
* **Simulation Types:** Use a Strategy pattern where `BaseMechanism` is inherited by `BirdieMechanism` and `OptimizationMechanism`.
* **Data Storage:** Store raw shot distributions in a structured format (e.g., nested NumPy arrays or Parquet) before fitting the final GP.
* **Sensitivity Params:** All simulation variables (shots-per-club, hazard-XY, dispersion-scale) must be passed via a `Config` dataclass.

## HPC & Technical Specs
* **Environment:** Python 3.10+, GPyTorch, NumPy.
* **Parallelism:** Ensure all classes are 'Pickleable' for `multiprocessing`.
* **Memory Management:** Be mindful of RAM when keeping "thousands of shots" for every point; use `float32` where possible.

## Style Guide
* Strict Type Hinting.
* Use `logging` instead of `print` for HPC compatibility.