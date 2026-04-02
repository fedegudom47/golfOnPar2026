"""
Golf simulation core: Player, Config, Mechanism hierarchy, and Simulator.

Architecture (from CLAUDE.md):
  - BaseMechanism  →  BirdieMechanism  (VariationalGP + BernoulliLikelihood)
                   →  OptimizationMechanism  (ExactGP)
  - Simulator uses a Strategy-pattern mechanism to evaluate shot outcomes.
  - Full shot distributions are preserved for every optimal coordinate so
    that variance / confidence intervals survive into the next GPR stage.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import gpytorch
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All tuneable parameters for one simulation run.

    Passing everything through a Config keeps the classes pickle-safe and
    makes HPC parameter sweeps trivial (one Config per grid point).
    """
    n_shots: int = 200
    dispersion_multiplier: float = 1.0   # scales club covariance matrices
    power_multiplier: float = 1.0        # scales club mean carry distances
    aim_range: tuple[float, float] = (-20.0, 20.0)
    aim_step: float = 2.0
    top_n_clubs: int = 5                 # clubs closest to target distance
    water_penalty: float = 2.0           # extra strokes for water drop
    gp_training_iter: int = 100
    gp_lr: float = 0.1
    n_inducing: int = 64                 # inducing points for VariationalGP


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """Encapsulates player-specific shot characteristics.

    dispersion: multiplier applied to every club's covariance matrix.
                > 1 → wilder, < 1 → tighter.
    power:      multiplier applied to every club's mean carry distance.
                > 1 → longer, < 1 → shorter.
    """
    dispersion: float = 1.0
    power: float = 1.0

    def apply_to_distribution(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, cov) scaled by player attributes (float32 output)."""
        scaled_mean = mean.copy().astype(np.float32)
        scaled_mean[1] *= self.power          # index 1 = carry distance

        # dispersion scales std-dev, so variance scales by dispersion^2
        scaled_cov = (cov * self.dispersion ** 2).astype(np.float32)
        return scaled_mean, scaled_cov


# ---------------------------------------------------------------------------
# Simulation result containers
# ---------------------------------------------------------------------------

@dataclass
class ShotOutcomes:
    """Raw outcomes for one (club, aim_offset) candidate."""
    club: str
    aim_offset: float
    strokes_array: np.ndarray   # shape (n_valid_shots,), dtype float32
    landing_points: np.ndarray  # shape (n_valid_shots, 2), dtype float32

    @property
    def mean(self) -> float:
        return float(np.mean(self.strokes_array)) if len(self.strokes_array) else float("nan")

    @property
    def variance(self) -> float:
        return float(np.var(self.strokes_array)) if len(self.strokes_array) else float("nan")


@dataclass
class SimulationResult:
    """The chosen optimal outcome for a single strategy coordinate.

    Crucially, strokes_array is the *full distribution* of the N simulated
    shots for the winning (club, aim_offset) pair — not just the mean.
    This enables variance estimation and confidence interval propagation
    when a second GPR is fitted to these optimal points.
    """
    start: tuple[float, float]
    club: str
    aim_offset: float
    strokes_array: np.ndarray   # shape (n_valid_shots,), dtype float32
    landing_points: np.ndarray  # shape (n_valid_shots, 2), dtype float32
    total_strokes_mean: float   # pre-computed for convenience (includes penalty)
    total_strokes_variance: float

    @classmethod
    def from_shot_outcomes(
        cls,
        start: tuple[float, float],
        outcomes: ShotOutcomes,
        penalty: float = 0.0,
    ) -> "SimulationResult":
        adjusted = outcomes.strokes_array + penalty
        return cls(
            start=start,
            club=outcomes.club,
            aim_offset=outcomes.aim_offset,
            strokes_array=adjusted.astype(np.float32),
            landing_points=outcomes.landing_points,
            total_strokes_mean=float(np.mean(adjusted)),
            total_strokes_variance=float(np.var(adjusted)),
        )


# ---------------------------------------------------------------------------
# Mechanism hierarchy
# ---------------------------------------------------------------------------

class BaseMechanism(ABC):
    """Strategy interface for evaluating the value of a landing point."""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the internal GP model to (X, y) training data."""

    @abstractmethod
    def evaluate(self, point: tuple[float, float]) -> float:
        """Return a scalar value for a 2-D landing coordinate."""

    @property
    @abstractmethod
    def optimize_direction(self) -> str:
        """'min' to minimise evaluate() (expected strokes),
           'max' to maximise it (birdie probability)."""

    def is_better(self, candidate: float, current_best: float) -> bool:
        if self.optimize_direction == "min":
            return candidate < current_best
        return candidate > current_best

    def worst_value(self) -> float:
        return float("inf") if self.optimize_direction == "min" else float("-inf")


# --- GPyTorch model definitions (module-level so they are pickleable) -------

class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class _VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor) -> None:
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        var_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True
        )
        super().__init__(var_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ---------------------------------------------------------------------------

class OptimizationMechanism(BaseMechanism):
    """Standard shot-value GPR.

    Fits an ExactGP (GaussianLikelihood) to (x, y) → expected_strokes data,
    then queries it per landing point to obtain E[strokes].
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: Optional[_ExactGPModel] = None
        self._likelihood: Optional[GaussianLikelihood] = None

    # --- BaseMechanism interface ---

    @property
    def optimize_direction(self) -> str:
        return "min"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        train_x = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(y, dtype=torch.float32)

        likelihood = GaussianLikelihood()
        model = _ExactGPModel(train_x, train_y, likelihood)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.gp_lr)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.config.gp_training_iter):
            optimizer.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                logger.debug("OptimizationMechanism train iter %d  loss=%.4f", i, loss.item())

        model.eval()
        likelihood.eval()
        self._model = model
        self._likelihood = likelihood
        logger.info("OptimizationMechanism training complete.")

    def evaluate(self, point: tuple[float, float]) -> float:
        if self._model is None:
            raise RuntimeError("Call train() before evaluate().")
        test_x = torch.tensor([[point[0], point[1]]], dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(test_x))
        return pred.mean.item()


# ---------------------------------------------------------------------------

class BirdieMechanism(BaseMechanism):
    """Binary birdie-probability model.

    Uses a VariationalGP + BernoulliLikelihood (Bernoulli–probit link) to
    produce P(birdie | landing_point). Requires binary labels (1 = birdie,
    0 = not birdie) as training targets.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: Optional[_VariationalGPModel] = None
        self._likelihood: Optional[BernoulliLikelihood] = None

    @property
    def optimize_direction(self) -> str:
        return "max"   # maximise birdie probability

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X : (N, 2) float array of (x, y) coordinates.
        y : (N,)   binary float array  (1.0 = birdie, 0.0 = no birdie).
        """
        train_x = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(y, dtype=torch.float32)

        # Select inducing points uniformly from training data
        n_ind = min(self.config.n_inducing, len(train_x))
        idx = torch.randperm(len(train_x))[:n_ind]
        inducing_points = train_x[idx].clone()

        likelihood = BernoulliLikelihood()
        model = _VariationalGPModel(inducing_points)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=self.config.gp_lr,
        )
        mll = VariationalELBO(likelihood, model, num_data=len(train_y))

        for i in range(self.config.gp_training_iter):
            optimizer.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                logger.debug("BirdieMechanism train iter %d  loss=%.4f", i, loss.item())

        model.eval()
        likelihood.eval()
        self._model = model
        self._likelihood = likelihood
        logger.info("BirdieMechanism training complete.")

    def evaluate(self, point: tuple[float, float]) -> float:
        """Return P(birdie) ∈ [0, 1]."""
        if self._model is None:
            raise RuntimeError("Call train() before evaluate().")
        test_x = torch.tensor([[point[0], point[1]]], dtype=torch.float32)
        with torch.no_grad():
            pred = self._likelihood(self._model(test_x))
        # BernoulliLikelihood.mean = P(y=1)
        return pred.mean.item()


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Runs the shot-by-shot Monte Carlo simulation for a set of strategy points.

    Key design guarantee
    --------------------
    For the winning (club, aim_offset) at each coordinate the *full array*
    of per-shot expected-stroke values is stored in SimulationResult.strokes_array.
    Callers MUST NOT reduce this to a scalar before fitting the next GPR stage —
    that would throw away the distributional information needed for confidence
    intervals and uncertainty propagation.
    """

    def __init__(
        self,
        player: Player,
        mechanism: BaseMechanism,
        config: Config,
    ) -> None:
        self.player = player
        self.mechanism = mechanism
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotation_translator(
        x_side: float,
        y_carry: float,
        angle_deg: float,
        starting_point: tuple[float, float],
        target: tuple[float, float],
    ) -> tuple[float, float]:
        """Rotate a (side, carry) shot vector into global (x, y) coordinates."""
        direction = np.array(target, dtype=np.float32) - np.array(starting_point, dtype=np.float32)
        unit_dir = direction / np.linalg.norm(direction)
        angle_rad = np.radians(angle_deg)
        rot = np.array(
            [[np.cos(angle_rad), -np.sin(angle_rad)],
             [np.sin(angle_rad),  np.cos(angle_rad)]],
            dtype=np.float32,
        )
        local_shot = np.array([x_side, y_carry], dtype=np.float32)
        rotated = rot @ local_shot
        global_vec = np.array(
            [unit_dir[0] * rotated[1] - unit_dir[1] * rotated[0],
             unit_dir[1] * rotated[1] + unit_dir[0] * rotated[0]],
            dtype=np.float32,
        )
        result = np.array(starting_point, dtype=np.float32) + global_vec
        return (float(result[0]), float(result[1]))

    def _simulate_club_aim(
        self,
        starting_point: tuple[float, float],
        target: tuple[float, float],
        club: str,
        aim_offset: float,
        club_dist: dict[str, dict],   # {"mean": ndarray, "cov": ndarray}
        evaluate_shot_fn,             # callable: (point, start, target) → float
    ) -> ShotOutcomes:
        """Simulate N shots for one (club, aim_offset) pair.

        Returns a ShotOutcomes whose strokes_array contains every valid shot
        value — no averaging performed here.
        """
        raw_mean, raw_cov = club_dist["mean"], club_dist["cov"]
        mean, cov = self.player.apply_to_distribution(raw_mean, raw_cov)

        # Aim-offset → angle
        dist_to_target = float(np.linalg.norm(
            np.array(target, dtype=np.float32) - np.array(starting_point, dtype=np.float32)
        ))
        angle_deg = float(np.degrees(np.arctan(aim_offset / dist_to_target))) if dist_to_target > 0 else 0.0

        raw_samples = np.random.multivariate_normal(
            mean.astype(np.float64), cov.astype(np.float64), size=self.config.n_shots
        ).astype(np.float32)

        strokes: list[float] = []
        landing_pts: list[tuple[float, float]] = []

        for shot in raw_samples:
            lp = self._rotation_translator(
                float(shot[0]), float(shot[1]), angle_deg, starting_point, target
            )
            val = evaluate_shot_fn(lp, starting_point, target)
            if not np.isnan(val):
                strokes.append(val)
                landing_pts.append(lp)

        return ShotOutcomes(
            club=club,
            aim_offset=aim_offset,
            strokes_array=np.array(strokes, dtype=np.float32),
            landing_points=np.array(landing_pts, dtype=np.float32).reshape(-1, 2),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_point(
        self,
        starting_point: tuple[float, float],
        target: tuple[float, float],
        club_distributions: dict[str, dict],
        evaluate_shot_fn,
        penalty: float = 0.0,
    ) -> Optional[SimulationResult]:
        """Find the optimal (club, aim) for one coordinate and return the
        full shot distribution for that winner.

        Parameters
        ----------
        starting_point : (x, y) coordinate of the current ball position.
        target         : (x, y) of the hole (or intermediate target).
        club_distributions : mapping club_name → {"mean": ndarray, "cov": ndarray}.
        evaluate_shot_fn   : callable(landing_pt, starting_pt, target) → float.
                             Uses self.mechanism.evaluate() internally or a
                             richer function that handles lie categories.
        penalty        : extra strokes to add (e.g. water_penalty from lie).

        Returns
        -------
        SimulationResult with the full strokes_array preserved, or None if
        no valid outcome was found.
        """
        dist_to_target = float(np.linalg.norm(
            np.array(target, dtype=np.float32) - np.array(starting_point, dtype=np.float32)
        ))

        # Select top-N clubs by proximity of mean carry to target distance
        club_diffs = [
            (club, abs(dist["mean"][1] * self.player.power - dist_to_target))
            for club, dist in club_distributions.items()
        ]
        top_clubs = [c for c, _ in sorted(club_diffs, key=lambda x: x[1])[: self.config.top_n_clubs]]

        aim_offsets = np.arange(
            self.config.aim_range[0],
            self.config.aim_range[1] + self.config.aim_step,
            self.config.aim_step,
        )

        best_value = self.mechanism.worst_value()
        best_outcomes: Optional[ShotOutcomes] = None

        for club in top_clubs:
            for aim in aim_offsets:
                outcomes = self._simulate_club_aim(
                    starting_point=starting_point,
                    target=target,
                    club=club,
                    aim_offset=float(aim),
                    club_dist=club_distributions[club],
                    evaluate_shot_fn=evaluate_shot_fn,
                )
                if len(outcomes.strokes_array) == 0:
                    continue

                # Add penalty before comparing so the mechanism sees the true cost
                adjusted_mean = outcomes.mean + penalty
                if self.mechanism.is_better(adjusted_mean, best_value):
                    best_value = adjusted_mean
                    best_outcomes = outcomes

        if best_outcomes is None:
            logger.warning("simulate_point: no valid outcomes for start=%s", starting_point)
            return None

        return SimulationResult.from_shot_outcomes(
            start=starting_point,
            outcomes=best_outcomes,
            penalty=penalty,
        )

    def simulate_all(
        self,
        strategy_points: list[tuple[float, float]],
        target: tuple[float, float],
        club_distributions: dict[str, dict],
        evaluate_shot_fn,
        get_penalty_fn=None,
    ) -> list[SimulationResult]:
        """Simulate every strategy point sequentially.

        Parameters
        ----------
        get_penalty_fn : optional callable(starting_point) → float.
                         Returns the stroke penalty for the lie at that point
                         (e.g. +1 for fairway, +2 for water drop).
                         Defaults to a constant +1 (normal shot overhead).
        """
        if get_penalty_fn is None:
            get_penalty_fn = lambda _: 1.0

        results: list[SimulationResult] = []
        n = len(strategy_points)
        for i, sp in enumerate(strategy_points):
            if i % max(1, n // 10) == 0:
                logger.info("simulate_all: %d / %d points done", i, n)
            penalty = get_penalty_fn(sp)
            result = self.simulate_point(
                starting_point=sp,
                target=target,
                club_distributions=club_distributions,
                evaluate_shot_fn=evaluate_shot_fn,
                penalty=penalty,
            )
            if result is not None:
                results.append(result)
        logger.info("simulate_all complete: %d / %d points returned results.", len(results), n)
        return results
