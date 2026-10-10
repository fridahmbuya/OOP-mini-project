"""Linear dispatch models for solar-battery and hybrid micro-grids."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, solve
from scipy.optimize import nnls


FloatArray = NDArray[np.float64]


class MicroGrid:
    """Solve daily solar and battery dispatch from two demand equations."""

    def __init__(
        self,
        coefficient_matrix: ArrayLike = ((3.0, 2.0), (4.0, 1.0)),
        solar_cost_per_kwh: float = 150.0,
        battery_cost_per_kwh: float = 450.0,
    ) -> None:
        matrix = np.asarray(coefficient_matrix, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("The coefficient matrix must have shape (2, 2).")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("The coefficient matrix must contain finite values.")
        costs = np.asarray([solar_cost_per_kwh, battery_cost_per_kwh], dtype=float)
        if not np.all(np.isfinite(costs)) or np.any(costs < 0):
            raise ValueError("Energy costs must be finite and nonnegative.")
        self.coefficient_matrix = matrix
        self.solar_cost_per_kwh = float(solar_cost_per_kwh)
        self.battery_cost_per_kwh = float(battery_cost_per_kwh)

    @property
    def determinant(self) -> float:
        """Return the determinant, which is -5 for the default system."""
        return float(np.linalg.det(self.coefficient_matrix))

    @property
    def condition_number(self) -> float:
        """Return the 2-norm condition number of the coefficient matrix."""
        return float(np.linalg.cond(self.coefficient_matrix))

    def _check_well_posed(self) -> None:
        if not np.isfinite(self.condition_number) or np.isclose(
            self.determinant, 0.0
        ):
            raise LinAlgError(
                "The demand system is singular or ill-conditioned; "
                "a reliable unique dispatch cannot be computed."
            )

    @staticmethod
    def _validate_demands(demands: ArrayLike, expected_rows: int) -> FloatArray:
        values = np.asarray(demands, dtype=float)
        if values.size == 0 and values.ndim == 1:
            return np.empty((0, expected_rows), dtype=float)
        if values.ndim != 2 or values.shape[1] != expected_rows:
            raise ValueError(
                f"Demands must be a two-dimensional array with "
                f"{expected_rows} columns."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Demands must contain finite values.")
        if np.any(values < 0):
            raise ValueError("Demands must be nonnegative.")
        return values

    def solve_day(self, d1: float, d2: float) -> FloatArray:
        """Return ``[solar_kwh, battery_kwh]`` for one day's demands."""
        demands = np.asarray([d1, d2], dtype=float)
        if not np.all(np.isfinite(demands)):
            raise ValueError("Demands must be finite numbers.")
        if np.any(demands < 0):
            raise ValueError("Demands must be nonnegative.")
        self._check_well_posed()
        return np.asarray(solve(self.coefficient_matrix, demands), dtype=float)

    def solve_days_loop(self, demands: ArrayLike) -> FloatArray:
        """Solve an ``(n_days, 2)`` array one day at a time."""
        values = self._validate_demands(demands, 2)
        self._check_well_posed()
        if len(values) == 0:
            return np.empty((0, 2), dtype=float)
        return np.vstack([self.solve_day(d1, d2) for d1, d2 in values])

    def solve_days_vectorized(self, demands: ArrayLike) -> FloatArray:
        """Solve an ``(n_days, 2)`` array with one ``(2, n_days)`` solve."""
        values = self._validate_demands(demands, 2)
        self._check_well_posed()
        if len(values) == 0:
            return np.empty((0, 2), dtype=float)
        right_hand_side = values.T
        return np.asarray(
            solve(self.coefficient_matrix, right_hand_side).T, dtype=float
        )

    def solve_nonnegative_day(self, d1: float, d2: float) -> tuple[FloatArray, float]:
        """Return the NNLS dispatch and its demand residual norm.

        This provides the closest nonnegative dispatch in the least-squares
        sense when an unconstrained solution is physically infeasible.
        """
        demands = np.asarray([d1, d2], dtype=float)
        if not np.all(np.isfinite(demands)):
            raise ValueError("Demands must be finite numbers.")
        if np.any(demands < 0):
            raise ValueError("Demands must be nonnegative.")
        self._check_well_posed()
        dispatch, residual_norm = nnls(self.coefficient_matrix, demands)
        return np.asarray(dispatch, dtype=float), float(residual_norm)

    def calculate_cost(self, solar_kwh: float, battery_kwh: float) -> float:
        """Calculate the daily illustrative energy cost in UGX."""
        usage = np.asarray([solar_kwh, battery_kwh], dtype=float)
        if not np.all(np.isfinite(usage)) or np.any(usage < 0):
            raise ValueError("Energy usage must be finite and nonnegative.")
        return float(
            usage[0] * self.solar_cost_per_kwh
            + usage[1] * self.battery_cost_per_kwh
        )


class HybridMicroGrid(MicroGrid):
    """Three-source model with a total-energy constraint for diesel dispatch."""

    def __init__(
        self,
        coefficient_matrix: ArrayLike = (
            (3.0, 2.0, 0.0),
            (4.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        solar_cost_per_kwh: float = 150.0,
        battery_cost_per_kwh: float = 450.0,
        diesel_cost_per_kwh: float = 900.0,
    ) -> None:
        matrix = np.asarray(coefficient_matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("The hybrid coefficient matrix must have shape (3, 3).")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("The coefficient matrix must contain finite values.")
        costs = np.asarray(
            [solar_cost_per_kwh, battery_cost_per_kwh, diesel_cost_per_kwh],
            dtype=float,
        )
        if not np.all(np.isfinite(costs)) or np.any(costs < 0):
            raise ValueError("Energy costs must be finite and nonnegative.")
        self.coefficient_matrix = matrix
        self.solar_cost_per_kwh = float(solar_cost_per_kwh)
        self.battery_cost_per_kwh = float(battery_cost_per_kwh)
        self.diesel_cost_per_kwh = float(diesel_cost_per_kwh)

    def solve_day(self, d1: float, d2: float, d3: float) -> FloatArray:
        """Return ``[solar, battery, diesel]`` for the three demands.

        The default third equation is ``solar + battery + diesel = d3``.
        """
        demands = np.asarray([d1, d2, d3], dtype=float)
        if not np.all(np.isfinite(demands)):
            raise ValueError("Demands must be finite numbers.")
        if np.any(demands < 0):
            raise ValueError("Demands must be nonnegative.")
        self._check_well_posed()
        return np.asarray(solve(self.coefficient_matrix, demands), dtype=float)

    def calculate_cost(
        self, solar_kwh: float, battery_kwh: float, diesel_kwh: float
    ) -> float:
        """Calculate daily cost for all three sources in UGX."""
        usage = np.asarray([solar_kwh, battery_kwh, diesel_kwh], dtype=float)
        if not np.all(np.isfinite(usage)) or np.any(usage < 0):
            raise ValueError("Energy usage must be finite and nonnegative.")
        return float(
            usage[0] * self.solar_cost_per_kwh
            + usage[1] * self.battery_cost_per_kwh
            + usage[2] * self.diesel_cost_per_kwh
        )
