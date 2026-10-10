"""Fish-stock dynamics, export-price simulation, and revenue-risk assessment."""

from __future__ import annotations

import statistics

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class FibonacciStockBaseline:
    """Generate the intentionally simplistic Fibonacci stock baseline."""

    @staticmethod
    def generate(count: int = 15) -> list[int]:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Count must be a positive integer.")
        values = [1, 1]
        while len(values) < count:
            values.append(values[-1] + values[-2])
        return values[:count]


class FishStock:
    """Simulate a harvested fish population with discrete logistic growth.

    Harvest rate is the fraction of the standing stock harvested each open
    week. Stock and harvest are measured in tonnes.
    """

    def __init__(
        self,
        harvest_rate: float,
        intrinsic_growth_rate: float = 0.4,
        carrying_capacity: float = 10_000.0,
        initial_stock: float = 4_000.0,
    ) -> None:
        parameters = np.asarray(
            [
                harvest_rate,
                intrinsic_growth_rate,
                carrying_capacity,
                initial_stock,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(parameters)):
            raise ValueError("Fish-stock parameters must be finite.")
        if not 0 <= harvest_rate <= 1:
            raise ValueError("Harvest rate must be between 0 and 1.")
        if not 0 < intrinsic_growth_rate <= 1:
            raise ValueError("Growth rate must be greater than 0 and at most 1.")
        if carrying_capacity <= 0:
            raise ValueError("Carrying capacity must be positive.")
        if not 0 <= initial_stock <= carrying_capacity:
            raise ValueError("Initial stock must be between zero and carrying capacity.")
        self.harvest_rate = float(harvest_rate)
        self.intrinsic_growth_rate = float(intrinsic_growth_rate)
        self.carrying_capacity = float(carrying_capacity)
        self.initial_stock = float(initial_stock)

    @property
    def maximum_sustainable_yield(self) -> float:
        """Return theoretical logistic MSY in tonnes per week (rK/4)."""
        return self.intrinsic_growth_rate * self.carrying_capacity / 4

    def simulate(
        self,
        weeks: int = 52,
        closed_season_weeks: int = 0,
        weeks_per_year: int = 52,
    ) -> tuple[FloatArray, FloatArray]:
        """Return stock trajectory and harvest for each week.

        The first ``closed_season_weeks`` of each year are closed. The stock
        trajectory includes the initial stock and therefore has ``weeks + 1``
        entries; harvest has one value per simulated week.
        """
        if not isinstance(weeks, int) or isinstance(weeks, bool) or weeks < 1:
            raise ValueError("Weeks must be a positive integer.")
        if (
            not isinstance(weeks_per_year, int)
            or isinstance(weeks_per_year, bool)
            or weeks_per_year < 1
        ):
            raise ValueError("Weeks per year must be a positive integer.")
        if (
            not isinstance(closed_season_weeks, int)
            or isinstance(closed_season_weeks, bool)
            or not 0 <= closed_season_weeks <= weeks_per_year
        ):
            raise ValueError(
                "Closed-season weeks must be between zero and weeks per year."
            )

        stocks = np.empty(weeks + 1, dtype=float)
        harvests = np.empty(weeks, dtype=float)
        stocks[0] = self.initial_stock
        for week in range(weeks):
            in_closed_season = week % weeks_per_year < closed_season_weeks
            rate = 0.0 if in_closed_season else self.harvest_rate
            current_stock = stocks[week]
            harvests[week] = rate * current_stock
            next_stock = (
                current_stock
                + self.intrinsic_growth_rate
                * current_stock
                * (1 - current_stock / self.carrying_capacity)
                - harvests[week]
            )
            stocks[week + 1] = next_stock
        return stocks, harvests


class PriceModel:
    """Simulate bounded weekly export prices with a seeded random walk."""

    def __init__(
        self,
        starting_price: float = 12_000.0,
        lower_bound: float = 9_000.0,
        upper_bound: float = 16_000.0,
        weekly_volatility: float = 500.0,
        seed: int = 42,
    ) -> None:
        parameters = np.asarray(
            [starting_price, lower_bound, upper_bound, weekly_volatility],
            dtype=float,
        )
        if not np.all(np.isfinite(parameters)):
            raise ValueError("Price-model parameters must be finite.")
        if lower_bound <= 0 or upper_bound <= lower_bound:
            raise ValueError("Price bounds must be positive and lower than upper.")
        if not lower_bound <= starting_price <= upper_bound:
            raise ValueError("Starting price must lie within the price bounds.")
        if weekly_volatility < 0:
            raise ValueError("Weekly volatility cannot be negative.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("Seed must be an integer.")
        self.starting_price = float(starting_price)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.weekly_volatility = float(weekly_volatility)
        self.seed = seed

    def simulate_prices(
        self,
        weeks: int = 52,
        paths: int = 1,
        seed: int | None = None,
    ) -> FloatArray:
        """Return a ``(paths, weeks)`` price array, including the initial price."""
        if not isinstance(weeks, int) or isinstance(weeks, bool) or weeks < 1:
            raise ValueError("Weeks must be a positive integer.")
        if not isinstance(paths, int) or isinstance(paths, bool) or paths < 1:
            raise ValueError("Paths must be a positive integer.")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        prices = np.empty((paths, weeks), dtype=float)
        prices[:, 0] = self.starting_price
        if weeks > 1:
            increments = rng.normal(
                0.0, self.weekly_volatility, size=(paths, weeks - 1)
            )
            prices[:, 1:] = self.starting_price + np.cumsum(increments, axis=1)
        return np.clip(prices, self.lower_bound, self.upper_bound)


class RiskAssessor:
    """Summarize export revenue and classify risk using coefficient of variation."""

    LOW_CV_MAX = 0.10
    MODERATE_CV_MAX = 0.25

    @staticmethod
    def summarize(revenue: ArrayLike) -> dict[str, float]:
        values = np.asarray(revenue, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("Revenue must be a one-dimensional series of at least 2 values.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Revenue values must be finite and nonnegative.")
        mean_revenue = statistics.mean(values.tolist())
        if mean_revenue == 0:
            raise ValueError("Coefficient of variation is undefined for zero mean revenue.")
        stdev_revenue = statistics.stdev(values.tolist())
        return {
            "mean": float(mean_revenue),
            "median": float(statistics.median(values.tolist())),
            "variance": float(statistics.variance(values.tolist())),
            "stdev": float(stdev_revenue),
            "cv": float(stdev_revenue / mean_revenue),
        }

    @classmethod
    def classify_cv(cls, coefficient_of_variation: float) -> str:
        if (
            not np.isfinite(coefficient_of_variation)
            or coefficient_of_variation < 0
        ):
            raise ValueError("Coefficient of variation must be finite and nonnegative.")
        if coefficient_of_variation <= cls.LOW_CV_MAX:
            return "Low"
        if coefficient_of_variation <= cls.MODERATE_CV_MAX:
            return "Moderate"
        return "High"

    @staticmethod
    def revenue_var(revenue: ArrayLike, confidence: float = 0.95) -> float:
        """Return lower-tail revenue VaR, i.e. the (1-confidence) quantile."""
        values = np.asarray(revenue, dtype=float)
        if values.ndim != 1 or values.size < 1:
            raise ValueError("Revenue must be a non-empty one-dimensional series.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Revenue values must be finite and nonnegative.")
        if not np.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("Confidence must be greater than 0 and less than 1.")
        return float(np.quantile(values, 1 - confidence))
