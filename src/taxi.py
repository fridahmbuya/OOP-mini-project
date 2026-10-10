"""Taxi route revenue, demand forecasting, and fleet planning."""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from math import ceil

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve


FloatArray = NDArray[np.float64]


class Route:
    """Passenger counts and fare for a taxi route."""

    def __init__(
        self,
        name: str,
        passenger_counts: ArrayLike,
        fare_ugx: float,
        vehicle_capacity: int = 14,
        trips_per_vehicle_per_day: int = 8,
    ) -> None:
        counts = np.asarray(passenger_counts, dtype=float)
        values = np.asarray(
            [fare_ugx, vehicle_capacity, trips_per_vehicle_per_day], dtype=float
        )
        if counts.ndim != 1 or counts.size < 1:
            raise ValueError("Passenger counts must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("Passenger counts must be finite and nonnegative.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Fare and fleet parameters must be finite.")
        if fare_ugx < 0:
            raise ValueError("Fare cannot be negative.")
        if (
            not isinstance(vehicle_capacity, int)
            or isinstance(vehicle_capacity, bool)
            or vehicle_capacity < 1
        ):
            raise ValueError("Vehicle capacity must be a positive integer.")
        if (
            not isinstance(trips_per_vehicle_per_day, int)
            or isinstance(trips_per_vehicle_per_day, bool)
            or trips_per_vehicle_per_day < 1
        ):
            raise ValueError("Daily trips per vehicle must be a positive integer.")
        if not name.strip():
            raise ValueError("Route name cannot be empty.")
        self.name = name
        self.passenger_counts = counts
        self.fare_ugx = float(fare_ugx)
        self.vehicle_capacity = vehicle_capacity
        self.trips_per_vehicle_per_day = trips_per_vehicle_per_day

    def daily_revenue(self) -> FloatArray:
        """Return observed daily gross revenue in UGX."""
        return self.passenger_counts * self.fare_ugx

    def total_revenue(self) -> float:
        """Return total observed gross revenue in UGX."""
        return float(np.sum(self.daily_revenue()))

    def descriptive_statistics(self) -> dict[str, float]:
        """Return mean, sample variance, and sample stdev of passenger counts."""
        values = self.passenger_counts.tolist()
        if len(values) < 2:
            raise ValueError("At least two passenger observations are required.")
        return {
            "mean": float(statistics.mean(values)),
            "variance": float(statistics.variance(values)),
            "stdev": float(statistics.stdev(values)),
        }

    def vehicles_required(
        self, forecast_passengers: float, buffer_fraction: float = 0.15
    ) -> int:
        """Size fleet for forecast passengers with an upward-rounded buffer."""
        if not np.isfinite(forecast_passengers) or forecast_passengers < 0:
            raise ValueError("Forecast passengers must be finite and nonnegative.")
        if not np.isfinite(buffer_fraction) or buffer_fraction < 0:
            raise ValueError("Buffer fraction must be finite and nonnegative.")
        daily_capacity = (
            self.vehicle_capacity * self.trips_per_vehicle_per_day
        )
        buffered_demand = forecast_passengers * (1 + buffer_fraction)
        return ceil(buffered_demand / daily_capacity)


class Forecaster(ABC):
    """Base class for passenger-demand forecasting models."""

    @abstractmethod
    def predict_next(self, history: ArrayLike) -> float:
        """Return a one-step-ahead forecast based on observed history."""

    def predict_series(self, history: ArrayLike, horizon: int) -> FloatArray:
        """Recursively forecast a nonempty series for a positive horizon."""
        values = self._validate_history(history, minimum_size=1)
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("Forecast horizon must be a positive integer.")
        forecasts = np.empty(horizon, dtype=float)
        for index in range(horizon):
            forecasts[index] = self.predict_next(values)
            values = np.append(values, forecasts[index])
        return forecasts

    @staticmethod
    def _validate_history(history: ArrayLike, minimum_size: int = 1) -> FloatArray:
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or values.size < minimum_size:
            raise ValueError(
                f"History must be one-dimensional with at least {minimum_size} values."
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("History values must be finite and nonnegative.")
        return values


class MovingAverageForecaster(Forecaster):
    """Forecast with the mean of the most recent three observations."""

    def __init__(self, window: int = 3) -> None:
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError("Moving-average window must be a positive integer.")
        self.window = window

    def predict_next(self, history: ArrayLike) -> float:
        values = self._validate_history(history, minimum_size=self.window)
        return float(np.mean(values[-self.window :]))


class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing with a configurable alpha in (0, 1]."""

    def __init__(self, alpha: float) -> None:
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Alpha must be greater than 0 and at most 1.")
        self.alpha = float(alpha)

    def predict_next(self, history: ArrayLike) -> float:
        values = self._validate_history(history, minimum_size=1)
        level = float(values[0])
        for observation in values[1:]:
            level = self.alpha * float(observation) + (1 - self.alpha) * level
        return level


class LinearTrendForecaster(Forecaster):
    """Least-squares linear trend over sequential observation indices."""

    def predict_next(self, history: ArrayLike) -> float:
        values = self._validate_history(history, minimum_size=2)
        indices = np.arange(values.size, dtype=float)
        slope, intercept = np.polyfit(indices, values, 1)
        return float(slope * values.size + intercept)


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the value from one seasonal period earlier."""

    def __init__(self, season_length: int = 7) -> None:
        if not isinstance(season_length, int) or isinstance(season_length, bool) or season_length < 1:
            raise ValueError("Season length must be a positive integer.")
        self.season_length = season_length

    def predict_next(self, history: ArrayLike) -> float:
        values = self._validate_history(history, minimum_size=self.season_length)
        return float(values[-self.season_length])


class SupplyDemandEquilibrium:
    """Solve the linear equilibrium for a route's demand and supply curves."""

    def __init__(
        self,
        demand_intercept: float = 120.0,
        demand_slope: float = 0.02,
        supply_intercept: float = 10.0,
        supply_slope: float = 0.03,
    ) -> None:
        parameters = np.asarray(
            [demand_intercept, demand_slope, supply_intercept, supply_slope],
            dtype=float,
        )
        if not np.all(np.isfinite(parameters)):
            raise ValueError("Supply and demand parameters must be finite.")
        if demand_slope <= 0 or supply_slope <= 0:
            raise ValueError("Demand and supply slope magnitudes must be positive.")
        self.coefficient_matrix = np.array(
            [[demand_slope, 1.0], [supply_slope, -1.0]], dtype=float
        )
        self.right_hand_side = np.array(
            [demand_intercept, -supply_intercept], dtype=float
        )

    def solve(self) -> tuple[float, float]:
        """Return equilibrium fare (UGX) and quantity (passengers/trip-hour)."""
        fare, quantity = solve(self.coefficient_matrix, self.right_hand_side)
        return float(fare), float(quantity)


class ForecasterEvaluator:
    """Walk-forward MAE evaluation and exponential-smoothing grid search."""

    @staticmethod
    def rolling_origin_mae(
        forecaster: Forecaster,
        observations: ArrayLike,
        start_index: int = 3,
    ) -> float:
        """Backtest one-step forecasts from ``start_index`` through final day."""
        values = Forecaster._validate_history(observations, minimum_size=2)
        if (
            not isinstance(start_index, int)
            or isinstance(start_index, bool)
            or not 1 <= start_index < values.size
        ):
            raise ValueError("Start index must select a valid day after the first.")
        predictions = [
            forecaster.predict_next(values[:index])
            for index in range(start_index, values.size)
        ]
        actual = values[start_index:]
        return float(np.mean(np.abs(actual - predictions)))

    @classmethod
    def evaluate(
        cls, observations: ArrayLike, forecasters: dict[str, Forecaster]
    ) -> dict[str, float]:
        """Return walk-forward MAE for a forecaster mapping."""
        return {
            name: cls.rolling_origin_mae(forecaster, observations)
            for name, forecaster in forecasters.items()
        }

    @classmethod
    def tune_alpha(
        cls,
        observations: ArrayLike,
        alpha_values: ArrayLike | None = None,
    ) -> tuple[float, dict[float, float]]:
        """Choose alpha with the lowest walk-forward MAE."""
        values = Forecaster._validate_history(observations, minimum_size=4)
        candidates = (
            np.linspace(0.1, 1.0, 10)
            if alpha_values is None
            else np.asarray(alpha_values, dtype=float)
        )
        if candidates.ndim != 1 or candidates.size == 0:
            raise ValueError("Alpha candidates must be a nonempty one-dimensional array.")
        if (
            not np.all(np.isfinite(candidates))
            or np.any(candidates <= 0)
            or np.any(candidates > 1)
        ):
            raise ValueError("Every alpha candidate must be in (0, 1].")
        scores = {
            float(alpha): cls.rolling_origin_mae(
                ExponentialSmoothingForecaster(float(alpha)), values
            )
            for alpha in candidates
        }
        best_alpha = min(scores, key=scores.get)
        return best_alpha, scores


class WeeklyDemandGenerator:
    """Create reproducible demand series with weekday/Friday/Sunday effects."""

    @staticmethod
    def generate(
        base_demand: float,
        days: int = 60,
        seed: int = 42,
        noise_std: float = 3.0,
        friday_multiplier: float = 1.25,
        sunday_multiplier: float = 0.7,
    ) -> FloatArray:
        parameters = np.asarray(
            [base_demand, noise_std, friday_multiplier, sunday_multiplier],
            dtype=float,
        )
        if not np.all(np.isfinite(parameters)):
            raise ValueError("Demand-generation parameters must be finite.")
        if base_demand < 0 or noise_std < 0:
            raise ValueError("Base demand and noise must be nonnegative.")
        if friday_multiplier < 0 or sunday_multiplier < 0:
            raise ValueError("Weekly multipliers must be nonnegative.")
        if not isinstance(days, int) or isinstance(days, bool) or days < 1:
            raise ValueError("Days must be a positive integer.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("Seed must be an integer.")
        rng = np.random.default_rng(seed)
        weekday_multipliers = np.ones(days, dtype=float)
        weekday_multipliers[np.arange(days) % 7 == 4] = friday_multiplier
        weekday_multipliers[np.arange(days) % 7 == 6] = sunday_multiplier
        demand = (
            base_demand * weekday_multipliers
            + rng.normal(0.0, noise_std, size=days)
        )
        return np.maximum(demand, 0.0)
