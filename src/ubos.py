"""Population data and forecasting model classes.

Population observations are represented in thousands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DistrictPopulation:
    """Population observations for one district, with population in thousands."""

    def __init__(self, name: str, years: np.ndarray, population: np.ndarray) -> None:
        self.name = name
        self.years = np.asarray(years, dtype=float)
        self.population = np.asarray(population, dtype=float)
        if self.years.ndim != 1 or self.population.ndim != 1:
            raise ValueError("Years and populations must be one-dimensional.")
        if len(self.years) != len(self.population):
            raise ValueError("Years and populations must have equal lengths.")
        if len(self.years) < 2:
            raise ValueError("At least two observations are required.")
        if not np.all(np.isfinite(self.years)) or not np.all(
            np.isfinite(self.population)
        ):
            raise ValueError("Years and populations must contain finite values.")
        if np.any(self.population < 0):
            raise ValueError("Population values cannot be negative.")
        if np.any(np.diff(self.years) <= 0):
            raise ValueError("Years must be strictly increasing.")

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.name!r}, "
            f"years={self.years.astype(int)!r}, "
            f"population={self.population!r})"
        )

    def __len__(self) -> int:
        return len(self.population)

    def get_years(self) -> np.ndarray:
        return self.years

    def get_population(self) -> np.ndarray:
        return self.population


class Forecaster(ABC):
    """Common interface for population forecasting models."""

    def __init__(self) -> None:
        self.years: np.ndarray | None = None
        self.population: np.ndarray | None = None

    @abstractmethod
    def fit(self, years: np.ndarray, population: np.ndarray) -> Forecaster:
        """Fit the model and return itself."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Predict the next ``horizon`` annual population values."""

    @abstractmethod
    def fitted_values(self) -> np.ndarray:
        """Return fitted values for the observations passed to fit()."""

    def _store_data(self, years: np.ndarray, population: np.ndarray) -> None:
        years_array = np.asarray(years, dtype=float)
        population_array = np.asarray(population, dtype=float)
        if years_array.ndim != 1 or population_array.ndim != 1:
            raise ValueError("Years and populations must be one-dimensional.")
        if len(years_array) != len(population_array) or len(years_array) < 2:
            raise ValueError("Fit requires equal-length arrays with at least two values.")
        if not np.all(np.isfinite(years_array)) or not np.all(
            np.isfinite(population_array)
        ):
            raise ValueError("Years and populations must contain finite values.")
        if np.any(population_array < 0) or np.any(np.diff(years_array) <= 0):
            raise ValueError("Populations must be nonnegative and years increasing.")
        self.years = years_array
        self.population = population_array

    def _check_horizon(self, horizon: int) -> None:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("Forecast horizon must be a positive integer.")
        if self.population is None:
            raise RuntimeError("Call fit() before predict().")


class LinearTrendForecaster(Forecaster):
    """Least-squares linear trend in calendar year."""

    def fit(self, years: np.ndarray, population: np.ndarray) -> LinearTrendForecaster:
        self._store_data(years, population)
        assert self.years is not None
        assert self.population is not None
        self._slope, self._intercept = np.polyfit(self.years, self.population, 1)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        assert self.years is not None
        future_years = self.years[-1] + np.arange(1, horizon + 1)
        return self._slope * future_years + self._intercept

    def fitted_values(self) -> np.ndarray:
        if self.years is None:
            raise RuntimeError("Call fit() before requesting fitted values.")
        return self._slope * self.years + self._intercept


class CAGRForecaster(Forecaster):
    """Constant compound annual growth from the first to last training value."""

    def fit(self, years: np.ndarray, population: np.ndarray) -> CAGRForecaster:
        self._store_data(years, population)
        assert self.years is not None
        assert self.population is not None
        if np.any(self.population <= 0):
            raise ValueError("CAGR forecasting requires strictly positive populations.")
        elapsed_years = self.years[-1] - self.years[0]
        self._annual_factor = (self.population[-1] / self.population[0]) ** (
            1 / elapsed_years
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        assert self.population is not None
        steps = np.arange(1, horizon + 1)
        return self.population[-1] * self._annual_factor**steps

    def fitted_values(self) -> np.ndarray:
        if self.years is None or self.population is None:
            raise RuntimeError("Call fit() before requesting fitted values.")
        steps = self.years - self.years[0]
        return self.population[0] * self._annual_factor**steps


class FibonacciRatioForecaster(Forecaster):
    """Scale each step by the next F(n+1)/F(n) Fibonacci ratio.

    The first multipliers are 1, 2, 1.5, 1.667, ... . This literal
    interpretation is intentionally retained as a heuristic for comparison,
    not as a demographic growth model.
    """

    def fit(
        self, years: np.ndarray, population: np.ndarray
    ) -> FibonacciRatioForecaster:
        self._store_data(years, population)
        self._fibonacci_values = [1, 1]
        return self

    def _multipliers(self, count: int) -> np.ndarray:
        while len(self._fibonacci_values) <= count:
            self._fibonacci_values.append(
                self._fibonacci_values[-1] + self._fibonacci_values[-2]
            )
        return np.array(
            [
                self._fibonacci_values[index + 1] / self._fibonacci_values[index]
                for index in range(count)
            ],
            dtype=float,
        )

    def predict(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        assert self.population is not None
        predictions = []
        value = self.population[-1]
        for multiplier in self._multipliers(horizon):
            value *= multiplier
            predictions.append(value)
        return np.asarray(predictions)

    def fitted_values(self) -> np.ndarray:
        if self.population is None:
            raise RuntimeError("Call fit() before requesting fitted values.")
        fitted = np.empty_like(self.population)
        fitted[0] = self.population[0]
        value = self.population[0]
        for index, multiplier in enumerate(self._multipliers(len(self.population) - 1)):
            value *= multiplier
            fitted[index + 1] = value
        return fitted
