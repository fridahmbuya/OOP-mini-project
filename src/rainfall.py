"""Rainfall summaries, crop screening rules, and regional comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine


FloatArray = NDArray[np.float64]
MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


class Region:
    """Monthly rainfall observations for one region in millimetres."""

    def __init__(self, name: str, rainfall: ArrayLike) -> None:
        values = np.asarray(rainfall, dtype=float)
        if values.shape != (12,):
            raise ValueError("Rainfall must contain exactly 12 monthly values.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Rainfall values must be finite.")
        if np.any(values < 0):
            raise ValueError("Rainfall values cannot be negative.")
        if not name.strip():
            raise ValueError("Region name cannot be empty.")
        self.name = name
        self.rainfall = values

    def annual_total(self) -> float:
        """Return annual rainfall in mm."""
        return float(np.sum(self.rainfall))

    def mean(self) -> float:
        """Return mean monthly rainfall in mm/month."""
        return float(np.mean(self.rainfall))

    def wettest_month(self) -> str:
        """Return the month with the greatest rainfall."""
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        """Return the month with the least rainfall."""
        return MONTHS[int(np.argmin(self.rainfall))]

    def coefficient_of_variation(self) -> float:
        """Return population standard deviation divided by mean."""
        mean_rainfall = self.mean()
        if mean_rainfall == 0:
            raise ValueError("Coefficient of variation is undefined for zero mean rainfall.")
        return float(np.std(self.rainfall) / mean_rainfall)


@dataclass(frozen=True)
class CropRule:
    """Illustrative monthly rainfall screen for a crop.

    Values between the minimum and maximum are labelled suitable. Rain below
    or above the screening range is labelled drought or waterlogging risk.
    """

    name: str
    minimum_mm: float
    maximum_mm: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Crop name cannot be empty.")
        if (
            not np.isfinite(self.minimum_mm)
            or not np.isfinite(self.maximum_mm)
            or self.minimum_mm < 0
            or self.maximum_mm <= self.minimum_mm
        ):
            raise ValueError("Crop rainfall bounds must be finite and increasing.")

    def classify(self, monthly_rainfall_mm: float) -> str:
        if not np.isfinite(monthly_rainfall_mm) or monthly_rainfall_mm < 0:
            raise ValueError("Monthly rainfall must be finite and nonnegative.")
        if monthly_rainfall_mm < self.minimum_mm:
            return "Drought risk"
        if monthly_rainfall_mm > self.maximum_mm:
            return "Waterlogging risk"
        return f"Good for {self.name}"


class RainfallAnalyzer:
    """Analyze rainfall similarity and rainy-season peak patterns."""

    DEFAULT_CROP_RULES: ClassVar[tuple[CropRule, ...]] = (
        CropRule("maize", 100.0, 250.0),
        CropRule("beans", 50.0, 150.0),
        CropRule("coffee", 100.0, 200.0),
    )

    @staticmethod
    def _validated_pair(first: Region, second: Region) -> tuple[FloatArray, FloatArray]:
        a = np.asarray(first.rainfall, dtype=float)
        b = np.asarray(second.rainfall, dtype=float)
        if a.shape != b.shape or a.ndim != 1:
            raise ValueError("Regions must have matching one-dimensional rainfall series.")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("Rainfall values must be finite.")
        return a, b

    @classmethod
    def cosine_similarity(cls, first: Region, second: Region) -> float:
        """Return vector cosine similarity, verified via SciPy cosine distance."""
        a, b = cls._validated_pair(first, second)
        norm_product = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm_product == 0:
            raise ValueError("Cosine similarity is undefined for a zero rainfall vector.")
        similarity = float(np.dot(a, b) / norm_product)
        scipy_similarity = 1.0 - float(cosine(a, b))
        if not np.isclose(similarity, scipy_similarity, rtol=1e-12, atol=1e-12):
            raise ArithmeticError("NumPy and SciPy cosine calculations did not agree.")
        return similarity

    @classmethod
    def pearson_correlation(cls, first: Region, second: Region) -> float:
        """Return Pearson's correlation between the two monthly rainfall series."""
        a, b = cls._validated_pair(first, second)
        if np.std(a) == 0 or np.std(b) == 0:
            raise ValueError("Pearson correlation is undefined for constant rainfall.")
        return float(np.corrcoef(a, b)[0, 1])

    @classmethod
    def euclidean_distance(cls, first: Region, second: Region) -> float:
        """Return Euclidean distance in monthly rainfall (mm)."""
        a, b = cls._validated_pair(first, second)
        return float(np.linalg.norm(a - b))

    @classmethod
    def similarity_matrices(
        cls, regions: list[Region]
    ) -> dict[str, tuple[list[str], FloatArray]]:
        """Return pairwise cosine, Pearson, and Euclidean matrices."""
        if not regions:
            raise ValueError("At least one region is required.")
        names = [region.name for region in regions]
        count = len(regions)
        cosine_matrix = np.empty((count, count), dtype=float)
        pearson_matrix = np.empty((count, count), dtype=float)
        distance_matrix = np.empty((count, count), dtype=float)
        for row, first in enumerate(regions):
            for column, second in enumerate(regions):
                cosine_matrix[row, column] = cls.cosine_similarity(first, second)
                pearson_matrix[row, column] = cls.pearson_correlation(first, second)
                distance_matrix[row, column] = cls.euclidean_distance(first, second)
        return {
            "cosine similarity": (names, cosine_matrix),
            "Pearson correlation": (names, pearson_matrix),
            "Euclidean distance (mm)": (names, distance_matrix),
        }

    @staticmethod
    def rainy_season_peaks(
        region: Region,
        prominence_mm: float = 15.0,
        minimum_separation_months: int = 3,
    ) -> tuple[list[str], str]:
        """Detect cyclic monthly rainfall peaks and return peak months/mode.

        The annual sequence is tiled three times to ensure December/January
        boundary peaks are detectable; only peaks in the middle year are kept.
        """
        if not np.isfinite(prominence_mm) or prominence_mm < 0:
            raise ValueError("Peak prominence must be finite and nonnegative.")
        if (
            not isinstance(minimum_separation_months, int)
            or isinstance(minimum_separation_months, bool)
            or minimum_separation_months < 1
        ):
            raise ValueError("Minimum peak separation must be a positive integer.")
        cyclic_rainfall = np.tile(region.rainfall, 3)
        peak_indices, _ = find_peaks(
            cyclic_rainfall,
            prominence=prominence_mm,
            distance=minimum_separation_months,
        )
        central_peaks = sorted(
            int(index - 12) for index in peak_indices if 12 <= index < 24
        )
        peak_months = [MONTHS[index] for index in central_peaks]
        season_type = "Bimodal" if len(central_peaks) >= 2 else "Unimodal"
        return peak_months, season_type
