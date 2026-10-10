import numpy as np
import pytest
from scipy.spatial.distance import cosine

from src.rainfall import CropRule, RainfallAnalyzer, Region


REGIONAL_RAINFALL = {
    "Kampala": [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130],
    "Gulu": [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15],
    "Mbarara": [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90],
}


@pytest.fixture
def regions() -> list[Region]:
    return [
        Region(name, rainfall)
        for name, rainfall in REGIONAL_RAINFALL.items()
    ]


def test_region_reports_rainfall_summaries() -> None:
    region = Region("Test", REGIONAL_RAINFALL["Kampala"])

    assert region.annual_total() == pytest.approx(1_600)
    assert region.mean() == pytest.approx(1600 / 12)
    assert region.wettest_month() == "May"
    assert region.driest_month() == "September"
    assert region.coefficient_of_variation() == pytest.approx(
        np.std(REGIONAL_RAINFALL["Kampala"]) / (1600 / 12)
    )


@pytest.mark.parametrize(
    "rainfall",
    [
        [1] * 11,
        [1] * 13,
        [[1] * 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.nan],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1],
    ],
)
def test_region_rejects_invalid_rainfall(rainfall: list) -> None:
    with pytest.raises(ValueError):
        Region("Test", rainfall)


@pytest.mark.parametrize("name", ["", "   "])
def test_region_rejects_blank_names(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        Region(name, [1] * 12)


def test_region_cv_is_undefined_for_zero_rainfall() -> None:
    with pytest.raises(ValueError, match="zero mean"):
        Region("Dry", [0] * 12).coefficient_of_variation()


def test_crop_rule_classifies_monthly_rainfall() -> None:
    maize = CropRule("maize", 100, 250)

    assert maize.classify(99) == "Drought risk"
    assert maize.classify(100) == "Good for maize"
    assert maize.classify(250) == "Good for maize"
    assert maize.classify(251) == "Waterlogging risk"


@pytest.mark.parametrize(
    ("name", "minimum", "maximum"),
    [("", 10, 20), ("maize", -1, 20), ("maize", 20, 20), ("maize", 10, np.inf)],
)
def test_crop_rule_rejects_invalid_bounds(
    name: str, minimum: float, maximum: float
) -> None:
    with pytest.raises(ValueError):
        CropRule(name, minimum, maximum)


@pytest.mark.parametrize("rainfall", [-1, np.nan, np.inf])
def test_crop_rule_rejects_invalid_rainfall(rainfall: float) -> None:
    with pytest.raises(ValueError):
        CropRule("maize", 100, 250).classify(rainfall)


def test_cosine_similarity_matches_scipy_cosine_distance(regions: list[Region]) -> None:
    first, second = regions[:2]

    result = RainfallAnalyzer.cosine_similarity(first, second)

    assert result == pytest.approx(
        1 - cosine(first.rainfall, second.rainfall)
    )
    assert -1 <= result <= 1


def test_cosine_similarity_is_invariant_to_positive_scaling() -> None:
    rainfall = np.arange(1, 13, dtype=float)

    assert RainfallAnalyzer.cosine_similarity(
        Region("Base", rainfall), Region("Scaled", rainfall * 10)
    ) == pytest.approx(1)


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero rainfall"):
        RainfallAnalyzer.cosine_similarity(
            Region("Dry", [0] * 12), Region("Wet", [1] * 12)
        )


def test_pairwise_matrices_have_expected_shape_and_symmetry(
    regions: list[Region],
) -> None:
    matrices = RainfallAnalyzer.similarity_matrices(regions)

    assert set(matrices) == {
        "cosine similarity",
        "Pearson correlation",
        "Euclidean distance (mm)",
    }
    for names, matrix in matrices.values():
        assert names == [region.name for region in regions]
        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(matrices["Euclidean distance (mm)"][1].diagonal(), 0)


def test_similarity_matrices_require_at_least_one_region() -> None:
    with pytest.raises(ValueError, match="At least one"):
        RainfallAnalyzer.similarity_matrices([])


def test_peak_detector_finds_two_cyclic_peaks() -> None:
    bimodal = Region(
        "Two seasons",
        [100, 120, 200, 100, 40, 30, 40, 110, 200, 110, 40, 80],
    )

    peak_months, season_type = RainfallAnalyzer.rainy_season_peaks(
        bimodal, prominence_mm=40, minimum_separation_months=3
    )

    assert peak_months == ["March", "September"]
    assert season_type == "Bimodal"


def test_peak_detector_handles_year_boundary_peak() -> None:
    region = Region(
        "Boundary peak",
        [200, 100, 50, 30, 20, 15, 10, 15, 20, 30, 60, 150],
    )

    peak_months, season_type = RainfallAnalyzer.rainy_season_peaks(region)

    assert peak_months == ["January"]
    assert season_type == "Unimodal"


@pytest.mark.parametrize(
    ("prominence", "separation"),
    [(-1, 3), (np.nan, 3), (15, 0), (15, 1.5)],
)
def test_peak_detector_rejects_invalid_parameters(
    prominence: float, separation: int
) -> None:
    with pytest.raises(ValueError):
        RainfallAnalyzer.rainy_season_peaks(
            Region("Test", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            prominence,
            separation,
        )
