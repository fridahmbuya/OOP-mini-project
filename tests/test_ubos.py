import numpy as np
import pytest

from src.ubos import (
    CAGRForecaster,
    DistrictPopulation,
    FibonacciRatioForecaster,
    Forecaster,
    LinearTrendForecaster,
)


FORECASTER_TYPES = (
    LinearTrendForecaster,
    CAGRForecaster,
    FibonacciRatioForecaster,
)


@pytest.mark.parametrize(
    ("years", "population", "message"),
    [
        ([[2019, 2020], [2021, 2022]], [10, 11], "one-dimensional"),
        ([2019, 2020], [10], "equal lengths"),
        ([2019], [10], "At least two"),
        ([2019, np.nan], [10, 11], "finite values"),
        ([2019, 2020], [10, np.inf], "finite values"),
        ([2019, 2020], [10, -1], "negative"),
        ([2020, 2019], [10, 11], "strictly increasing"),
        ([2019, 2019], [10, 11], "strictly increasing"),
    ],
)
def test_district_population_rejects_invalid_observations(
    years: list, population: list, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DistrictPopulation("Example", years, population)


def test_district_population_stores_data_and_exposes_observations() -> None:
    years = np.array([2019, 2020, 2021])
    population = np.array([100, 110, 120])

    district = DistrictPopulation("Example", years, population)

    assert district.name == "Example"
    assert len(district) == 3
    np.testing.assert_array_equal(district.get_years(), years)
    np.testing.assert_array_equal(district.get_population(), population)
    assert "Example" in repr(district)


def test_forecaster_is_abstract() -> None:
    with pytest.raises(TypeError):
        Forecaster()


@pytest.mark.parametrize("forecaster_type", FORECASTER_TYPES)
def test_forecasters_require_fitting_before_prediction_or_fitted_values(
    forecaster_type: type[Forecaster],
) -> None:
    forecaster = forecaster_type()

    with pytest.raises(RuntimeError, match="fit"):
        forecaster.predict(1)
    with pytest.raises(RuntimeError, match="fit"):
        forecaster.fitted_values()


@pytest.mark.parametrize(
    ("years", "population", "message"),
    [
        ([2019, 2020], [10], "equal-length arrays"),
        ([2019], [10], "equal-length arrays"),
        ([2019, 2020], [10, -1], "nonnegative"),
        ([2020, 2019], [10, 11], "increasing"),
        ([2019, np.nan], [10, 11], "finite values"),
        ([2019, 2020], [10, np.inf], "finite values"),
        ([[2019, 2020]], [10, 11], "one-dimensional"),
    ],
)
@pytest.mark.parametrize("forecaster_type", FORECASTER_TYPES)
def test_forecasters_reject_invalid_training_data(
    forecaster_type: type[Forecaster],
    years: list,
    population: list,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        forecaster_type().fit(years, population)


@pytest.mark.parametrize("forecaster_type", FORECASTER_TYPES)
def test_forecasters_reject_invalid_horizons(
    forecaster_type: type[Forecaster],
) -> None:
    forecaster = forecaster_type().fit([2019, 2020], [100, 110])

    for horizon in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(horizon)


def test_linear_trend_forecaster_fits_and_predicts_linear_data() -> None:
    years = np.array([2019, 2020, 2021])
    population = np.array([100, 110, 120])
    forecaster = LinearTrendForecaster()

    assert forecaster.fit(years, population) is forecaster
    np.testing.assert_allclose(forecaster.fitted_values(), population)
    np.testing.assert_allclose(forecaster.predict(2), [130, 140])


def test_cagr_forecaster_predicts_compound_growth() -> None:
    years = np.array([2019, 2020, 2021])
    population = np.array([100, 110, 121])
    forecaster = CAGRForecaster()

    assert forecaster.fit(years, population) is forecaster
    np.testing.assert_allclose(forecaster.fitted_values(), population)
    np.testing.assert_allclose(forecaster.predict(2), [133.1, 146.41])


def test_cagr_forecaster_rejects_zero_population() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        CAGRForecaster().fit([2019, 2020], [0, 10])


def test_fibonacci_forecaster_uses_fibonacci_ratios_for_predictions() -> None:
    forecaster = FibonacciRatioForecaster().fit(
        [2019, 2020, 2021], [10, 20, 30]
    )

    np.testing.assert_allclose(forecaster.fitted_values(), [10, 10, 20])
    np.testing.assert_allclose(forecaster.predict(4), [30, 60, 90, 150])
