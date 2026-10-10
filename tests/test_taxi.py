import numpy as np
import pytest

from src.taxi import (
    ExponentialSmoothingForecaster,
    Forecaster,
    ForecasterEvaluator,
    LinearTrendForecaster,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    SupplyDemandEquilibrium,
    WeeklyDemandGenerator,
)


def test_route_calculates_daily_and_total_revenue() -> None:
    route = Route("Test", [10, 20, 30], fare_ugx=2_000)

    np.testing.assert_array_equal(route.daily_revenue(), [20_000, 40_000, 60_000])
    assert route.total_revenue() == pytest.approx(120_000)


def test_route_reports_sample_descriptive_statistics() -> None:
    stats = Route("Test", [1, 2, 3], 100).descriptive_statistics()

    assert stats == {"mean": 2.0, "variance": 1.0, "stdev": 1.0}


@pytest.mark.parametrize(
    ("passengers", "fare", "message"),
    [
        ([], 100, "non-empty"),
        ([[1, 2]], 100, "one-dimensional"),
        ([1, np.nan], 100, "finite and nonnegative"),
        ([1, -1], 100, "finite and nonnegative"),
        ([1, 2], -1, "Fare"),
        ([1, 2], np.inf, "finite"),
    ],
)
def test_route_rejects_invalid_data(
    passengers: list, fare: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Route("Test", passengers, fare)


def test_route_fleet_capacity_includes_buffer_and_rounds_up() -> None:
    route = Route("Test", [50, 55], 2_000)

    assert route.vehicles_required(112, buffer_fraction=0) == 1
    assert route.vehicles_required(113, buffer_fraction=0) == 2
    assert route.vehicles_required(100, buffer_fraction=0.15) == 2


@pytest.mark.parametrize("forecast", [-1, np.nan, np.inf])
def test_route_fleet_planning_rejects_invalid_forecast(forecast: float) -> None:
    with pytest.raises(ValueError):
        Route("Test", [1], 1).vehicles_required(forecast)


def test_three_forecasters_produce_expected_one_step_predictions() -> None:
    history = [1, 2, 3, 4]

    assert MovingAverageForecaster().predict_next(history) == pytest.approx(3)
    assert ExponentialSmoothingForecaster(1.0).predict_next(history) == pytest.approx(4)
    assert LinearTrendForecaster().predict_next(history) == pytest.approx(5)


def test_exponential_smoothing_uses_alpha_recursively() -> None:
    assert ExponentialSmoothingForecaster(0.5).predict_next([10, 20, 30]) == pytest.approx(
        22.5
    )


def test_forecaster_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        Forecaster()


@pytest.mark.parametrize(
    ("forecaster", "history"),
    [
        (MovingAverageForecaster(), [1, 2]),
        (LinearTrendForecaster(), [1]),
    ],
)
def test_forecasters_reject_insufficient_history(
    forecaster: Forecaster, history: list[int]
) -> None:
    with pytest.raises(ValueError):
        forecaster.predict_next(history)


@pytest.mark.parametrize("alpha", [0, -0.1, 1.1, np.nan])
def test_smoothing_forecaster_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha)


def test_predict_series_recursively_forecasts() -> None:
    predictions = LinearTrendForecaster().predict_series([10, 12, 14], 2)

    np.testing.assert_allclose(predictions, [16, 18])


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_predict_series_rejects_invalid_horizon(horizon: int) -> None:
    with pytest.raises(ValueError):
        LinearTrendForecaster().predict_series([1, 2], horizon)


def test_walk_forward_evaluation_scores_origins_day_four_through_last() -> None:
    observations = [10, 10, 10, 20, 20, 20, 20]

    score = ForecasterEvaluator.rolling_origin_mae(
        MovingAverageForecaster(), observations, start_index=3
    )

    assert score == pytest.approx(5.0)


def test_alpha_grid_search_returns_lowest_error_candidate() -> None:
    history = [10, 10, 10, 20, 20, 20, 20]

    best_alpha, scores = ForecasterEvaluator.tune_alpha(history, [0.2, 0.8, 1.0])

    assert best_alpha in scores
    assert scores[best_alpha] == pytest.approx(min(scores.values()))


@pytest.mark.parametrize(
    "candidates",
    [[], [[0.5]], [0, 0.5], [0.5, 1.1], [np.nan]],
)
def test_alpha_grid_search_rejects_invalid_candidates(candidates: list) -> None:
    with pytest.raises(ValueError):
        ForecasterEvaluator.tune_alpha([1, 2, 3, 4], candidates)


def test_equilibrium_solves_supply_demand_system() -> None:
    fare, quantity = SupplyDemandEquilibrium().solve()

    assert fare == pytest.approx(2_200)
    assert quantity == pytest.approx(76)
    assert quantity == pytest.approx(120 - 0.02 * fare)
    assert quantity == pytest.approx(10 + 0.03 * fare)


def test_seasonal_naive_uses_value_one_week_earlier() -> None:
    history = [10, 11, 12, 13, 14, 30, 5, 20, 21, 22, 23, 24, 50, 4]

    assert SeasonalNaiveForecaster(7).predict_next(history) == pytest.approx(20)


def test_seasonal_naive_beats_moving_average_on_weekly_pattern() -> None:
    demand = WeeklyDemandGenerator.generate(
        base_demand=50, days=60, seed=22, noise_std=1
    )
    moving_mae = ForecasterEvaluator.rolling_origin_mae(
        MovingAverageForecaster(), demand, start_index=7
    )
    seasonal_mae = ForecasterEvaluator.rolling_origin_mae(
        SeasonalNaiveForecaster(7), demand, start_index=7
    )

    assert seasonal_mae < moving_mae


def test_weekly_demand_generator_is_reproducible_and_has_weekly_pattern() -> None:
    first = WeeklyDemandGenerator.generate(50, days=60, seed=10)
    second = WeeklyDemandGenerator.generate(50, days=60, seed=10)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (60,)
    assert np.mean(first[4::7]) > np.mean(first[6::7])
