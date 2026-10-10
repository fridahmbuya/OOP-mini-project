import numpy as np
import pytest

from src.fish_stock import FibonacciStockBaseline, FishStock, PriceModel, RiskAssessor


def test_fibonacci_baseline_generates_requested_count() -> None:
    assert FibonacciStockBaseline.generate(15) == [
        1,
        1,
        2,
        3,
        5,
        8,
        13,
        21,
        34,
        55,
        89,
        144,
        233,
        377,
        610,
    ]


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_fibonacci_baseline_rejects_invalid_count(count: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        FibonacciStockBaseline.generate(count)


def test_fish_stock_simulates_logistic_growth_with_harvest() -> None:
    model = FishStock(harvest_rate=0.1)

    stocks, harvests = model.simulate(weeks=3)

    assert stocks.shape == (4,)
    assert harvests.shape == (3,)
    assert stocks[0] == pytest.approx(4_000)
    assert harvests[0] == pytest.approx(400)
    assert stocks[1] == pytest.approx(4_560)


def test_fish_stock_reports_theoretical_maximum_sustainable_yield() -> None:
    assert FishStock(harvest_rate=0.2).maximum_sustainable_yield == pytest.approx(
        1_000
    )


def test_closed_season_pauses_harvest_and_allows_stock_growth() -> None:
    model = FishStock(harvest_rate=0.1)

    stocks, harvests = model.simulate(weeks=10, closed_season_weeks=2)
    no_harvest_stocks, _ = FishStock(harvest_rate=0.0).simulate(weeks=3)

    np.testing.assert_array_equal(harvests[:2], [0, 0])
    assert stocks[1] > stocks[0]
    assert harvests[2] > 0
    assert stocks[3] < no_harvest_stocks[3]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"harvest_rate": -0.1}, "Harvest rate"),
        ({"harvest_rate": 1.1}, "Harvest rate"),
        ({"harvest_rate": np.nan}, "finite"),
        ({"harvest_rate": 0.1, "intrinsic_growth_rate": 0}, "Growth rate"),
        ({"harvest_rate": 0.1, "carrying_capacity": 0}, "Carrying capacity"),
        ({"harvest_rate": 0.1, "initial_stock": 10_001}, "Initial stock"),
    ],
)
def test_fish_stock_rejects_invalid_parameters(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FishStock(**kwargs)


@pytest.mark.parametrize(
    ("weeks", "closed_season_weeks", "weeks_per_year"),
    [(0, 0, 52), (2, -1, 52), (2, 53, 52), (2, 0, 0), (1.5, 0, 52)],
)
def test_fish_stock_rejects_invalid_simulation_periods(
    weeks: int, closed_season_weeks: int, weeks_per_year: int
) -> None:
    with pytest.raises(ValueError):
        FishStock(0.1).simulate(weeks, closed_season_weeks, weeks_per_year)


def test_price_model_is_seeded_bounded_and_starts_at_configured_price() -> None:
    model = PriceModel(seed=7)

    first = model.simulate_prices(weeks=52, paths=10)
    second = model.simulate_prices(weeks=52, paths=10)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], np.full(10, 12_000.0))
    assert np.all(first >= model.lower_bound)
    assert np.all(first <= model.upper_bound)
    assert first.shape == (10, 52)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lower_bound": 0}, "Price bounds"),
        ({"lower_bound": 13_000, "upper_bound": 12_000}, "Price bounds"),
        ({"starting_price": 8_000}, "Starting price"),
        ({"weekly_volatility": -1}, "volatility"),
        ({"seed": 1.5}, "Seed"),
    ],
)
def test_price_model_rejects_invalid_parameters(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PriceModel(**kwargs)


@pytest.mark.parametrize(("weeks", "paths"), [(0, 1), (1, 0), (1.5, 1), (1, True)])
def test_price_simulation_rejects_invalid_dimensions(weeks: int, paths: int) -> None:
    with pytest.raises(ValueError):
        PriceModel().simulate_prices(weeks=weeks, paths=paths)


def test_revenue_summary_returns_statistics_and_cv() -> None:
    summary = RiskAssessor.summarize([1, 2, 3])

    assert summary["mean"] == pytest.approx(2)
    assert summary["median"] == pytest.approx(2)
    assert summary["variance"] == pytest.approx(1)
    assert summary["stdev"] == pytest.approx(1)
    assert summary["cv"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("cv", "risk"),
    [(0, "Low"), (0.1, "Low"), (0.11, "Moderate"), (0.25, "Moderate"), (0.26, "High")],
)
def test_risk_classification_uses_cv_thresholds(cv: float, risk: str) -> None:
    assert RiskAssessor.classify_cv(cv) == risk


@pytest.mark.parametrize("cv", [-0.1, np.nan, np.inf])
def test_risk_classification_rejects_invalid_cv(cv: float) -> None:
    with pytest.raises(ValueError):
        RiskAssessor.classify_cv(cv)


@pytest.mark.parametrize("revenue", [[], [1], [[1, 2]], [1, np.nan], [1, -1], [0, 0]])
def test_revenue_summary_rejects_invalid_revenue(revenue: list) -> None:
    with pytest.raises(ValueError):
        RiskAssessor.summarize(revenue)


def test_revenue_var_returns_lower_tail_quantile() -> None:
    revenue = np.arange(1, 101, dtype=float)

    assert RiskAssessor.revenue_var(revenue, confidence=0.95) == pytest.approx(5.95)


@pytest.mark.parametrize(
    ("revenue", "confidence"),
    [([], 0.95), ([[1, 2]], 0.95), ([1, np.nan], 0.95), ([1, -1], 0.95), ([1, 2], 1)],
)
def test_revenue_var_rejects_invalid_inputs(revenue: list, confidence: float) -> None:
    with pytest.raises(ValueError):
        RiskAssessor.revenue_var(revenue, confidence)
