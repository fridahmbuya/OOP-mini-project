import numpy as np
import pytest
from scipy.linalg import LinAlgError

from src.solar import HybridMicroGrid, MicroGrid


def test_default_system_determinant_and_condition_number() -> None:
    grid = MicroGrid()

    assert grid.determinant == pytest.approx(-5.0)
    assert grid.condition_number == pytest.approx(
        np.linalg.cond(np.array([[3.0, 2.0], [4.0, 1.0]]))
    )
    assert np.isfinite(grid.condition_number)


def test_solve_day_returns_dispatch_satisfying_both_equations() -> None:
    grid = MicroGrid()

    dispatch = grid.solve_day(50, 50)

    np.testing.assert_allclose(dispatch, [10.0, 10.0])
    np.testing.assert_allclose(grid.coefficient_matrix @ dispatch, [50, 50])


def test_solve_day_rejects_invalid_demands() -> None:
    grid = MicroGrid()

    for demands in ((-1, 10), (10, -1), (np.nan, 10), (10, np.inf)):
        with pytest.raises(ValueError):
            grid.solve_day(*demands)


def test_singular_system_is_rejected_before_solving() -> None:
    grid = MicroGrid(((1, 2), (2, 4)))

    assert grid.determinant == pytest.approx(0.0)
    with pytest.raises(LinAlgError, match="singular or ill-conditioned"):
        grid.solve_day(10, 20)


def test_loop_and_vectorized_solves_match_and_satisfy_constraints() -> None:
    grid = MicroGrid()
    demands = np.array([[50, 50], [60, 55], [45, 40]], dtype=float)

    loop_dispatch = grid.solve_days_loop(demands)
    vectorized_dispatch = grid.solve_days_vectorized(demands)

    np.testing.assert_allclose(loop_dispatch, vectorized_dispatch)
    np.testing.assert_allclose(
        (grid.coefficient_matrix @ vectorized_dispatch.T).T, demands
    )


@pytest.mark.parametrize(
    "demands",
    [
        np.array([50, 50]),
        np.array([[50, 50, 60]]),
        np.array([[50, np.nan]]),
        np.array([[50, -1]]),
    ],
)
def test_batch_methods_reject_invalid_demands(demands: np.ndarray) -> None:
    grid = MicroGrid()

    with pytest.raises(ValueError):
        grid.solve_days_loop(demands)
    with pytest.raises(ValueError):
        grid.solve_days_vectorized(demands)


def test_batch_methods_return_empty_dispatch_for_no_days() -> None:
    grid = MicroGrid()

    assert grid.solve_days_loop([]).shape == (0, 2)
    assert grid.solve_days_vectorized([]).shape == (0, 2)


def test_nonnegative_solver_handles_infeasible_unconstrained_dispatch() -> None:
    grid = MicroGrid()
    unconstrained = grid.solve_day(40, 10)

    assert unconstrained[0] < 0
    nonnegative, residual_norm = grid.solve_nonnegative_day(40, 10)

    assert np.all(nonnegative >= 0)
    assert residual_norm > 0


def test_cost_calculation_uses_illustrative_rates() -> None:
    assert MicroGrid().calculate_cost(10, 5) == pytest.approx(3_750)


@pytest.mark.parametrize(
    "usage",
    [(-1, 2), (1, -2), (np.nan, 1), (1, np.inf)],
)
def test_cost_calculation_rejects_invalid_usage(usage: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        MicroGrid().calculate_cost(*usage)


def test_hybrid_system_solves_three_constraints() -> None:
    grid = HybridMicroGrid()

    dispatch = grid.solve_day(50, 50, 30)

    np.testing.assert_allclose(dispatch, [10.0, 10.0, 10.0])
    np.testing.assert_allclose(grid.coefficient_matrix @ dispatch, [50, 50, 30])
    assert grid.calculate_cost(*dispatch) == pytest.approx(15_000)


def test_hybrid_system_rejects_linearly_dependent_third_equation() -> None:
    grid = HybridMicroGrid(
        (
            (3, 2, 0),
            (4, 1, 0),
            (7, 3, 0),
        )
    )

    assert grid.determinant == pytest.approx(0.0)
    with pytest.raises(LinAlgError, match="singular or ill-conditioned"):
        grid.solve_day(50, 50, 100)


def test_hybrid_system_rejects_invalid_matrix_shape() -> None:
    with pytest.raises(ValueError, match="shape \\(3, 3\\)"):
        HybridMicroGrid(((1, 2), (3, 4)))
