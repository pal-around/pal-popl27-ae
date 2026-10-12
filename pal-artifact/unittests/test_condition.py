"""Tests for the ``condition`` statement."""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, condition, sample

Fin3 = Fin(3)


# Shared setup for the tests below:
#   x ~ [0.5, 0.3, 0.2], y ~ [0.1, 0.4, 0.5], independent.
# With PAL's convention, ctxt.variables ends up [x, y] so the joint
# likelihood has shape (n_y, n_x): likelihood[i, j] = p(x=j) * p(y=i).
#
# After masking on `x >= y`:
#   y=0: [0.05, 0.03, 0.02]   (sum 0.10)
#   y=1: [0.00, 0.12, 0.08]   (sum 0.20)
#   y=2: [0.00, 0.00, 0.10]   (sum 0.10)
# Total surviving mass: 0.40.


def test_condition_is_a_pure_mask():
    """`condition(x >= y)` multiplies the joint by the mask; the mass is P(x >= y)."""

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        y: Fin3 = sample([0.1, 0.4, 0.5])
        condition(x >= y)
        return x, y

    r = compile_pal_fn(f)()
    # Return is (x, y) so output[j, i] = p(x=j, y=i) * mask(x>=y).
    expected = np.array(
        [
            [0.05, 0.00, 0.00],
            [0.03, 0.12, 0.00],
            [0.02, 0.08, 0.10],
        ]
    )
    np.testing.assert_allclose(r, expected)
    assert pytest.approx(r.sum(), abs=1e-6) == 0.40


def test_condition_rejects_keyword_arguments():
    """`condition()` takes no keyword arguments."""

    def g():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        condition(x >= 1, bogus=True)
        return x

    with pytest.raises(SyntaxError, match="Unexpected keyword argument"):
        compile_pal_fn(g)


def test_condition_marginal_after_mask():
    """Marginalizing x out of the masked joint yields the y-marginal."""

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        y: Fin3 = sample([0.1, 0.4, 0.5])
        condition(x >= y)
        return y

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.10, 0.20, 0.10])


def test_condition_disjunction_mask_only():
    """`condition(x or y)` masks the joint by the disjunction. Pre-mask the
    joint is uniform at 0.25; the mask zeroes only the (False, False) cell,
    leaving the other three cells untouched."""

    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.5, 0.5])
        condition(x or y)
        return x, y

    r = compile_pal_fn(f)()
    expected = np.array(
        [
            [0.0, 0.25],
            [0.25, 0.25],
        ]
    )
    np.testing.assert_allclose(r, expected)
    assert pytest.approx(r.sum(), abs=1e-6) == 0.75


def test_condition_equality_keeps_diagonal():
    """`condition(x == y)` retains the diagonal of the joint over (x, y);
    off-diagonal cells are masked to zero."""

    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.5, 0.5])
        condition(x == y)
        return x, y

    r = compile_pal_fn(f)()
    expected = np.array(
        [
            [0.25, 0.0],
            [0.0, 0.25],
        ]
    )
    np.testing.assert_allclose(r, expected)
    assert pytest.approx(r.sum(), abs=1e-6) == 0.5


def test_condition_chained_comparison():
    """`condition(a <= x <= b)` masks via the desugared AND of pairwise comparisons."""

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        y: Fin3 = sample([0.1, 0.4, 0.5])
        condition(0 <= x <= 1)
        return x, y

    r = compile_pal_fn(f)()
    # Pre-mask joint[j, i] = p(x=j) * p(y=i). Mask keeps only x in {0, 1}:
    #   row x=0: [0.05, 0.20, 0.25]
    #   row x=1: [0.03, 0.12, 0.15]
    #   row x=2: zeroed
    expected = np.array(
        [
            [0.05, 0.20, 0.25],
            [0.03, 0.12, 0.15],
            [0.00, 0.00, 0.00],
        ]
    )
    np.testing.assert_allclose(r, expected)
    assert pytest.approx(r.sum(), abs=1e-6) == 0.80
