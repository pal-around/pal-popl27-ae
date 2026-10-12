"""Tests for every PAL expression form parsed by ``parse_expr``."""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, sample

Fin3 = Fin(3)
Fin4 = Fin(4)


def test_int_literal_weight():
    def f():
        x: Bool = sample([3, 1])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.75, 0.25])


def test_float_literal_weight():
    def f():
        x: Bool = sample([0.25, 0.75])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.25, 0.75])


def test_bool_literal_weight():
    def f():
        x: Bool = sample([True, False])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [1.0, 0.0])


def test_arithmetic_add_sub_mul_div():
    def f():
        x: Bool = sample([1 + 1, 4 - 2 * 1])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.5, 0.5])


def test_mod_and_pow():
    """``%`` and ``**`` should be available as BinOps."""

    def f():
        # 2 ** 2 == 4; 7 % 3 == 1 ; normalized: 0.8, 0.2
        x: Bool = sample([2**2, 7 % 3])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.8, 0.2])


def test_unary_minus():
    def f():
        x: Bool = sample([-(-3), -(-1)])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.75, 0.25])


def test_comparisons():
    """Test ``<``, the assignment-from-expression syntax and that the
    deterministic relation it encodes is preserved."""

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        y: Bool = x < 2
        return x, y

    r = compile_pal_fn(f)()
    # y == (x < 2), so (x=0,y=1)=0.5, (x=1,y=1)=0.3, (x=2,y=0)=0.2
    np.testing.assert_allclose(r[:, 0], [0.0, 0.0, 0.2])
    np.testing.assert_allclose(r[:, 1], [0.5, 0.3, 0.0])


def test_neq_and_gte():
    def f():
        x: Fin3 = sample([0.4, 0.4, 0.2])
        y: Bool = x != 1
        z: Bool = x >= 1
        return x, y, z

    r = compile_pal_fn(f)()
    # y is an indicator for x != 1
    np.testing.assert_allclose(r.sum(axis=2)[:, 1], [0.4, 0.0, 0.2])
    # z is an indicator for x >= 1
    np.testing.assert_allclose(r.sum(axis=1)[:, 1], [0.0, 0.4, 0.2])


def test_chained_comparison_two_ops():
    """``a <= x <= b`` desugars to ``(a <= x) and (x <= b)``."""

    def f():
        x: Fin4 = sample([0.1, 0.2, 0.3, 0.4])
        y: Bool = 1 <= x <= 2
        return x, y

    r = compile_pal_fn(f)()
    # y == 1 iff x in {1, 2}
    np.testing.assert_allclose(r[:, 1], [0.0, 0.2, 0.3, 0.0])
    np.testing.assert_allclose(r[:, 0], [0.1, 0.0, 0.0, 0.4])


def test_chained_comparison_three_ops():
    """A 3-op chain ``a <= x <= y <= b`` ANDs all three pairwise comparisons."""

    def f():
        x: Fin4 = sample([0.25, 0.25, 0.25, 0.25])
        y: Fin4 = sample([0.25, 0.25, 0.25, 0.25])
        z: Bool = 1 <= x <= y <= 2
        return x, y, z

    r = compile_pal_fn(f)()
    # z == 1 iff (x >= 1) and (x <= y) and (y <= 2). Enumerate:
    #   x=1,y=1 ; x=1,y=2 ; x=2,y=2  -> each contributes 0.25*0.25 = 0.0625
    assert pytest.approx(r[..., 1].sum(), abs=1e-6) == 3 * 0.0625
    # All other (x,y) pairs must have z == 0
    assert pytest.approx(r[..., 0].sum(), abs=1e-6) == 1.0 - 3 * 0.0625


def test_chained_comparison_mixed_ops():
    """Chains may mix operators, e.g. ``a < x <= b``."""

    def f():
        x: Fin4 = sample([0.1, 0.2, 0.3, 0.4])
        y: Bool = 0 < x <= 2
        return x, y

    r = compile_pal_fn(f)()
    # y == 1 iff x in {1, 2}
    np.testing.assert_allclose(r[:, 1], [0.0, 0.2, 0.3, 0.0])
    np.testing.assert_allclose(r[:, 0], [0.1, 0.0, 0.0, 0.4])


def test_boolop_and_or():
    """Boolean ``and`` / ``or`` are parsed via ExprBoolOp."""

    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.5, 0.5])
        z: Bool = (x == 1) and (y == 1)
        w: Bool = (x == 1) or (y == 1)
        return x, y, z, w

    r = compile_pal_fn(f)()
    assert r.shape == (2, 2, 2, 2)
    # P(z=1) == P(x=1, y=1) == 0.25
    assert pytest.approx(r[..., 1, :].sum(), abs=1e-6) == 0.25
    # P(w=1) == 1 - P(x=0, y=0) == 0.75
    assert pytest.approx(r[..., 1].sum(), abs=1e-6) == 0.75


def test_inline_if_then_else_expression():
    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample(0.9 if x == 0 else 0.1)
        return x, y

    r = compile_pal_fn(f)()
    # P(y=0) == 1 always (since wpp is a scalar in y, normalized to uniform)
    np.testing.assert_allclose(r.sum(axis=0), [0.5, 0.5])


def test_probability_table_chain():
    """``[a, b, c]`` desugars to a chained if-then-else on the implicit
    variable ``_`` being sampled."""

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.5, 0.3, 0.2])


def test_singleton_probability_table():
    """A single-element list is allowed and is treated as a constant weight."""

    def f():
        x: Bool = sample([0.5])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.5, 0.5])


def test_conditional_table_via_gt_syntax():
    """``var > [...]`` selects the wpp value based on ``var``'s value."""

    def f():
        x: Fin3 = sample([1 / 3, 1 / 3, 1 / 3])
        y: Bool = sample(x > [0.9, 0.5, 0.1])
        return x, y

    r = compile_pal_fn(f)()
    # weights are scalar in y, so each row of the joint is uniform over y
    np.testing.assert_allclose(r.sum(axis=1), [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(r[:, 0], [1 / 6, 1 / 6, 1 / 6])


def test_unary_not():
    """``not <expr>`` (NOT) flips a Bool expression."""

    def f():
        x: Bool = sample([0.7, 0.3])
        y: Bool = not (x == 0)
        return x, y

    r = compile_pal_fn(f)()
    # y == 1 iff x == 1
    expected = np.array([[0.7, 0.0], [0.0, 0.3]])
    np.testing.assert_allclose(r, expected)
