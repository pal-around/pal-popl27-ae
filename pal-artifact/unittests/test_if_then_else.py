"""Tests for the ``with <vars>: if ...: yield ...`` construct."""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, sample

Fin3 = Fin(3)
Fin6 = Fin(6)


def test_simple_if_else():
    def f():
        cond: Bool = sample([0.6, 0.4])
        with out:
            if cond == 0:
                a: Bool = 0
                yield a
            else:
                b: Bool = 1
                yield b

        return out

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.6, 0.4])


def test_if_elif_else_three_way_branch():
    def f():
        cond: Fin3 = sample([0.5, 0.3, 0.2])
        with out:
            if cond == 0:
                a: Fin3 = 0
                yield a
            elif cond == 1:
                b: Fin3 = 1
                yield b
            else:
                c: Fin3 = 2
                yield c

        return out

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.5, 0.3, 0.2])


def test_if_then_else_yields_multiple_variables():
    def f():
        cond: Bool = sample([0.5, 0.5])
        with x, y:
            if cond == 0:
                a: Bool = 0
                b: Bool = 1
                yield a, b
            else:
                c: Bool = 1
                d: Bool = 0
                yield c, d

        return x, y

    r = compile_pal_fn(f)()
    # On branch 0 we get (0, 1); on branch 1 we get (1, 0). Each branch
    # has 0.5 prior probability, so the joint mass is 0.5 on each.
    expected = np.array([[0.0, 0.5], [0.5, 0.0]])
    np.testing.assert_allclose(r, expected)


def test_if_then_else_block_can_sample_inside():
    def f():
        cond: Bool = sample([0.5, 0.5])
        with out:
            if cond == 0:
                a: Bool = sample([0.9, 0.1])
                yield a
            else:
                b: Bool = sample([0.1, 0.9])
                yield b

        return out

    r = compile_pal_fn(f)()
    # P(out=0) = 0.5 * 0.9 + 0.5 * 0.1 = 0.5
    # P(out=1) = 0.5 * 0.1 + 0.5 * 0.9 = 0.5
    np.testing.assert_allclose(r, [0.5, 0.5])
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


# The next three tests share the same prior over (cond, shared, shared2):
#   cond    ~ [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
#   shared  ~ [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
#   shared2 ~ [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
# (For `Fin6 = sample([w0, ..., wk])` the trailing weights are repeated for
# the remaining support, then the table is normalized.)
SHARED_PRIOR = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])


def test_truthy_cond_branch_yields_shared_else_yields_one():
    """`if cond:` (truthy on a Fin6) routes the 5 nonzero values to a
    ``shared`` sample and the zero value to the constant ``1``."""

    def f():
        cond: Fin6 = sample([0.4, 0.2, 0.1])
        shared: Fin6 = sample([0.3, 0.3, 0.1])
        shared2: Fin6 = sample([0.4, 0.2, 0.1])

        with x:
            if cond:
                a: Fin6 = shared
                temp = cond
                yield a
            else:
                b: Fin6 = 1
                yield b
        return x

    r = compile_pal_fn(f)()
    # P(cond == 0) = 0.4 → x = 1.
    # P(cond != 0) = 0.6 → x = shared.
    expected = 0.6 * SHARED_PRIOR.copy()
    expected[1] += 0.4
    np.testing.assert_allclose(r, expected)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


def test_if_elif_else_three_way_yields_pair():
    """Three-way branch on ``cond`` with each arm yielding a (y, z) pair."""

    def f():
        cond: Fin6 = sample([0.4, 0.2, 0.1])
        shared: Fin6 = sample([0.3, 0.3, 0.1])
        shared2: Fin6 = sample([0.4, 0.2, 0.1])

        with y, z:
            if cond == 0:
                a: Fin6 = 0
                b: Fin6 = shared
                temp2: Fin6 = cond
                temp3: Fin6 = 2
                yield a, b
            elif cond == 1:
                temp: Fin6 = 1
                c: Fin6 = shared
                d: Fin6 = 3
                yield c, d
            else:
                c: Fin6 = 4
                d: Fin6 = 5
                yield c, d
        return y, z

    r = compile_pal_fn(f)()
    # cond=0 (P=0.4): (y, z) = (0, shared) → row 0 += 0.4 * shared
    # cond=1 (P=0.2): (y, z) = (shared, 3) → column 3 += 0.2 * shared
    # cond>=2 (P=0.4): (y, z) = (4, 5)     → cell [4, 5] += 0.4
    expected = np.zeros((6, 6))
    expected[0, :] += 0.4 * SHARED_PRIOR
    expected[:, 3] += 0.2 * SHARED_PRIOR
    expected[4, 5] += 0.4
    np.testing.assert_allclose(r, expected)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


def test_nested_if_then_else_in_else_branch():
    """Outer ``if cond == 0`` branches normally; the else branch nests a
    second if-then-else block that itself yields the (y, z) pair."""

    def f():
        cond: Fin6 = sample([0.4, 0.2, 0.1])
        shared: Fin6 = sample([0.3, 0.3, 0.1])
        shared2: Fin6 = sample([0.4, 0.2, 0.1])

        with y, z:
            if cond == 0:
                a: Fin6 = 0
                b: Fin6 = shared
                yield a, b
            else:
                with c, d:
                    if cond == 1:
                        c: Fin6 = shared
                        d: Fin6 = 3
                        yield c, d
                    else:
                        c: Fin6 = 4
                        d: Fin6 = 5
                        yield c, d
                yield c, d
        return y, z

    r = compile_pal_fn(f)()
    # Same per-branch distribution as the flat if-elif-else version above.
    expected = np.zeros((6, 6))
    expected[0, :] += 0.4 * SHARED_PRIOR
    expected[:, 3] += 0.2 * SHARED_PRIOR
    expected[4, 5] += 0.4
    np.testing.assert_allclose(r, expected)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0
