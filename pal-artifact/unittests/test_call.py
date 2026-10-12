"""Tests for function-call expressions (``ExprCall``)."""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, sample

Fin3 = Fin(3)
Fin4 = Fin(4)


# Module-level helpers and constants used by the PAL programs below. They
# must live at module level (not nested in a test) because the compiled PAL
# program resolves these names through ``f.__globals__``.
arr = np.array([1, 2, 3, 4]).astype(np.float32)


def add1(x):
    return x + 1


def const():
    return arr


def test_call_no_args_returns_constant_weights():
    """A no-arg call returning a vector is used directly as a wpp."""

    def f():
        x: Fin4 = sample(const())
        return x

    r = compile_pal_fn(f)()
    # const() returns [1, 2, 3, 4]; normalized: [0.1, 0.2, 0.3, 0.4].
    np.testing.assert_allclose(r, [0.1, 0.2, 0.3, 0.4])


def test_call_with_scalar_param_argument():
    """A call whose argument is a scalar parameter is emitted verbatim."""

    def f(arr):
        x: Fin4 = sample(np.cumsum(arr))
        return x

    r = compile_pal_fn(f)(arr)
    # cumsum([1,2,3,4]) = [1,3,6,10]; normalized: [0.05, 0.15, 0.3, 0.5].
    np.testing.assert_allclose(r, [0.05, 0.15, 0.3, 0.5])


def test_call_with_sampled_variable_argument():
    """A call whose argument is a previously sampled variable is compiled
    against the (marginal) likelihood tensor of that variable, placed on the
    axis of the variable being sampled."""

    def f():
        x: Bool = sample([0.6, 0.4])
        y: Bool = sample(add1(x))
        return x, y

    r = compile_pal_fn(f)()

    # weight_y is add1(P(x)) = [1.6, 1.4] laid out on the y-axis (independent
    # of x), so P(y=0) = 1.6/3.0 and P(y=1) = 1.4/3.0.
    py = np.array([1.6, 1.4]) / 3.0
    expected = np.outer([0.6, 0.4], py)
    np.testing.assert_allclose(r, expected, atol=1e-7)


def test_call_np_exp_on_sampled_variable():
    """``np.exp`` of a sampled variable: same semantics as a user-defined
    call, but routed through a dotted name (``np.exp``)."""

    def f():
        x: Bool = sample([0.7, 0.3])
        y: Bool = sample(np.exp(x))
        return x, y

    r = compile_pal_fn(f)()
    w = np.exp([0.7, 0.3])
    py = w / w.sum()
    expected = np.outer([0.7, 0.3], py)
    np.testing.assert_allclose(r, expected, atol=1e-7)


def test_exp_softmax():
    def f(logits):
        x: Bool = sample(np.exp(logits))
        return x

    r = compile_pal_fn(f)(np.array([0.0, 1.0]))
    np.testing.assert_allclose(r, [1 / (1 + np.e), np.e / (1 + np.e)])


def test_log_weight():
    def f(weights):
        x: Bool = sample(np.log(weights))
        return x

    # log([e, e]) = [1, 1] -> uniform
    r = compile_pal_fn(f)(np.array([np.e, np.e]))
    np.testing.assert_allclose(r, [0.5, 0.5], atol=1e-6)


def test_abs_weight():
    def f(weights):
        x: Bool = sample(np.abs(weights))
        return x

    # abs([-2, 2]) = [2, 2] -> uniform
    r = compile_pal_fn(f)(np.array([-2.0, 2.0]))
    np.testing.assert_allclose(r, [0.5, 0.5])
