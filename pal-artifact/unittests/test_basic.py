"""Basic sample/return/assign tests using the annotated-assignment syntax."""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, sample

Fin3 = Fin(3)
Fin4 = Fin(4)


def test_single_fair_coin():
    def f():
        x: Bool = sample([0.5, 0.5])
        return x

    r = compile_pal_fn(f)()
    assert r.shape == (2,)
    np.testing.assert_allclose(r, [0.5, 0.5])


def test_biased_coin():
    def f():
        x: Bool = sample([0.7, 0.3])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.7, 0.3])


def test_weights_are_normalized():
    """Weights are normalized to a probability distribution, so a [2, 8]
    weighting should produce the same distribution as a [0.2, 0.8] one."""

    def f():
        x: Bool = sample([2, 8])
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.2, 0.8])


def test_uniform_via_constant_weight():
    def f():
        x: Bool = sample(1)
        return x

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.5, 0.5])


def test_two_independent_fair_coins():
    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.5, 0.5])
        return x, y

    r = compile_pal_fn(f)()
    assert r.shape == (2, 2)
    np.testing.assert_allclose(r, np.full((2, 2), 0.25))


def test_finite_support():
    def f():
        x: Fin4 = sample([0.4, 0.3, 0.2, 0.1])
        return x

    r = compile_pal_fn(f)()
    assert r.shape == (4,)
    np.testing.assert_allclose(r, [0.4, 0.3, 0.2, 0.1])


def test_dirac_via_assign_single():
    """``y: T = x`` copies the distribution of ``x`` into a perfectly
    correlated variable ``y``."""

    def f():
        x: Bool = sample([0.7, 0.3])
        y: Bool = x
        return x, y

    r = compile_pal_fn(f)()
    expected = np.array([[0.7, 0.0], [0.0, 0.3]])
    np.testing.assert_allclose(r, expected)


def test_assign_expression():
    """``y: T = <expr>`` is an assignment from a deterministic expression."""

    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = 1 - x
        return x, y

    r = compile_pal_fn(f)()
    expected = np.array([[0.0, 0.5], [0.5, 0.0]])
    np.testing.assert_allclose(r, expected)


def test_dirac_correlation_via_indicator():
    """Sampling ``y`` with a wpp that is 1 only when ``x == y`` produces a
    Dirac on the diagonal."""

    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample(1 if x == y else 0)
        return x, y

    r = compile_pal_fn(f)()
    expected = np.array([[0.5, 0.0], [0.0, 0.5]])
    np.testing.assert_allclose(r, expected)


def test_scalar_parameter():
    def f(p=0.5):
        x: Bool = sample([1 - p, p])
        return x

    compiled = compile_pal_fn(f)
    np.testing.assert_allclose(compiled(), [0.5, 0.5])
    np.testing.assert_allclose(compiled(0.1), [0.9, 0.1])
    np.testing.assert_allclose(compiled(0.8), [0.2, 0.8])


def test_marginal_distribution_sums_to_one():
    def f():
        x: Fin3 = sample([0.2, 0.3, 0.5])
        y: Fin3 = sample([0.1, 0.4, 0.5])
        return x, y

    r = compile_pal_fn(f)()
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0
    np.testing.assert_allclose(r.sum(axis=1), [0.2, 0.3, 0.5])
    np.testing.assert_allclose(r.sum(axis=0), [0.1, 0.4, 0.5])


def test_jax_jit_backend_matches_numpy_backend():
    """Smoke test that the default JAX+JIT backend produces the same answer
    as the NumPy backend used in the rest of the test suite."""

    def f():
        x: Bool = sample([0.7, 0.3])
        y: Bool = sample(0.9 if x == 0 else 0.2)
        return x, y

    r = compile_pal_fn(f)()
    r_jax = compile_pal_fn(f, numpy=False, jit=True)()

    np.testing.assert_allclose(np.asarray(r_jax), r, atol=1e-6)
