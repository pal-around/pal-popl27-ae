"""Tests for the ``markov`` construct."""

import contextlib
import io

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, init, markov, sample

Fin3 = Fin(3)

# Transition matrix used by the init-block tests: row = current state.
T = np.array([[0.6, 0.4], [0.7, 0.3]])
LVA = pytest.mark.parametrize("lva", [True, False])


def _peak_cells(prog, **kw) -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_pal_fn(prog, print_static_ndim_peak=True, **kw)
    for line in buf.getvalue().splitlines():
        if "Static peak cells" in line:
            return int(line.rsplit(":", 1)[1])
    raise AssertionError("peak cells not reported")


def test_markov_zero_repeats_is_identity():
    def f():
        x: Bool = sample([0.8, 0.2])
        with markov(y := x, repeat=0):
            z: Bool = sample(1 if z == y else 0)
            yield z

        return y

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.8, 0.2])


def test_markov_dirac_preserves_distribution():
    """Each step copies the previous value into the next; the marginal is
    therefore unchanged."""

    def f():
        x: Bool = sample([0.9, 0.1])
        with markov(y := x, repeat=5):
            z: Bool = sample(1 if z == y else 0)
            yield z

        return y

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.9, 0.1])


def test_markov_flip_at_every_step():
    """A two-state chain that deterministically flips at every step. After
    an even number of steps the marginal equals the prior; after an odd
    number it is reflected."""

    def f_even():
        x: Bool = sample([0.7, 0.3])
        with markov(y := x, repeat=4):
            z: Bool = sample(1 if z != y else 0)
            yield z

        return y

    def f_odd():
        x: Bool = sample([0.7, 0.3])
        with markov(y := x, repeat=3):
            z: Bool = sample(1 if z != y else 0)
            yield z

        return y

    np.testing.assert_allclose(compile_pal_fn(f_even)(), [0.7, 0.3])
    np.testing.assert_allclose(compile_pal_fn(f_odd)(), [0.3, 0.7])


def test_markov_repeat_via_scalar_parameter():
    def f(n=2):
        x: Bool = sample([0.6, 0.4])
        with markov(y := x, repeat=n):
            z: Bool = sample(1 if z == y else 0)
            yield z

        return y

    compiled = compile_pal_fn(f)
    np.testing.assert_allclose(compiled(), [0.6, 0.4])
    np.testing.assert_allclose(compiled(0), [0.6, 0.4])
    np.testing.assert_allclose(compiled(7), [0.6, 0.4])


def test_markov_with_multiple_state_variables():
    def f():
        a: Bool = sample([0.5, 0.5])
        b: Bool = sample([0.9, 0.1])
        with markov(aa := a, bb := b, repeat=3):
            c: Bool = sample(1 if c == bb else 0)
            d: Bool = sample(1 if d == aa else 0)
            yield c, d

        return aa, bb

    r = compile_pal_fn(f)()
    assert r.shape == (2, 2)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0
    # an odd number of swaps swaps the marginals
    np.testing.assert_allclose(r.sum(axis=0), [0.5, 0.5])
    np.testing.assert_allclose(r.sum(axis=1), [0.9, 0.1])


# ---------------------------------------------------------------------------
# Init blocks
# ---------------------------------------------------------------------------


@LVA
def test_markov_full_init_block_form(lva):
    def f():
        with markov(x, repeat=3):
            with init():
                x0: Bool = sample([0.9, 0.1])
                yield x0
            xn: Bool = sample(x > [[0.6, 0.4], [0.7, 0.3]])
            yield xn
        return x

    expected = np.array([0.9, 0.1]) @ np.linalg.matrix_power(T, 3)
    np.testing.assert_allclose(compile_pal_fn(f, lva=lva)(), expected)


@LVA
def test_markov_init_block_local_is_dropped(lva):
    # `u` is local to the init block: it is marginalized before the loop starts in
    # both modes, so the loop itself only ever holds the state and its successor.
    def f():
        with markov(x, repeat=2):
            with init():
                u: Fin3 = sample([0.2, 0.3, 0.5])
                x0: Bool = u == 2
                yield x0
            xn: Bool = sample(x > [[0.6, 0.4], [0.7, 0.3]])
            yield xn
        return x

    expected = np.array([0.5, 0.5]) @ np.linalg.matrix_power(T, 2)
    np.testing.assert_allclose(compile_pal_fn(f, lva=lva)(), expected)
    # peak = u (3) x x0 (2) inside the init block; the loop holds x x xn = 4 cells
    assert _peak_cells(f, lva=lva) == 6


def test_markov_init_block_marginalizes_dead_outer_variable():
    # `w` is read only by the init block and is dead afterwards, so eager
    # marginalization sums it out inside the init block.
    def f():
        w: Fin3 = sample([0.2, 0.3, 0.5])
        with markov(x, repeat=2):
            with init():
                x0: Bool = w == 2
                yield x0
            xn: Bool = sample(x > [[0.6, 0.4], [0.7, 0.3]])
            yield xn
        return x

    expected = np.array([0.5, 0.5]) @ np.linalg.matrix_power(T, 2)
    for lva in (True, False):
        np.testing.assert_allclose(compile_pal_fn(f, lva=lva)(), expected)
    assert _peak_cells(f, lva=True) == 6
    # Without eager marginalization `w` stays in scope through the loop.
    assert _peak_cells(f, lva=False) == 12


@LVA
def test_markov_body_yields_permuted_carries(lva):
    # The body yields the state in swapped order, so every iteration swaps the two
    # variables' axes.
    def f():
        a: Bool = sample([0.5, 0.5])
        b: Bool = sample([0.9, 0.1])
        with markov(aa := a, bb := b, repeat=3):
            yield bb, aa
        return aa, bb

    np.testing.assert_allclose(compile_pal_fn(f, lva=lva)(), np.outer([0.9, 0.1], [0.5, 0.5]))


@LVA
def test_markov_init_block_yields_permuted_locals(lva):
    def f():
        with markov(aa, bb, repeat=2):
            with init():
                p: Bool = sample([0.5, 0.5])
                q: Bool = sample([0.9, 0.1])
                yield q, p
            c: Bool = aa
            d: Bool = bb
            yield c, d
        return aa, bb

    np.testing.assert_allclose(compile_pal_fn(f, lva=lva)(), np.outer([0.9, 0.1], [0.5, 0.5]))


@LVA
def test_markov_identity_yield_of_carry(lva):
    def f():
        a: Bool = sample([0.3, 0.7])
        b: Bool = sample([0.6, 0.4])
        with markov(aa := a, bb := b, repeat=4):
            yield aa, bb
        return aa, bb

    np.testing.assert_allclose(compile_pal_fn(f, lva=lva)(), np.outer([0.3, 0.7], [0.6, 0.4]))


# ---------------------------------------------------------------------------
# Random loop counts: a mixture over the count's values
# ---------------------------------------------------------------------------


@LVA
def test_rv_bound_correlated_with_state(lva):
    # The count also initializes the state: s ends at n + n (mod 3), so the answer
    # depends on the correlation between the count and the state.
    def f():
        n: Fin3 = sample([0.2, 0.3, 0.5])
        with markov(s, repeat=n):
            with init():
                s0: Fin3 = n
                yield s0
            t: Fin3 = (s + 1) % 3
            yield t
        return s

    np.testing.assert_allclose(compile_pal_fn(f, lva=lva)(), [0.2, 0.5, 0.3])


@LVA
def test_rv_bound_returned_jointly(lva):
    # s flips once per iteration, so the joint of (n, s) pairs each count with the
    # state after that many flips.
    def f():
        n: Fin3 = sample([0.2, 0.3, 0.5])
        s0: Bool = sample([0.6, 0.4])
        with markov(s := s0, repeat=n):
            t: Bool = 1 - s
            yield t
        return n, s

    p_n = np.array([0.2, 0.3, 0.5])
    states = [np.array([0.6, 0.4]), np.array([0.4, 0.6]), np.array([0.6, 0.4])]
    expected = np.stack([p_n[k] * states[k] for k in range(3)])
    np.testing.assert_allclose(compile_pal_fn(f, lva=lva)(), expected)
