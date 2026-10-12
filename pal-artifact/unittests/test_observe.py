"""Tests for the ``observe`` statement.

``observe(x, v)`` is surface sugar for ``condition(x == v)`` (a list of values is
the disjunction over them), so it only masks: the likelihood tensor is left
**unnormalized**, its total mass is the evidence probability, and ``x``'s axis is
marginalized by kill insertion when ``x`` dies.
``test_observe_preserves_evidence_probability`` and
``test_observe_in_markov_matches_condition`` pin the equivalence to ``condition``.
"""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, condition, markov, observe, sample

Fin3 = Fin(3)


def test_observe_literal_marginalizes_the_observed_variable():
    """The surviving joint is scaled by P(x == 1) = 0.3, not renormalized."""

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        y: Fin3 = sample([0.1, 0.4, 0.5])
        observe(x, 1)
        return y

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, 0.3 * np.array([0.1, 0.4, 0.5]))


def test_observe_list_keeps_subset_mass():
    """Observing ``x in [0, 1]`` leaves total mass P(x in [0, 1]) = 0.8."""

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        y: Bool = sample(0.9 if x == 0 else 0.1)
        observe(x, [0, 1])
        return y

    r = compile_pal_fn(f)()
    assert pytest.approx(r.sum(), abs=1e-6) == 0.8


def test_observe_variable_equality():
    """``observe(x, y)`` conditions on ``x == y``; the diagonal has mass 0.5."""

    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.5, 0.5])
        observe(x, y)
        return y

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.25, 0.25])


def test_observe_constrains_disjunction():
    """Classic ``x or y`` constraint: rules out (0, 0), leaving three outcomes
    of weight 0.25 each -- uniform once normalized, mass 0.75 before."""

    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.5, 0.5])
        disj: Bool = x or y
        observe(disj, True)
        return x, y

    r = compile_pal_fn(f)()
    expected = np.array([[0.0, 0.25], [0.25, 0.25]])
    np.testing.assert_allclose(r, expected)
    np.testing.assert_allclose(r / r.sum(), np.array([[0.0, 1 / 3], [1 / 3, 1 / 3]]))


def test_observe_preserves_evidence_probability():
    """The total mass after ``observe`` is P(evidence), matching ``condition``.

    The two constructs therefore give ``--print-final-likelihood`` the same meaning.
    """

    def with_observe():
        a: Bool = sample([0.5, 0.5])
        b: Bool = sample([0.5, 0.5])
        s: Bool = a and b
        observe(s, 1)
        return a

    def with_condition():
        a: Bool = sample([0.5, 0.5])
        b: Bool = sample([0.5, 0.5])
        s: Bool = a and b
        condition(s == 1)
        return a

    r_obs = compile_pal_fn(with_observe)()
    r_cond = compile_pal_fn(with_condition)()

    # P(s = 1) = P(a = 1, b = 1) = 0.25
    assert pytest.approx(float(np.sum(r_obs)), abs=1e-9) == 0.25
    np.testing.assert_allclose(r_obs, r_cond)


def test_observe_in_branches_keeps_relative_branch_weights():
    """``observe`` inside an ``if`` keeps each branch's mass.

    The branches are summed at the merge, so their relative masses (0.05 and 0.25)
    determine the posterior over ``c``: [1/6, 5/6].
    """

    def f():
        c: Bool = sample([0.5, 0.5])
        with out:
            if c == 0:
                x: Bool = sample([0.9, 0.1])
                observe(x, 1)
                a: Bool = 1
                yield a
            else:
                y: Bool = sample([0.5, 0.5])
                observe(y, 1)
                b: Bool = 1
                yield b
        return c

    r = compile_pal_fn(f)()
    # branch c=0 mass = 0.5 * 0.1 = 0.05; branch c=1 mass = 0.5 * 0.5 = 0.25
    np.testing.assert_allclose(r, [0.05, 0.25])
    np.testing.assert_allclose(r / r.sum(), [1 / 6, 5 / 6])


def test_observe_in_markov_matches_condition():
    """Inside a scan, ``observe`` and ``condition`` agree exactly, including mass."""

    def with_observe():
        h0: Bool = sample([0.5, 0.5])
        with markov(h := h0, repeat=3):
            e: Bool = sample((0.8 if _ == 1 else 0.2) if h == 1 else (0.3 if _ == 1 else 0.7))
            observe(e, 1)
            hn: Bool = h
            yield hn
        return h

    def with_condition():
        h0: Bool = sample([0.5, 0.5])
        with markov(h := h0, repeat=3):
            e: Bool = sample((0.8 if _ == 1 else 0.2) if h == 1 else (0.3 if _ == 1 else 0.7))
            condition(e == 1)
            hn: Bool = h
            yield hn
        return h

    r_obs = compile_pal_fn(with_observe)()
    r_cond = compile_pal_fn(with_condition)()
    np.testing.assert_allclose(r_obs, r_cond)
    # P(h=0) * 0.3^3 and P(h=1) * 0.8^3
    np.testing.assert_allclose(r_obs, [0.5 * 0.3**3, 0.5 * 0.8**3])


@pytest.mark.parametrize("lva", [True, False])
@pytest.mark.parametrize("reorder", [True, False])
def test_branch_dropping_outer_variable_keeps_siblings_correct(lva, reorder):
    """A branch that drops an outside variable must not corrupt its siblings.

    Here ``x`` is observed in both branches and is dead afterwards, so eager
    marginalization kills it inside each branch. Each branch is compiled from the
    pre-if context, and a variable's value tensor is rebuilt wherever its axis
    position changed, so the second branch never works against the first branch's
    shifted axes.
    """

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        y: Bool = sample([0.4, 0.6])
        with z:
            if y == 0:
                observe(x, 1)
                a: Bool = 1
                yield a
            else:
                observe(x, 2)
                b: Bool = 0
                yield b
        return z

    r = compile_pal_fn(f, lva=lva, reorder=reorder)()
    # z=1 via branch y=0: P(y=0)*P(x=1) = 0.4*0.3; z=0 via branch y=1: 0.6*0.2
    np.testing.assert_allclose(r, [0.6 * 0.2, 0.4 * 0.3], atol=1e-12)
