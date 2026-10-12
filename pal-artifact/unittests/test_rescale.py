"""Scale tracking (``@pal(rescale=True)``).

The compiled program maintains

    true likelihood  ==  likelihood_tensor * exp(log_scale)

so it can divide the tensor down into a representable range without losing the
evidence probability.

Every test compares against the unscaled compilation of the *same* program, so
the oracle is PAL itself with rescaling off.
"""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, condition, init, markov, observe, sample

Fin3 = Fin(3)
Fin4 = Fin(4)


def _both(prog, **kw):
    """(unscaled tensor, rescaled tensor, log_scale) for the same program."""
    plain = np.asarray(compile_pal_fn(prog, **kw)(), dtype=np.float64)
    scaled, log_scale = compile_pal_fn(prog, rescale=True, return_log_scale=True, **kw)()
    return plain, np.asarray(scaled, dtype=np.float64), float(log_scale)


def _assert_equivalent(prog, **kw):
    plain, scaled, log_scale = _both(prog, **kw)
    # Same posterior.
    np.testing.assert_allclose(scaled / scaled.sum(), plain / plain.sum(), rtol=1e-9, atol=1e-12)
    # Same evidence probability, recovered through the tracked scale.
    assert pytest.approx(float(scaled.sum()) * np.exp(log_scale), rel=1e-9) == float(plain.sum())
    return plain, scaled, log_scale


# ---------------------------------------------------------------------------
# Programs covering each construct that has to carry or rebase the scale.
# ---------------------------------------------------------------------------

def straight_line():
    x: Fin3 = sample([0.5, 0.3, 0.2])
    y: Fin3 = sample([0.1, 0.4, 0.5])
    condition(x >= y)
    return x, y


def with_observe():
    x: Fin3 = sample([0.5, 0.3, 0.2])
    y: Fin3 = sample([0.1, 0.4, 0.5])
    observe(x, 1)
    return y


def markov_chain():
    h0: Bool = sample([0.5, 0.5])
    with markov(h := h0, repeat=6):
        e: Bool = sample((0.8 if _ == 1 else 0.2) if h == 1 else (0.3 if _ == 1 else 0.7))
        condition(e == 1)
        hn: Bool = h
        yield hn
    return h


def markov_random_bound():
    n: Fin3 = sample([0.2, 0.3, 0.5])
    h0: Bool = sample([0.5, 0.5])
    with markov(h := h0, repeat=n):
        e: Bool = sample((0.8 if _ == 1 else 0.2) if h == 1 else (0.3 if _ == 1 else 0.7))
        condition(e == 1)
        hn: Bool = h
        yield hn
    return h


def if_branches():
    c: Bool = sample([0.5, 0.5])
    with out:
        if c == 0:
            x: Bool = sample([0.9, 0.1])
            condition(x == 1)
            a: Bool = 1
            yield a
        else:
            y: Bool = sample([0.5, 0.5])
            condition(y == 1)
            b: Bool = 1
            yield b
    return c


# The case that forces a real rebase: only one branch rescales, because only one
# contains a loop, so the branches reach the merge at different scales.
def markov_inside_branch():
    c: Bool = sample([0.5, 0.5])
    with out:
        if c == 0:
            h0: Bool = sample([0.5, 0.5])
            with markov(h := h0, repeat=4):
                e: Bool = sample((0.7 if _ == 1 else 0.3) if h == 1 else (0.2 if _ == 1 else 0.8))
                condition(e == 1)
                hn: Bool = h
                yield hn
            a: Bool = h
            yield a
        else:
            b: Bool = sample([0.4, 0.6])
            yield b
    return c, out


def branch_inside_markov():
    key: Bool = sample([0.5, 0.5])
    with markov(k := key, repeat=4):
        f: Bool = sample([0.6, 0.4])
        with nk:
            if f:
                t1: Bool = k
                yield t1
            else:
                t2: Bool = sample([0.5, 0.5])
                yield t2
        condition(nk == 1)
        yield nk
    return k


def markov_init_block_with_evidence():
    # Evidence inside the init block as well as in the body.
    with markov(h, repeat=5):
        with init():
            h0: Bool = sample([0.5, 0.5])
            e0: Bool = sample((0.8 if _ == 1 else 0.2) if h0 == 1 else (0.3 if _ == 1 else 0.7))
            condition(e0 == 1)
            yield h0
        e: Bool = sample((0.8 if _ == 1 else 0.2) if h == 1 else (0.3 if _ == 1 else 0.7))
        condition(e == 1)
        hn: Bool = h
        yield hn
    return h


def rv_bound_correlated_with_evidence():
    # A random loop count that also determines the initial state, with evidence in
    # every iteration: the accumulator must be rescaled together with the carry.
    n: Fin3 = sample([0.2, 0.3, 0.5])
    with markov(h, repeat=n):
        with init():
            h0: Bool = n == 1
            yield h0
        e: Bool = sample((0.8 if _ == 1 else 0.2) if h == 1 else (0.3 if _ == 1 else 0.7))
        condition(e == 1)
        hn: Bool = 1 - h
        yield hn
    return h


PROGRAMS = [
    rv_bound_correlated_with_evidence,
    markov_init_block_with_evidence,
    straight_line,
    with_observe,
    markov_chain,
    markov_random_bound,
    if_branches,
    markov_inside_branch,
    branch_inside_markov,
]


@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
@pytest.mark.parametrize("lva", [True, False])
@pytest.mark.parametrize("reorder", [True, False])
def test_rescale_equivalent_under_all_passes(prog, lva, reorder):
    """Rescaling must compose with eager marginalization and reordering.

    ``markov_inside_branch`` pins scope-exit marginalization (``lva=False``): the
    loop state must be marginalized at the end of the loop body and the branch's
    locals at the end of the branch, or the branch ends holding a variable its
    sibling does not have and the merge is rejected.
    """
    _assert_equivalent(prog, lva=lva, reorder=reorder)


def test_rescale_off_emits_no_scale_tracking():
    """rescale=False must emit no scale-tracking code at all, so programs that do
    not opt in are unaffected."""
    import contextlib
    import io

    for prog in PROGRAMS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            compile_pal_fn(prog, print_compiled=True)
        src = buf.getvalue()
        assert "__log_scale" not in src, f"{prog.__name__} emitted scale tracking with rescale=False"
        assert "__rescale_m" not in src


def test_rescale_emits_tracking_and_rebase_only_where_needed():
    """The branch rebase costs a multiply per branch, so it must appear only when
    a branch actually rescaled."""
    import contextlib
    import io

    def src_of(prog):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            compile_pal_fn(prog, rescale=True, print_compiled=True)
        return buf.getvalue()

    # A loop rescales once per iteration.
    assert "__rescale_m" in src_of(markov_chain)
    # Branches with no loop inside end at the entry scale -> plain sum, no rebase.
    assert "merge_max" not in src_of(if_branches)
    # A loop inside one branch forces the rebase.
    assert "merge_max" in src_of(markov_inside_branch)


def test_rescale_prevents_underflow_where_unscaled_dies():
    """The point of the feature: a chain long enough to underflow in float64
    still yields a usable posterior, and the true log evidence is recovered."""

    def long_chain():
        h0: Bool = sample([0.5, 0.5])
        with markov(h := h0, repeat=400):
            e: Bool = sample((0.001 if _ == 1 else 0.999) if h == 1 else (0.0005 if _ == 1 else 0.9995))
            condition(e == 1)
            hn: Bool = h
            yield hn
        return h

    plain = np.asarray(compile_pal_fn(long_chain)(), dtype=np.float64)
    scaled, log_scale = compile_pal_fn(long_chain, rescale=True, return_log_scale=True)()
    scaled, log_scale = np.asarray(scaled, dtype=np.float64), float(log_scale)

    # Unscaled underflows to nothing; rescaled stays finite and normalizable.
    assert plain.sum() == 0.0
    assert scaled.sum() > 0.0
    assert np.isfinite(scaled).all()

    # Exact posterior: P(h=1) / P(h=0) = (0.001/0.0005)^400 = 2^400, so h=1 has
    # essentially all the mass.
    post = scaled / scaled.sum()
    np.testing.assert_allclose(post, [0.0, 1.0], atol=1e-12)

    # log evidence = log(0.5 * 0.0005^400 + 0.5 * 0.001^400)
    expected = np.log(0.5) + 400 * np.log(0.001) + np.log1p(0.5**400)
    assert pytest.approx(float(np.log(scaled.sum()) + log_scale), rel=1e-9) == expected


def test_rescale_survives_zero_mass():
    """Contradictory evidence gives an all-zero tensor; the max is 0 and the
    rescale must not divide by it (that would be NaN)."""

    def contradiction():
        x: Bool = sample([0.5, 0.5])
        condition(x == 1)
        condition(x == 0)
        return x

    scaled, log_scale = compile_pal_fn(contradiction, rescale=True, return_log_scale=True)()
    scaled = np.asarray(scaled, dtype=np.float64)
    assert np.isfinite(scaled).all()
    assert scaled.sum() == 0.0
    assert np.isfinite(float(log_scale))
