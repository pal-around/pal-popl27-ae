"""Eager marginalization must not change results.

The live-variable analysis (`lva`) inserts `kill` (marginalization) statements to
sum out dead variables early. This is purely a peak-memory optimization: the final
marginal distribution must be identical whether `lva` is on or off. `lva=False`
(marginalize each block's variables only where the block ends) is the reference.

This module compiles a battery of representative programs both ways and asserts the
returned likelihood tensors match.
"""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, condition, init, markov, observe, sample

Fin3 = Fin(3)
Fin6 = Fin(6)


# ---------------------------------------------------------------------------
# Representative programs (module-level so both configs compile the same source).
# Each is a zero-argument (or default-argument) @pal-able function.
# ---------------------------------------------------------------------------

def straight_line_dead_intermediates():
    # x and y are dead after their draws; only z is returned.
    x: Fin3 = sample([0.5, 0.3, 0.2])
    y: Fin3 = sample([0.1, 0.4, 0.5])
    z: Bool = sample(0.9 if x == y else 0.1)
    return z


def condition_mask_only():
    x: Fin3 = sample([0.5, 0.3, 0.2])
    y: Fin3 = sample([0.1, 0.4, 0.5])
    condition(x >= y)
    return y


def condition_then_marginalize_chained():
    x: Fin3 = sample([0.5, 0.3, 0.2])
    y: Fin3 = sample([0.1, 0.4, 0.5])
    z: Fin3 = sample([0.2, 0.3, 0.5])
    condition(x >= y)
    condition(y >= z)
    return y


def observe_literal():
    x: Fin3 = sample([0.5, 0.3, 0.2])
    y: Fin3 = sample([0.1, 0.4, 0.5])
    observe(x, 1)
    return y


def observe_variable_equality():
    x: Bool = sample([0.5, 0.5])
    y: Bool = sample([0.5, 0.5])
    observe(x, y)
    return y


def markov_dirac():
    x: Bool = sample([0.9, 0.1])
    with markov(y := x, repeat=5):
        z: Bool = sample(1 if z == y else 0)
        yield z
    return y


def markov_flip():
    x: Bool = sample([0.7, 0.3])
    with markov(y := x, repeat=4):
        z: Bool = sample(1 if z != y else 0)
        yield z
    return y


def markov_multi_state():
    a: Bool = sample([0.5, 0.5])
    b: Bool = sample([0.9, 0.1])
    with markov(aa := a, bb := b, repeat=3):
        c: Bool = sample(1 if c == bb else 0)
        d: Bool = sample(1 if d == aa else 0)
        yield c, d
    return aa, bb


def markov_repeat_param(n=3):
    x: Bool = sample([0.6, 0.4])
    with markov(y := x, repeat=n):
        z: Bool = sample(1 if z == y else 0)
        yield z
    return y


def markov_body_reads_enclosing():
    # w is sampled before the loop, read inside the body on every iteration, and
    # is dead after the loop (only the carry y is returned). The K_l analysis must
    # keep w live *through* the loop (fv(body) term) and marginalize it *after* --
    # marginalizing it inside would corrupt later iterations. `lva=False` is the
    # reference for the correct joint.
    w: Bool = sample([0.7, 0.3])
    x: Bool = sample([0.5, 0.5])
    with markov(y := x, repeat=3):
        z: Bool = sample(0.8 if z == w else 0.2)
        yield z
    return y


def if_simple():
    cond: Bool = sample([0.6, 0.4])
    with out:
        if cond == 0:
            a: Bool = 0
            yield a
        else:
            b: Bool = 1
            yield b
    return out


def if_sample_inside():
    cond: Bool = sample([0.5, 0.5])
    with out:
        if cond == 0:
            a: Bool = sample([0.9, 0.1])
            yield a
        else:
            b: Bool = sample([0.1, 0.9])
            yield b
    return out


def if_branch_reads_shared():
    # `shared` is read only by the truthy branch; `shared2` by neither. Exercises
    # the sibling-read handling.
    cond: Fin6 = sample([0.4, 0.2, 0.1])
    shared: Fin6 = sample([0.3, 0.3, 0.1])
    shared2: Fin6 = sample([0.4, 0.2, 0.1])
    with x:
        if cond:
            a: Fin6 = shared
            yield a
        else:
            b: Fin6 = 1
            yield b
    return x


def if_nested():
    cond: Fin6 = sample([0.4, 0.2, 0.1])
    shared: Fin6 = sample([0.3, 0.3, 0.1])
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


def markov_with_rv_bound_in_branch():
    # A markov whose repeat count is a random variable, inside one branch of an
    # if-then-else. The bound is read by the loop *statement* but never by its
    # body, so it is not in fv(body); the loop's trailing marginalization must
    # pick it up via fv(bound), or this branch ends holding `n` while its sibling
    # does not and the two no longer share a post-context.
    x: Bool = sample([0.3, 0.7])
    n: Fin3 = sample([0.2, 0.3, 0.5])
    with z:
        if x == 0:
            t1: Bool = sample([0.4, 0.6])
            yield t1
        else:
            s0: Bool = sample([0.6, 0.4])
            with markov(sm := s0, repeat=n):
                t2: Bool = sample(0.9 if t2 == sm else 0.1)
                yield t2
            yield sm
    return z


def rv_bound_dead_after_loop():
    # Same mechanism without a branch: `n` is the loop bound and is dead after
    # the loop, so the trailing marginalization must release it.
    n: Fin3 = sample([0.2, 0.3, 0.5])
    s0: Bool = sample([0.6, 0.4])
    with markov(sm := s0, repeat=n):
        t: Bool = sample(0.8 if t == sm else 0.2)
        yield t
    return sm


def markov_with_if_inside():
    # A loop whose body branches (caesar-like structure without conditioning).
    key: Bool = sample([0.5, 0.5])
    with markov(k := key, repeat=3):
        flip: Bool = sample([0.6, 0.4])
        with newk:
            if flip:
                t1: Bool = k
                yield t1
            else:
                t2: Bool = sample([0.5, 0.5])
                yield t2
        yield newk
    return k


def markov_init_block_full_form():
    # Full init-block form: the init block has a local of its own and reads an
    # outer variable that is dead after it.
    w: Fin3 = sample([0.2, 0.3, 0.5])
    with markov(x, repeat=3):
        with init():
            u: Bool = sample(0.8 if u == (w == 2) else 0.2)
            yield u
        xn: Bool = sample(0.7 if xn == x else 0.3)
        yield xn
    return x


def markov_permuted_yield():
    # The body yields the old `bb` as the new `aa` and a fresh draw as the new `bb`.
    a: Bool = sample([0.5, 0.5])
    b: Bool = sample([0.9, 0.1])
    with markov(aa := a, bb := b, repeat=3):
        c: Bool = sample(0.7 if c == aa else 0.3)
        yield bb, c
    return aa, bb


def rv_bound_correlated_with_state():
    # The loop count also initializes the state, so the mixture over counts must
    # slice the joint distribution by the count (Lean's `repeat`).
    n: Fin3 = sample([0.2, 0.3, 0.5])
    with markov(s, repeat=n):
        with init():
            s0: Fin3 = n
            yield s0
        t: Fin3 = (s + 1) % 3
        yield t
    return n, s


PROGRAMS = [
    rv_bound_correlated_with_state,
    markov_init_block_full_form,
    markov_permuted_yield,
    straight_line_dead_intermediates,
    condition_mask_only,
    condition_then_marginalize_chained,
    observe_literal,
    observe_variable_equality,
    markov_dirac,
    markov_flip,
    markov_multi_state,
    markov_repeat_param,
    markov_body_reads_enclosing,
    if_simple,
    if_sample_inside,
    if_branch_reads_shared,
    if_nested,
    markov_with_rv_bound_in_branch,
    rv_bound_dead_after_loop,
    markov_with_if_inside,
]


@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
def test_lva_preserves_marginals(prog):
    """Eager marginalization (lva=True) yields the same distribution as the
    scope-exit reference (lva=False)."""
    with_lva = compile_pal_fn(prog, lva=True)()
    without_lva = compile_pal_fn(prog, lva=False)()
    np.testing.assert_allclose(
        np.asarray(with_lva), np.asarray(without_lva), rtol=1e-6, atol=1e-9
    )
