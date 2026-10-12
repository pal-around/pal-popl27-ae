"""Finer-grained eager marginalization for if-then-else (the paper's
branch-entry ``marg(P_i)`` rule).

Each branch marginalizes, at its entry, the variables live at the if that this
branch does not read and that are not live after the if -- the condition's reads
and the sibling branches' reads::

    P_i = (fv(e) | U_{j!=i} fv(blk_j)) \\ (fv(blk_i) | (L \\ x))

The tests check that ``lva=True`` computes the same distribution as ``lva=False``
across many if-shapes, and that branch-entry marginalization keeps the peak tensor
small.
"""

import contextlib
import io

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, condition, markov, observe, sample

Fin3 = Fin(3)
Fin6 = Fin(6)


# ---------------------------------------------------------------------------
# Correctness fixtures: lva=True must equal lva=False.
# ---------------------------------------------------------------------------

def if_condition_only_dead():
    # `route` is read only by the condition and is dead after the if: P_i = {route}
    # in *both* branches, so it is summed out at each branch entry.
    route: Fin6 = sample([0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
    with out:
        if route == 0:
            a: Bool = sample([0.7, 0.3])
            yield a
        else:
            b: Bool = sample([0.2, 0.8])
            yield b
    return out


def if_sibling_only_read():
    # `s1` is read only by the truthy branch, `s2` only by the falsy branch; both
    # dead after. Each is in the *other* branch's P_i (entry marg) and killed after
    # use in its own branch.
    cond: Bool = sample([0.5, 0.5])
    s1: Bool = sample([0.6, 0.4])
    s2: Bool = sample([0.3, 0.7])
    with out:
        if cond == 0:
            a: Bool = sample(0.9 if a == s1 else 0.1)
            yield a
        else:
            b: Bool = sample(0.9 if b == s2 else 0.1)
            yield b
    return out


def if_condition_live_after():
    # `cond` is returned, hence live after the if: it must NOT be marginalized in
    # any branch (cond in L\x removes it from every P_i).
    cond: Bool = sample([0.6, 0.4])
    with out:
        if cond == 0:
            a: Bool = sample([0.7, 0.3])
            yield a
        else:
            b: Bool = sample([0.2, 0.8])
            yield b
    return out, cond


def if_outside_read_both():
    # `w` is read by both branches, dead after: in neither P_i (both read it); each
    # branch kills it after its last use via the ordinary block translation.
    cond: Bool = sample([0.5, 0.5])
    w: Bool = sample([0.7, 0.3])
    with out:
        if cond == 0:
            a: Bool = sample(0.8 if a == w else 0.2)
            yield a
        else:
            b: Bool = sample(0.2 if b == w else 0.8)
            yield b
    return out


def if_outside_read_and_live_after():
    # `w` is read by one branch AND live after: stays live through the if.
    cond: Bool = sample([0.5, 0.5])
    w: Bool = sample([0.7, 0.3])
    with out:
        if cond == 0:
            a: Bool = sample(0.8 if a == w else 0.2)
            yield a
        else:
            b: Bool = sample([0.5, 0.5])
            yield b
    return out, w


def if_three_way():
    # elif chain: three branches, each with its own sibling read.
    sel: Fin3 = sample([0.5, 0.3, 0.2])
    p: Bool = sample([0.6, 0.4])
    q: Bool = sample([0.1, 0.9])
    with out:
        if sel == 0:
            a: Bool = sample(0.9 if a == p else 0.1)
            yield a
        elif sel == 1:
            b: Bool = sample(0.9 if b == q else 0.1)
            yield b
        else:
            c: Bool = sample([0.5, 0.5])
            yield c
    return out


def if_condition_multivar():
    # Condition reads two variables, both dead after.
    x: Bool = sample([0.5, 0.5])
    y: Bool = sample([0.6, 0.4])
    with out:
        if x == y:
            a: Bool = sample([0.7, 0.3])
            yield a
        else:
            b: Bool = sample([0.2, 0.8])
            yield b
    return out


def if_nested_condition_dead():
    # Outer and inner conditions both dead after their ifs.
    outer: Fin3 = sample([0.5, 0.3, 0.2])
    inner: Bool = sample([0.4, 0.6])
    with y:
        if outer == 0:
            a: Bool = sample([0.9, 0.1])
            yield a
        else:
            with z:
                if inner == 0:
                    c: Bool = sample([0.3, 0.7])
                    yield c
                else:
                    d: Bool = sample([0.5, 0.5])
                    yield d
            yield z
    return y


def if_in_loop_condition_dead():
    # caesar-like: an if inside a markov loop whose condition (`flip`) is a
    # body-local, dead after the if. P_i marginalizes it inside each branch on
    # every iteration.
    key: Bool = sample([0.5, 0.5])
    with markov(k := key, repeat=4):
        flip: Bool = sample([0.6, 0.4])
        with newk:
            if flip == 0:
                t1: Bool = sample(0.9 if t1 == k else 0.1)
                yield t1
            else:
                t2: Bool = sample([0.5, 0.5])
                yield t2
        yield newk
    return k


PROGRAMS = [
    if_condition_only_dead,
    if_sibling_only_read,
    if_condition_live_after,
    if_outside_read_both,
    if_outside_read_and_live_after,
    if_three_way,
    if_condition_multivar,
    if_nested_condition_dead,
    if_in_loop_condition_dead,
]


@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
def test_finer_grained_if_preserves_marginals(prog):
    """Eager marginalization (lva=True) yields the same distribution as the
    scope-exit reference (lva=False)."""
    with_lva = np.asarray(compile_pal_fn(prog, lva=True)())
    without_lva = np.asarray(compile_pal_fn(prog, lva=False)())
    np.testing.assert_allclose(with_lva, without_lva, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Peak memory: the condition's and sibling branches' variables are summed out at the
# start of each branch, so they never coexist with the branch body's draws.
# ---------------------------------------------------------------------------

Fin20 = Fin(20)


def if_bigcond_bigbody():
    # A high-cardinality condition variable, and a high-cardinality draw in each
    # branch: route(20) is summed out before v(20) is drawn, so the peak is a single
    # 20-cell axis, not 20 x 20.
    route: Fin20 = sample([1.0 / 20] * 20)
    with out:
        if route == 0:
            v: Fin20 = sample([1.0 / 20] * 20)
            yield v
        else:
            w: Fin20 = sample([1.0 / 20] * 20)
            yield w
    return out


def _peak_cells(prog) -> int:
    """Actual static peak memory (product of cardinalities) under eager marg."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_pal_fn(prog, lva=True, print_static_ndim_peak=True)
    for line in buf.getvalue().splitlines():
        if "Static peak cells" in line:
            return int(line.rsplit(":", 1)[1])
    raise AssertionError("peak cells were not reported")


@pytest.mark.parametrize(
    "prog, finer_cells, coarse_cells",
    [
        # route(Fin6) summed out inside each branch: peak is route alone (6), never
        # route x drawn-value (6 x 2 = 12).
        (if_condition_only_dead, 6, 12),
        # 20x reduction: route(20) alone vs route(20) x v(20).
        (if_bigcond_bigbody, 20, 400),
    ],
)
def test_finer_grained_if_reduces_peak_memory(prog, finer_cells, coarse_cells):
    # coarse_cells is the size of the condition's and the drawn variable's axes
    # together, computed by hand.
    assert finer_cells < coarse_cells
    assert _peak_cells(prog) == finer_cells
