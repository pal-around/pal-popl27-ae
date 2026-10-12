"""Eager marginalization of loops (pal/src/eager.py): the previous loop state is an
ordinary variable of the body, marginalized where it dies."""

import contextlib
import io

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn, parse_ir

from pal import Bool, Fin, init, markov, sample
from pal.src.syntax import StmtKill, StmtMarkov, StmtYield

from test_lva_equivalence import markov_multi_state

Fin3 = Fin(3)


def _loop(stmts):
    return next(s for s in stmts if isinstance(s, StmtMarkov))


def _position(block, var):
    """Index of the kill of ``var`` in ``block``, or None."""
    return next((i for i, s in enumerate(block) if isinstance(s, StmtKill) and s.var == var), None)


def _peak_cells(prog, **kw) -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_pal_fn(prog, print_static_ndim_peak=True, **kw)
    for line in buf.getvalue().splitlines():
        if "Static peak cells" in line:
            return int(line.rsplit(":", 1)[1])
    raise AssertionError("peak cells were not reported")


def state_read_then_dead():
    x: Bool = sample([0.3, 0.7])
    with markov(y := x, repeat=3):
        z: Bool = sample(0.9 if z == y else 0.2)
        w: Fin3 = sample([0.2, 0.3, 0.5] if z == 0 else [0.6, 0.3, 0.1])
        v: Bool = sample(0.5 if w == 2 else 0.1)
        yield v
    return y


def test_previous_state_is_killed_after_its_last_read():
    loop = _loop(parse_ir(state_read_then_dead, reorder=False))
    body = loop.statements
    k = _position(body, "y")
    assert k is not None, body
    assert isinstance(body[k - 1], type(body[0])) and body[k - 1].var == "z"
    assert isinstance(body[-1], StmtYield)
    np.testing.assert_allclose(
        compile_pal_fn(state_read_then_dead, lva=True)(), compile_pal_fn(state_read_then_dead, lva=False)()
    )


def state_read_only_by_branch_condition():
    x: Bool = sample([0.3, 0.7])
    with markov(y := x, repeat=3):
        with z:
            if y == 0:
                a: Bool = sample([0.9, 0.1])
                yield a
            else:
                b: Bool = sample([0.4, 0.6])
                yield b
        yield z
    return y


def test_state_read_only_by_condition_is_killed_at_branch_entry():
    loop = _loop(parse_ir(state_read_only_by_branch_condition, reorder=False))
    branch = loop.statements[0]
    for block in branch.blocks:
        assert _position(block, "y") == 0, block
    np.testing.assert_allclose(
        compile_pal_fn(state_read_only_by_branch_condition, lva=True)(),
        compile_pal_fn(state_read_only_by_branch_condition, lva=False)(),
    )


def unread_state_live_after_loop():
    # The body never reads `aa`, but it is returned, so it cannot be pruned; the
    # previous value dies at body entry and the body yields a fresh one.
    a: Bool = sample([0.3, 0.7])
    b: Bool = sample([0.5, 0.5])
    with markov(aa := a, bb := b, repeat=2):
        c: Bool = sample(0.8 if c == bb else 0.1)
        d: Bool = sample([0.6, 0.4])
        yield d, c
    return aa, bb


def test_unread_state_is_killed_at_body_entry():
    loop = _loop(parse_ir(unread_state_live_after_loop, reorder=False))
    assert _position(loop.statements, "aa") == 0, loop.statements
    r = compile_pal_fn(unread_state_live_after_loop, lva=True)()
    np.testing.assert_allclose(r, compile_pal_fn(unread_state_live_after_loop, lva=False)())
    # The returned tensor is indexed [aa, bb]; aa is a fresh [0.6, 0.4] draw.
    np.testing.assert_allclose(np.asarray(r).sum(axis=1), [0.6, 0.4])


def init_source_dead_after_init():
    w: Fin3 = sample([0.2, 0.3, 0.5])
    with markov(x, repeat=2):
        with init():
            x0: Bool = sample(0.9 if w == 0 else 0.3)
            yield x0
        xn: Bool = sample(0.7 if xn == x else 0.3)
        yield xn
    return x


def test_init_source_is_killed_inside_the_init_block():
    stmts = parse_ir(init_source_dead_after_init, reorder=False)
    loop = _loop(stmts)
    assert _position(loop.init_block, "w") is not None, loop.init_block
    assert all(not (isinstance(s, StmtKill) and s.var == "w") for s in stmts)
    np.testing.assert_allclose(
        compile_pal_fn(init_source_dead_after_init, lva=True)(),
        compile_pal_fn(init_source_dead_after_init, lva=False)(),
    )


def four_state_ring():
    # Four loop variables, each next value reading a different previous one. Every
    # previous value dies at its single reader, so the tensor never holds more than the
    # state plus one new draw.
    a: Bool = sample([0.5, 0.5])
    b: Bool = sample([0.4, 0.6])
    c: Bool = sample([0.3, 0.7])
    d: Bool = sample([0.2, 0.8])
    with markov(aa := a, bb := b, cc := c, dd := d, repeat=3):
        na: Bool = sample(0.9 if na == bb else 0.1)
        nb: Bool = sample(0.8 if nb == cc else 0.2)
        nc: Bool = sample(0.7 if nc == dd else 0.3)
        nd: Bool = sample(0.6 if nd == aa else 0.4)
        yield na, nb, nc, nd
    return aa, bb, cc, dd


def test_loop_state_peak_stays_one_axis_above_the_state():
    """A k-variable loop peaks at k+1 axes: each previous state variable is summed out
    once it is last read, so for k = 4 the peak is 2^5 = 32 cells."""
    assert _peak_cells(four_state_ring, lva=True, reorder=False) == 32
    np.testing.assert_allclose(
        compile_pal_fn(four_state_ring, lva=True)(), compile_pal_fn(four_state_ring, lva=False)()
    )


def test_multi_state_loop_peak_is_one_state_axis_below_naive():
    # Each state is killed as soon as it is read, so the two state axes never
    # coexist with both new draws: the peak is 2^3 cells.
    assert _peak_cells(markov_multi_state, lva=True, reorder=False) == 8
