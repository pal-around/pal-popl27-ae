"""Marginalization at scope exit (pal/src/scope.py), used when eager marginalization
is off, and yield normalization, used in both modes."""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn, parse_ir

from pal import Bool, init, markov, sample
from pal.src.syntax import StmtAssignSingle, StmtIfThenElse, StmtKill, StmtMarkov, StmtReturn, StmtYield


def _kills_before_terminal(block):
    """Variables killed by the run of kills right before the block's last statement."""
    assert isinstance(block[-1], (StmtYield, StmtReturn))
    killed = []
    for s in reversed(block[:-1]):
        if not isinstance(s, StmtKill):
            break
        killed.append(s.var)
    return list(reversed(killed))


def _all_kills(block):
    return [s.var for s in block if isinstance(s, StmtKill)]


def test_kills_before_return_in_sorted_order():
    def f():
        b: Bool = sample([0.5, 0.5])
        a: Bool = sample(0.9 if a == b else 0.1)
        c: Bool = sample(0.8 if c == a else 0.2)
        return c

    stmts = parse_ir(f, lva=False, reorder=False)
    assert _kills_before_terminal(stmts) == ["a", "b"]
    assert _all_kills(stmts) == ["a", "b"]


def test_loop_body_kills_its_locals_and_the_previous_state_before_the_yield():
    def f():
        x: Bool = sample([0.5, 0.5])
        with markov(y := x, repeat=2):
            w: Bool = sample([0.5, 0.5])
            z: Bool = sample(0.9 if z == y else 0.1)
            yield z
        return y

    stmts = parse_ir(f, lva=False, reorder=False)
    loop = next(s for s in stmts if isinstance(s, StmtMarkov))
    assert _kills_before_terminal(loop.statements) == ["w", "y"]
    assert _all_kills(loop.statements) == ["w", "y"]
    assert _kills_before_terminal(stmts) == ["x"]


def test_init_block_kills_its_locals_before_the_yield():
    def f():
        with markov(x, repeat=2):
            with init():
                u: Bool = sample([0.5, 0.5])
                x0: Bool = sample(0.9 if x0 == u else 0.1)
                yield x0
            xn: Bool = sample(0.8 if xn == x else 0.2)
            yield xn
        return x

    stmts = parse_ir(f, lva=False, reorder=False)
    loop = next(s for s in stmts if isinstance(s, StmtMarkov))
    assert _kills_before_terminal(loop.init_block) == ["u"]
    assert _kills_before_terminal(loop.statements) == ["x"]


def test_each_branch_kills_only_its_own_locals():
    def f():
        c: Bool = sample([0.5, 0.5])
        w: Bool = sample([0.3, 0.7])
        with out:
            if c == 0:
                t: Bool = sample(0.9 if t == w else 0.1)
                s: Bool = sample(0.8 if s == t else 0.2)
                yield s
            else:
                r: Bool = sample([0.5, 0.5])
                yield r
        return out

    stmts = parse_ir(f, lva=False, reorder=False)
    branch = next(s for s in stmts if isinstance(s, StmtIfThenElse))
    assert _kills_before_terminal(branch.blocks[0]) == ["t"]
    assert _kills_before_terminal(branch.blocks[1]) == []
    outer = {"c", "w"}
    for block in branch.blocks:
        assert not outer & set(_all_kills(block)), "a branch killed an outer variable"


@pytest.mark.parametrize("lva", [True, False])
def test_if_branch_yields_enclosing_variable(lva):
    # Branch 1 yields `x` itself, which is still in scope after the if. The yield is
    # rewritten to yield a fresh copy of it.
    def f():
        x: Bool = sample([0.3, 0.7])
        c: Bool = sample([0.5, 0.5])
        with z:
            if c == 0:
                yield x
            else:
                t: Bool = 1 - x
                yield t
        return z, x

    stmts = parse_ir(f, lva=lva, reorder=False)
    branch = next(s for s in stmts if isinstance(s, StmtIfThenElse))
    copies = [s for s in branch.blocks[0] if isinstance(s, StmtAssignSingle) and s.internal]
    assert len(copies) == 1 and copies[0].source_var == "x"
    assert branch.blocks[0][-1].vars == [copies[0].target_var]

    # z = x on c = 0 and z = 1 - x on c = 1, each with probability 1/2.
    r = compile_pal_fn(f, lva=lva)()
    np.testing.assert_allclose(r, [[0.15, 0.35], [0.15, 0.35]])
