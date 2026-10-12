"""Constant weight tables (``--table-lowering``, pal/src/compile.py).

A weight expression that is a pure function of the variables it reads -- every
Bayesian-network conditional probability table -- is computed once at compile time and
read with a single gather. The table reaches the compiled program as a global.

These tests pin that the lowered program computes the same distribution as the
expression form, that the table really is out of the source, and that the unsupported
cases fall back instead of compiling something wrong."""

import contextlib
import io
import re

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, condition, markov, sample

Fin3 = Fin(3)
Fin4 = Fin(4)
Shifted = [1, 2, 3]  # a support whose values are not their own positions

BANNER = "==================== Compiled program ===================="
TABLES_BANNER = "==================== Weight tables ===================="


def _printed(prog, **kw) -> str:
    """Everything the compiler prints: the program, then the table listing."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_pal_fn(prog, print_compiled=True, **kw)
    return buf.getvalue().split(BANNER, 1)[1]


def _program(prog, **kw) -> str:
    """The generated program alone, without the table listing that follows it."""
    return _printed(prog, **kw).split(TABLES_BANNER, 1)[0]


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

def flat_bayes_net():
    pollution: Fin3 = sample([0.5, 0.4, 0.1])
    smoker: Bool = sample([0.3, 0.7])
    cancer: Bool = sample(
        (0.030001 if cancer == 0 else 0.969999)
        if pollution == 0 and smoker == 0
        else (0.050001 if cancer == 0 else 0.949999)
    )
    xray: Bool = sample(0.900001 if xray == cancer else 0.099999)
    return xray


def guarded_table():
    parent: Bool = sample([0.25, 0.75])
    child: Fin3 = sample(parent > [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    return child


def non_adjacent_parent():
    # `unrelated` sits between the parent and the sampled variable in the context, so
    # the gather's index arrays must land on their own axes.
    parent: Bool = sample([0.4, 0.6])
    unrelated: Fin4 = sample([0.1, 0.2, 0.3, 0.4])
    child: Fin3 = sample(0.7 if child == parent else 0.15)
    return child, unrelated, parent


def table_in_loop_body():
    x: Bool = sample([0.3, 0.7])
    with markov(y := x, repeat=3):
        z: Bool = sample(0.8 if z == y else 0.2)
        yield z
    return y


def table_in_branches():
    c: Bool = sample([0.5, 0.5])
    with out:
        if c == 0:
            a: Fin3 = sample([0.2, 0.3, 0.5])
            yield a
        else:
            b: Fin3 = sample(0.6 if b == 1 else 0.2)
            yield b
    return out


def with_evidence():
    x: Fin3 = sample([0.5, 0.3, 0.2])
    y: Bool = sample(0.9 if x == 0 else 0.1)
    condition(y == 1)
    return x


PROGRAMS = [
    flat_bayes_net,
    guarded_table,
    non_adjacent_parent,
    table_in_loop_body,
    table_in_branches,
    with_evidence,
]


# ---------------------------------------------------------------------------
# The lowered program computes the same thing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
@pytest.mark.parametrize("lva", [True, False])
@pytest.mark.parametrize("reorder", [True, False])
def test_table_matches_expression_path(prog, lva, reorder):
    lowered = np.asarray(compile_pal_fn(prog, lva=lva, reorder=reorder)())
    plain = np.asarray(compile_pal_fn(prog, lva=lva, reorder=reorder, table_lowering=False)())
    np.testing.assert_allclose(lowered, plain, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
def test_table_matches_expression_path_on_jax(prog):
    lowered = np.asarray(compile_pal_fn(prog, numpy=False, jit=True)())
    plain = np.asarray(compile_pal_fn(prog, numpy=False, jit=True, table_lowering=False)())
    np.testing.assert_allclose(lowered, plain, rtol=1e-12, atol=1e-15)


def test_float32_paths_agree():
    lowered = np.asarray(compile_pal_fn(flat_bayes_net, dtype="float32")())
    plain = np.asarray(compile_pal_fn(flat_bayes_net, dtype="float32", table_lowering=False)())
    # The table is evaluated in double and cast once; the expression path computes in
    # float32 throughout, so the two agree only to float32 precision.
    np.testing.assert_allclose(lowered, plain, rtol=1e-6)


# ---------------------------------------------------------------------------
# The table is out of the source
# ---------------------------------------------------------------------------

def test_weights_are_gathered_and_not_in_the_source():
    src = _program(flat_bayes_net)
    assert re.search(r"weight_cancer = __tbl_\d+_cancer\[", src), src
    # No table is assigned in the program: it arrives through the globals.
    assert not re.search(r"^\s*__tbl_\w+\s*=", src, flags=re.M), src
    # And the probabilities themselves are gone from the text.
    for literal in ("0.030001", "0.969999", "0.900001"):
        assert literal not in src


def test_expression_path_keeps_the_literals():
    src = _program(flat_bayes_net, table_lowering=False)
    assert "0.030001" in src
    assert "__tbl_" not in src


def test_lowering_shrinks_the_program():
    assert len(_program(flat_bayes_net)) < len(_program(flat_bayes_net, table_lowering=False))


def test_table_listing_is_printed():
    out = _printed(flat_bayes_net)
    assert TABLES_BANNER in out
    assert re.search(r"__tbl_\d+_xray: shape \(2, 2\), 4 cells", out), out


def test_gather_appears_inside_the_scan_body():
    src = _program(table_in_loop_body)
    body = src[src.index("def __scan") :]
    assert re.search(r"weight_z = __tbl_\d+_z\[", body), body


# ---------------------------------------------------------------------------
# Fallbacks: cases that must keep the expression form
# ---------------------------------------------------------------------------

def scalar_param_weight(p=0.3):
    x: Bool = sample([p, 1 - p])
    return x


def non_positional_support():
    x: Shifted = sample([0.2, 0.3, 0.5])
    return x


@pytest.mark.parametrize("prog", [scalar_param_weight, non_positional_support], ids=lambda p: p.__name__)
def test_unsupported_weights_fall_back(prog):
    src = _program(prog)
    assert "__tbl_" not in src
    np.testing.assert_allclose(
        np.asarray(compile_pal_fn(prog)()),
        np.asarray(compile_pal_fn(prog, table_lowering=False)()),
        rtol=1e-12,
    )


def test_cell_limit_falls_back(monkeypatch):
    import pal.src.consts as consts

    # 1 cell: every table in the program, down to a parentless two-value one, is too big.
    monkeypatch.setattr(consts, "TABLE_CELL_LIMIT", 1)
    src = _program(flat_bayes_net)
    assert "__tbl_" not in src
    np.testing.assert_allclose(
        np.asarray(compile_pal_fn(flat_bayes_net)()),
        np.asarray(compile_pal_fn(flat_bayes_net, table_lowering=False)()),
        rtol=1e-12,
    )


def test_assign_is_not_lowered():
    # `x`'s weights are a table, but the assign's one-hot weights are not worth one.
    def f():
        x: Bool = sample([0.3, 0.7])
        y: Bool = x
        return y

    src = _program(f)
    assert re.search(r"weight_y = \(np\.ones_like", src), src
    assert not re.search(r"weight_y = __tbl_", src), src
