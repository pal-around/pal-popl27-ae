"""Pruning of barren factors (``--prune``, pal/src/prune.py).

A factor no returned or yielded variable and no evidence depends on may be dropped, but
only if it marginalizes to 1. An ``assign`` always does; a ``sample`` does only when its
weights cannot sum to zero, because PAL's normalization is total and a zero row
contributes zero mass, exactly like ``condition(False)``.

The pass is off by default, so these tests ask for it explicitly, and one checks that it
stays off unless asked."""

import contextlib
import io

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn, parse_ir

from pal import Bool, Fin, markov, sample
from pal.src.syntax import StmtSample

Fin3 = Fin(3)
Fin4 = Fin(4)


def _peak_cells(prog, **kw) -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_pal_fn(prog, print_static_ndim_peak=True, **kw)
    for line in buf.getvalue().splitlines():
        if "Static peak cells" in line:
            return int(line.rsplit(":", 1)[1])
    raise AssertionError("peak cells not reported")


def _has_sample(stmts, name: str) -> bool:
    return any(isinstance(s, StmtSample) and s.var == name for s in stmts)


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

def flat_with_barren():
    # `dead` is not upstream of the return, and its weights are positive.
    root: Fin4 = sample([0.4, 0.3, 0.2, 0.1])
    used: Bool = sample(0.9 if used == root else 0.1)
    dead: Bool = sample(0.5 if dead == root else 0.5)
    return used


def barren_before_markov():
    # `noise` is barren and sits before a loop: the pass must drop it and recurse into
    # the loop without breaking it.
    seed: Bool = sample([0.6, 0.4])
    noise: Fin4 = sample([0.25, 0.25, 0.25, 0.25])
    with markov(y := seed, repeat=3):
        z: Bool = sample(1 if z == y else 0)
        yield z
    return y


def barren_with_ancestor():
    # `mid` is read only by the barren `dead`, so dropping `dead` makes `mid` barren
    # too: the closure must remove both.
    root: Bool = sample([0.3, 0.7])
    kept: Bool = sample(0.8 if kept == root else 0.2)
    mid: Bool = sample(0.6 if mid == root else 0.4)
    dead: Bool = sample(0.5 if dead == mid else 0.5)
    return kept


def barren_nonnormalizing():
    # `dead` is barren, but its weights are all zero unless root == 0, so it contributes
    # zero mass there: dropping it would put that mass back and change the answer.
    root: Fin3 = sample([0.5, 0.3, 0.2])
    dead: Bool = sample(1 if root == 0 else 0)
    return root


def barren_positive():
    root: Fin3 = sample([0.5, 0.3, 0.2])
    dead: Bool = sample([0.25, 0.75])
    return root


PROGRAMS = [flat_with_barren, barren_before_markov, barren_with_ancestor, barren_positive]


# ---------------------------------------------------------------------------
# Off unless asked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
def test_barren_factors_are_kept_by_default(prog):
    assert _has_sample(parse_ir(prog), "dead") or _has_sample(parse_ir(prog), "noise")


# ---------------------------------------------------------------------------
# What pruning removes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
def test_pruning_drops_the_barren_factor(prog):
    stmts = parse_ir(prog, prune=True)
    assert not _has_sample(stmts, "dead")
    assert not _has_sample(stmts, "noise")


def test_pruning_drops_the_barren_factor_s_ancestors():
    stmts = parse_ir(barren_with_ancestor, prune=True)
    assert not _has_sample(stmts, "dead")
    assert not _has_sample(stmts, "mid")
    assert _has_sample(stmts, "kept") and _has_sample(stmts, "root")


@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
@pytest.mark.parametrize("lva", [True, False])
@pytest.mark.parametrize("reorder", [True, False])
def test_pruning_preserves_the_marginal(prog, lva, reorder):
    ref = np.asarray(compile_pal_fn(prog, prune=False, reorder=False, lva=False)())
    got = np.asarray(compile_pal_fn(prog, prune=True, reorder=reorder, lva=lva)())
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_pruning_lowers_the_peak_where_marginalization_is_deferred():
    """With eager marginalization the peak is the same either way: a barren draw dies at
    its own sample, so it never coexists with anything. What pruning saves there is the
    work -- a weight tensor, a normalization, a multiply and a sum. The peak win shows
    up with ``lva=False``, where a barren variable stays live until the return."""
    assert _peak_cells(flat_with_barren, prune=True, lva=True) == _peak_cells(
        flat_with_barren, prune=False, lva=True
    )
    assert _peak_cells(flat_with_barren, prune=True, lva=False) < _peak_cells(
        flat_with_barren, prune=False, lva=False
    )


# ---------------------------------------------------------------------------
# Soundness: a barren sample that does not marginalize to 1 is kept
# ---------------------------------------------------------------------------

def test_barren_sample_with_a_zero_row_is_kept():
    stmts = parse_ir(barren_nonnormalizing, prune=True, reorder=True, lva=True)
    assert _has_sample(stmts, "dead"), stmts
    # Mass survives only where root = 0, with or without the pass.
    ref = np.asarray(compile_pal_fn(barren_nonnormalizing, prune=False, reorder=False, lva=False)())
    got = np.asarray(compile_pal_fn(barren_nonnormalizing, prune=True, reorder=True, lva=True)())
    np.testing.assert_allclose(got, ref, rtol=1e-12)
    np.testing.assert_allclose(ref, [0.5, 0.0, 0.0], rtol=1e-12)
