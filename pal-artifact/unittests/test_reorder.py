"""Cardinality-weighted reordering of factor blocks (pal/src/reorder.py).

The pass reorders each block -- including loop bodies, init blocks and branches -- to a
lower-peak topological order. Reordering respects data dependencies, so it must not
change the distribution; it only lowers peak memory. These tests check both:
``--reorder`` yields the same marginal as no reordering (and composes with eager
marginalization), and it reduces the peak on a badly-ordered block.
"""

import contextlib
import io

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn, parse_ir

from pal import Bool, Fin, condition, markov, observe, sample

# Reuse the representative program batteries from the other suites: reordering
# (with eager marginalization) must not change any of their marginals. This
# exercises the pass on loops, branches, observations and conditioning, which it
# recurses into or treats as factor nodes.
from test_lva_equivalence import PROGRAMS as LVA_PROGRAMS
from test_eager_if import PROGRAMS as IF_PROGRAMS

Fin4 = Fin(4)
Fin3 = Fin(3)


# ---------------------------------------------------------------------------
# Fixtures. `*_bad_order` sample all the roots first, then their consumers --
# the order that keeps every root live at once (high peak). Reordering should
# interleave root/consumer pairs.
# ---------------------------------------------------------------------------

def flat_bad_order():
    a0: Fin4 = sample([0.4, 0.3, 0.2, 0.1])
    a1: Fin4 = sample([0.1, 0.2, 0.3, 0.4])
    a2: Fin4 = sample([0.25, 0.25, 0.25, 0.25])
    b0: Bool = sample(0.9 if b0 == a0 else 0.1)
    b1: Bool = sample(0.9 if b1 == a1 else 0.1)
    b2: Bool = sample(0.9 if b2 == a2 else 0.1)
    e: Bool = sample(0.7 if b0 == b1 else 0.3)
    return e, b2


def flat_with_barren():
    # `dead` is not upstream of the return; reordering must still preserve the marginal.
    root: Fin4 = sample([0.4, 0.3, 0.2, 0.1])
    used: Bool = sample(0.9 if used == root else 0.1)
    dead: Bool = sample(0.5 if dead == root else 0.5)
    return used


def collider_chain():
    x0: Bool = sample([0.6, 0.4])
    x1: Bool = sample([0.5, 0.5])
    x2: Bool = sample([0.3, 0.7])
    y0: Bool = sample(0.8 if y0 == x0 else 0.2)
    y1: Bool = sample(0.8 if y1 == x1 else 0.2)
    z: Bool = sample(0.9 if y0 == y1 else 0.1)
    w: Bool = sample(0.9 if w == x2 else 0.1)
    return z, w


def markov_program():
    x: Bool = sample([0.7, 0.3])
    with markov(y := x, repeat=4):
        z: Bool = sample(1 if z == y else 0)
        yield z
    return y


def flat_with_evidence():
    # Bayes-net-shaped block with an observation. Reordering must keep the
    # evidence and its ancestors and still yield the same conditioned marginal.
    a: Fin4 = sample([0.4, 0.3, 0.2, 0.1])
    b: Fin4 = sample([0.1, 0.2, 0.3, 0.4])
    g: Bool = sample(0.8 if g == a else 0.2)
    h: Bool = sample(0.7 if h == b else 0.3)
    observe(g, 1)
    return h


def observe_after_reader():
    # `x` is read by `y` and observed; whichever order the pass picks must give the
    # same conditioned marginal.
    x: Fin4 = sample([0.4, 0.3, 0.2, 0.1])
    y: Bool = sample(0.8 if y == x else 0.2)
    observe(x, 1)
    return y


def flat_with_condition():
    a: Fin4 = sample([0.4, 0.3, 0.2, 0.1])
    b: Fin4 = sample([0.1, 0.2, 0.3, 0.4])
    condition(a >= b)
    return b


def flat_barren_before_markov():
    # `noise` is barren (never read, not returned) and sits before a loop: the pass must
    # recurse into the loop without breaking it.
    seed: Bool = sample([0.6, 0.4])
    noise: Fin4 = sample([0.25, 0.25, 0.25, 0.25])
    with markov(y := seed, repeat=3):
        z: Bool = sample(1 if z == y else 0)
        yield z
    return y


def multiple_conditions():
    # Every condition is a plain multiplicative factor, so the pass is free to
    # reorder them among the samples they depend on.
    a: Fin3 = sample([0.5, 0.3, 0.2])
    b: Fin3 = sample([0.1, 0.4, 0.5])
    c: Fin3 = sample([0.2, 0.3, 0.5])
    condition(a >= b)
    condition(b >= c)
    return b


FLAT_PROGRAMS = [flat_bad_order, flat_with_barren, collider_chain]
EXTRA_PROGRAMS = [
    markov_program,
    flat_with_evidence,
    observe_after_reader,
    flat_with_condition,
    flat_barren_before_markov,
    multiple_conditions,
]
ALL_PROGRAMS = FLAT_PROGRAMS + EXTRA_PROGRAMS


def _peak_cells(prog, **kw) -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_pal_fn(prog, print_static_ndim_peak=True, **kw)
    for line in buf.getvalue().splitlines():
        if "Static peak cells" in line:
            return int(line.rsplit(":", 1)[1])
    raise AssertionError("peak cells not reported")


@pytest.mark.parametrize("prog", ALL_PROGRAMS, ids=lambda p: p.__name__)
def test_reorder_preserves_marginal(prog):
    """Reordering must not change the distribution, on its own or combined with
    eager marginalization."""
    ref = np.asarray(compile_pal_fn(prog, reorder=False, lva=False)())
    for reorder in (True, False):
        for lva in (True, False):
            got = np.asarray(compile_pal_fn(prog, reorder=reorder, lva=lva)())
            np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "prog", LVA_PROGRAMS + IF_PROGRAMS, ids=lambda p: p.__name__
)
def test_reorder_preserves_marginal_on_representative_programs(prog):
    """The full representative battery (loops, branches, observe, condition,
    nested) must be unchanged by reordering combined with eager marginalization."""
    ref = np.asarray(compile_pal_fn(prog, reorder=False, lva=False)())
    got = np.asarray(compile_pal_fn(prog, reorder=True, lva=True)())
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_reorder_reduces_peak_on_bad_order():
    without = _peak_cells(flat_bad_order, reorder=False)
    with_reorder = _peak_cells(flat_bad_order, reorder=True)
    assert with_reorder < without


def test_reorder_is_noop_on_markov_peak():
    # The statements around the loop are already in their best order, so reordering
    # (which recurses into the loop's blocks) leaves the peak unchanged.
    assert _peak_cells(markov_program, reorder=True) == _peak_cells(markov_program, reorder=False)


# Dropping barren factors is a separate, opt-in pass; see test_prune.py.
