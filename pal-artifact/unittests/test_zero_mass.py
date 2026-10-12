"""The opt-in zero-mass report (``--print-zero-mass``, pal/src/compile.py).

PAL's normalization is total: a parent assignment whose weights do not sum to a
positive, finite total contributes zero mass, exactly like ``condition(False)``. That
keeps an unreachable, unspecified table row harmless, but it also means a mis-specified
weight table shows up as reduced evidence rather than as an error. The flag reports
where that happens, without changing what the program computes."""

import contextlib
import io
import re

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, condition, markov, sample

Fin3 = Fin(3)

BANNER = "==================== Compiled program ===================="


def _program(prog, **kw) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_pal_fn(prog, print_compiled=True, **kw)
    return buf.getvalue().split(BANNER, 1)[1]


def _run(prog, **kw) -> tuple[object, str]:
    """Run the compiled program, returning its result and whatever it printed."""
    buf = io.StringIO()
    fn = compile_pal_fn(prog, **kw)
    with contextlib.redirect_stdout(buf):
        result = fn()
    return result, buf.getvalue()


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

def healthy():
    x: Fin3 = sample([0.5, 0.3, 0.2])
    y: Bool = sample(0.9 if x == 0 else 0.1)
    return y


def zero_row_reachable():
    # For root != 0 every weight of `bad` is zero, and those values of root carry mass,
    # so this statement drops 0.5 of the total.
    root: Fin3 = sample([0.5, 0.3, 0.2])
    bad: Bool = sample(1 if root == 0 else 0)
    return bad


def zero_row_unreachable():
    # The condition removes every root except 0 before `bad` is sampled, so the rows
    # whose weights are all zero carry no mass and nothing is dropped.
    root: Fin3 = sample([0.5, 0.3, 0.2])
    condition(root == 0)
    bad: Bool = sample(1 if root == 0 else 0)
    return bad


def zero_row_in_loop():
    x: Bool = sample([1.0, 0.0])
    with markov(y := x, repeat=3):
        z: Bool = sample(1 if y == 1 else 0)
        yield z
    return y


def assign_only():
    x: Bool = sample([0.3, 0.7])
    y: Bool = x
    return y


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prog", [healthy, zero_row_reachable, assign_only], ids=lambda p: p.__name__)
def test_off_emits_no_check(prog):
    src = _program(prog)
    assert "zm_lost" not in src
    assert "wsum" not in src


def test_off_is_the_default_program():
    assert _program(zero_row_reachable) == _program(zero_row_reachable, print_zero_mass=False)


# ---------------------------------------------------------------------------
# On: what gets reported
# ---------------------------------------------------------------------------

def test_reports_the_variable_line_and_amount():
    result, out = _run(zero_row_reachable, print_zero_mass=True)
    assert "PAL zero mass: sample bad" in out, out
    # The message names this file and the line of the offending sample.
    assert re.search(r"test_zero_mass\.py:\d+\) dropped\s+0\.5", out), out
    # `bad`'s weights do not mention `bad`, so where they are not all zero (root = 0)
    # both of its values are equally likely: 0.5 of the mass survives, split evenly.
    np.testing.assert_allclose(np.asarray(result), [0.25, 0.25])


def test_healthy_program_reports_nothing():
    _, out = _run(healthy, print_zero_mass=True)
    assert "zero mass" not in out


def test_nothing_reported_when_the_zero_row_carries_no_mass():
    _, out = _run(zero_row_unreachable, print_zero_mass=True)
    assert "zero mass" not in out, out


def test_no_check_for_an_assign():
    src = _program(assign_only, print_zero_mass=True)
    assert "zm_lost_y" not in src
    assert "zm_lost_x" in src  # the sample is still checked


def test_reports_inside_a_loop_body():
    src = _program(zero_row_in_loop, print_zero_mass=True)
    body = src[src.index("def __scan") :]
    assert "zm_lost_z" in body, body


# ---------------------------------------------------------------------------
# The result never changes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prog", [healthy, zero_row_reachable, zero_row_unreachable, zero_row_in_loop], ids=lambda p: p.__name__
)
def test_the_flag_does_not_change_the_result(prog):
    with_flag, _ = _run(prog, print_zero_mass=True)
    without, _ = _run(prog)
    np.testing.assert_allclose(np.asarray(with_flag), np.asarray(without), rtol=1e-12)


def test_reports_under_jit():
    import jax

    fn = compile_pal_fn(zero_row_reachable, numpy=False, jit=True, print_zero_mass=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
        jax.effects_barrier()  # jax.debug.print is asynchronous
    assert "PAL zero mass: sample bad" in buf.getvalue(), buf.getvalue()
