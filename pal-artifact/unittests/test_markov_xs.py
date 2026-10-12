"""Per-iteration inputs to a markov block (``xs=(p := theta)``).

``xs`` gives the body one row of a parameter per iteration (one trainable probability
per step, say): the parameter's leading axis is the iteration, as ``jax.lax.scan``
slices its inputs. These tests check the semantics against the unrolled model, on both
backends, and what is rejected."""

import re

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn, parse_ir

from pal import Bool, Fin, markov, sample

Fin3 = Fin(3)
Fin4 = Fin(4)

THETA = np.array([0.1, 0.25, 0.5, 0.75])
ROWS = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

def chain_loop(theta):
    # Step i flips with probability theta[i] of staying True once True.
    x0: Bool = False
    with markov(x := x0, repeat=4, xs=(p := theta)):
        y: Bool = sample(p if x else [1 - p, p])
        yield y
    return x


def chain_unrolled(a, b, c, d):
    x0: Bool = False
    x1: Bool = sample(a if x0 else [1 - a, a])
    x2: Bool = sample(b if x1 else [1 - b, b])
    x3: Bool = sample(c if x2 else [1 - c, c])
    x4: Bool = sample(d if x3 else [1 - d, d])
    return x4


def table_loop(rows):
    # A row of numbers is the whole weight: this step's table over the domain.
    x0: Fin4 = 0
    with markov(x := x0, repeat=3, xs=(w := rows)):
        y: Fin4 = sample(w)
        yield y
    return x


def two_inputs(theta, rows):
    x0: Fin4 = 0
    t0: Bool = False
    with markov(x := x0, t := t0, repeat=3, xs=(p := theta, w := rows)):
        y: Fin4 = sample(w)
        u: Bool = sample([1 - p, p])
        yield y, u
    return x, t


# ---------------------------------------------------------------------------
# The loop equals the unrolled model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "xla"])
@pytest.mark.parametrize("lva", [True, False])
def test_scalar_rows_match_the_unrolled_model(numpy, lva):
    got = np.asarray(compile_pal_fn(chain_loop, numpy=numpy, lva=lva)(THETA))
    want = np.asarray(compile_pal_fn(chain_unrolled, numpy=numpy, lva=lva)(*THETA))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "xla"])
def test_a_row_of_numbers_is_the_step_s_table(numpy):
    # Every step's weights are independent of the state, so the answer is the last row.
    got = np.asarray(compile_pal_fn(table_loop, numpy=numpy)(ROWS))
    np.testing.assert_allclose(got, ROWS[-1], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "xla"])
def test_two_inputs_are_bound_in_order(numpy):
    got = np.asarray(compile_pal_fn(two_inputs, numpy=numpy)(THETA[:3], ROWS))
    # The two carries are independent: the last row for x, and theta[-1] for t.
    np.testing.assert_allclose(got.sum(axis=0), [1 - THETA[2], THETA[2]], rtol=1e-12)
    np.testing.assert_allclose(got.sum(axis=1), ROWS[-1], rtol=1e-12)


def test_gradients_match_the_unrolled_model():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    loop = compile_pal_fn(chain_loop, numpy=False)
    unrolled = compile_pal_fn(chain_unrolled, numpy=False)
    theta = jnp.asarray(THETA)
    got = jax.grad(lambda t: loop(t)[1])(theta)
    want = jax.grad(lambda t: unrolled(t[0], t[1], t[2], t[3])[1])(theta)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# What the emitted program looks like
# ---------------------------------------------------------------------------

def test_the_inputs_become_the_scan_s_xs(capsys):
    compile_pal_fn(chain_loop, numpy=True, print_compiled=True)
    program = capsys.readouterr().out
    assert "xs=(theta,)" in program
    # The body takes the row tuple and unpacks this iteration's row from it.
    assert re.search(r"def __scan\d+\(carry, __xs\d+\):", program)
    assert re.search(r"\(p,\) = __xs\d+", program)


def test_a_loop_without_inputs_is_unchanged(capsys):
    def plain():
        x: Bool = sample([0.7, 0.3])
        with markov(y := x, repeat=3):
            z: Bool = sample(1 if z == y else 0)
            yield z
        return y

    compile_pal_fn(plain, numpy=True, print_compiled=True)
    program = capsys.readouterr().out
    assert "xs=None" in program
    assert re.search(r"def __scan\d+\(carry, _\):", program)
    assert "__xs" not in program


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def test_a_row_inside_a_larger_expression_must_be_a_scalar():
    def bad(rows):
        x0: Fin4 = 0
        with markov(x := x0, repeat=3, xs=(w := rows)):
            y: Fin4 = sample(w if x == 0 else 1)
            yield y
        return x

    # A scalar row is fine in an expression; an array row is caught where its shape is
    # known -- as the program runs, or as JAX traces it.
    compile_pal_fn(bad, numpy=True)(np.array([0.5, 0.25, 0.75]))
    with pytest.raises(AssertionError, match="use it as the whole weight"):
        compile_pal_fn(bad, numpy=True)(ROWS)


def test_a_wrong_leading_dimension_is_an_error():
    with pytest.raises(AssertionError, match="the loop runs 4 times"):
        compile_pal_fn(chain_loop, numpy=True)(THETA[:2])


def test_a_random_repeat_count_cannot_take_inputs():
    def bad(theta):
        n: Fin3 = sample([0.2, 0.3, 0.5])
        x0: Bool = False
        with markov(x := x0, repeat=n, xs=(p := theta)):
            y: Bool = sample([1 - p, p])
            yield y
        return x

    with pytest.raises(SyntaxError, match="needs a fixed repeat count"):
        parse_ir(bad)


def test_the_source_must_be_a_parameter():
    def bad():
        x0: Bool = False
        with markov(x := x0, repeat=2, xs=(p := THETA)):
            y: Bool = sample([1 - p, p])
            yield y
        return x

    with pytest.raises(SyntaxError, match="must be a parameter of the model function"):
        parse_ir(bad)


def test_a_row_name_cannot_be_reused_by_a_variable():
    def bad(theta):
        x0: Bool = False
        with markov(x := x0, repeat=2, xs=(p := theta)):
            p: Bool = sample([0.5, 0.5])
            y: Bool = sample([0.5, 0.5])
            yield y
        return x

    with pytest.raises(SyntaxError, match="shadow a markov xs name"):
        parse_ir(bad)


def test_unknown_keywords_are_rejected():
    def bad(theta):
        x0: Bool = False
        with markov(x := x0, repeat=2, ys=theta):
            y: Bool = sample([0.5, 0.5])
            yield y
        return x

    with pytest.raises(SyntaxError, match="optional 'xs' keyword"):
        parse_ir(bad)
