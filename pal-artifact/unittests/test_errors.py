"""Tests that PAL surfaces expected syntax and compilation errors.

These call ``_pal_impl`` directly (rather than going through ``@pal``) so the
original ``SyntaxError`` / ``CompilationError`` propagates out without being
turned into a ``SystemExit`` by the user-facing wrapper.
"""

import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, init, markov, sample
from pal.src.compile import CompilationError

Fin3 = Fin(3)


# ---------------------------------------------------------------------------
# Top-level structural errors
# ---------------------------------------------------------------------------


def test_missing_return_statement_is_rejected():
    def f():
        x: Bool = sample([0.5, 0.5])

    with pytest.raises(SyntaxError, match="PAL functions must have a return statement"):
        compile_pal_fn(f)


def test_statements_after_return_are_rejected():
    def f():
        x: Bool = sample([0.5, 0.5])
        return x
        y: Bool = sample([0.5, 0.5])

    with pytest.raises(SyntaxError, match="No statements allowed after return statement"):
        compile_pal_fn(f)


def test_return_must_be_variables_only():
    def f():
        x: Bool = sample([0.5, 0.5])
        return 42  # numeric literal is not a variable

    with pytest.raises(SyntaxError, match="Return values must be variables"):
        compile_pal_fn(f)


def test_return_variable_names_must_be_unique():
    def f():
        x: Bool = sample([0.5, 0.5])
        return x, x

    with pytest.raises(SyntaxError, match="Return variable names must be unique"):
        compile_pal_fn(f)


# ---------------------------------------------------------------------------
# Expression-level parse errors
# ---------------------------------------------------------------------------


def test_empty_probability_table_is_rejected():
    def f():
        x: Bool = sample([])
        return x

    with pytest.raises(SyntaxError, match="Probability table cannot be empty"):
        compile_pal_fn(f)


# ---------------------------------------------------------------------------
# Markov errors
# ---------------------------------------------------------------------------


def test_markov_repeat_must_be_nonnegative():
    def f():
        x: Bool = sample([0.5, 0.5])
        with markov(y := x, repeat=-1):
            z: Bool = sample(1 if z == y else 0)
            yield z

        return y

    with pytest.raises(SyntaxError, match="Markov repeat must be a non-negative integer literal or a scalar parameter"):
        compile_pal_fn(f)


def test_nested_markov_blocks_are_rejected():
    def f():
        x: Bool = sample([0.5, 0.5])
        with markov(y := x, repeat=2):
            with markov(z := y, repeat=2):
                w: Bool = sample(1 if w == z else 0)
                yield w

            yield y

        return y

    with pytest.raises(SyntaxError, match="Nested markov blocks are not supported"):
        compile_pal_fn(f)


def test_markov_plain_arguments_require_init_block():
    def f():
        x: Bool = sample([0.5, 0.5])
        with markov(y, repeat=2):
            z: Bool = sample([0.5, 0.5])
            yield z

        return y

    with pytest.raises(SyntaxError, match=r"must begin with a 'with init\(\):' block"):
        compile_pal_fn(f)


def test_markov_mixed_argument_forms_are_rejected():
    def f():
        x: Bool = sample([0.5, 0.5])
        with markov(y := x, z, repeat=2):
            a: Bool = sample([0.5, 0.5])
            b: Bool = sample([0.5, 0.5])
            yield a, b

        return y

    with pytest.raises(SyntaxError, match="must be all of the form 'var := init_var' or all plain variable names"):
        compile_pal_fn(f)


def test_markov_walrus_arguments_with_init_block_are_rejected():
    def f():
        x: Bool = sample([0.5, 0.5])
        with markov(y := x, repeat=2):
            with init():
                x0: Bool = sample([0.5, 0.5])
                yield x0
            z: Bool = sample([0.5, 0.5])
            yield z

        return y

    with pytest.raises(SyntaxError, match=r"cannot also have a 'with init\(\):' block"):
        compile_pal_fn(f)


def test_init_block_outside_markov_is_rejected():
    def f():
        with init():
            x0: Bool = sample([0.5, 0.5])
            yield x0
        return x0

    with pytest.raises(SyntaxError, match="only allowed as the first statement of a markov block"):
        compile_pal_fn(f)


def test_init_block_later_in_markov_body_is_rejected():
    def f():
        with markov(y, repeat=2):
            with init():
                y0: Bool = sample([0.5, 0.5])
                yield y0
            z: Bool = sample([0.5, 0.5])
            with init():
                w: Bool = sample([0.5, 0.5])
                yield w
            yield z

        return y

    with pytest.raises(SyntaxError, match="only allowed as the first statement of a markov block"):
        compile_pal_fn(f)


def test_init_block_yield_arity_must_match_markov_variables():
    def f():
        with markov(y, z, repeat=2):
            with init():
                y0: Bool = sample([0.5, 0.5])
                yield y0
            a: Bool = sample([0.5, 0.5])
            b: Bool = sample([0.5, 0.5])
            yield a, b

        return y

    with pytest.raises(SyntaxError, match="Number of variables yielded by the init block must match"):
        compile_pal_fn(f)


def test_init_block_must_end_with_yield():
    def f():
        with markov(y, repeat=2):
            with init():
                y0: Bool = sample([0.5, 0.5])
            z: Bool = sample([0.5, 0.5])
            yield z

        return y

    with pytest.raises(SyntaxError, match="Init block must end with a yield statement"):
        compile_pal_fn(f)


def test_markov_inside_init_block_is_rejected():
    def f():
        x: Bool = sample([0.5, 0.5])
        with markov(y, repeat=2):
            with init():
                with markov(u := x, repeat=2):
                    v: Bool = sample(1 if v == u else 0)
                    yield v
                yield u
            z: Bool = sample([0.5, 0.5])
            yield z

        return y

    with pytest.raises(SyntaxError, match="Nested markov blocks are not supported"):
        compile_pal_fn(f)


def test_init_and_body_must_yield_the_same_types():
    def f():
        with markov(y, repeat=2):
            with init():
                y0: Fin3 = sample([0.2, 0.3, 0.5])
                yield y0
            z: Bool = sample([0.5, 0.5])
            yield z

        return y

    with pytest.raises(CompilationError, match="markov body yields values of types"):
        compile_pal_fn(f)


def test_markov_block_must_contain_yield():
    def f():
        x: Bool = sample([0.5, 0.5])
        with markov(y := x, repeat=3):
            z: Bool = sample(1 if z == y else 0)

        return y

    with pytest.raises(SyntaxError, match="Markov block must contain at least one yield statement"):
        compile_pal_fn(f)


# ---------------------------------------------------------------------------
# If-then-else errors
# ---------------------------------------------------------------------------


def test_if_then_else_must_yield_at_end():
    def f():
        cond: Bool = sample([0.5, 0.5])
        with out:
            if cond == 0:
                a: Bool = 0
            else:
                b: Bool = 1
                yield b

        return out

    with pytest.raises(SyntaxError, match="If-then-else block must end with a yield statement"):
        compile_pal_fn(f)


def test_if_then_else_yield_arity_must_match_return_vars():
    def f():
        cond: Bool = sample([0.5, 0.5])
        with x, y:
            if cond == 0:
                a: Bool = 0
                yield a  # only one variable yielded, but two return vars declared
            else:
                b: Bool = 0
                c: Bool = 1
                yield b, c

        return x, y

    with pytest.raises(
        SyntaxError,
        match="Number of yielded variables in if-then-else blocks must match number of return variables",
    ):
        compile_pal_fn(f)


# ---------------------------------------------------------------------------
# Compile-time errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lva", [True, False])
def test_redefining_a_variable_in_scope_is_rejected(lva):
    """Variable names must be fresh within their scope, whether or not eager
    marginalization is enabled; the check runs right after yield normalization,
    before reordering and marginalization."""
    from pal.src.eager import IRWellFormednessError

    def f():
        x: Bool = sample([0.5, 0.5])
        x: Bool = sample([0.3, 0.7])
        return x

    with pytest.raises(IRWellFormednessError, match=r"\['x'\] already defined"):
        compile_pal_fn(f, lva=lva)


def test_reserved_double_underscore_prefix_is_rejected():
    def f():
        __x: Bool = sample([0.5, 0.5])
        return __x

    with pytest.raises(CompilationError, match=r"Variable __x cannot begin with reserved prefix '__'"):
        compile_pal_fn(f)
