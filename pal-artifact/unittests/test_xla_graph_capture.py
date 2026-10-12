"""Tests for the opt-in ``xla_graph_capture`` kwarg.

The option lowers XLA's CUDA-graph capture threshold for the compiled function
(``xla_gpu_graph_min_graph_size``) and takes effect only on a top-level ``jax.jit``.
The tests check that a nested call still computes the right answer and that PAL
reports that the option does not apply. They assert on the emitted program text
where possible, so they need no GPU.
"""

import io
import warnings
from contextlib import redirect_stdout

import jax
import jax.numpy as jnp
import pytest

from pal import Bool, Fin, sample
from pal import _pal_impl
from pal.src.main import _GraphCaptureGuard

Fin3 = Fin(3)


def _emit(func, **overrides):
    """Compile with the JAX backend and return the generated program text."""
    kwargs = {"jit": True, "numpy": False, "print_compiled": True, **overrides}
    buf = io.StringIO()
    with redirect_stdout(buf):
        _pal_impl(func, **kwargs)
    return buf.getvalue()


def _prog():
    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.5, 0.5])
        return x

    return f


# ---------------------------------------------------------------------------
# What gets emitted
# ---------------------------------------------------------------------------


def test_default_emits_plain_jit():
    """Without xla_graph_capture the decorator is a bare @jax.jit."""
    text = _emit(_prog())
    assert "@jax.jit" in text
    assert "compiler_options" not in text


def test_opt_in_emits_compiler_options():
    text = _emit(_prog(), xla_graph_capture=True)
    assert "compiler_options" in text
    assert "xla_gpu_graph_min_graph_size" in text
    assert "partial(jax.jit" in text


def test_default_and_disabled_emit_identical_programs():
    assert _emit(_prog()) == _emit(_prog(), xla_graph_capture=False)


def test_numpy_backend_never_emits_the_option():
    """numpy=True forces jit=False, so there is no decorator to carry it."""
    text = _emit(_prog(), numpy=True, jit=False, xla_graph_capture=True)
    assert "compiler_options" not in text


# ---------------------------------------------------------------------------
# Composition under an outer trace
# ---------------------------------------------------------------------------


def test_default_composes_under_an_outer_jit():
    f = _pal_impl(_prog(), jit=True, numpy=False)
    outer = jax.jit(lambda: jnp.sum(f()))
    assert float(outer()) == pytest.approx(float(jnp.sum(f())))


def test_default_composes_under_grad():
    def g(t: float):
        x: Bool = sample([t, 1.0])
        return x

    f = _pal_impl(g, jit=True, numpy=False)
    d = jax.jit(jax.grad(lambda t: jnp.sum(f(t) * jnp.arange(2, dtype=jnp.float64))))
    assert jnp.isfinite(d(2.0))


def test_opt_in_nested_still_computes_the_right_answer_and_says_so():
    """A nested call computes the same answer, and PAL reports that the option does
    not apply: a PAL warning when JAX ignores the option, a PAL error when JAX
    rejects it."""
    f = _pal_impl(_prog(), jit=True, numpy=False, xla_graph_capture=True)
    expected = float(jnp.sum(f()))  # top level: the option applies

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            got = float(jax.jit(lambda: jnp.sum(f()))())
        except ValueError as e:
            # JAX rejected the option on the nested jit; PAL's message names it.
            assert "xla_graph_capture" in str(e)
            assert "top-level" in str(e)
            return

    assert got == pytest.approx(expected)
    assert any("xla_graph_capture" in str(w.message) for w in caught), (
        "nested call silently lost the option without warning"
    )


def test_guard_rewrites_a_rejecting_jax_error():
    """The guard's message for a rejected option, checked against a stub."""

    def raises_like_a_rejecting_jax(*args, **kwargs):
        raise ValueError("compiler_options can only be passed to top-level jax.jit")

    guard = _GraphCaptureGuard(raises_like_a_rejecting_jax, "my_prog")

    with pytest.raises(ValueError, match="my_prog") as excinfo:
        guard()
    message = str(excinfo.value)
    assert "xla_graph_capture" in message
    assert "top-level" in message
    # The original is kept as the cause rather than discarded.
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_guard_passes_through_an_unrelated_value_error_unchanged():
    def raises_something_else(*args, **kwargs):
        raise ValueError("something else entirely")

    guard = _GraphCaptureGuard(raises_something_else, "my_prog")
    with pytest.raises(ValueError, match="something else entirely"):
        guard()


def test_opt_in_still_works_at_top_level():
    f = _pal_impl(_prog(), jit=True, numpy=False, xla_graph_capture=True)
    assert float(jnp.sum(f())) == pytest.approx(1.0)


def test_guard_delegates_attributes():
    """The proxy must not hide the jitted-function API."""
    f = _pal_impl(_prog(), jit=True, numpy=False, xla_graph_capture=True)
    assert hasattr(f, "lower")
    assert callable(f.lower)


def test_guard_does_not_swallow_unrelated_value_errors():
    f = _pal_impl(_prog(), jit=True, numpy=False, xla_graph_capture=True)
    with pytest.raises(TypeError):
        f(1, 2, 3)  # wrong arity -> must propagate, not be rewritten
