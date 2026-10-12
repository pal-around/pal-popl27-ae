"""Shared helpers for PAL unit tests.

Compiles PAL functions with the NumPy backend (no JIT, no printing) so that
tests can assert on returned arrays directly.
"""

from pal import _pal_impl

DEFAULT_KWARGS = dict(
    print_final_likelihood=False,
    numpy=True,
    jit=False,
)


def compile_pal_fn(func, **overrides):
    kwargs = {**DEFAULT_KWARGS, **overrides}
    return _pal_impl(func, **kwargs)


def parse_ir(func, **overrides):
    """The statements PAL hands to the code generator for ``func``: after parsing,
    yield normalization, well-formedness checking, reordering and marginalization.
    For structural assertions about the front end."""
    from pal.src.main import front_end

    kwargs = {**DEFAULT_KWARGS, **overrides}
    parsing_ctxt, _, _ = front_end(func, kwargs)
    return parsing_ctxt.pal_stmts
