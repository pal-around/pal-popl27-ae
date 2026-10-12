from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .syntax import VarName

if TYPE_CHECKING:
    from .parse import ParsingContext


@dataclass
class SourceInfo:
    """Tracks the original source file/lines so error messages can reference them.

    `lines[0]` corresponds to file line `line_offset`.
    """

    filename: str = "<unknown>"
    lines: list[str] = field(default_factory=list)
    line_offset: int = 1

    def file_lineno(self, lineno: int) -> int:
        return self.line_offset + lineno - 1

    def source_line(self, lineno: int) -> str:
        if 0 < lineno <= len(self.lines):
            return self.lines[lineno - 1].rstrip("\n")
        return ""

    def format_error(self, lineno: int, message: str) -> str:
        line = self.source_line(lineno)
        header = f'File "{self.filename}", line {self.file_lineno(lineno)}'
        if line:
            return f"{header}\n    {line.lstrip()}\n{message}"
        return f"{header}\n{message}"


@contextmanager
def source_error_context(source: SourceInfo, lineno: int):
    """Catch parse/compile errors and decorate them with source context.

    Only the innermost call adds context; outer wrappers re-raise unchanged.
    """
    try:
        yield
    except Exception as e:
        if getattr(e, "_pal_source_attached", False):
            raise
        formatted = source.format_error(lineno, str(e))
        try:
            new_err = type(e)(formatted)
        except Exception:
            new_err = type(e).__new__(type(e))
            new_err.args = (formatted,)
        setattr(new_err, "_pal_source_attached", True)
        raise new_err from None


def np_scan(f, init, xs, length=None):
    """jax.lax.scan for the NumPy backend, including a tuple of per-iteration inputs.

    Each input's leading axis is the iteration, and the body sees one row of each per
    iteration -- a tuple, in the same order, as JAX gives it for a pytree of ``xs``."""
    if xs is None:
        assert length is not None
        rows = [None] * length
    elif isinstance(xs, tuple):
        lengths = {len(x) for x in xs}
        assert len(lengths) == 1, f"markov xs have different leading dimensions: {sorted(lengths)}"
        assert length is None or lengths == {length}, (
            f"markov xs have leading dimension {lengths.pop()}, but the loop runs {length} times"
        )
        rows = list(zip(*xs))
    else:
        assert length is None or len(xs) == length
        rows = xs
    carry = init
    for x in rows:
        carry, _ = f(carry, x)
    return carry, None


def generate_param_list(ctxt: ParsingContext) -> str:
    params = []
    for var, default in ctxt.scalar_params.items():
        if default is None:
            params.append(f"{var}")
        else:
            params.append(f"{var}={default}")
    return ", ".join(params)


def generate_static_argnames_kwarg(static_argnames: set[VarName]) -> str | None:
    if not static_argnames:
        return None
    args = ", ".join(f'"{arg}"' for arg in static_argnames)
    return f"static_argnames=({args},)"


def static_vars(**kwargs):
    def decorator(f: Callable) -> Callable:
        for k in kwargs:
            setattr(f, k, kwargs[k])
        return f

    return decorator
