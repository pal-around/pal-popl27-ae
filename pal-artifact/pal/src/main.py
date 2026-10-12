import argparse
import ast
import inspect
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path
from pprint import pprint
from typing import Callable

from .compile import LIKELIHOOD_TENSOR, LOG_SCALE_VAR, CompileContext, compile_pal
from .helpers import (
    SourceInfo,
    generate_param_list,
    generate_static_argnames_kwarg,
    np_scan,
    static_vars,
)
from .eager import lva
from .parse import ParsingContext, parse_pal, parse_pal_parameters

Bool = [False, True]


def Fin(n: int) -> list[int]:
    """Finite type with n elements."""
    return list(range(n))


""" Bool should be equivalent to Fin(2) """
assert Fin(2) == Bool

Fin2 = Fin(2)
Fin3 = Fin(3)
Fin4 = Fin(4)
Fin5 = Fin(5)
Fin6 = Fin(6)
Fin7 = Fin(7)
Fin8 = Fin(8)
Fin9 = Fin(9)


def _jit_compiler_options(kwargs: dict) -> str:
    """XLA options for the emitted ``@jax.jit``, as source text.

    With ``@pal(xla_graph_capture=True)``, XLA captures scan bodies with fewer than five
    kernels in CUDA graphs, which cuts their per-iteration launch overhead on the GPU.
    The option takes effect only on a top-level ``jax.jit`` (see ``_GraphCaptureGuard``).
    """
    if not kwargs.get("xla_graph_capture", False):
        return ""
    return 'compiler_options={"xla_gpu_graph_min_graph_size": 1}'


def _inside_an_outer_trace() -> bool:
    """True when called under an outer ``jax.jit`` / ``grad`` / ``vmap`` trace.

    ``trace_state_clean()`` is private JAX API; if it cannot be used, this reports
    False, which only skips the warning in ``_GraphCaptureGuard``.
    """
    try:
        from jax._src.core import trace_state_clean

        return not trace_state_clean()
    except Exception:
        return False


class _GraphCaptureGuard:
    """Wraps a function compiled with ``xla_graph_capture=True`` and warns when it is
    called inside an outer trace, where JAX ignores the option. ``__getattr__``
    delegates the rest of the jitted-function API (``.lower()``, ``.trace()``, ...)."""

    __slots__ = ("_pal_fn", "_pal_name")

    def __init__(self, fn, name):
        self._pal_fn = fn
        self._pal_name = name

    def __getattr__(self, item):
        return getattr(self._pal_fn, item)

    def _nested_use_message(self, verb: str) -> str:
        return (
            f"PAL: `{self._pal_name}` was compiled with xla_graph_capture=True, which "
            f"sets an XLA compiler option that only takes effect on a top-level "
            f"jax.jit -- but this call is inside an outer jax.jit / jax.grad / "
            f"jax.vmap, so the option {verb}. Either drop xla_graph_capture (PAL "
            "then composes normally, at the cost of CUDA-graph capture for small "
            'scan bodies), or pass compiler_options={"xla_gpu_graph_min_graph_size": 1} '
            "at your own outermost jax.jit instead."
        )

    def __call__(self, *args, **kwargs):
        if _inside_an_outer_trace():
            # JAX ignores the option here; warn, then run normally (the answers are
            # unaffected).
            warnings.warn(self._nested_use_message("is being ignored"), stacklevel=2)
        try:
            return self._pal_fn(*args, **kwargs)
        except ValueError as e:
            if "compiler_options" not in str(e):
                raise
            # JAX rejected the option because the call is nested; say why.
            raise ValueError(self._nested_use_message("cannot be applied")) from e


def _pal(f: Callable, **kwargs) -> Callable:
    try:
        return _pal_impl(f, **kwargs)
    except Exception as e:
        if getattr(e, "_pal_source_attached", False):
            # Print the decorated message without a traceback.
            sys.exit(f"{type(e).__name__}: {e}")
        raise


# Parsed source files, keyed by (path, length, hash of the text).
_MODULE_AST_CACHE: dict[tuple[str, int, int], ast.Module] = {}
_MODULE_AST_CACHE_MAX = 8


def _parse_module_cached(file_lines: list[str], src_file: str) -> ast.Module:
    """Parse a source file once and share the module among the ``@pal`` functions and
    recompilations that read it. The parsed module is only read, never modified."""
    src = "".join(file_lines)
    key = (src_file, len(src), hash(src))
    module = _MODULE_AST_CACHE.get(key)
    if module is None:
        module = ast.parse(src, filename=src_file)
        if len(_MODULE_AST_CACHE) >= _MODULE_AST_CACHE_MAX:
            _MODULE_AST_CACHE.clear()
        _MODULE_AST_CACHE[key] = module
    return module


def _find_function_ast(f: Callable, src_file: str) -> tuple[ast.stmt, SourceInfo]:
    """Return the AST FunctionDef for ``f`` and a SourceInfo for error reporting.

    Fast path: read the source file (via inspect's linecache-backed
    ``findsource``, which only runs a cheap regex, not the tokenizer) and parse
    the whole file once. The matching FunctionDef is found by name and starting
    line (``co_firstlineno`` points at the first decorator, or the ``def`` line
    if undecorated). Node line numbers are absolute, so ``line_offset`` is 1.

    Falls back to ``inspect.getsourcelines`` when the source can't be located or
    matched (e.g. functions defined in a string / REPL).
    """
    try:
        file_lines, _ = inspect.findsource(f)
        co_firstlineno = f.__code__.co_firstlineno
        target_name = f.__name__
        module = _parse_module_cached(file_lines, src_file)
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                if start == co_firstlineno:
                    return node, SourceInfo(filename=src_file, lines=file_lines, line_offset=1)
    except (OSError, TypeError, SyntaxError):
        pass

    # Fallback: tokenizer-based extraction (slower; used when the source file cannot
    # be parsed or the function is not found in it).
    raw_src_lines, src_start_lineno = inspect.getsourcelines(f)
    src = textwrap.dedent("".join(raw_src_lines))
    src_tree = ast.parse(src, filename=src_file).body[0]
    return src_tree, SourceInfo(filename=src_file, lines=src.splitlines(), line_offset=src_start_lineno)


def front_end(f: Callable, kwargs: dict) -> tuple[ParsingContext, SourceInfo, int | None]:
    """Parse ``f`` and run the front-end passes. Returns the parsing context, whose
    ``pal_stmts`` are ready for code generation, the source info, and the static lva
    peak (``None`` when eager marginalization is off).

    Passes, in order: parse; ``normalize_yields`` (every yield names variables of
    its own block); ``check_wellformed`` (use before definition, fresh names,
    scoping, yield arity); ``prune`` (optional, off by default); ``reorder``
    (optional); and kill insertion -- eager marginalization, or marginalization at
    scope exit when ``lva`` is off."""
    from .eager import check_wellformed
    from .scope import insert_scope_exit_kills, normalize_yields

    src_file = inspect.getsourcefile(f)
    assert src_file is not None

    # Locate the decorated function in the parsed source file. AST line numbers are
    # absolute file line numbers, so the source line offset is 1.
    src_tree, source_info = _find_function_ast(f, src_file)

    # Parse
    parsing_ctxt = ParsingContext(
        f.__globals__,
        source=source_info,
    )
    parse_pal_parameters(f, parsing_ctxt)
    parse_pal(src_tree, parsing_ctxt)

    if kwargs.get("print_parsed", False):
        print("==================== Parsed statements ====================\n")
        for s in parsing_ctxt.pal_stmts:
            pprint(s)
        print()

    # A markov block's per-iteration inputs are data, like scalar parameters, so every
    # front-end pass treats them the same way: opaque, and invisible to liveness.
    scalar_params = set(parsing_ctxt.scalar_params) | set(parsing_ctxt.xs_params)
    normalize_yields(parsing_ctxt.pal_stmts)

    # Well-formedness of the source IR (use before definition, fresh names, block
    # scoping, yield arity), in both lva modes; reorder and kill insertion rely on
    # names being fresh within a block.
    check_wellformed(parsing_ctxt.pal_stmts, scalar_params=scalar_params)

    if kwargs.get("prune", False):
        # Drop factors that cannot affect the query (off by default).
        from .prune import prune

        parsing_ctxt.pal_stmts = prune(
            parsing_ctxt.pal_stmts,
            scalar_params,
            f.__globals__,
            frozenset(parsing_ctxt.non_deterministic_params),
        )

    if kwargs.get("reorder", True):
        # Reorder factor blocks to a lower-peak topological order. Runs before any
        # kill is inserted.
        from .reorder import reorder

        parsing_ctxt.pal_stmts = reorder(
            parsing_ctxt.pal_stmts,
            scalar_params,
            f.__globals__,
            frozenset(parsing_ctxt.non_deterministic_params),
        )

    lva_peak: int | None = None
    if kwargs.get("lva", True):
        # Eager marginalization: kill each variable right after its last use.
        lva_peak = lva(parsing_ctxt.pal_stmts, scalar_params=scalar_params)
        if kwargs.get("print_lva_peak", False):
            print(f"LVA peak live variables: {lva_peak}")
        header = "After kill injection"
    else:
        # Marginalize each block's variables where they go out of scope.
        insert_scope_exit_kills(parsing_ctxt.pal_stmts)
        if kwargs.get("print_lva_peak", False):
            print("LVA peak live variables: (lva disabled)")
        header = "After scope-exit kill insertion"

    if kwargs.get("print_parsed", False):
        print(f"==================== {header} ====================\n")
        for s in parsing_ctxt.pal_stmts:
            pprint(s)
        print()

    return parsing_ctxt, source_info, lva_peak


def _pal_impl(f: Callable, **kwargs) -> Callable:
    n_pal_warmup = kwargs.pop("n_pal_warmup", 0)

    # The NumPy backend and JIT are mutually exclusive: `numpy=True` swaps `np`
    # for NumPy in the generated program, and wrapping that in @jax.jit would
    # trace it back into JAX arrays -- silently in float32, since x64 is only
    # enabled on the JAX path below.
    if kwargs.get("numpy", False):
        kwargs["jit"] = False
    else:
        import jax

        jax.config.update("jax_enable_x64", kwargs.get("dtype", "float64") == "float64")

    if n_pal_warmup > 0:
        if not kwargs.get("numpy", False):
            try:
                import jax
                import jax.numpy as jnp

                jax.devices()
                jax.jit(lambda x: x + 1)(jnp.zeros(1)).block_until_ready()
            except Exception:
                pass
        warmup_kwargs = {k: (False if k.startswith("print_") else v) for k, v in kwargs.items()}
        for _ in range(n_pal_warmup):
            _pal_impl(f, **warmup_kwargs)

    _pal_compile_start = time.perf_counter()
    parsing_ctxt, source_info, _ = front_end(f, kwargs)

    # Compile
    compile_ctxt = CompileContext(
        kwargs=kwargs,
        non_deterministic_params=parsing_ctxt.non_deterministic_params,
        scalar_params=parsing_ctxt.scalar_params,
        xs_params=set(parsing_ctxt.xs_params),
        source=source_info,
        func_globals=f.__globals__,
    )
    if kwargs.get("numpy", False):
        compile_ctxt.compiled_lines.append('np.seterr(invalid="ignore")')
    compile_pal(compile_ctxt, parsing_ctxt.pal_stmts)

    if kwargs.get("print_static_ndim_peak", False):
        print(f"Static peak ndim: {compile_ctxt.peak_ndim}")
        _itemsize = 8 if kwargs.get("dtype", "float64") == "float64" else 4
        print(f"Static peak cells: {compile_ctxt.peak_cells}")
        print(f"Static peak memory bytes: {compile_ctxt.peak_cells * _itemsize}")

    if kwargs.get("print_device", False):
        if kwargs.get("numpy", False):
            print("Device: numpy (CPU)")
        else:
            import jax

            print(f"JAX backend: {jax.default_backend()}")
            print(f"JAX devices: {jax.devices()}")

    # Print and return likelihood tensor.
    #
    # Under ``rescale=True`` the tensor is divided down to a representable range and
    # the divisors are accumulated in LOG_SCALE_VAR, so that
    # ``true likelihood == likelihood_tensor * exp(log_scale)``. The returned tensor is
    # the rescaled one, proportional to the true likelihood; ``print_final_likelihood``
    # reports the true log evidence log(sum) + log_scale, and ``return_log_scale``
    # returns log_scale.
    rescaling = kwargs.get("rescale", False)
    if kwargs.get("print_final_likelihood", False):
        if kwargs.get("numpy", False):
            compile_ctxt.compiled_lines.append(f"print(f'{{{LIKELIHOOD_TENSOR}}} : {{{LIKELIHOOD_TENSOR}.dtype}}')")
            if rescaling:
                compile_ctxt.compiled_lines.append(
                    f"print(f'log evidence: {{float(np.log({LIKELIHOOD_TENSOR}.sum()) + {LOG_SCALE_VAR})}}')"
                )
        else:
            compile_ctxt.compiled_lines.append(f"jax.debug.print('{{ll}} : {{ll.dtype}}', ll={LIKELIHOOD_TENSOR})")
            if rescaling:
                compile_ctxt.compiled_lines.append(
                    f"jax.debug.print('log evidence: {{le}}', "
                    f"le=np.log({LIKELIHOOD_TENSOR}.sum()) + {LOG_SCALE_VAR})"
                )
    if rescaling and kwargs.get("return_log_scale", False):
        compile_ctxt.compiled_lines.append(f"return {LIKELIHOOD_TENSOR}, {LOG_SCALE_VAR}")
    else:
        compile_ctxt.compiled_lines.append(f"return {LIKELIHOOD_TENSOR}")

    # Generate program
    program = ""

    if kwargs.get("jit", True):
        static_argnames = generate_static_argnames_kwarg(parsing_ctxt.static_argnames)
        jit_kwargs = [a for a in (static_argnames, _jit_compiler_options(kwargs)) if a]
        if jit_kwargs:
            program += f"@partial(jax.jit, {', '.join(jit_kwargs)})\n"
        else:
            program += "@jax.jit\n"

    program += f"""def _compiled_{f.__name__}({generate_param_list(parsing_ctxt)}):
{textwrap.indent("\n".join(compile_ctxt.compiled_lines), "    ")}
"""

    if kwargs.get("print_compiled", False):
        try:  # Try to format the program nicely if ruff is installed
            if shutil.which("ruff") is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / "tmp.py"
                    tmp_path.write_text(program)
                    subprocess.run(
                        ["ruff", "format", str(tmp_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    program = tmp_path.read_text()
        except Exception:
            pass
        print("==================== Compiled program ====================")
        print(program)
        if compile_ctxt.tables:
            print("==================== Weight tables ====================")
            for _table_name, _table in compile_ctxt.tables.items():
                print(f"{_table_name}: shape {_table.shape}, {_table.size} cells")
                if _table.size <= 64:
                    print(textwrap.indent(repr(_table.tolist()), "    "))

    locals = {}
    globals = f.__globals__.copy()

    if kwargs.get("numpy", False):
        import numpy as np

        globals["np"] = np
        globals["scan"] = np_scan
    else:
        from functools import partial

        import jax
        import jax.numpy as np

        globals["np"] = np
        globals["jax"] = jax
        globals["scan"] = jax.lax.scan
        globals["partial"] = partial

    # Constant weight tables reach the compiled program as globals. They are converted
    # with the backend's asarray: a traced index cannot index a NumPy array, and JAX
    # keeps a float64 table only with x64 enabled.
    _table_dtype = "float32" if kwargs.get("dtype", "float64") == "float32" else "float64"
    for _table_name, _table in compile_ctxt.tables.items():
        globals[_table_name] = np.asarray(_table, dtype=getattr(np, _table_dtype))

    exec(program, globals, locals)

    if kwargs.get("print_pal_compilation_time", False):
        print(f"PAL compilation time: {time.perf_counter() - _pal_compile_start:.10f}")

    compiled = locals[f"_compiled_{f.__name__}"]
    if kwargs.get("xla_graph_capture", False):
        compiled = _GraphCaptureGuard(compiled, f.__name__)
    if kwargs.get("print_device", False):
        is_numpy = kwargs.get("numpy", False)

        def _pal_print_device_wrapper(*args, **kw):
            return compiled(*args, **kw)

        _pal_print_device_wrapper._pal_print_device = True  # type: ignore[attr-defined]
        _pal_print_device_wrapper._pal_numpy = is_numpy  # type: ignore[attr-defined]
        return _pal_print_device_wrapper
    return compiled


def pal(f: Callable | None = None, /, **kwargs) -> Callable:
    """Compile a PAL program: ``@pal`` or ``@pal(option=value, ...)``.

    A program defined in the script being run (module ``__main__``) also takes the
    options given on the command line (see :func:`add_arguments`), e.g. ``--numpy``;
    options passed to the decorator take precedence. Other command-line arguments are
    left to the script.

    options:
        print_parsed: default = False
        print_compiled: default = False
        lva: default = True
        reorder: default = True
        prune: default = False
        table_lowering: default = True
        rescale: default = False
        return_log_scale: default = False
        print_zero_mass: default = False
        print_final_likelihood: default = False
        print_intermediate_likelihoods: default = False
        print_lva_peak: default = False
        print_static_ndim_peak: default = False
        print_intermediate_ndim_peak: default = False
        print_intermediate_ndim: default = False
        print_intermediate_shape: default = False
        print_device: default = False
        jit: default = True
        numpy: default = False (if True, overrides jit to False)
        dtype: default = "float64" (one of "float32", "float64")
        xla_graph_capture: default = False. GPU only; see _jit_compiler_options.
    """

    def decorate(f: Callable) -> Callable:
        options = _command_line_options() if f.__module__ == "__main__" else {}
        return _pal(f, **{**options, **kwargs})

    return decorate if f is None else decorate(f)


def _command_line_options() -> dict:
    """The compiler options given on the command line. Options not given, and arguments
    that are not compiler options, are left out."""
    parser = add_arguments(_GivenOnlyParser(argument_default=argparse.SUPPRESS, allow_abbrev=False))
    given, _ = parser.parse_known_args()
    return vars(given)


class _GivenOnlyParser(argparse.ArgumentParser):
    """A parser whose result holds only the options given."""

    def add_argument(self, *args, **kwargs):
        kwargs.pop("default", None)
        return super().add_argument(*args, **kwargs)


# ========================= Helpers =========================


@static_vars(counter=0)
def time_execution(f: Callable, *args):
    start_time = time.perf_counter()
    ret = f(*args)
    # JAX dispatches asynchronously; wait for the device so that the time covers the
    # execution. NumPy arrays have no block_until_ready and pass through. (Checking for
    # the method keeps `import jax` out of the timed region.)
    for _out in ret if isinstance(ret, tuple) else (ret,):
        _block = getattr(_out, "block_until_ready", None)
        if _block is not None:
            _block()
    end_time = time.perf_counter()

    time_execution.counter += 1  # type: ignore
    print(f"Run {time_execution.counter}: {end_time - start_time:.10f}")  # type: ignore

    if getattr(f, "_pal_print_device", False):
        if getattr(f, "_pal_numpy", False):
            print("Device: numpy (CPU)")
        else:
            import jax

            jax.debug.print("Device: {d}", d=ret.devices())
    return ret


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--print-parsed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the parsed PAL statements.",
    )
    parser.add_argument(
        "--print-compiled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the compiled Python program.",
    )
    parser.add_argument(
        "--lva",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eager marginalization: marginalize each random variable right after its "
        "last use (on by default). --no-lva marginalizes each block's variables only "
        "where the block ends (the end of a branch, loop body or init block, or the "
        "return); it computes the same distribution with a larger peak tensor.",
    )
    parser.add_argument(
        "--reorder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reorder factor blocks to a lower-peak topological order "
        "(cardinality-weighted). On by default.",
    )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop factors that cannot affect the queried variables (a barren sample "
        "or assign, and anything only they depend on). Off by default. A barren sample "
        "is dropped only when its weights provably cannot sum to zero.",
    )
    parser.add_argument(
        "--print-zero-mass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report each sample statement that drops mass: a parent assignment whose "
        "weights do not sum to a positive, finite total contributes zero mass (PAL's "
        "normalization is total). Off by default. Under JAX the report is an "
        "asynchronous jax.debug.print.",
    )
    parser.add_argument(
        "--table-lowering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute a weight expression that is a pure function of the variables it "
        "reads (every Bayesian-network conditional probability table) once at compile "
        "time and read it with a single gather. On by default.",
    )
    parser.add_argument(
        "--rescale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Divide the likelihood tensor by its max once per markov iteration and "
        "accumulate the log of the divisor, so that long chains do not underflow. Off by "
        "default. The returned tensor is proportional to the likelihood, and "
        "--print-final-likelihood reports the true log evidence.",
    )
    parser.add_argument(
        "--return-log-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --rescale, return (likelihood_tensor, log_scale) instead of just "
        "the tensor, so the caller can recover the absolute likelihood.",
    )
    parser.add_argument(
        "--print-final-likelihood",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the final likelihood tensor.",
    )
    parser.add_argument(
        "--print-intermediate-likelihoods",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print intermediate likelihood tensors.",
    )
    parser.add_argument(
        "--print-lva-peak",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print peak number of simultaneously-live variables found by the static live-variable analysis (one number, printed before compilation).",
    )
    parser.add_argument(
        "--print-static-ndim-peak",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print peak number of likelihood_tensor.ndim tracked at compile time across all scopes (one number, printed before exec).",
    )
    parser.add_argument(
        "--print-intermediate-ndim-peak",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print peak likelihood_tensor.ndim observed at runtime (numpy) / jax trace time. Single number printed at the end of the compiled function.",
    )
    parser.add_argument(
        "--print-intermediate-ndim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print likelihood_tensor.ndim at each step of the compiled program (top-level statements, markov bodies, and if-then-else blocks). Fires at jax trace time / numpy runtime.",
    )
    parser.add_argument(
        "--print-intermediate-shape",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print likelihood_tensor.shape at each step of the compiled program. Fires at jax trace time / numpy runtime.",
    )
    parser.add_argument(
        "--print-device",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the JAX backend and device list at compile time, and inject a per-call device print into the compiled program.",
    )
    parser.add_argument(
        "--jit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable JIT compilation.",
    )
    parser.add_argument(
        "--numpy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use NumPy instead of JAX (overrides JIT to False).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Floating-point precision for the likelihood tensor (and all derived float arrays).",
    )
    parser.add_argument(
        "--xla-graph-capture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture small scan bodies in CUDA graphs (GPU only; top-level calls only).",
    )
    parser.add_argument(
        "--print-pal-compilation-time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print 'PAL compilation time: <seconds>' once the @pal decorator finishes (excludes JAX JIT trace).",
    )
    return parser


def parse_command_line_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)

    ret = vars(parser.parse_args())
    if ret["numpy"]:
        ret["jit"] = False
    return ret


def argument_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)

    return parser


MAIN_KEYS = {
    "print_parsed",
    "print_compiled",
    "lva",
    "reorder",
    "prune",
    "table_lowering",
    "print_zero_mass",
    "print_final_likelihood",
    "print_intermediate_likelihoods",
    "print_lva_peak",
    "print_static_ndim_peak",
    "print_intermediate_ndim_peak",
    "print_intermediate_ndim",
    "print_intermediate_shape",
    "print_device",
    "jit",
    "numpy",
    "dtype",
    "rescale",
    "return_log_scale",
    "xla_graph_capture",
}


def arguments(args) -> dict:
    d = vars(args)
    return d


# To avoid 'Undefined name' errors
def sample(): ...
def observe(): ...
def condition(): ...
def markov(): ...
def init(): ...
