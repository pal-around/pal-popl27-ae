import textwrap
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

from .consts import TABLE_MIN_PARENTS_FOR_LOWERING, ConstEnv, table_for
from .live import get_all_vars
from .helpers import SourceInfo, source_error_context, static_vars
from .syntax import (
    BinOp,
    BoolOp,
    Domain,
    Expr,
    ExprBinOp,
    ExprBoolOp,
    ExprCall,
    ExprIfThenElse,
    ExprUnaryOp,
    Literal,
    Stmt,
    StmtAssignSingle,
    StmtCondition,
    StmtIfThenElse,
    StmtKill,
    StmtMarkov,
    StmtReturn,
    StmtSample,
    StmtYield,
    UnaryOp,
    VarName,
)

RESERVED_PREFIX = "__"
# A random variable's value tensor ("enum tensor"): its support reshaped onto the
# variable's axis. Following the paper's C-Var, it depends only on the variable's
# domain and axis position, and is (re)built wherever a statement reads the
# variable at a position no existing tensor matches.
ENUM_TENSOR_PREFIX = "enum_"
TABLE_PREFIX = f"{RESERVED_PREFIX}tbl_"
SHAPE_PREFIX = "shape_"
WEIGHT_PREFIX = "weight_"
PROB_PREFIX = "prob_"
TEMP_PREFIX = "temp_"
LIKELIHOOD_TENSOR = "likelihood_tensor"
PEAK_NDIM_VAR = f"{RESERVED_PREFIX}peak_ndim"
# Scale tracking (``@pal(rescale=True)``). The compiled program maintains
#
#     true likelihood  ==  likelihood_tensor * exp(LOG_SCALE_VAR)
#
# so the tensor can be divided down into a representable range while the
# evidence probability stays recoverable. The divisor is the tensor's max.
LOG_SCALE_VAR = f"{RESERVED_PREFIX}log_scale"
NP = "np"


class CompilationError(Exception):
    pass


@dataclass
class CompileContext:
    kwargs: dict[str, Any]
    non_deterministic_params: set[VarName]
    scalar_params: dict[VarName, Any]
    # Names a markov block's `xs=` binds to one row per iteration: data, like a scalar
    # parameter, and emitted verbatim.
    xs_params: set[VarName] = field(default_factory=set)

    # Source info (filename + lines + line offset) for error reporting
    source: SourceInfo = field(default_factory=SourceInfo)

    # Compiled program
    compiled_lines: list[str] = field(default_factory=list)

    # Live variables; store variable name and support, e.g. ("s0", bool)
    variables: list[tuple[str, Domain]] = field(default_factory=list)

    # Current variable name being sampled
    current_var: str = field(default="")

    # The decorated function's module globals, used to resolve a domain name
    # (e.g. "Discrete3", "Bool") to its cardinality for peak-memory accounting.
    func_globals: dict[str, Any] = field(default_factory=dict)

    # Peak number of likelihood_tensor dimensions (= max len(variables))
    # tracked at compile time across all scopes.
    peak_ndim: int = 0

    # Peak number of likelihood_tensor cells (= max product of the live
    # variables' domain sizes), tracked at compile time across all scopes.
    # Peak memory in bytes is peak_cells * dtype itemsize.
    peak_cells: int = 0

    # Set while compiling a construct that emitted a rescale, so an enclosing
    # if-then-else knows its branches may have ended at different scales and must
    # be rebased before they are summed.
    rescaled: bool = False

    # For each random variable, the axis index at which the Python local ``enum_<v>``
    # currently holds its value tensor in the emitted code's current scope. A read at
    # any other position re-emits the tensor. A name's domain cannot change while its
    # entry is live: entries are reset per branch and per loop body, dropped when the
    # variable is marginalized or renamed at a yield, and overwritten when it is
    # sampled.
    enum_valid: dict[VarName, int] = field(default_factory=dict)

    # Enum tensors emitted while compiling the current statement's expressions;
    # the statement places them before the lines that use them.
    enum_prelude: list[str] = field(default_factory=list)

    # Variable name -> axis index in ``variables``, refreshed by _begin_expr.
    var_index: dict[VarName, int] = field(default_factory=dict)

    # Constant weight tables (see _lower_constant_table), handed to the generated
    # program through its globals. Name -> float64 array, in emission order.
    tables: dict[str, Any] = field(default_factory=dict)
    # Table content -> its name, so identical tables share one array.
    table_names: dict[Any, str] = field(default_factory=dict)

    # Source line of the statement being compiled, for messages the generated program
    # prints at run time.
    current_lineno: int = 0

    @property
    def rescale_enabled(self) -> bool:
        return bool(self.kwargs.get("rescale", False))

    def const_env(self) -> ConstEnv:
        """The environment for compile-time weight evaluation: the live variables'
        domains, the module globals that resolve them, and the names that are never
        constant."""
        return ConstEnv(
            domains=dict(self.variables),
            func_globals=self.func_globals,
            opaque=frozenset(self.scalar_params)
            | frozenset(self.non_deterministic_params)
            | frozenset(self.xs_params),
        )


def _domain_size(ctxt: CompileContext, dom: Any) -> int:
    """Cardinality of a domain. Domains are stored as name strings (e.g.
    "Discrete3", "Bool") that resolve at runtime to sized objects in the
    function's module globals; resolve them here for compile-time accounting."""
    obj = ctxt.func_globals.get(dom, dom) if isinstance(dom, str) else dom
    try:
        return len(obj)
    except TypeError:
        return 1  # unresolved/unsized domain; do not inflate the product


def _same_domain(ctxt: CompileContext, a: Any, b: Any) -> bool:
    """Whether two domains denote the same support (e.g. ``Bool`` and ``Fin2``)."""
    if a == b:
        return True
    ra = ctxt.func_globals.get(a, a) if isinstance(a, str) else a
    rb = ctxt.func_globals.get(b, b) if isinstance(b, str) else b
    try:
        return list(ra) == list(rb)
    except TypeError:
        return False


def _bump_peak(ctxt: CompileContext) -> None:
    """Update peak_ndim / peak_cells from the current ctxt.variables."""
    n = len(ctxt.variables)
    if n > ctxt.peak_ndim:
        ctxt.peak_ndim = n
    cells = 1
    for _, dom in ctxt.variables:
        cells *= _domain_size(ctxt, dom)
    if cells > ctxt.peak_cells:
        ctxt.peak_cells = cells


def _enum_line(var: VarName, dom: Domain, idx: int) -> str:
    return f"{ENUM_TENSOR_PREFIX}{var} = {NP}.array({dom}).reshape(-1{', 1' * idx})"


def _begin_expr(ctxt: CompileContext) -> None:
    """Prepare to compile the expressions of one statement against the current
    context: refresh the axis-index map and start an empty enum prelude."""
    ctxt.var_index = {v: i for i, (v, _) in enumerate(ctxt.variables)}
    ctxt.enum_prelude = []


def _ensure_enum(ctxt: CompileContext, var: VarName) -> str:
    """Name of ``var``'s value tensor, emitting it into the prelude unless a tensor
    for it at its current axis position already exists in this scope (C-Var)."""
    idx = ctxt.var_index.get(var)
    if idx is None:
        raise CompilationError(f"Variable {var} not defined")
    if ctxt.enum_valid.get(var) != idx:
        ctxt.enum_prelude.append(_enum_line(var, ctxt.variables[idx][1], idx))
        ctxt.enum_valid[var] = idx
    return ENUM_TENSOR_PREFIX + var


# Operator spellings for compile_expr, which runs once per expression node
# (millions of nodes in generated programs), so the table is built once at import.
_BIN_OP_STR = {
    BinOp.ADD: "+",
    BinOp.SUB: "-",
    BinOp.MUL: "*",
    BinOp.DIV: "/",
    BinOp.MOD: "%",
    BinOp.POW: "**",
    BinOp.EQ: "==",
    BinOp.NEQ: "!=",
    BinOp.LT: "<",
    BinOp.LTE: "<=",
    BinOp.GT: ">",
    BinOp.GTE: ">=",
}


def compile_expr(expr: Expr, ctxt: CompileContext) -> str:
    """Compile an expression to a tensor expression over the current context.

    Callers must call :func:`_begin_expr` first and emit ``ctxt.enum_prelude``
    before the line that uses the result."""
    # Dispatch on the concrete type by identity, most frequent types first.
    cls = type(expr)
    if cls is str:  # VarName
        var: VarName = expr  # type: ignore[assignment]
        if var == "NaN":
            # resolve NaN as a special constant
            return "np.nan"
        elif var in ctxt.scalar_params or var in ctxt.xs_params:
            return var
        if var == "_":
            var = VarName(ctxt.current_var)
        # Fast path: the value tensor already exists at the variable's current axis.
        idx = ctxt.var_index.get(var)
        if idx is not None and ctxt.enum_valid.get(var) == idx:
            return ENUM_TENSOR_PREFIX + var
        return _ensure_enum(ctxt, var)
    elif cls is ExprBinOp:
        binop = cast(ExprBinOp, expr)
        op = _BIN_OP_STR[binop.op]
        return f"({compile_expr(binop.left, ctxt)}) {op} ({compile_expr(binop.right, ctxt)})"
    elif cls is ExprIfThenElse:
        ite = cast(ExprIfThenElse, expr)
        return f"{NP}.where({compile_expr(ite.cond, ctxt)}, {compile_expr(ite.then_branch, ctxt)}, {compile_expr(ite.else_branch, ctxt)})"
    elif cls is int or cls is float or cls is bool:  # Literal
        return str(expr)
    elif cls is ExprBoolOp:
        boolop = cast(ExprBoolOp, expr)
        match boolop.op:
            case BoolOp.AND:
                return "(" + ") & (".join(compile_expr(v, ctxt) for v in boolop.values) + ")"
            case BoolOp.OR:
                return "(" + ") | (".join(compile_expr(v, ctxt) for v in boolop.values) + ")"
    elif cls is ExprUnaryOp:
        unaryop = cast(ExprUnaryOp, expr)
        match unaryop.op:
            case UnaryOp.INVERT:
                return f"~({compile_expr(unaryop.operand, ctxt)})"
            case UnaryOp.NOT:
                return f"{NP}.logical_not({compile_expr(unaryop.operand, ctxt)})"
            case UnaryOp.USUB:
                return f"-({compile_expr(unaryop.operand, ctxt)})"
    elif cls is ExprCall:
        call = cast(ExprCall, expr)
        var_names = tuple(v for v, _ in ctxt.variables)

        n_existing = len(var_names)
        # Check if the variable being assigned to is already in the typing context
        if var_names and var_names[-1] == ctxt.current_var:
            n_existing -= 1

        def compile_call_arg(arg: VarName) -> str:
            if arg in ctxt.scalar_params:
                return arg
            if arg not in var_names:
                raise CompilationError(f"Variable {arg} not defined")

            if n_existing <= 1:
                return LIKELIHOOD_TENSOR

            axes_to_sum = tuple(-(j + 1) for j in range(n_existing) if j != var_names.index(arg))
            return f"{LIKELIHOOD_TENSOR}.sum({axes_to_sum})"

        call_args = [compile_call_arg(a) for a in call.args]
        for k, v in call.kwargs.items():
            if k is None:
                call_args.append(f"**{compile_call_arg(v)}")
            else:
                call_args.append(f"{k}={compile_call_arg(v)}")

        compiled_call = f"{call.func.replace('jnp.', 'np.')}({', '.join(call_args)})"
        return f"{NP}.reshape({compiled_call}, {_ensure_enum(ctxt, VarName(ctxt.current_var))}.shape)"
    else:
        raise CompilationError("Expression type could not be matched")


def _table_global(ctxt: CompileContext, var: VarName, table) -> str:
    """Register ``table`` under a name the generated program reads from its globals.

    Names are numbered in emission order, since variable names are fresh only within a
    block (two branches may both define ``temp``). Globals are visible inside scan
    bodies."""
    key = (table.values.shape, table.values.tobytes())
    name = ctxt.table_names.get(key)
    if name is None:
        name = f"{TABLE_PREFIX}{len(ctxt.tables)}_{var}"
        ctxt.table_names[key] = name
        ctxt.tables[name] = table.values
    return name


def _lower_constant_table(
    ctxt: CompileContext, var: VarName, support: Domain, wpp: Expr
) -> tuple[str, tuple[VarName, ...], tuple[int, ...]] | None:
    """Prepare the gather for a weight expression that is a pure function of enumerated
    variables -- every Bayesian-network conditional probability table.

    The table is computed once at compile time and read with a single gather. Returns
    the table's global name, its
    parent order and its shape, or None when the weights are not constant (see
    :mod:`pal.src.consts`), in which case nothing has been emitted."""
    table = table_for(var, support, wpp, ctxt.const_env())
    if table is None or len(table.axes) < TABLE_MIN_PARENTS_FOR_LOWERING:
        return None
    if any(p not in ctxt.var_index for p in table.axes):
        return None
    return _table_global(ctxt, var, table), table.axes, table.values.shape


def compile_sample(
    ctxt: CompileContext, var: VarName, support: Domain, wpp: Expr, is_assign=False, bypass_reserved_prefix=False
) -> list[str]:
    compiled_lines = []

    ctxt.current_var = var

    # Check that variable is not already defined
    if var in (v for v, _ in ctxt.variables):
        raise CompilationError(f"Variable {var} already defined")

    # Check that variable does not begin with reserved prefix
    if var.startswith(RESERVED_PREFIX) and not bypass_reserved_prefix:
        raise CompilationError(f"Variable {var} cannot begin with reserved prefix '{RESERVED_PREFIX}'")
    # Update typing context
    ctxt.variables.append((var, support))
    _bump_peak(ctxt)

    compiled_lines.append(f"# {'Assign' if is_assign else 'Sample'} {var}")

    # Enumeration tensor of the new variable, on the newest axis
    idx = len(ctxt.variables) - 1
    compiled_lines.append(_enum_line(var, support, idx))
    ctxt.enum_valid[var] = idx

    # Weight tensor, from a precomputed table where the weights are constant given the
    # variables they read, and from the expression itself otherwise.
    dtype = ctxt.kwargs.get("dtype", "float64")
    _begin_expr(ctxt)
    lowered = None
    if ctxt.kwargs.get("table_lowering", True) and not is_assign:
        lowered = _lower_constant_table(ctxt, var, support, wpp)
    if lowered is not None:
        table_name, axes, shape = lowered
        # The index arrays are the variables' own value tensors, each already shaped for
        # its axis, so the gather broadcasts to exactly the shape the expression path
        # builds. Reading them through _ensure_enum keeps C-Var's invariant: a tensor is
        # built where it is read, and rebuilt inside a scan body.
        index = [_ensure_enum(ctxt, p) for p in axes] + [_ensure_enum(ctxt, VarName(var))]
        compiled_lines.extend(ctxt.enum_prelude)
        compiled_lines.append(f"# Weight table {table_name}: shape {shape}, axes {tuple(axes) + (var,)}")
        gather = ", ".join(f"{name}.astype(int)" for name in index)
        compiled_lines.append(f"{WEIGHT_PREFIX}{var} = {table_name}[{gather}].astype({NP}.{dtype})")
    elif isinstance(wpp, str) and wpp in ctxt.xs_params:
        # The whole weight is one row of a markov block's per-iteration input. A row of
        # numbers is this step's table over the variable's domain, so it is laid on the
        # variable's own axis; a scalar row broadcasts as any scalar weight does. Which
        # of the two it is depends on the parameter's shape, known when the program runs
        # (and when JAX traces it), not when it is compiled.
        compiled_lines.extend(ctxt.enum_prelude)
        row = f"{NP}.reshape({wpp}, {ENUM_TENSOR_PREFIX}{var}.shape) if {NP}.ndim({wpp}) else {wpp}"
        compiled_lines.append(
            f"{WEIGHT_PREFIX}{var} = ({NP}.ones_like({ENUM_TENSOR_PREFIX}{var}, dtype={NP}.{dtype}) "
            f"* ({row})).astype({NP}.{dtype})"
        )
    else:
        weight_expr = compile_expr(wpp, ctxt)
        compiled_lines.extend(ctxt.enum_prelude)
        # A row used inside a larger expression must be a scalar: an array would
        # broadcast onto whichever axis happened to match. Checked where the shape is
        # known -- as the program runs, or as JAX traces it -- and free afterwards.
        for name in sorted(get_all_vars(wpp) & ctxt.xs_params) if ctxt.xs_params else []:
            compiled_lines.append(
                f"assert {NP}.ndim({name}) == 0, "
                f"{f'markov xs row {name!r} is an array; use it as the whole weight of a sample'!r}"
            )
        compiled_lines.append(
            f"{WEIGHT_PREFIX}{var} = ({NP}.ones_like({ENUM_TENSOR_PREFIX}{var}, dtype={NP}.{dtype}) * ({weight_expr})).astype({NP}.{dtype})"
        )

    # Probability tensor. Normalization is total: where the weights sum to zero, or to
    # something non-finite, the quotient is not a distribution and the assignment
    # contributes zero mass -- the same as condition(False). nan_to_num is what makes
    # that so, and it keeps a parent assignment the program can never reach (an
    # unspecified table row) from poisoning the result, since 0 * NaN is NaN.
    sum_axis = -len(ctxt.variables)
    if ctxt.kwargs.get("print_zero_mass", False) and not is_assign:
        # Report the mass this statement drops: rows whose weights do not normalize and
        # that the incoming tensor gives nonzero mass. Two scalar reductions, and the
        # weight sum is the one the normalization needs anyway.
        total = f"{RESERVED_PREFIX}wsum_{var}"
        dropped = f"{RESERVED_PREFIX}zm_lost_{var}"
        where = f"{ctxt.source.filename}:{ctxt.current_lineno}"
        message = f"PAL zero mass: sample {var} ({where}) dropped "
        compiled_lines.append(f"{total} = {NP}.sum({WEIGHT_PREFIX}{var}, axis={sum_axis})")
        compiled_lines.append(
            f"{dropped} = {NP}.sum({NP}.where({NP}.logical_not(({total} > 0) & {NP}.isfinite({total})), "
            f"{LIKELIHOOD_TENSOR}, 0))"
        )
        if ctxt.kwargs.get("numpy", False):
            compiled_lines.append(
                f"if {dropped} > 0: print({message!r}, float({dropped}), 'of', "
                f"float({NP}.sum({LIKELIHOOD_TENSOR})), 'incoming mass')"
            )
        else:
            compiled_lines.append(
                f"jax.lax.cond({dropped} > 0, lambda: jax.debug.print({message + '{lost} of {total} incoming mass'!r}, "
                f"lost={dropped}, total={NP}.sum({LIKELIHOOD_TENSOR})), lambda: None)"
            )
        compiled_lines.append(f"{PROB_PREFIX}{var} = {WEIGHT_PREFIX}{var} / {total}")
    else:
        compiled_lines.append(
            f"{PROB_PREFIX}{var} = {WEIGHT_PREFIX}{var} / {NP}.sum({WEIGHT_PREFIX}{var}, axis={sum_axis})"
        )
    compiled_lines.append(f"{PROB_PREFIX}{var} = {NP}.nan_to_num({PROB_PREFIX}{var})")

    # Update likelihood tensor
    compiled_lines.append(f"{LIKELIHOOD_TENSOR} = {LIKELIHOOD_TENSOR} * {PROB_PREFIX}{var}")

    return compiled_lines + ["", f"# Types: {ctxt.variables}", ""]


def compile_assign_single(
    ctxt: CompileContext, target_var: VarName, source_var: VarName, bypass_reserved_prefix=False
) -> list[str]:
    """
    assign(target_var in dom(source_var), source_var)
    """
    # Check that source variable is defined; and get its support
    try:
        support = next(support for var, support in ctxt.variables if var == source_var)
    except StopIteration:
        raise CompilationError(f"Variable {source_var} not defined")

    return compile_sample(
        ctxt,
        target_var,
        support,
        ExprBinOp(VarName("_"), BinOp.EQ, VarName(source_var)),
        is_assign=True,
        bypass_reserved_prefix=bypass_reserved_prefix,
    )


def compile_rescale(ctxt: CompileContext, extra_tensors: tuple[str, ...] = ()) -> list[str]:
    """Divide the likelihood tensor by its maximum and add the log of that
    divisor to ``LOG_SCALE_VAR``, preserving ``tensor * exp(log_scale)``.

    Dividing by the max keeps the largest entry at 1.0.

    ``extra_tensors`` are other tensors held at the same scale (the random-loop-
    bound accumulator) and are divided by the same factor so they stay
    commensurable.

    An all-zero tensor (contradictory evidence) has max 0; dividing would give
    NaN, so that case is left untouched and contributes nothing to the scale.
    """
    if not ctxt.rescale_enabled:
        return []
    ctxt.rescaled = True
    m = f"{RESERVED_PREFIX}rescale_m"
    lines = [
        "# Rescale (scale-tracked): keep max at 1, accumulate the log divisor",
        f"{m} = {NP}.max({LIKELIHOOD_TENSOR})",
        f"{m} = {NP}.where({m} > 0, {m}, 1.0)",
        f"{LIKELIHOOD_TENSOR} = {LIKELIHOOD_TENSOR} / {m}",
    ]
    lines += [f"{t} = {t} / {m}" for t in extra_tensors]
    lines.append(f"{LOG_SCALE_VAR} = {LOG_SCALE_VAR} + {NP}.log({m})")
    return lines


def compile_condition(ctxt: CompileContext, expr: Expr) -> list[str]:
    compiled_lines = ["# Condition"]

    # Multiply likelihood tensor by the boolean mask produced by the expression.
    _begin_expr(ctxt)
    mask_expr = compile_expr(expr, ctxt)
    compiled_lines.extend(ctxt.enum_prelude)
    compiled_lines.append(f"mask = {mask_expr}")
    compiled_lines.append(f"{LIKELIHOOD_TENSOR} = {LIKELIHOOD_TENSOR} * mask")

    return compiled_lines + ["", f"# Types: {ctxt.variables}", ""]


def compile_kills(ctxt: CompileContext, vars: list[VarName]) -> list[str]:
    """Marginalize ``vars`` with a single sum over their axes. Value tensors are left
    alone -- a later variable whose axis moved gets its tensor rebuilt when it is
    next read."""
    var_names = [v for v, _ in ctxt.variables]
    absent = [v for v in vars if v not in var_names]
    if absent or len(set(vars)) != len(vars):
        raise CompilationError(f"internal error: marginalizing {vars}, but the context is {var_names}")
    if not vars:
        return []
    present = list(vars)

    axes = sorted(-var_names.index(v) - 1 for v in present)
    axis = axes[0] if len(axes) == 1 else tuple(axes)
    compiled_lines = [
        f"# Kill {', '.join(present)}",
        f"{LIKELIHOOD_TENSOR} = {NP}.sum({LIKELIHOOD_TENSOR}, axis={axis})",
    ]
    for v in present:
        ctxt.variables.pop([name for name, _ in ctxt.variables].index(v))
        ctxt.enum_valid.pop(v, None)

    return compiled_lines + ["", f"# Types: {ctxt.variables}", ""]


def compile_return(ctxt: CompileContext, vars: list[VarName]) -> list[str]:
    """``return x1, ..., xk``: the context holds exactly the returned variables (the
    front end has marginalized everything else); move their axes into the return
    order."""
    compiled_lines = []

    # Check uniqueness of returned variables
    if len(vars) != len(set(vars)):
        raise CompilationError("Returned variables must be unique")

    var_names = [v for v, _ in ctxt.variables]
    if set(var_names) != set(vars):
        raise CompilationError(f"internal error: returning {vars}, but the context is {var_names}")
    indices = [var_names.index(v) for v in vars]

    compiled_lines.append(f"# Return {', '.join(vars)}")

    # Transpose axes accordingly
    compiled_lines.append(
        f"{LIKELIHOOD_TENSOR} = {LIKELIHOOD_TENSOR}.transpose({tuple(len(ctxt.variables) - i - 1 for i in indices)})"
    )

    return compiled_lines + ["", f"# Types: {ctxt.variables}", ""]


indent4 = partial(textwrap.indent, prefix="    ")


def _compile_stmts(ctxt: CompileContext, stmts: list[Stmt]) -> list[str]:
    """Compile a statement sequence. A run of consecutive kills becomes one sum over
    all their axes, so explicit marginalization never costs more tensor passes than
    summing the same variables out together."""
    lines: list[str] = []
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        lines.extend(_emit_intermediate_prints(ctxt))
        if isinstance(stmt, StmtKill):
            j = i
            while j < len(stmts) and isinstance(stmts[j], StmtKill):
                j += 1
            run = [cast(StmtKill, s).var for s in stmts[i:j]]
            with source_error_context(ctxt.source, getattr(stmt, "lineno", 0)):
                lines.extend(compile_kills(ctxt, run))
            i = j
        else:
            lines.extend(compile_stmt(ctxt, stmt))
            i += 1
    return lines


def compile_block(
    ctxt: CompileContext,
    stmts: list[Stmt],
    out_names: list[VarName],
    expected_outer: tuple | None = None,
    expected_ydoms: tuple | None = None,
    what: str = "block",
) -> tuple[list[str], tuple, tuple]:
    """Compile a block ``s; yield y1, ..., yk`` whose yielded values are renamed to
    ``out_names`` (paper C-Block, with Lean's ``Block.compileAs`` permutation).

    The block's statements are compiled in order; the yield marginalizes nothing
    (the front end has already inserted every marginalization). The yielded
    variables must then be the newest variables of the context, in any order: if
    their order differs from the yield's, one transpose moves their axes into the
    yield's order before they are renamed.

    Returns the lines, the outer context the block ends in (everything below the
    yielded values) and the yielded values' domains. ``expected_outer`` and
    ``expected_ydoms`` check a block against a sibling or the loop state it must
    reproduce."""
    last = stmts[-1] if stmts else None
    if not isinstance(last, StmtYield):
        raise CompilationError(f"internal error: {what} does not end with a yield")
    yield_vars = list(last.vars)

    entry_names = {v for v, _ in ctxt.variables}
    lines = _compile_stmts(ctxt, stmts[:-1])

    names = [v for v, _ in ctxt.variables]
    n_outer = len(names) - len(yield_vars)
    if n_outer < 0 or set(names[n_outer:]) != set(yield_vars):
        raise CompilationError(
            f"internal error: {what} yields {yield_vars}, which are not the newest variables of "
            f"its context {names}"
        )
    outer = tuple(ctxt.variables[:n_outer])
    unmarginalized = [v for v, _ in outer if v not in entry_names]
    if unmarginalized:
        raise CompilationError(
            f"internal error: {what} ends with its local variable(s) {unmarginalized} not marginalized"
        )
    if expected_outer is not None and outer != tuple(expected_outer):
        raise CompilationError(
            f"internal error: {what} ends in context {list(outer)}, expected {list(expected_outer)}"
        )
    dom_of = dict(ctxt.variables)
    ydoms = tuple(dom_of[v] for v in yield_vars)
    if expected_ydoms is not None and (
        len(ydoms) != len(expected_ydoms) or not all(_same_domain(ctxt, a, b) for a, b in zip(ydoms, expected_ydoms))
    ):
        raise CompilationError(
            f"{what} yields values of types {list(ydoms)}, but {list(expected_ydoms)} are required"
        )

    if names[n_outer:] != yield_vars:
        # Variable index i is tensor axis r-1-i. The new order keeps the outer
        # context and puts the yielded variables in the yield's order.
        new_order = [v for v, _ in outer] + yield_vars
        old_index = {v: i for i, v in enumerate(names)}
        r = len(names)
        axes = tuple(r - 1 - old_index[new_order[r - 1 - a]] for a in range(r))
        lines.append(f"# Yield {', '.join(yield_vars)}: move the yielded axes into the yield's order")
        lines.append(f"{LIKELIHOOD_TENSOR} = {LIKELIHOOD_TENSOR}.transpose({axes})")

    for v in yield_vars:
        ctxt.enum_valid.pop(v, None)
    for v in out_names:
        ctxt.enum_valid.pop(v, None)
    ctxt.variables = list(outer) + [(VarName(o), d) for o, d in zip(out_names, ydoms)]
    return lines, outer, ydoms


@static_vars(counter=0)
def compile_markov(
    ctxt: CompileContext,
    markov_vars: list[VarName],
    init_block: list[Stmt],
    repeat: int | VarName,
    statements: list[Stmt],
    xs: list[tuple[VarName, VarName]] | None = None,
) -> list[str]:
    """Paper C-Repeat / Lean ``repeat``: the init block runs once and yields the
    initial loop state; the body -- a block over the outer context and the state --
    runs ``repeat`` times in a scan and yields the next state."""
    compiled_lines = []

    # Counter for name mangling
    compile_markov.counter += 1  # type: ignore
    current_counter = compile_markov.counter  # type: ignore

    # Init block: establishes the loop context (outer variables that survive it,
    # then the loop state).
    compiled_lines.append("# ========== Markov init block ==========")
    init_lines, loop_outer, carry_doms = compile_block(
        ctxt,
        init_block,
        markov_vars,
        what="markov init block",
    )
    compiled_lines.extend(init_lines)

    if repeat == 0:
        return compiled_lines + ["", f"# Types: {ctxt.variables}", ""]

    loop_ctx = ctxt.variables[:]
    loop_names = tuple(v for v, _ in loop_ctx)

    # A random loop count n : Fin c is a mixture over its values (Lean `repeat`):
    # run the body k times on the slice of the distribution where n == k, for every
    # k, and sum. n is constant through the loop and the body is linear, so a single
    # scan computes it: after k iterations the tensor holds body^k of the whole
    # input, and its n == k slice is added to an accumulator.
    using_rv_bound: bool = isinstance(repeat, str) and repeat in loop_names
    if isinstance(repeat, str) and not using_rv_bound and repeat not in ctxt.scalar_params:
        raise CompilationError(f"internal error: loop count {repeat} is not live at the start of the loop")
    xs = xs or []
    if xs and using_rv_bound:
        raise CompilationError("internal error: a random loop count cannot be combined with xs")
    onehot_var = f"{RESERVED_PREFIX}markov{current_counter}_onehot"
    acc_var = f"{RESERVED_PREFIX}markov{current_counter}_acc"
    slice_shape = ""

    if using_rv_bound:
        n_index = loop_names.index(repeat)
        n_domain = loop_ctx[n_index][1]
        slice_shape = f"(-1{', 1' * n_index})"
        compiled_lines.append(f"# Loop count {repeat}: accumulate the slice n == k after k iterations")
        compiled_lines.append(f"{onehot_var} = {NP}.eye(len({n_domain}), dtype={LIKELIHOOD_TENSOR}.dtype)")
        compiled_lines.append(f"{acc_var} = {LIKELIHOOD_TENSOR} * {onehot_var}[0].reshape{slice_shape}")

    # Define function for scan. Under rescaling the accumulated log scale is part
    # of the carry: each iteration divides the tensor down and adds the log of
    # the divisor, so the scale must survive to the next iteration.
    rescaling = ctxt.rescale_enabled
    if using_rv_bound:
        compiled_lines.append(f"def {RESERVED_PREFIX}scan{current_counter}(carry, k_onehot):")
        carry_names = f"{LIKELIHOOD_TENSOR}, {acc_var}" + (f", {LOG_SCALE_VAR}" if rescaling else "")
        compiled_lines.append(f"    {carry_names} = carry")
    else:
        # With per-iteration inputs the scan's second argument is this iteration's row
        # of each of them (a tuple, in the order they were bound, when there is more
        # than one) -- exactly what jax.lax.scan hands the body for a pytree of xs.
        row_arg = f"{RESERVED_PREFIX}xs{current_counter}" if xs else "_"
        compiled_lines.append(f"def {RESERVED_PREFIX}scan{current_counter}(carry, {row_arg}):")
        carry_names = LIKELIHOOD_TENSOR + (f", {LOG_SCALE_VAR}" if rescaling else "")
        compiled_lines.append(f"    {carry_names} = carry")
        if xs:
            rows = ", ".join(row for row, _ in xs)
            compiled_lines.append(f"    {rows}{',' if len(xs) == 1 else ''} = {row_arg}")
    if ctxt.kwargs.get("print_intermediate_ndim_peak", False):
        compiled_lines.append(f"    nonlocal {PEAK_NDIM_VAR}")
    compiled_lines.append("")

    # The scan body is its own Python function: a name assigned anywhere in it is
    # local to all of it, so no enclosing value tensor may be relied on inside --
    # every tensor the body reads is built in the body.
    outer_enum_valid = ctxt.enum_valid
    ctxt.enum_valid = {}

    body_lines, _, _ = compile_block(
        ctxt,
        statements,
        markov_vars,
        expected_outer=loop_outer,
        expected_ydoms=carry_doms,
        what="markov body",
    )
    compiled_lines.extend(map(indent4, body_lines))
    ctxt.variables = loop_ctx[:]

    if using_rv_bound:
        compiled_lines.append(
            f"    {acc_var} = {acc_var} + {LIKELIHOOD_TENSOR} * k_onehot.reshape{slice_shape}"
        )
        # The accumulator holds terms from earlier iterations, so it must be
        # divided by the same factor to stay at the carry's scale.
        compiled_lines.extend(map(indent4, compile_rescale(ctxt, (acc_var,))))
        tail = f", {LOG_SCALE_VAR}" if rescaling else ""
        compiled_lines.append(f"    return ({LIKELIHOOD_TENSOR}, {acc_var}{tail}), None")
    else:
        compiled_lines.extend(map(indent4, compile_rescale(ctxt)))
        tail = f", {LOG_SCALE_VAR}" if rescaling else ""
        compiled_lines.append(f"    return ({LIKELIHOOD_TENSOR}{tail}), None")
    compiled_lines.append("")
    compiled_lines.append(f"    # Types after return: {ctxt.variables}")
    compiled_lines.append("")

    # Back in the enclosing scope, whose value tensors the body did not touch.
    ctxt.enum_valid = outer_enum_valid

    # Call scan
    if using_rv_bound:
        # Iteration k (1-based) sees row k of the identity: the slice n == k.
        out = f"(_, {LIKELIHOOD_TENSOR}, {LOG_SCALE_VAR})" if rescaling else f"(_, {LIKELIHOOD_TENSOR})"
        init = (
            f"({LIKELIHOOD_TENSOR}, {acc_var}, {LOG_SCALE_VAR})"
            if rescaling
            else f"({LIKELIHOOD_TENSOR}, {acc_var})"
        )
        compiled_lines.append(
            f"# Markov scan\n"
            f"{out}, _ = scan(\n"
            f"    {RESERVED_PREFIX}scan{current_counter},\n"
            f"    init={init},\n"
            f"    xs={onehot_var}[1:],\n"
            f")"
        )
    else:
        out = f"({LIKELIHOOD_TENSOR}, {LOG_SCALE_VAR})" if rescaling else LIKELIHOOD_TENSOR
        init = f"({LIKELIHOOD_TENSOR}, {LOG_SCALE_VAR})" if rescaling else LIKELIHOOD_TENSOR
        # A tuple even for one input, so the body unpacks the same way either way.
        # `length` is the loop's iteration count; with both given, scan rejects a
        # parameter whose leading axis has another length.
        xs_arg = "(" + "".join(f"{source}, " for _, source in xs) + ")" if xs else "None"
        compiled_lines.append(
            f"# Markov scan\n"
            f"{out}, _ = scan(\n"
            f"    {RESERVED_PREFIX}scan{current_counter},\n"
            f"    init={init},\n"
            f"    xs={xs_arg},\n"
            f"    length={repeat},\n"
            f")"
        )

    return compiled_lines + ["", f"# Types: {ctxt.variables}", ""]


@static_vars(counter=0)
def compile_if_then_else(
    ctxt: CompileContext,
    return_vars: list[VarName],
    conditions: list[Expr],
    blocks: list[list[Stmt]],
) -> list[str]:
    """Paper C-If: each branch is a block run on the pre-if tensor masked by its
    condition, and the branch tensors are summed.

    Every branch must end at the same outer context followed by its yielded values;
    the front end's kill insertion (eager, or at scope exit) guarantees it, and a
    violation is an internal error. A branch may marginalize outer variables (eager
    marginalization does): each branch is compiled from the pre-if context, and a
    value tensor whose axis moved is rebuilt where it is read, so nothing emitted by
    one branch is relied on by another."""
    compiled_lines = []

    # Counter for name mangling
    compile_if_then_else.counter += 1  # type: ignore
    current_counter = compile_if_then_else.counter  # type: ignore

    # Save initial state to be recovered later
    initial_ctxt = ctxt.variables[:]
    initial_enum_valid = dict(ctxt.enum_valid)
    initial_likelihood_name = f"{RESERVED_PREFIX}if{current_counter}_initial_{LIKELIHOOD_TENSOR}"

    compiled_lines.append(f"# ========== If-then-else #{current_counter}: pre-if tensor ==========")
    compiled_lines.append(f"{initial_likelihood_name} = {LIKELIHOOD_TENSOR}")

    branch_likelihood_names = []
    branch_log_scale_names = []
    # Snapshot the scale at the if entry: every branch starts from it, and a
    # branch that rescales internally ends at its own scale.
    initial_log_scale_name = f"{RESERVED_PREFIX}if{current_counter}_initial_{LOG_SCALE_VAR}"
    if ctxt.rescale_enabled:
        compiled_lines.append(f"{initial_log_scale_name} = {LOG_SCALE_VAR}")
    compiled_lines.append("")
    any_branch_rescaled = False
    first_outer: tuple | None = None
    first_ydoms: tuple | None = None

    for branch_num, block in enumerate(blocks, 1):
        compiled_lines.append(f"# ========== If-then-else #{current_counter} branch {branch_num} ==========")

        # Reset state to the initial state
        ctxt.variables = initial_ctxt[:]
        ctxt.enum_valid = dict(initial_enum_valid)
        compiled_lines.append(f"{LIKELIHOOD_TENSOR} = {initial_likelihood_name}")
        if ctxt.rescale_enabled:
            compiled_lines.append(f"{LOG_SCALE_VAR} = {initial_log_scale_name}")
        outer_rescaled, ctxt.rescaled = ctxt.rescaled, False

        # Mask likelihood tensor with branch condition
        if len(conditions) == 1:
            if branch_num == 1:
                branch_cond_expr = conditions[0]
            else:
                branch_cond_expr = ExprUnaryOp(UnaryOp.NOT, conditions[0])
        else:
            if branch_num == 1:
                branch_cond_expr = conditions[0]
            elif branch_num <= len(conditions):
                branch_cond_expr = ExprBoolOp(
                    BoolOp.AND,
                    [ExprUnaryOp(UnaryOp.NOT, c) for c in conditions[: branch_num - 1]] + [conditions[branch_num - 1]],
                )
            else:
                branch_cond_expr = ExprBoolOp(
                    BoolOp.AND, [ExprUnaryOp(UnaryOp.NOT, c) for c in conditions[: branch_num - 1]]
                )

        _begin_expr(ctxt)
        cond_expr = compile_expr(branch_cond_expr, ctxt)
        compiled_lines.extend(ctxt.enum_prelude)
        compiled_lines.append(f"mask = {NP}.logical_not({NP}.logical_not({cond_expr}))")
        compiled_lines.append(f"{LIKELIHOOD_TENSOR} = {LIKELIHOOD_TENSOR} * mask")
        compiled_lines.append("")

        # The branch body, as a block yielding the if's return variables.
        branch_lines, outer, ydoms = compile_block(
            ctxt,
            block,
            return_vars,
            expected_outer=first_outer,
            expected_ydoms=first_ydoms,
            what=f"if-then-else branch {branch_num}",
        )
        compiled_lines.extend(branch_lines)
        if branch_num == 1:
            first_outer, first_ydoms = outer, ydoms

        # Save final likelihood tensor for this branch
        branch_likelihood_name = f"{RESERVED_PREFIX}if{current_counter}_branch{branch_num}_{LIKELIHOOD_TENSOR}"
        compiled_lines.append(f"{branch_likelihood_name} = {LIKELIHOOD_TENSOR}")
        branch_likelihood_names.append(branch_likelihood_name)
        if ctxt.rescale_enabled:
            branch_log_scale_name = f"{RESERVED_PREFIX}if{current_counter}_branch{branch_num}_{LOG_SCALE_VAR}"
            compiled_lines.append(f"{branch_log_scale_name} = {LOG_SCALE_VAR}")
            branch_log_scale_names.append(branch_log_scale_name)
        any_branch_rescaled = any_branch_rescaled or ctxt.rescaled
        ctxt.rescaled = outer_rescaled or ctxt.rescaled
        compiled_lines.append("")

    # Consolidate branches by summing their likelihood tensors.
    compiled_lines.append(f"# ========== If-then-else #{current_counter}: sum branches ==========")
    if any_branch_rescaled:
        # A branch rescaled internally, so the branches sit at different scales
        # and cannot be added directly. Rebase each onto the largest scale first
        # (the logsumexp shift, applied once here).
        m = f"{RESERVED_PREFIX}if{current_counter}_merge_max"
        compiled_lines.append(f"{m} = {NP}.maximum.reduce([{', '.join(branch_log_scale_names)}])"
                              if len(branch_log_scale_names) > 2
                              else f"{m} = {NP}.maximum({branch_log_scale_names[0]}, {branch_log_scale_names[1]})"
                              if len(branch_log_scale_names) == 2
                              else f"{m} = {branch_log_scale_names[0]}")
        terms = [f"{n} * {NP}.exp({s} - {m})" for n, s in zip(branch_likelihood_names, branch_log_scale_names)]
        compiled_lines.append(f"{LIKELIHOOD_TENSOR} = {' + '.join(terms)}")
        compiled_lines.append(f"{LOG_SCALE_VAR} = {m}")
    else:
        compiled_lines.append(f"{LIKELIHOOD_TENSOR} = {' + '.join(branch_likelihood_names)}")
        if ctxt.rescale_enabled:
            compiled_lines.append(f"{LOG_SCALE_VAR} = {initial_log_scale_name}")
    compiled_lines.append("")

    # The context after the if: the common outer context, then the returned variables.
    assert first_outer is not None and first_ydoms is not None
    ctxt.variables = list(first_outer) + [(return_var, dom) for return_var, dom in zip(return_vars, first_ydoms)]
    # Value tensors emitted in a branch are not relied on after the if. The pre-if
    # entries stay correct for every variable whose axis did not move (no branch
    # re-emitted those); for one whose axis moved, the recorded index no longer
    # matches and the tensor is rebuilt on its next read.
    ctxt.enum_valid = initial_enum_valid

    return compiled_lines + ["", f"# Types: {ctxt.variables}", ""]


def compile_stmt(ctxt: CompileContext, stmt: Stmt) -> list[str]:
    ctxt.current_lineno = getattr(stmt, "lineno", 0)
    with source_error_context(ctxt.source, getattr(stmt, "lineno", 0)):
        match stmt:
            case StmtSample(var, support, wpp, is_assign):
                return compile_sample(ctxt, var, support, wpp, is_assign)
            case StmtAssignSingle(target_var, source_var, internal):
                return compile_assign_single(ctxt, target_var, source_var, bypass_reserved_prefix=internal)
            case StmtCondition(expr):
                return compile_condition(ctxt, expr)
            case StmtKill(var):
                return compile_kills(ctxt, [var])
            case StmtReturn(vars):
                return compile_return(ctxt, vars)
            case StmtMarkov(
                vars=vars, init_block=init_block, repeat=repeat, statements=statements, xs=markov_xs
            ):
                return compile_markov(ctxt, vars, init_block, repeat, statements, markov_xs)
            case StmtIfThenElse(return_vars=return_vars, conditions=conditions, blocks=blocks):
                return compile_if_then_else(ctxt, return_vars, conditions, blocks)
            case _:
                raise CompilationError("Invalid statement")


def _emit_intermediate_prints(ctxt: CompileContext) -> list[str]:
    """Lines to insert before each compiled step (top-level, markov body,
    if-then-else block) based on --print-intermediate-* flags."""
    lines: list[str] = []
    if ctxt.kwargs.get("print_intermediate_likelihoods", False):
        lines.append("# Print intermediate likelihood")
        if ctxt.kwargs.get("numpy", False):
            lines.append(f"print({LIKELIHOOD_TENSOR})")
            lines.append(f"print({LIKELIHOOD_TENSOR}.sum())")
        else:
            lines.append(f"jax.debug.print('{{likelihood_tensor}}', likelihood_tensor={LIKELIHOOD_TENSOR})")
            lines.append(f"jax.debug.print('{{t}}', t={LIKELIHOOD_TENSOR}.sum())")
    if ctxt.kwargs.get("print_intermediate_ndim", False):
        lines.append(f"print('ndim:', {LIKELIHOOD_TENSOR}.ndim)")
    if ctxt.kwargs.get("print_intermediate_shape", False):
        lines.append(f"print('shape:', {LIKELIHOOD_TENSOR}.shape)")
    if ctxt.kwargs.get("print_intermediate_ndim_peak", False):
        lines.append(f"{PEAK_NDIM_VAR} = max({PEAK_NDIM_VAR}, {LIKELIHOOD_TENSOR}.ndim)")
    if lines:
        lines.append("")
    return lines


def compile_pal(ctxt: CompileContext, pal_stmts: list[Stmt]):
    # Initial likelihood tensor
    dtype = ctxt.kwargs.get("dtype", "float64")
    ctxt.compiled_lines.append("")
    ctxt.compiled_lines.append("# Initial likelihood tensor")
    ctxt.compiled_lines.append(f"{LIKELIHOOD_TENSOR} = {NP}.array(1.0, dtype={NP}.{dtype})")
    if ctxt.rescale_enabled:
        ctxt.compiled_lines.append(f"{LOG_SCALE_VAR} = {NP}.array(0.0, dtype={NP}.{dtype})")

    if ctxt.kwargs.get("print_intermediate_ndim_peak", False):
        ctxt.compiled_lines.append(f"{PEAK_NDIM_VAR} = 0")

    ctxt.compiled_lines.append("")
    ctxt.compiled_lines.extend(_compile_stmts(ctxt, pal_stmts))

    if ctxt.kwargs.get("print_intermediate_ndim_peak", False):
        ctxt.compiled_lines.append(f"{PEAK_NDIM_VAR} = max({PEAK_NDIM_VAR}, {LIKELIHOOD_TENSOR}.ndim)")
        ctxt.compiled_lines.append(f"print('runtime peak ndim:', {PEAK_NDIM_VAR})")
