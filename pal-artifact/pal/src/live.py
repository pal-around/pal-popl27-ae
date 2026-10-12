from collections.abc import Collection
from typing import cast

from .syntax import (
    Expr,
    ExprBinOp,
    ExprBoolOp,
    ExprCall,
    ExprIfThenElse,
    ExprUnaryOp,
    Stmt,
    StmtAssignSingle,
    StmtCondition,
    StmtIfThenElse,
    StmtMarkov,
    StmtReturn,
    StmtSample,
    StmtYield,
    VarName,
)

def _collect_vars(expr: Expr, out: set[VarName]) -> None:
    # Accumulate variable names into `out` in a single pass: one shared set, so
    # large generated expressions allocate no per-node sets.
    cls = type(expr)
    if cls is str:  # VarName
        out.add(cast(VarName, expr))
    elif cls is ExprBinOp:
        binop = cast(ExprBinOp, expr)
        _collect_vars(binop.left, out)
        _collect_vars(binop.right, out)
    elif cls is ExprIfThenElse:
        ite = cast(ExprIfThenElse, expr)
        _collect_vars(ite.cond, out)
        _collect_vars(ite.then_branch, out)
        _collect_vars(ite.else_branch, out)
    elif cls is ExprBoolOp:
        for v in cast(ExprBoolOp, expr).values:
            _collect_vars(v, out)
    elif cls is ExprUnaryOp:
        _collect_vars(cast(ExprUnaryOp, expr).operand, out)
    elif cls is ExprCall:
        call = cast(ExprCall, expr)
        for a in call.args:
            _collect_vars(a, out)
        for v in call.kwargs.values():
            _collect_vars(v, out)
    # else: a Literal (int/float/bool) — contributes no variables.


def get_all_vars(expr: Expr, scalar_params: Collection[VarName] = ()) -> set[VarName]:
    """Random variables the expression reads. ``scalar_params`` are the function's
    inputs: they carry no tensor axis, so they are not random variables."""
    vars_set: set[VarName] = set()
    _collect_vars(expr, vars_set)
    # `NaN` is a constant (compile_expr emits np.nan), not a random variable.
    return vars_set - set(scalar_params) - {VarName("NaN")}


def get_use_set(stmt: Stmt, scalar_params: Collection[VarName] = ()) -> frozenset[VarName]:
    """Random variables ``stmt`` reads from the scope it sits in. For a loop or an
    if-then-else these are the free variables of the construct: names bound inside
    its blocks, and a loop's state inside its body, are local to it."""
    cls = type(stmt)
    if cls is StmtSample or cls is StmtCondition:
        # The expression of a sample or a condition never changes after parsing, and
        # several front-end passes (well-formedness, reorder, eager marginalization)
        # ask for its use set, so the expression walk is done once per statement. The
        # key is the scalar params, since they decide what the walk filters out.
        key = frozenset(scalar_params)
        cached = stmt.__dict__.get("_use_set")
        if cached is not None and cached[0] == key:
            return cached[1]
        if cls is StmtSample:
            sample = cast(StmtSample, stmt)
            result = frozenset(get_all_vars(sample.wpp, key) - {sample.var, VarName("_")})
        else:
            result = frozenset(get_all_vars(cast(StmtCondition, stmt).expr, key) - {VarName("_")})
        stmt.__dict__["_use_set"] = (key, result)
        return result

    use_set = set()
    match stmt:
        case StmtAssignSingle(source_var=var):
            use_set.add(var)
        case StmtReturn(vars=vars) | StmtYield(vars=vars):
            use_set.update(vars)
        case StmtMarkov(vars=vars, init_block=init_block, repeat=repeat, statements=statements):
            use_set.update(block_free_vars(init_block, scalar_params))
            if isinstance(repeat, str):  # VarName
                use_set.add(repeat)
            use_set.update(block_free_vars(statements, scalar_params) - set(vars))
        case StmtIfThenElse(conditions=conditions, blocks=blocks):
            for condition in conditions:
                use_set.update(get_all_vars(condition, scalar_params))
            for block in blocks:
                use_set.update(block_free_vars(block, scalar_params))
    return frozenset(use_set - {VarName("_")} - set(scalar_params))


def block_free_vars(stmts: list[Stmt], scalar_params: Collection[VarName] = ()) -> set[VarName]:
    """Random variables a block reads from its enclosing scope."""
    free: set[VarName] = set()
    bound: set[VarName] = set()
    for s in stmts:
        if s is None:
            continue
        free |= get_use_set(s, scalar_params) - bound
        bound |= get_def_set(s)
    return free


def get_def_set(stmt: Stmt) -> set[VarName]:
    def_set = set()
    match stmt:
        case StmtSample(var=var):
            def_set.add(var)
        case StmtAssignSingle(target_var=var):
            def_set.add(var)
        case StmtMarkov(vars=vars):
            def_set.update(vars)
        case StmtIfThenElse(return_vars=return_vars):
            def_set.update(return_vars)
    return def_set - {VarName("_")}
