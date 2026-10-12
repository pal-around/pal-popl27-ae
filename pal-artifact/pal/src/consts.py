"""Compile-time evaluation of weight expressions.

A weight expression is a *constant table* when it is a pure function of enumerated
random variables: every Bayesian-network conditional probability table is one. Two
passes use that:

* the code generator lowers such a weight to one array plus a gather
  (:mod:`pal.src.compile`);
* ``prune`` drops a barren ``sample`` only if it marginalizes to 1, which holds exactly
  when no weight row sums to zero; :func:`provably_normalizing` decides that.

PAL's normalization is total: a parent assignment whose weights sum to zero, or are
non-finite, contributes zero mass (``compile_sample`` divides and then applies
``nan_to_num``).

Evaluation happens in Python and the generated program computes in NumPy, so a cell is
accepted only when the two agree. On a rejection the caller uses the generic code path
(the code generator) or keeps the statement (prune).
"""

from __future__ import annotations

import itertools
import operator as _op
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .syntax import (
    BinOp,
    BoolOp,
    Domain,
    Expr,
    ExprBinOp,
    ExprBoolOp,
    ExprIfThenElse,
    ExprUnaryOp,
    Stmt,
    StmtAssignSingle,
    StmtSample,
    UnaryOp,
    VarName,
)

# A table is materialized as float64, so the limit caps it at 8 MiB.
TABLE_CELL_LIMIT = 1 << 20

# The code generator lowers a weight to a table only when it reads at least this many
# parents. provably_normalizing builds tables regardless of this bound.
TABLE_MIN_PARENTS_FOR_LOWERING = 1

# Names that are not random variables: `_` is the variable being sampled and `NaN` is
# a constant (``compile_expr`` emits ``np.nan``).
SELF = VarName("_")
NAN = VarName("NaN")


class NotConstant(Exception):
    """The expression is not a pure function of enumerated variables."""


def _invert(v: Any) -> Any:
    # numpy's ~ on a bool array is logical not, while Python's ~True is -2.
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return ~v
    raise NotConstant


_BIN_OP_EVAL = {
    BinOp.ADD: _op.add,
    BinOp.SUB: _op.sub,
    BinOp.MUL: _op.mul,
    BinOp.DIV: _op.truediv,
    BinOp.MOD: _op.mod,
    BinOp.POW: _op.pow,
    BinOp.EQ: _op.eq,
    BinOp.NEQ: _op.ne,
    BinOp.LT: _op.lt,
    BinOp.LTE: _op.le,
    BinOp.GT: _op.gt,
    BinOp.GTE: _op.ge,
}
_UNARY_OP_EVAL = {UnaryOp.INVERT: _invert, UnaryOp.NOT: _op.not_, UnaryOp.USUB: _op.neg}


@dataclass(frozen=True)
class ConstEnv:
    """What the evaluator needs, available to both the front end and the code generator.

    ``domains`` maps the names in scope to their domains, ``func_globals`` resolves a
    domain name (``"Fin3"``, ``"Bool"``) to its support, and ``opaque`` holds the names
    that are never compile-time constants: the function's scalar and non-deterministic
    parameters."""

    domains: Mapping[VarName, Domain]
    func_globals: Mapping[str, Any]
    opaque: frozenset[VarName] = frozenset()


@dataclass(frozen=True)
class ConstTable:
    """The weights of ``var`` for every assignment of ``axes``.

    Dimension ``k`` of ``values`` runs over ``axes[k]``'s support and the last one over
    ``var``'s, so indexing it with each variable's value tensor reproduces the weight
    tensor the generic path builds."""

    var: VarName
    axes: tuple[VarName, ...]
    values: np.ndarray

    def rows_all_positive(self) -> bool:
        """Whether every row sums to a positive, finite total, i.e. the sample
        marginalizes to 1 for every assignment of ``axes``."""
        totals = self.values.reshape(-1, self.values.shape[-1]).sum(axis=-1)
        return bool(np.all(np.isfinite(totals) & (totals > 0)))


def support_values(env: ConstEnv, dom: Any) -> list | None:
    """The support as a list of values, or None when it does not resolve to one."""
    obj = env.func_globals.get(dom, dom) if isinstance(dom, str) else dom
    return list(obj) if isinstance(obj, (list, tuple)) else None


def is_positional(values: list) -> bool:
    """Whether a value's position in the support equals the value, so the value tensor
    indexes the table directly (``Fin(n)``, and ``Bool == [False, True]``)."""
    return all(isinstance(v, (int, bool)) and int(v) == i for i, v in enumerate(values))


def read_names(expr: Expr) -> set[VarName]:
    """The variables the expression reads, excluding ``_`` and ``NaN``."""
    out: set[VarName] = set()
    _collect(expr, out)
    return out - {SELF, NAN}


def _collect(expr: Expr, out: set[VarName]) -> None:
    cls = type(expr)
    if cls is str:
        out.add(expr)  # type: ignore[arg-type]
    elif cls is ExprBinOp:
        _collect(expr.left, out)  # type: ignore[union-attr]
        _collect(expr.right, out)  # type: ignore[union-attr]
    elif cls is ExprIfThenElse:
        _collect(expr.cond, out)  # type: ignore[union-attr]
        _collect(expr.then_branch, out)  # type: ignore[union-attr]
        _collect(expr.else_branch, out)  # type: ignore[union-attr]
    elif cls is ExprBoolOp:
        for v in expr.values:  # type: ignore[union-attr]
            _collect(v, out)
    elif cls is ExprUnaryOp:
        _collect(expr.operand, out)  # type: ignore[union-attr]
    elif cls is not int and cls is not float and cls is not bool:
        # A call (or anything else) is not constant; report it as a read of a name that
        # cannot be bound, so the caller bails out.
        raise NotConstant


def eval_const(expr: Expr, vals: Mapping[VarName, Any]) -> Any:
    """Evaluate the expression with each variable bound to a concrete value.

    Raises :class:`NotConstant` for a call, an unbound name, or an operand numpy would
    treat differently."""
    cls = type(expr)
    if cls is str:
        try:
            return vals[expr]  # type: ignore[index]
        except KeyError:
            raise NotConstant
    if cls is int or cls is float or cls is bool:
        return expr
    if cls is ExprIfThenElse:
        cond = eval_const(expr.cond, vals)  # type: ignore[union-attr]
        if not isinstance(cond, (bool, np.bool_)):
            # The generated program branches with np.where, whose condition is a mask.
            cond = bool(cond)
        branch = expr.then_branch if cond else expr.else_branch  # type: ignore[union-attr]
        return eval_const(branch, vals)
    if cls is ExprBinOp:
        left = eval_const(expr.left, vals)  # type: ignore[union-attr]
        right = eval_const(expr.right, vals)  # type: ignore[union-attr]
        return _BIN_OP_EVAL[expr.op](left, right)  # type: ignore[union-attr]
    if cls is ExprUnaryOp:
        return _UNARY_OP_EVAL[expr.op](eval_const(expr.operand, vals))  # type: ignore[union-attr]
    if cls is ExprBoolOp:
        values = [eval_const(v, vals) for v in expr.values]  # type: ignore[union-attr]
        # compile_expr emits & and |, which on floats raise in numpy and on ints are
        # bitwise; only boolean operands give the same answer both ways.
        if not all(isinstance(v, bool) for v in values):
            raise NotConstant
        out = values[0]
        for v in values[1:]:
            out = (out and v) if expr.op is BoolOp.AND else (out or v)  # type: ignore[union-attr]
        return out
    raise NotConstant


def _cell(v: Any) -> float:
    """One weight, as the generated program would compute it."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, int):
        if abs(v) > 2**53:
            raise NotConstant  # numpy's int64 would overflow or lose precision
        return float(v)
    if isinstance(v, float):
        return v
    raise NotConstant  # complex (e.g. (-2) ** 0.5, which numpy makes nan), or worse


def constant_table(env: ConstEnv, var: VarName, support: Domain, wpp: Expr) -> ConstTable | None:
    """The weights of ``var`` as a table over the variables ``wpp`` reads, or None when
    the weights are not a compile-time constant or the table would be too large."""
    self_values = support_values(env, support)
    if self_values is None or not is_positional(self_values):
        return None

    try:
        # The sampled variable is not a parent: the annotated syntax writes it by name
        # in its own weight (``xray == cancer``), and it already owns the last axis.
        names = sorted(read_names(wpp) - {var})
    except NotConstant:
        return None

    axes: list[tuple[VarName, list]] = []
    cells = len(self_values)
    for name in names:
        if name in env.opaque or name not in env.domains:
            return None
        values = support_values(env, env.domains[name])
        if values is None or not is_positional(values):
            return None
        axes.append((name, values))
        cells *= len(values)

    if cells > TABLE_CELL_LIMIT:
        return None

    flat: list[float] = []
    vals: dict[VarName, Any] = {NAN: float("nan")}
    try:
        for assignment in itertools.product(*(values for _, values in axes)):
            for name, value in zip((name for name, _ in axes), assignment):
                vals[name] = value
            for self_value in self_values:
                vals[var] = self_value
                vals[SELF] = self_value
                flat.append(_cell(eval_const(wpp, vals)))
    except (NotConstant, KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None

    shape = tuple(len(values) for _, values in axes) + (len(self_values),)
    return ConstTable(var, tuple(name for name, _ in axes), np.array(flat, dtype=np.float64).reshape(shape))


def _table_key(env: ConstEnv, var: VarName, support: Domain, wpp: Expr) -> tuple | None:
    """Identity of the table for this statement *in this environment*: the resolved
    supports. The front end resolves fewer domains than the code generator, so keying
    on them keeps a front-end "unknown" from being reused later."""
    self_values = support_values(env, support)
    if self_values is None:
        return None
    try:
        names = sorted(read_names(wpp) - {var})
    except NotConstant:
        return None
    parents = []
    for name in names:
        if name in env.opaque or name not in env.domains:
            return None
        values = support_values(env, env.domains[name])
        if values is None:
            return None
        parents.append((name, tuple(values)))
    return (tuple(self_values), tuple(parents))


def table_for(
    var: VarName,
    support: Domain,
    wpp: Expr,
    env: ConstEnv,
    cache_on: Stmt | None = None,
) -> ConstTable | None:
    """:func:`constant_table`, memoized on the statement so prune's check and the code
    generator build each table once."""
    key = _table_key(env, var, support, wpp)
    if cache_on is not None:
        cached = cache_on.__dict__.get("_const_table")
        if cached is not None and cached[0] == key:
            return cached[1]
    table = constant_table(env, var, support, wpp) if key is not None else None
    if cache_on is not None:
        cache_on.__dict__["_const_table"] = (key, table)
    return table


def _value_leaves_all_positive(expr: Expr) -> bool:
    """Whether every weight the expression can produce is positive, looking only at
    value positions (an if-then-else's condition is not one: ``[0.5, 0.5]`` parses to
    ``ExprIfThenElse(_ == 0, 0.5, 0.5)``, whose ``0`` is part of the test).

    True implies every row sums to something positive, whatever the domains are, which
    is what makes this usable where the domains are unknown."""
    cls = type(expr)
    if cls is bool:
        return bool(expr)
    if cls is int or cls is float:
        return expr > 0  # type: ignore[operator]
    if cls is ExprIfThenElse:
        return _value_leaves_all_positive(expr.then_branch) and _value_leaves_all_positive(  # type: ignore[union-attr]
            expr.else_branch  # type: ignore[union-attr]
        )
    return False


def provably_normalizing(stmt: Stmt, env: ConstEnv) -> bool:
    """Whether the statement is guaranteed to marginalize to 1, so a barren instance of
    it may be dropped."""
    if isinstance(stmt, StmtAssignSingle):
        return True  # one-hot weights over the source's domain
    if not isinstance(stmt, StmtSample):
        return False
    if _value_leaves_all_positive(stmt.wpp):
        return True
    table = table_for(stmt.var, stmt.support, stmt.wpp, env, cache_on=stmt)
    return table is not None and table.rows_all_positive()
