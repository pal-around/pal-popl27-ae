"""Compile-time evaluation of weight expressions (pal/src/consts.py).

The generated program computes weights in NumPy, so a cell may only be accepted when
Python and NumPy agree on it; every other case must be reported as "not a table", never
as a wrong number. The tests below check both, and the normalization proof that
pruning relies on."""

import numpy as np
import pytest

from pal import Bool, Fin
from pal.src.consts import (
    ConstEnv,
    NotConstant,
    constant_table,
    eval_const,
    is_positional,
    provably_normalizing,
    support_values,
    table_for,
)
from pal.src.syntax import (
    BinOp,
    BoolOp,
    Domain,
    ExprBinOp,
    ExprBoolOp,
    ExprCall,
    ExprIfThenElse,
    ExprUnaryOp,
    StmtAssignSingle,
    StmtSample,
    UnaryOp,
    VarName,
)

Fin3 = Fin(3)
GLOBALS = {"Bool": Bool, "Fin3": Fin3, "Shifted": [1, 2, 3]}


def env(**domains) -> ConstEnv:
    return ConstEnv(domains={VarName(k): Domain(v) for k, v in domains.items()}, func_globals=GLOBALS)


def var(name: str) -> VarName:
    return VarName(name)


# ---------------------------------------------------------------------------
# Supports
# ---------------------------------------------------------------------------

def test_support_values_resolves_domain_names():
    e = env()
    assert support_values(e, Domain("Bool")) == [False, True]
    assert support_values(e, Domain("Fin3")) == [0, 1, 2]
    assert support_values(e, Domain("Nope")) is None


def test_is_positional():
    assert is_positional([0, 1, 2])
    assert is_positional([False, True])  # Bool indexes like [0, 1]
    assert not is_positional([1, 2, 3])
    assert not is_positional(["a", "b"])


# ---------------------------------------------------------------------------
# eval_const agrees with NumPy, or refuses
# ---------------------------------------------------------------------------

def test_arithmetic_and_comparisons_match_numpy():
    x = var("x")
    cases = [
        (ExprBinOp(x, BinOp.ADD, 2), 5),
        (ExprBinOp(x, BinOp.SUB, 1), 2),
        (ExprBinOp(x, BinOp.MUL, 4), 12),
        (ExprBinOp(x, BinOp.DIV, 2), 1.5),
        (ExprBinOp(x, BinOp.MOD, 2), 1),
        (ExprBinOp(x, BinOp.POW, 2), 9),
        (ExprBinOp(x, BinOp.EQ, 3), True),
        (ExprBinOp(x, BinOp.NEQ, 3), False),
        (ExprBinOp(x, BinOp.LT, 4), True),
        (ExprBinOp(x, BinOp.LTE, 3), True),
        (ExprBinOp(x, BinOp.GT, 3), False),
        (ExprBinOp(x, BinOp.GTE, 3), True),
        (ExprUnaryOp(UnaryOp.USUB, x), -3),
    ]
    for expr, expected in cases:
        got = eval_const(expr, {x: 3})
        assert got == expected, expr
        # The generated program computes the same thing on an array.
        arr = eval_const(expr, {x: np.array([3])})
        np.testing.assert_array_equal(np.asarray(arr), np.asarray([expected]))


def test_invert_follows_numpy_on_booleans():
    # Python's ~True is -2; the generated program applies ~ to a boolean mask.
    b = var("b")
    assert eval_const(ExprUnaryOp(UnaryOp.INVERT, b), {b: True}) is False
    assert eval_const(ExprUnaryOp(UnaryOp.INVERT, b), {b: False}) is True
    assert eval_const(ExprUnaryOp(UnaryOp.INVERT, b), {b: 5}) == ~5


def test_boolops_require_boolean_operands():
    a, b = var("a"), var("b")
    assert eval_const(ExprBoolOp(BoolOp.AND, [a, b]), {a: True, b: False}) is False
    assert eval_const(ExprBoolOp(BoolOp.OR, [a, b]), {a: True, b: False}) is True
    # `&` on floats raises in NumPy, so Python's `and` is not a faithful model.
    with pytest.raises(NotConstant):
        eval_const(ExprBoolOp(BoolOp.AND, [a, b]), {a: True, b: 0.7})


def test_if_then_else_picks_a_branch():
    x = var("x")
    expr = ExprIfThenElse(ExprBinOp(x, BinOp.EQ, 0), 0.9, 0.1)
    assert eval_const(expr, {x: 0}) == 0.9
    assert eval_const(expr, {x: 1}) == 0.1


def test_unbound_name_and_call_are_refused():
    x = var("x")
    with pytest.raises(NotConstant):
        eval_const(x, {})
    with pytest.raises(NotConstant):
        eval_const(ExprCall("np.sum", [x], {}), {x: 1})


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def test_table_over_one_parent():
    x, y = var("x"), var("y")
    # 0.7 where y == x, else 0.15, over y: Fin3 given x: Bool.
    wpp = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, x), 0.7, 0.15)
    table = constant_table(env(x="Bool"), y, Domain("Fin3"), wpp)
    assert table is not None
    assert table.axes == (x,)
    np.testing.assert_allclose(table.values, [[0.7, 0.15, 0.15], [0.15, 0.7, 0.15]])
    assert table.rows_all_positive()


def test_table_without_parents():
    y = var("y")
    wpp = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, 0), 0.25, 0.75)
    table = constant_table(env(), y, Domain("Bool"), wpp)
    assert table is not None and table.axes == ()
    np.testing.assert_allclose(table.values, [0.25, 0.75])


def test_no_table_for_unresolvable_or_non_positional_domains():
    y = var("y")
    wpp = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, 0), 0.25, 0.75)
    assert constant_table(env(), y, Domain("Nope"), wpp) is None
    assert constant_table(env(), y, Domain("Shifted"), wpp) is None
    # A parent whose support is not positional cannot index the table either.
    x = var("x")
    wpp_parent = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, x), 0.7, 0.3)
    assert constant_table(env(x="Shifted"), y, Domain("Bool"), wpp_parent) is None


def test_no_table_for_opaque_names_or_unknown_variables():
    x, y = var("x"), var("y")
    wpp = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, x), 0.7, 0.3)
    opaque = ConstEnv(domains={x: Domain("Bool")}, func_globals=GLOBALS, opaque=frozenset({x}))
    assert constant_table(opaque, y, Domain("Bool"), wpp) is None  # scalar parameter
    assert constant_table(env(), y, Domain("Bool"), wpp) is None  # x not in scope


def test_cell_limit(monkeypatch):
    import pal.src.consts as consts

    y = var("y")
    wpp = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, 0), 0.25, 0.75)
    monkeypatch.setattr(consts, "TABLE_CELL_LIMIT", 1)
    assert consts.constant_table(env(), y, Domain("Bool"), wpp) is None


def test_division_by_zero_is_not_a_table():
    x, y = var("x"), var("y")
    wpp = ExprBinOp(1, BinOp.DIV, x)  # x == 0 divides by zero
    assert constant_table(env(x="Bool"), y, Domain("Bool"), wpp) is None


def test_table_is_cached_on_the_statement():
    stmt = StmtSample(var("y"), Domain("Bool"), ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, 0), 0.25, 0.75))
    e = env()
    first = table_for(stmt.var, stmt.support, stmt.wpp, e, cache_on=stmt)
    second = table_for(stmt.var, stmt.support, stmt.wpp, e, cache_on=stmt)
    assert first is not None and first is second


# ---------------------------------------------------------------------------
# provably_normalizing: what pruning relies on
# ---------------------------------------------------------------------------

def sample(var_name: str, domain: str, wpp) -> StmtSample:
    return StmtSample(var(var_name), Domain(domain), wpp)


def test_positive_weights_need_no_domains():
    # [0.5, 0.5] parses to an if-then-else whose condition holds a 0, which is not a
    # weight; only the value positions matter.
    wpp = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, 0), 0.5, 0.5)
    assert provably_normalizing(sample("y", "Bool", wpp), ConstEnv({}, GLOBALS))
    assert provably_normalizing(sample("y", "Bool", 1), ConstEnv({}, GLOBALS))


def test_zero_weight_rows_are_not_provable():
    # sample([0, 0]) has no mass at all, and `x == 2` is all-zero for x: Bool.
    zeros = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, 0), 0, 0)
    assert not provably_normalizing(sample("y", "Bool", zeros), env())
    x_eq_2 = ExprBinOp(var("x"), BinOp.EQ, 2)
    assert not provably_normalizing(sample("y", "Bool", x_eq_2), env(x="Bool"))
    assert not provably_normalizing(sample("y", "Bool", 0), env())


def test_zero_cells_are_fine_when_every_row_has_mass():
    # A weight of 0 for some values is normal; only an all-zero row is a problem.
    wpp = ExprIfThenElse(ExprBinOp(VarName("_"), BinOp.EQ, 0), 0, 1)
    assert provably_normalizing(sample("y", "Fin3", wpp), env())


def test_non_constant_weights_are_not_provable():
    e = ConstEnv(domains={var("p"): Domain("Bool")}, func_globals=GLOBALS, opaque=frozenset({var("p")}))
    assert not provably_normalizing(sample("y", "Bool", var("p")), e)
    call = ExprCall("np.sum", [var("x")], {})
    assert not provably_normalizing(sample("y", "Bool", call), env(x="Bool"))


def test_assign_is_always_normalizing():
    assert provably_normalizing(StmtAssignSingle(var("y"), var("x")), ConstEnv({}, GLOBALS))
