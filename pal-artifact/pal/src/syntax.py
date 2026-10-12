from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NewType

VarName = NewType("VarName", str)
Literal = bool | int | float
Domain = NewType("Domain", str)


class UnaryOp(Enum):
    INVERT = auto()
    NOT = auto()
    USUB = auto()


class BinOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    POW = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LTE = auto()
    GT = auto()
    GTE = auto()


class BoolOp(Enum):
    AND = auto()
    OR = auto()


@dataclass(slots=True)
class ExprUnaryOp:
    op: UnaryOp
    operand: Expr


@dataclass(slots=True)
class ExprBinOp:
    left: Expr
    op: BinOp
    right: Expr


@dataclass(slots=True)
class ExprBoolOp:
    op: BoolOp
    values: Sequence[Expr]


@dataclass(slots=True)
class ExprIfThenElse:
    cond: Expr
    then_branch: Expr
    else_branch: Expr


@dataclass(slots=True)
class ExprCall:
    func: str
    args: Sequence[VarName]
    kwargs: dict[str | None, VarName]


Expr = VarName | Literal | ExprUnaryOp | ExprBinOp | ExprBoolOp | ExprIfThenElse | ExprCall


@dataclass
class StmtSample:
    var: VarName
    support: Domain
    wpp: Expr
    is_assign: bool = False


@dataclass
class StmtCondition:
    expr: Expr


@dataclass
class StmtKill:
    var: VarName


@dataclass
class StmtReturn:
    vars: list[VarName]


@dataclass
class StmtYield:
    vars: list[VarName]


@dataclass
class StmtAssignSingle:
    target_var: VarName
    source_var: VarName
    # Compiler-generated (e.g. a yield copy), so its target may use the reserved prefix.
    internal: bool = False


@dataclass
class StmtMarkov:
    """``vars <- repeat n init { init_block } do { statements }`` (paper/Lean ``repeat``).

    ``init_block`` runs once and yields the initial loop state; ``statements`` (the
    body) runs ``repeat`` times, reading the state from ``vars`` and yielding the next
    one. Both blocks end in a ``StmtYield`` of ``len(vars)`` variables."""

    vars: list[VarName]
    init_block: list[Stmt]
    repeat: int | VarName
    statements: list[Stmt]
    # Per-iteration inputs, as (name bound in the body, parameter holding the rows).
    # The parameter's leading axis is the iteration, so the body sees one row per
    # iteration -- the scan's ``xs``.
    xs: list[tuple[VarName, VarName]] = field(default_factory=list)


@dataclass
class StmtIfThenElse:
    return_vars: list[VarName]
    conditions: list[Expr]
    blocks: list[list[Stmt]]
    yields: list[list[VarName]]


Stmt = (
    None
    | StmtSample
    | StmtCondition
    | StmtKill
    | StmtReturn
    | StmtYield
    | StmtAssignSingle
    | StmtMarkov
    | StmtIfThenElse
)
