"""Front-end passes that make block scoping explicit in the IR.

* :func:`normalize_yields` -- every ``yield`` names variables bound by its own block
  (for a loop body, its statements or the loop state). A yield of an enclosing
  variable becomes a yield of a fresh local copy, so that at the yield the yielded
  variables are always the block's newest ones, as the block compilation rule (paper
  C-Block, Lean ``Block``) requires.
* :func:`insert_scope_exit_kills` -- the kill-insertion pass used when eager
  marginalization is off: each block marginalizes the variables it introduced and
  does not yield right before its ``yield`` (the program, before its ``return``).
  A loop body's variables include the previous loop state.

After either kill-insertion pass (this one or eager marginalization) every block ends
in exactly the context it is typed to end in, and the code generator compiles only
explicit ``StmtKill`` statements.
"""

from __future__ import annotations

from itertools import count

from .live import get_def_set
from .syntax import (
    Stmt,
    StmtAssignSingle,
    StmtIfThenElse,
    StmtKill,
    StmtMarkov,
    StmtReturn,
    StmtYield,
    VarName,
)

_COPY_PREFIX = "__y"


def _at(stmt: Stmt, like: Stmt) -> Stmt:
    """Give a synthesized statement the source line of the statement it serves."""
    setattr(stmt, "lineno", getattr(like, "lineno", 0))
    return stmt


def normalize_yields(stmts: list[Stmt]) -> None:
    """Rewrite, in place, every yield of a variable its block does not bind into a
    yield of a fresh local copy of it."""
    fresh = count(1)

    def fix_block(block: list[Stmt], state: tuple[VarName, ...] | list[VarName]) -> None:
        for s in block:
            if isinstance(s, StmtMarkov):
                fix_block(s.init_block, ())
                fix_block(s.statements, s.vars)
            elif isinstance(s, StmtIfThenElse):
                for b in s.blocks:
                    fix_block(b, ())
                s.yields = [list(b[-1].vars) if b and isinstance(b[-1], StmtYield) else y
                            for b, y in zip(s.blocks, s.yields)]

        last = block[-1] if block else None
        if not isinstance(last, StmtYield):
            return
        bound: set[VarName] = set(state)
        for s in block[:-1]:
            bound |= get_def_set(s)
        new_vars: list[VarName] = []
        for v in last.vars:
            if v in bound:
                new_vars.append(v)
            else:
                copy = VarName(f"{_COPY_PREFIX}{next(fresh)}_{v}")
                block.insert(len(block) - 1, _at(StmtAssignSingle(copy, v, internal=True), last))
                new_vars.append(copy)
        if new_vars != list(last.vars):
            block[-1] = _at(StmtYield(new_vars), last)

    fix_block(stmts, ())


def insert_scope_exit_kills(stmts: list[Stmt]) -> None:
    """Insert, in place, a kill for every variable a block introduces but does not
    yield, right before the block's ``yield`` (the program's ``return``), in sorted
    order."""

    def fix_block(block: list[Stmt], state: tuple[VarName, ...] | list[VarName]) -> None:
        for s in block:
            if isinstance(s, StmtMarkov):
                fix_block(s.init_block, ())
                fix_block(s.statements, s.vars)
            elif isinstance(s, StmtIfThenElse):
                for b in s.blocks:
                    fix_block(b, ())

        last = block[-1] if block else None
        if not isinstance(last, (StmtYield, StmtReturn)):
            return
        local: set[VarName] = set(state)
        for s in block[:-1]:
            local |= get_def_set(s)
        kills = [_at(StmtKill(v), last) for v in sorted(local - set(last.vars))]
        block[len(block) - 1 : len(block) - 1] = kills

    fix_block(stmts, ())
