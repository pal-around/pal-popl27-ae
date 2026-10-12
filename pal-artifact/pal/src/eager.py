"""Eager marginalization: insert ``kill`` statements (marginalizations) into the IR.

One backward pass over the structured IR computes liveness (a loop's fixed point has
the closed form ``K_l = (L | fv(body)) - carry``) and inserts ``StmtKill(v)`` where
each variable dies, so the tensor at every program point holds exactly the variables
live there: a statement is followed by the kills of what dies at it, and every block
begins by marginalizing what is in the tensor on entry but not live in it. The pass is
linear in the size of the IR (times the nesting depth).

:func:`lva` returns the static peak number of simultaneously live variables.
"""

from __future__ import annotations

from .syntax import (
    Stmt,
    StmtIfThenElse,
    StmtKill,
    StmtMarkov,
    StmtReturn,
    StmtYield,
    VarName,
)
from collections.abc import Collection

from .live import (
    block_free_vars,
    get_all_vars,
    get_def_set,
    get_use_set,
)

# ---------------------------------------------------------------------------
# Use/def/free-variable sets. live.get_use_set memoizes the expensive part (walking
# a sample's or condition's expression) on the statement, so these are thin wrappers.
# ---------------------------------------------------------------------------

def _use(stmt: Stmt, scalar_params: Collection[VarName]) -> frozenset[VarName]:
    return get_use_set(stmt, scalar_params)


def _def(stmt: Stmt) -> frozenset[VarName]:
    return frozenset(get_def_set(stmt))


def _fv(stmts: list[Stmt], scalar_params: Collection[VarName]) -> frozenset[VarName]:
    """Random variables the block reads from its enclosing scope."""
    return frozenset(block_free_vars(stmts, scalar_params))


def _live_before(
    stmt: Stmt, live_after: frozenset[VarName], scalar_params: Collection[VarName]
) -> frozenset[VarName]:
    """Backward transfer ``live(stmt, L) = use(stmt) | (L - def(stmt))``
    (the gen/kill form)."""
    return _use(stmt, scalar_params) | (live_after - _def(stmt))


# ---------------------------------------------------------------------------
# Well-formedness of the IR: use before definition, fresh names, block scoping, and
# yield agreement. check_wellformed runs before reorder and kill insertion.
# ---------------------------------------------------------------------------
class IRWellFormednessError(Exception):
    pass


def _yield_vars_of(block: list[Stmt]) -> list[VarName]:
    last = block[-1] if block else None
    return list(last.vars) if isinstance(last, StmtYield) else []


def _check_block(stmts: list[Stmt], defined: set[VarName], scalar_params: set[VarName]) -> None:
    """Forward well-formedness check of one block. ``defined`` = RVs in scope on entry.

    A statement may not bind a name already in scope, even a dead one; a name bound
    inside a branch, init block or loop body may be reused outside that block."""
    scope = set(defined)
    for s in stmts:
        if s is None:
            continue
        # A compound statement's use set holds only its free reads; reads of its
        # blocks' locals are checked in the recursion below.
        missing = _use(s, scalar_params) - scope
        if missing:
            raise IRWellFormednessError(
                f"variable(s) {sorted(missing)} used before definition in {type(s).__name__}"
            )
        clash = _def(s) & scope
        if clash:
            raise IRWellFormednessError(
                f"variable(s) {sorted(clash)} already defined in this scope; variable names "
                "must be fresh (even when the earlier variable is no longer used) -- rename "
                "the new variable"
            )
        if isinstance(s, StmtMarkov):
            # The init block runs in the enclosing scope and yields the initial
            # state; its locals are not visible in the body.
            _check_block(s.init_block, scope, scalar_params)
            iv = _yield_vars_of(s.init_block)
            if len(iv) != len(s.vars):
                raise IRWellFormednessError(
                    f"markov init block yields {len(iv)} value(s) but the loop has "
                    f"{len(s.vars)} carry variable(s)"
                )
            # The body sees the enclosing scope plus the state; freshness inside it
            # rules out rebinding an enclosing variable or the state, which the
            # closed-form pass-through K_l = (L | fv(body)) - carry relies on
            # (the def(blk) = 0 specialization).
            _check_block(s.statements, scope | set(s.vars), scalar_params)
            yv = _yield_vars_of(s.statements)
            if len(yv) != len(s.vars):
                raise IRWellFormednessError(
                    f"markov yields {len(yv)} value(s) but has {len(s.vars)} carry variable(s)"
                )
        elif isinstance(s, StmtIfThenElse):
            arities = {len(_yield_vars_of(b)) for b in s.blocks}
            if len(arities) > 1 or (arities and next(iter(arities)) != len(s.return_vars)):
                raise IRWellFormednessError(
                    "if-then-else branches disagree on the number of yielded values"
                )
            for b in s.blocks:
                _check_block(b, scope, scalar_params)
        scope |= _def(s)


def check_wellformed(stmts: list[Stmt], scalar_params: set[VarName] | None = None) -> None:
    """Reject ill-formed IR (see :func:`_check_block`). Runs after yield normalization
    and before every other front-end pass, whether or not eager marginalization is
    enabled."""
    sp: set[VarName] = set(scalar_params) if scalar_params is not None else set()
    _check_block(stmts, set(), sp)


# ---------------------------------------------------------------------------
# The translation: backward over each block, inserting StmtKill in place. `carried`
# holds the enclosing variables live across the whole block; it feeds the peak only.
# ---------------------------------------------------------------------------

def _emar_block(
    stmts: list[Stmt],
    live_after_block: frozenset[VarName],
    entry: frozenset[VarName] | None,
    carried: frozenset[VarName],
    scalar_params: set[VarName],
) -> tuple[frozenset[VarName], int]:
    """Translate a block (``stmts`` ending in StmtYield, or the top-level program).

    ``live_after_block`` is what is live *after* the block excluding the yielded
    values. ``entry`` is the set of variables in the tensor when the block starts;
    those not live at its first statement are marginalized right there. Returns
    (live_in, peak)."""
    live_in, peak = _emar_seq(stmts, live_after_block | frozenset(_yield_vars_of(stmts)), carried, scalar_params)
    if entry is not None:
        peak = max(peak, len(entry | carried))
        # sorted(): set iteration order varies with PYTHONHASHSEED, and this order
        # reaches the emitted program (hence float summation order).
        stmts[0:0] = [StmtKill(v) for v in sorted(entry - live_in - scalar_params)]
    return live_in, peak


def _emar_seq(
    stmts: list[Stmt],
    live_after: frozenset[VarName],
    carried: frozenset[VarName],
    scalar_params: set[VarName],
) -> tuple[frozenset[VarName], int]:
    """Translate a statement sequence backward, threading the live-after set.

    Mutates ``stmts`` (inserts StmtKill). Returns (live_in, peak)."""
    peak = len(live_after | carried)
    L = live_after
    for i in range(len(stmts) - 1, -1, -1):
        s = stmts[i]
        if s is None:
            continue
        L_before = _live_before(s, L, scalar_params)
        peak = max(peak, len(L_before | carried), len(L | carried))

        # Recurse into compound statements first (they insert kills into their own
        # blocks and report what dies at them).
        dying: frozenset[VarName]
        if isinstance(s, StmtMarkov):
            dying, sub_peak = _emar_markov(s, L, carried, scalar_params)
            peak = max(peak, sub_peak)
        elif isinstance(s, StmtIfThenElse):
            dying, sub_peak = _emar_if(s, L, carried, scalar_params)
            peak = max(peak, sub_peak)
        elif isinstance(s, (StmtReturn, StmtYield)):
            # Terminal markers: their "use" is the block/program output, never
            # marginalized. They only make their variables live (via _live_before).
            dying = frozenset()
        else:
            # Straight-line statement: kill exactly the variables that die here,
            # (use | def) - live-after.
            dying = _use(s, scalar_params) | _def(s)

        for v in sorted(dying - L - scalar_params):  # sorted(): see _emar_block
            stmts.insert(i + 1, StmtKill(v))

        L = L_before
    return L, peak


def _emar_markov(
    stmt: StmtMarkov,
    live_after: frozenset[VarName],
    carried: frozenset[VarName],
    scalar_params: set[VarName],
) -> tuple[frozenset[VarName], int]:
    """Translate a bounded loop with the pass-through
    ``K_l = (L | fv(body) | fv(bound)) - carry``; the init block and the body are both
    translated with live-after ``K_l``. Returns the variables that die at the loop, and
    the sub-peak."""
    # Prune state variables that neither the body reads nor anything after the
    # loop needs.
    keep = _fv(stmt.statements, scalar_params) | live_after
    if not all(v in keep for v in stmt.vars):
        iv = _yield_vars_of(stmt.init_block)
        yv = _yield_vars_of(stmt.statements)
        kept = [k for k, var in enumerate(stmt.vars) if var in keep]
        stmt.vars = [stmt.vars[k] for k in kept]
        for blk, ys in ((stmt.init_block, iv), (stmt.statements, yv)):
            old = blk[-1]
            blk[-1] = StmtYield([ys[k] for k in kept])
            setattr(blk[-1], "lineno", getattr(old, "lineno", 0))

    carry = frozenset(stmt.vars)
    bound_fv: frozenset[VarName] = frozenset(
        {stmt.repeat} if isinstance(stmt.repeat, str) else ()
    ) - scalar_params
    body_fv = _fv(stmt.statements, scalar_params)

    K = (live_after | body_fv | bound_fv) - carry  # loop pass-through, K_l
    entry = _live_before(stmt, live_after, scalar_params)

    _, body_peak = _emar_block(stmt.statements, K, entry=K | carry, carried=carried | K, scalar_params=scalar_params)
    _, init_peak = _emar_block(stmt.init_block, K, entry=entry, carried=carried, scalar_params=scalar_params)

    # What dies at the loop: the state, the body's enclosing reads that K held live
    # across iterations, and the count.
    return (body_fv | bound_fv | carry), max(body_peak, init_peak)


def _emar_if(
    stmt: StmtIfThenElse,
    live_after: frozenset[VarName],
    carried: frozenset[VarName],
    scalar_params: set[VarName],
) -> tuple[frozenset[VarName], int]:
    """Translate an if-then-else. Each branch marginalizes at its entry what it does
    not need, ``P_i = live(if) \\ live_in(blk_i)``, and is translated with live-after
    ``L \\ x``, so all branches end in the same context. Return variables dead after
    the if die at the if."""
    ret = frozenset(stmt.return_vars)
    L_minus_x = live_after - ret
    entry = _live_before(stmt, live_after, scalar_params)

    peak = 0
    for block in stmt.blocks:
        _, bpeak = _emar_block(block, L_minus_x, entry=entry, carried=carried | L_minus_x, scalar_params=scalar_params)
        peak = max(peak, bpeak)

    return ret, peak


# ---------------------------------------------------------------------------
# Public entry.
# ---------------------------------------------------------------------------
def _assert_no_kills(stmts: list[Stmt]) -> None:
    for s in stmts:
        assert not isinstance(s, StmtKill), "eager marginalization expects IR without marginalization"
        if isinstance(s, StmtMarkov):
            _assert_no_kills(s.init_block)
            _assert_no_kills(s.statements)
        elif isinstance(s, StmtIfThenElse):
            for b in s.blocks:
                _assert_no_kills(b)


def lva(
    stmts: list[Stmt],
    scalar_params: set[VarName] | None = None,
) -> int:
    """Insert eager marginalizations into ``stmts`` (in place) and return the
    static peak simultaneously-live count. ``stmts`` must already have passed
    :func:`check_wellformed`."""
    sp: set[VarName] = set(scalar_params) if scalar_params is not None else set()
    _assert_no_kills(stmts)
    _, peak = _emar_block(stmts, frozenset(), entry=None, carried=frozenset(), scalar_params=sp)
    return peak
