"""Pruning of barren factors.

A PAL program denotes a product of factors. A factor that cannot affect the queried
variables -- a *barren* node, one that no returned or yielded variable and no evidence
depends on -- may be dropped, provided it marginalizes to 1:

* an ``assign`` always does: its weights are one-hot over the source's domain;
* a ``sample`` normalizes by its weight row's sum, and PAL's normalization is total, so
  a row of zeros (or a non-finite one) contributes **zero** mass rather than 1. Dropping
  such a factor would silently put its mass back, so it is kept unless
  :func:`pal.src.consts.provably_normalizing` proves the weights cannot sum to zero.

Evidence (``condition``, and ``observe``, which desugars to it) and compound statements
(``markov``, ``if``) are never dropped: evidence changes the mass, and a compound
statement's blocks may contain either.

This is off by default (``--prune``).

Ordering is a separate concern, in :mod:`pal.src.reorder`.
"""

from __future__ import annotations

from .consts import ConstEnv, provably_normalizing
from .live import get_def_set, get_use_set
from .syntax import (
    Domain,
    Stmt,
    StmtAssignSingle,
    StmtIfThenElse,
    StmtKill,
    StmtMarkov,
    StmtReturn,
    StmtSample,
    StmtYield,
    VarName,
)


def prune(
    stmts: list[Stmt],
    scalar_params: set[VarName],
    func_globals: dict,
    non_deterministic_params: frozenset[VarName] = frozenset(),
) -> list[Stmt]:
    """Drop the barren factors of ``stmts`` and, recursively, of the init blocks,
    bodies and branches of any loops and ifs."""
    env = ConstEnv(
        domains={},
        func_globals=func_globals,
        opaque=frozenset(scalar_params) | frozenset(non_deterministic_params),
    )
    return _prune_block(stmts, func_globals, env)


def _block_domains(stmts: list[Stmt], func_globals: dict) -> dict[VarName, Domain]:
    """The domains this block's statements make known, for the normalization proof.
    Only what is visible syntactically: a sample's support, an assign's source, a
    loop's state (from its init block's yields), and an if's return variables when
    every branch yields the same domains."""
    known: dict[VarName, Domain] = {}
    for s in stmts:
        if isinstance(s, StmtSample):
            known[s.var] = s.support
        elif isinstance(s, StmtAssignSingle):
            if s.source_var in known:
                known[s.target_var] = known[s.source_var]
        elif isinstance(s, StmtMarkov):
            inner = _block_domains(s.init_block, func_globals)
            last = s.init_block[-1] if s.init_block else None
            yielded = last.vars if isinstance(last, StmtYield) else []
            for carry, y in zip(s.vars, yielded):
                if y in inner:
                    known[carry] = inner[y]
        elif isinstance(s, StmtIfThenElse):
            per_branch = []
            for block in s.blocks:
                inner = _block_domains(block, func_globals)
                last = block[-1] if block else None
                yielded = last.vars if isinstance(last, StmtYield) else []
                per_branch.append([inner.get(y) for y in yielded])
            if per_branch and all(b == per_branch[0] for b in per_branch):
                for ret, dom in zip(s.return_vars, per_branch[0]):
                    if dom is not None:
                        known[ret] = dom
    return known


def _prune_block(stmts: list[Stmt], func_globals: dict, env: ConstEnv) -> list[Stmt]:
    # Recurse into compound statements' blocks first (bottom-up), so a block that
    # becomes barren inside a loop or branch is dropped there too.
    outer = dict(env.domains) | _block_domains(stmts, func_globals)
    for s in stmts:
        assert not isinstance(s, StmtKill), "prune runs before any marginalization is inserted"
        if isinstance(s, StmtMarkov):
            s.init_block[:] = _prune_block(s.init_block, func_globals, ConstEnv(outer, func_globals, env.opaque))
            body_env = ConstEnv(outer | _block_domains(s.init_block, func_globals), func_globals, env.opaque)
            s.statements[:] = _prune_block(s.statements, func_globals, body_env)
        elif isinstance(s, StmtIfThenElse):
            branch_env = ConstEnv(outer, func_globals, env.opaque)
            for k in range(len(s.blocks)):
                s.blocks[k][:] = _prune_block(s.blocks[k], func_globals, branch_env)

    body = [s for s in stmts if s is not None and not isinstance(s, (StmtReturn, StmtYield))]
    terminal = [s for s in stmts if isinstance(s, (StmtReturn, StmtYield))]
    if not body:
        return stmts

    sink_vars: set[VarName] = set()
    for t in terminal:
        sink_vars |= set(t.vars)

    return _keep_useful(body, sink_vars, ConstEnv(outer, func_globals, env.opaque)) + terminal


def _keep_useful(body: list[Stmt], sink_vars: set[VarName], env: ConstEnv) -> list[Stmt]:
    """The statements of one block that can affect its output, in their original order."""
    n = len(body)

    # Variable defined by each statement -> its index. check_wellformed guarantees
    # names are fresh within a block, so every variable has exactly one definition here.
    def_of: dict[VarName, int] = {}
    defs: list[frozenset[VarName]] = []
    for i, s in enumerate(body):
        d = frozenset(get_def_set(s))
        defs.append(d)
        for v in d:
            def_of[v] = i

    # Data-dependency parents, over the variables defined at this level (a loop's body
    # reads of its own state are not dependencies on the loop statement itself).
    parents: list[set[int]] = []
    for i, s in enumerate(body):
        reads = get_use_set(s, env.opaque)
        parents.append({def_of[v] for v in reads if v in def_of and def_of[v] != i})

    def close_over_ancestors(keep: set[int]) -> set[int]:
        work = list(keep)
        while work:
            i = work.pop()
            for p in parents[i]:
                if p not in keep:
                    keep.add(p)
                    work.append(p)
        return keep

    # Keep every evidence and compound node, everything defining an output variable,
    # and all their ancestors.
    prunable = [isinstance(s, (StmtSample, StmtAssignSingle)) for s in body]
    keep: set[int] = {i for i in range(n) if not prunable[i]}
    keep |= {i for i in range(n) if defs[i] & sink_vars}
    keep = close_over_ancestors(keep)

    # Of the rest, a sample is dropped only if it provably marginalizes to 1. Checked
    # only on the statements pruning would actually drop, so no weight table is built
    # for a statement that is kept anyway.
    kept_back = {
        i
        for i in range(n)
        if i not in keep and isinstance(body[i], StmtSample) and not provably_normalizing(body[i], env)
    }
    if kept_back:
        keep = close_over_ancestors(keep | kept_back)

    return [body[i] for i in sorted(keep)]
