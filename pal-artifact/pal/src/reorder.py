"""Cardinality-weighted reordering of factor blocks.

Each statement contributes a factor, so any order that respects data dependencies
computes the same distribution, while the peak tensor depends on the order. This pass
reorders each block (recursing into loop bodies, init blocks and branches) to a
topological order with a low simulated peak, chosen among the input order, a
cardinality-weighted greedy order and DFS orders. It runs after ``check_wellformed``
and before marginalization is inserted. ``PAL_REORDER_DEBUG`` prints each block's
candidate orders and their peaks.
"""

from __future__ import annotations

import math
import os

from .syntax import (
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
from .live import get_def_set, get_use_set


def _cardinality(support, func_globals) -> int:
    obj = func_globals.get(support, support) if isinstance(support, str) else support
    try:
        return len(obj)
    except TypeError:
        return 1


def reorder(
    stmts: list[Stmt],
    scalar_params: set[VarName],
    func_globals: dict,
    non_deterministic_params: frozenset[VarName] = frozenset(),
) -> list[Stmt]:
    """Reorder ``stmts`` (and, recursively, the init blocks, bodies and branches of any
    loops and ifs) to a lower-peak order that preserves the distribution."""
    opaque = frozenset(scalar_params) | frozenset(non_deterministic_params)
    return _reorder_block(stmts, func_globals, opaque)


def _reorder_block(stmts: list[Stmt], func_globals: dict, opaque: frozenset[VarName]) -> list[Stmt]:
    # 1. Recurse into compound statements' blocks first (bottom-up).
    for s in stmts:
        assert not isinstance(s, StmtKill), "reorder runs before any marginalization is inserted"
        if isinstance(s, StmtMarkov):
            s.init_block[:] = _reorder_block(s.init_block, func_globals, opaque)
            s.statements[:] = _reorder_block(s.statements, func_globals, opaque)
        elif isinstance(s, StmtIfThenElse):
            for k in range(len(s.blocks)):
                s.blocks[k][:] = _reorder_block(s.blocks[k], func_globals, opaque)

    body = [s for s in stmts if s is not None and not isinstance(s, (StmtReturn, StmtYield))]
    terminal = [s for s in stmts if isinstance(s, (StmtReturn, StmtYield))]
    if len(body) < 2:
        return stmts

    sink_vars: set[VarName] = set()
    for t in terminal:
        sink_vars |= set(t.vars)

    return _order_level(body, sink_vars, func_globals, opaque) + terminal


def _carry_logcard(s: StmtMarkov, logcard: dict[VarName, float], func_globals: dict) -> dict[VarName, float]:
    """Log cardinality of each carry variable, from what the init block yields."""
    local: dict[VarName, float] = {}
    for t in s.init_block:
        if isinstance(t, StmtSample):
            local[t.var] = math.log(max(_cardinality(t.support, func_globals), 1))
        elif isinstance(t, StmtAssignSingle):
            local[t.target_var] = local.get(t.source_var, logcard.get(t.source_var, 0.0))
    last = s.init_block[-1] if s.init_block else None
    yielded = last.vars if isinstance(last, StmtYield) else []
    return {c: local.get(y, logcard.get(y, 0.0)) for c, y in zip(s.vars, yielded)}


def _order_level(
    body: list[Stmt], sink_vars: set[VarName], func_globals: dict, opaque: frozenset[VarName]
) -> list[Stmt]:
    """Reorder a block's factor statements (the block without its terminal
    return/yield)."""
    n = len(body)

    # Variable defined by each statement -> its index; and per-statement read set.
    # check_wellformed guarantees names are fresh within a block, so every variable
    # has exactly one defining statement here.
    def_of: dict[VarName, int] = {}
    defs: list[frozenset[VarName]] = []
    reads: list[frozenset[VarName]] = []
    for i, s in enumerate(body):
        d = frozenset(get_def_set(s))
        defs.append(d)
        for v in d:
            def_of[v] = i
    for i, s in enumerate(body):
        reads.append(get_use_set(s, opaque))

    # Cardinality (log) of each locally-defined variable, for the peak weighting.
    logcard: dict[VarName, float] = {}
    for i, s in enumerate(body):
        if isinstance(s, StmtSample):
            (v,) = tuple(defs[i])
            logcard[v] = math.log(max(_cardinality(s.support, func_globals), 1))
        elif isinstance(s, StmtAssignSingle):
            logcard[s.target_var] = logcard.get(s.source_var, 0.0)
        elif isinstance(s, StmtMarkov):
            logcard.update(_carry_logcard(s, logcard, func_globals))
        else:  # if-then-else return vars: cardinality unknown here -> neutral
            for v in defs[i]:
                logcard.setdefault(v, 0.0)

    # Local reads = read vars defined at this level (drop self-defs to avoid a
    # markov's body reads of its own carries becoming self-loops).
    lreads: list[frozenset[VarName]] = [
        frozenset(v for v in reads[i] if v in def_of and def_of[v] != i) for i in range(n)
    ]

    # Data-dependency parents.
    parents: list[set[int]] = [set(def_of[v] for v in lreads[i]) for i in range(n)]

    # Every statement is ordered (pal.src.prune drops factors that cannot affect the
    # query).
    keep: set[int] = set(range(n))

    children: list[set[int]] = [set() for _ in range(n)]
    for j in keep:
        for p in parents[j]:
            if p in keep:
                children[p].add(j)

    # Per-node ancestor count / cardinality mass, for DFS parent-ordering.
    anc_count: dict[int, float] = {}
    anc_mass: dict[int, float] = {}
    for i in _topo_indices(keep, parents):
        c = 1.0
        m = sum(logcard.get(v, 0.0) for v in defs[i])
        for p in parents[i]:
            if p in keep:
                c += anc_count[p]
                m += anc_mass[p]
        anc_count[i], anc_mass[i] = c, m

    # DFS seeds: kept nodes with no kept consumer (the block's "outputs" --
    # sink-defining and evidence nodes).
    seeds = [i for i in keep if not (children[i] & keep)]

    args = (keep, parents, children, defs, lreads, logcard, sink_vars)
    named = [
        # The program's own order. Definitions precede uses, so it is a valid
        # topological order, and keeping it as a candidate means the pass can never
        # choose a worse order than the one it was given -- generators that already
        # emit a good order (and differ only in how they break ties) then reorder to
        # the same peak.
        ("input", sorted(keep)),
        ("greedy", _greedy_order(*args)),
        ("dfs-bngen", _dfs_postorder(keep, parents, seeds, anc_count, descending=True)),
        ("dfs-mass-desc", _dfs_postorder(keep, parents, seeds, anc_mass, descending=True)),
        ("dfs-mass-asc", _dfs_postorder(keep, parents, seeds, anc_mass, descending=False)),
    ]
    sims = [_simulate_peak(o, keep, defs, lreads, logcard, sink_vars) for _, o in named]
    if os.environ.get("PAL_REORDER_DEBUG"):
        print("REORDER candidates (sim peak cells):", {nm: round(math.exp(s)) for (nm, _), s in zip(named, sims)})
    best = named[min(range(len(named)), key=lambda k: sims[k])][1]
    return [body[i] for i in best]


# ---------------------------------------------------------------------------
# Order construction (var-set based: a node may define 0, 1, or several vars).
# ---------------------------------------------------------------------------

def _consumer_counts(keep, lreads):
    """var -> number of kept statements that read it."""
    cnt: dict[VarName, int] = {}
    for i in keep:
        for v in lreads[i]:
            cnt[v] = cnt.get(v, 0) + 1
    return cnt


def _greedy_order(keep, parents, children, defs, lreads, logcard, sink_vars):
    """Forward greedy: at each step place the ready node leaving the smallest
    weighted live set."""
    remaining = _consumer_counts(keep, lreads)
    for i in keep:
        for v in defs[i]:
            remaining.setdefault(v, 0)
    unmet = {i: len(parents[i] & keep) for i in keep}
    ready = {i for i in keep if unmet[i] == 0}
    placed: set[int] = set()
    order: list[int] = []
    live: set[VarName] = set()

    def wsum(vs):
        return sum(logcard.get(v, 0.0) for v in vs)

    while len(placed) < len(keep):
        best, best_key = None, None
        for i in ready:
            newlive = live | defs[i]
            dead = set()
            for v in lreads[i]:
                if remaining.get(v, 0) == 1 and v not in sink_vars:
                    dead.add(v)
            for v in defs[i]:
                if remaining.get(v, 0) == 0 and v not in sink_vars:
                    dead.add(v)
            key = (wsum(newlive - dead), wsum(defs[i]), i)
            if best_key is None or key < best_key:
                best, best_key = i, key
        i = best
        placed.add(i)
        order.append(i)
        ready.discard(i)
        live |= defs[i]
        for v in lreads[i]:
            if v in remaining:
                remaining[v] -= 1
                if remaining[v] == 0 and v not in sink_vars:
                    live.discard(v)
        for v in defs[i]:
            if remaining.get(v, 0) == 0 and v not in sink_vars:
                live.discard(v)
        for c in children[i]:
            if c in keep and c not in placed:
                unmet[c] -= 1
                if unmet[c] == 0:
                    ready.add(c)
    return order


def _dfs_postorder(keep, parents, seeds, weight, descending):
    """Depth-first post-order from the output nodes over parents, visiting the
    heaviest- (or lightest-) weight parent first. Iterative.

    With ``anc_count`` as the weight, a node's weight sums its parents' counts (a shared
    ancestor counts once per path); ties are broken by statement position."""
    sign = -1.0 if descending else 1.0

    def key(i):
        return sign * weight[i]

    visited: set[int] = set()
    order: list[int] = []
    for seed in sorted(seeds, key=key):
        if seed in visited:
            continue
        stack = [(seed, False)]
        while stack:
            node, expanded = stack.pop()
            if node in visited:
                continue
            if expanded:
                visited.add(node)
                order.append(node)
                continue
            stack.append((node, True))
            for p in sorted((p for p in parents[node] if p in keep), key=key, reverse=True):
                if p not in visited:
                    stack.append((p, False))
    # Any kept node not reachable from a seed (e.g. an isolated evidence node with
    # no ancestors that was itself a seed is already covered; guard anyway).
    for i in keep:
        if i not in visited:
            order.append(i)
            visited.add(i)
    return order


def _topo_indices(keep, parents):
    unmet = {i: len(parents[i] & keep) for i in keep}
    children: dict[int, list[int]] = {i: [] for i in keep}
    for j in keep:
        for p in parents[j]:
            if p in keep:
                children[p].append(j)
    q = [i for i in keep if unmet[i] == 0]
    out = []
    while q:
        i = q.pop()
        out.append(i)
        for c in children[i]:
            unmet[c] -= 1
            if unmet[c] == 0:
                q.append(c)
    return out


def _simulate_peak(order, keep, defs, lreads, logcard, sink_vars):
    """Log of the peak weighted live set for a given order under eager kill."""
    remaining = _consumer_counts(keep, lreads)
    for i in keep:
        for v in defs[i]:
            remaining.setdefault(v, 0)
    live: set[VarName] = set()
    w = 0.0
    peak = 0.0
    for i in order:
        for v in defs[i]:
            if v not in live:
                live.add(v)
                w += logcard.get(v, 0.0)
        if w > peak:
            peak = w
        for v in lreads[i]:
            if v in remaining:
                remaining[v] -= 1
                if remaining[v] == 0 and v not in sink_vars and v in live:
                    live.discard(v)
                    w -= logcard.get(v, 0.0)
        for v in defs[i]:
            if remaining.get(v, 0) == 0 and v not in sink_vars and v in live:
                live.discard(v)
                w -= logcard.get(v, 0.0)
    return peak
