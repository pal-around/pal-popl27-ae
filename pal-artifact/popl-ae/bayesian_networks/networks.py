"""Which variable each Bayesian-network benchmark queries.

Both generators (PAL, Dice) resolve their query here, so both answer the same
question. A query is an index into the network's variables in ``.bif`` declaration
order -- the numbering ``bngen`` uses.
"""

import os

from bif import parse_bif

# Per-network query node.
NODE_IDS = {
    "cancer": 3,
    "survey": 5,
    "alarm": 25,
    "insurance": 19,
    "hepar2": 24,
    "pigs": 3,
    "water": 26,
    "munin": 1040,
    "andes": 220,
    "link": 15,
    "diabetes": 411,
    "hailfinder": 43,
}


def resolve_node(bif: str, order: list[str], node: int | None = None) -> int:
    """The query's index: ``node`` if given, else the network's entry in NODE_IDS, else
    the last variable."""
    if node is None:
        name = os.path.splitext(os.path.basename(bif))[0]
        node = NODE_IDS.get(name, len(order) - 1)
    if not 0 <= node < len(order):
        raise SystemExit(f"node {node} out of range for {bif} ({len(order)} variables)")
    return node


def resolve_target(bif: str, order: list[str], node: int | None = None) -> str:
    """The query variable's name."""
    return order[resolve_node(bif, order, node)]


def load(bif: str, node: int | None = None):
    """Parse ``bif`` and resolve its query: ``(order, values, cpt, target)``."""
    order, values, cpt = parse_bif(bif)
    return order, values, cpt, resolve_target(bif, order, node)
