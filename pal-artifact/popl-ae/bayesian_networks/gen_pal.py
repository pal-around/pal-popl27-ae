"""Emit a PAL program for one Bayesian network.

Each conditional probability table is written as a guarded table over its parents:
``x > [e0, e1, ...]`` means ``e0 if x == 0 else e1 if x == 1 else ...``, and a bare list
is a row of weights for the variable being sampled::

    Cancer: Discrete2 = sample(Pollution > [Smoker > [[0.030000, 0.970000], [0.001000, 0.999000]], ...])

Only the query's ancestors are emitted, in bngen's order (see bngen.py), which is the
order gen_dice.py's Dice program binds them in; the elimination order is the compiler's
(its reorder pass is on by default). Weights are written as bngen writes them: bngen
keeps each weight as its log and prints the exp of that with ``%f``. The query comes from ``--node`` or from
``networks.NODE_IDS``.

Writes the program to stdout.
"""

import argparse
import math
import sys

from bngen import program_order
from networks import load


def bngen_weight(w: float) -> str:
    """The weight as bngen's Dice program writes it: ``%f`` of exp(log(w)), so 0.0049505
    reads 0.004950."""
    return "%f" % (math.exp(math.log(w)) if w > 0 else 0.0)


def format_row(weights: list[float]) -> str:
    return "[" + ", ".join(bngen_weight(w) for w in weights) + "]"


def format_weights(var: str, values: dict[str, list[str]], cpt: dict) -> str:
    """The variable's table as nested guarded tables over its parents."""
    parents, rows = cpt[var]
    if not parents:
        return format_row(rows[()])

    def nest(i: int, prefix: list[str]) -> str:
        if i == len(parents):
            return format_row(rows[tuple(prefix)])
        branches = ", ".join(nest(i + 1, prefix + [v]) for v in values[parents[i]])
        return f"{parents[i]} > [{branches}]"

    return nest(0, [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bif", required=True, help="path to the .bif network file")
    parser.add_argument("--node", type=int, default=None,
                        help="node id to query; default = NODE_IDS[<network>] (falls back to the last node)")
    args = parser.parse_args()

    _, values, cpt, target = load(args.bif, args.node)
    variables = program_order(args.bif, cpt, target)
    domains = sorted({len(values[v]) for v in variables})

    lines = ["from pal import pal, time_execution, Fin", ""]
    lines += [f"Discrete{n} = Fin({n})" for n in domains]
    lines += ["", "@pal", "def model():"]
    for v in variables:
        lines.append(f"    {v}: Discrete{len(values[v])} = sample({format_weights(v, values, cpt)})")
    lines += [f"    return {target}", "", "time_execution(model)", "time_execution(model)"]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
