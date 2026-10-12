"""Emit a Dice program for one Bayesian network: bngen's program, pruned to the query.

    python gen_dice.py --bif FILE.bif [--node N]

The program is the ``let`` bindings that bngen (see bngen.py) writes with ``sym`` for the
query's ancestors, character for character and in bngen's order, followed by the query
variable. gen_pal.py samples the same variables in the same order. The query comes from
``--node`` or from ``networks.NODE_IDS``.

Writes the program to stdout.
"""

import argparse
import sys

from bif import ancestors
from bngen import sym_bindings
from networks import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bif", required=True, help="path to the .bif network file")
    parser.add_argument("--node", type=int, default=None,
                        help="node id to query; default = NODE_IDS[<network>] (falls back to the last node)")
    args = parser.parse_args()

    _, _, cpt, target = load(args.bif, args.node)
    needed = ancestors(target, cpt)
    lines = [line for name, line in sym_bindings(args.bif) if name in needed]
    lines.append(target)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
