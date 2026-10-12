"""The variable order and Dice text that bngen gives a network.

bngen is https://github.com/SHoltzen/bngen (`bngen-upstream` in the dev shell; see
nix/bngen-upstream.nix). Its `sym` command writes every variable of a ``.bif`` network as
``let <name> = <table> in``, in the network's topological order (``make_topo_vars`` in its
``bnType.ml``), and ends with a tuple of all of them.
"""

import subprocess

from bif import ancestors

BNGEN = "bngen-upstream"


def sym_bindings(bif: str) -> list[tuple[str, str]]:
    """``(name, line)`` for each ``let`` line of ``bngen sym``, in bngen's order."""
    out = subprocess.run([BNGEN, "sym", bif], capture_output=True, text=True, check=True).stdout
    bindings = []
    for line in out.splitlines():
        if line.startswith("let "):
            assert line.endswith(" in"), line
            bindings.append((line[len("let "):line.index(" = ")], line))
    return bindings


def program_order(bif: str, cpt: dict, target: str) -> list[str]:
    """The query's ancestors (itself included), in bngen's order."""
    needed = ancestors(target, cpt)
    return [name for name, _ in sym_bindings(bif) if name in needed]
