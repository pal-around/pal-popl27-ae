"""Minimal ``.bif`` reader for the networks under ``bayesian_networks/bif``.

Only what a generator needs: each variable's values in declaration order, and each
conditional probability table as ``(parents, rows)`` where ``rows`` maps a tuple of
parent *values* to the variable's weights."""

import re


def parse_bif(path: str) -> tuple[list[str], dict[str, list[str]], dict[str, tuple[list[str], dict]]]:
    """Return ``(declaration order, values per variable, cpt per variable)``."""
    text = open(path, errors="ignore").read()

    values: dict[str, list[str]] = {}
    order: list[str] = []
    for m in re.finditer(
        r"variable\s+(\S+)\s*\{\s*type\s+discrete\s*\[\s*(\d+)\s*\]\s*\{([^}]*)\}", text, re.S
    ):
        name = m.group(1)
        vals = [s.strip() for s in m.group(3).split(",") if s.strip()]
        assert len(vals) == int(m.group(2)), (name, vals)
        values[name] = vals
        order.append(name)

    cpt: dict[str, tuple[list[str], dict]] = {}
    for m in re.finditer(r"probability\s*\(\s*([^|)]+?)\s*(?:\|\s*([^)]*))?\)\s*\{(.*?)\}", text, re.S):
        name = m.group(1).strip()
        parents = [p.strip() for p in (m.group(2) or "").split(",") if p.strip()]
        body = m.group(3)
        rows: dict[tuple[str, ...], list[float]] = {}
        if not parents:
            table = re.search(r"table\s*([^;]*);", body).group(1)
            rows[()] = [float(x) for x in table.replace(",", " ").split()]
        else:
            for row in re.finditer(r"\(([^)]*)\)\s*([^;]*);", body):
                key = tuple(s.strip() for s in row.group(1).split(","))
                rows[key] = [float(x) for x in row.group(2).replace(",", " ").split()]
        cpt[name] = (parents, rows)

    return order, values, cpt


def ancestors(v: str, cpt: dict[str, tuple[list[str], dict]]) -> set[str]:
    """``v`` and everything it depends on."""
    out: set[str] = set()
    stack = [v]
    while stack:
        u = stack.pop()
        if u in out:
            continue
        out.add(u)
        stack.extend(cpt[u][0])
    return out
