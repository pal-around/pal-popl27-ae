"""Check that every configuration that finished a case gives the same answer.

    python report/check.py RESULTS [--tol 1e-5]

For each suite and case, compares the normalized distributions in the ``answer`` column
of every configuration's CSV, prints the cases that disagree by more than ``--tol``, and
exits with status 1 if there are any. The Bayesian networks' weights are written with
six decimals, so a few rows of their tables sum to 1 only within 1e-6, and PAL and Dice
handle such a row differently; the default tolerance allows for that.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()

    answers: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
    for path in sorted(a.results.glob("*/*.csv")):
        for r in csv.DictReader(path.open()):
            if r.get("status") == "ok" and r.get("answer"):
                answers[(r["suite"], r["case_id"])][r["config"]] = [float(x) for x in r["answer"].split()]

    compared = disagree = 0
    for (suite, case), by_config in sorted(answers.items()):
        if len(by_config) < 2:
            continue
        compared += 1
        configs = sorted(by_config)
        ref = by_config[configs[0]]
        for config in configs[1:]:
            other = by_config[config]
            width = max(len(ref), len(other))
            diff = max(abs((ref[i] if i < len(ref) else 0.0) - (other[i] if i < len(other) else 0.0))
                       for i in range(width))
            if diff > a.tol:
                disagree += 1
                print(f"DISAGREE {suite} {case}: {configs[0]} vs {config}, max difference {diff:.3g}")
    if compared == 0:
        print("No case was run by two or more configurations; nothing to compare.")
    else:
        print(f"{compared} cases run by two or more configurations; "
              f"{'all agree' if disagree == 0 else f'{disagree} disagreements'} (tolerance {a.tol:g}).")
    sys.exit(1 if disagree else 0)


if __name__ == "__main__":
    main()
