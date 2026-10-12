"""The Bayes-net generators give PAL and Dice the same program: bngen's
(SHoltzen/bngen), pruned to the query, with the same variables in bngen's order and the
same weights. The generators run as the artifact runs them, from the command line."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BN = Path(__file__).resolve().parent.parent / "bayesian_networks"
NETWORKS = sorted(p.stem for p in (BN / "bif").glob("*.bif"))

pytestmark = pytest.mark.skipif(shutil.which("bngen-upstream") is None, reason="bngen-upstream is not on PATH")


def run(*argv: str) -> str:
    return subprocess.run(argv, cwd=BN, capture_output=True, text=True, check=True).stdout


def query_ancestors(net: str) -> set[str]:
    code = ("import sys; from bif import ancestors; from networks import load; "
            "_, _, cpt, t = load(sys.argv[1]); print(' '.join(sorted(ancestors(t, cpt))))")
    return set(run(sys.executable, "-c", code, f"bif/{net}.bif").split())


@pytest.mark.parametrize("net", NETWORKS)
def test_dice_program_is_bngens_pruned_to_the_query(net):
    full = [ln for ln in run("bngen-upstream", "sym", f"bif/{net}.bif").splitlines() if ln.startswith("let ")]
    dice = run(sys.executable, "gen_dice.py", "--bif", f"bif/{net}.bif").splitlines()
    needed = query_ancestors(net)
    assert dice[:-1] == [ln for ln in full if ln.split()[1] in needed]
    assert dice[-1] in needed


@pytest.mark.parametrize("net", NETWORKS)
def test_pal_program_samples_the_dice_programs_variables_in_order(net):
    pal = run(sys.executable, "gen_pal.py", "--bif", f"bif/{net}.bif")
    dice = run(sys.executable, "gen_dice.py", "--bif", f"bif/{net}.bif")
    assert re.findall(r"^    (\S+): \S+ = sample\(", pal, re.M) == re.findall(r"^let (\S+) = ", dice, re.M)
    assert re.search(r"^    return (\S+)$", pal, re.M).group(1) == dice.strip().splitlines()[-1]
    # The same weights, in the same order (PAL's tables and Dice's if-chains both
    # enumerate parent values first to last).
    assert re.findall(r"[0-9]+\.[0-9]+", pal) == re.findall(r"[0-9]+\.[0-9]+", dice)
