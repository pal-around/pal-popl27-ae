"""Every program under ``examples/`` runs and prints the right answer.

Each example runs in its own process, the way a reader would run it, on the NumPy
backend and on JAX (CPU). Its last line of output ends with the distribution it
computes, which is compared with an answer computed here directly in NumPy."""

import ast
import os
import subprocess
import sys
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def ladder():
    # Forward over the routers (x1, x2) of a layer; the packet starts at x1.
    a = 0.9
    dist = {(True, False): 1.0}
    for _ in range(10):
        nxt = {}
        for (x1, x2), w in dist.items():
            for s11, s12, s21, s22 in product([False, True], repeat=4):
                ws = np.prod([a if s else 1 - a for s in (s11, s12, s21, s22)])
                key = ((x1 and s11) or (x2 and s21), (x1 and s12) or (x2 and s22))
                nxt[key] = nxt.get(key, 0.0) + w * ws
        dist = nxt
    success = sum(w for (x1, x2), w in dist.items() if x1 or x2)
    return [1 - success, success]


def casino():
    # Joint over (cheating, loaded die at the end), by the forward algorithm.
    faces = np.array([3, 6, 1, 5, 2, 6, 6, 4, 6, 6, 6, 3, 6, 6, 2, 6, 1, 4, 3, 5]) - 1
    honest = 0.5 * (1 / 6) ** len(faces)
    trans = np.array([[0.95, 0.05], [0.1, 0.9]])
    emit = np.array([[1 / 6] * 6, [0.1] * 5 + [0.5]])
    alpha = np.array([1.0, 0.0])
    for f in faces:
        alpha = (alpha @ trans) * emit[:, f]
    joint = np.array([[honest, 0.0], 0.5 * alpha])
    return joint / joint.sum()


def delivery(observed=8):
    def total(leg, legs):
        dist = np.array([1.0])
        for _ in range(legs):
            dist = np.convolve(dist, leg)
        return dist[observed] if observed < len(dist) else 0.0

    joint = 0.5 * np.array([total([0.3, 0.5, 0.2], 6), total([0, 0.1, 0.3, 0.4, 0.2], 2)])
    return joint / joint.sum()


def hmm():
    # Forward algorithm over [no rain, rain].
    trans = np.array([[0.7, 0.3], [0.3, 0.7]])
    emit = np.array([[0.8, 0.2], [0.1, 0.9]])
    alpha = np.array([0.5, 0.5])
    for u in [1, 1, 0, 1, 1, 0, 0, 1, 1, 1]:
        alpha = (alpha @ trans) * emit[:, u]
    return alpha / alpha.sum()


def gamblers_ruin():
    p = 0.45
    step = np.zeros((7, 7))
    step[0, 0] = step[6, 6] = 1
    for c in range(1, 6):
        step[c, c + 1], step[c, c - 1] = p, 1 - p
    return np.linalg.matrix_power(step, 20)[3]


def random_trials():
    p = 0.6
    attempts = [0.1, 0.2, 0.3, 0.25, 0.15]
    dist = np.zeros(5)
    for n, w in enumerate(attempts):
        for k in range(n + 1):
            dist[k] += w * comb(n, k) * p**k * (1 - p) ** (n - k)
    return dist


ANSWERS = {
    "ladder.py": ladder,
    "casino.py": casino,
    "delivery.py": delivery,
    "hmm.py": hmm,
    "gamblers_ruin.py": gamblers_ruin,
    "random_trials.py": random_trials,
}


def test_every_example_has_an_answer():
    assert sorted(p.name for p in EXAMPLES.glob("*.py")) == sorted(ANSWERS)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_example_prints_the_answer(name, backend):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    flags = ["--numpy"] if backend == "numpy" else []
    result = subprocess.run(
        [sys.executable, name, *flags],
        cwd=EXAMPLES,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Run 2:" in result.stdout, result.stdout[-2000:]
    last = result.stdout.strip().splitlines()[-1]
    printed = ast.literal_eval(last[last.index("[") :])
    np.testing.assert_allclose(printed, ANSWERS[name](), rtol=0, atol=1e-9)
