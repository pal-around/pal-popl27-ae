"""The dicefig1 model in memo: a Boolean chain x0 ~ [0.9, 0.1], then n steps of
x_{i+1} ~ [0.6, 0.4] if not x_i else [0.7, 0.3]; the query is the last variable.

    python memo_dicefig1.py --n N

Calls the model twice in one process and prints each call's time (``Run 1``, ``Run 2``;
the first includes JAX's compilation) and the distribution of the last variable.
"""

import argparse
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

from memo import memo  # noqa: E402

Bool = [False, True]


@memo
def f[z: Bool](n):
    world: given(prev in Bool, wpp=(0.1 if prev else 0.9) if n == 0 else f[prev](n - 1))  # noqa: F821
    world: given(curr in Bool, wpp=(prev == curr) if n == 0 else (  # noqa: F821
        (0.3 if curr else 0.7) if prev else  # noqa: F821
        (0.4 if curr else 0.6)  # noqa: F821
    ))
    return Pr[world.curr == z]  # noqa: F821


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()
    if args.n < 0:
        parser.error("--n must be non-negative")
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * args.n + 1000))

    ret = None
    for run in (1, 2):
        start = time.perf_counter()
        ret = jax.block_until_ready(f(args.n))
        print(f"Run {run}: {time.perf_counter() - start:.10f}")
    print("POPL-AE device:", jax.devices()[0].device_kind)
    print("POPL-AE dtype:", ret.dtype)
    print("POPL-AE answer:", " ".join(format(float(v), ".17g") for v in ret.ravel()))


if __name__ == "__main__":
    main()
