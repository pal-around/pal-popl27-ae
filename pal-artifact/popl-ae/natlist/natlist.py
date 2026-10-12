"""NatList-N: tune a random list generator so its length distribution becomes uniform.

    python natlist.py [--n 100] [--epochs 1000] [--lr 0.3] [--output-csv path]

The generator stops at each of N positions with a trainable probability sigmoid(theta_i)
and otherwise conses an element (drawn uniformly, and never read again). Training minimizes KL(uniform over 0..N ‖ length distribution) from theta = 0.

The loss is the exact KL, computed from log-probabilities, and the update is Adam.

The model is one `markov` loop whose per-iteration input is the step's probability, so
its size is a parameter.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.special
import optax

from pal import Bool, Fin, pal  # noqa: F401

EPOCHS = 1000
LR = 0.3
N_DEFAULT = 100
ELEM_DEFAULT = 100

# PAL resolves a domain name in the model's module globals when it compiles, so the two
# domains are set from --n and --elem before the model is compiled (see `compile_model`).
T: list = []
LEN: list = []


def gen_natlist(theta, n):
    stopped0: Bool = False  # type: ignore
    len0: LEN = 0  # type: ignore
    with markov(stopped := stopped0, lenv := len0, repeat=n, xs=(p := theta)):  # type: ignore # noqa: F821
        x: T = sample(1)  # type: ignore # noqa: F821,F841
        s: Bool = sample((1 if s == 1 else 0) if stopped else [1 - p, p])  # type: ignore # noqa: F821
        new_len: LEN = lenv if s else lenv + 1  # type: ignore # noqa: F821
        yield s, new_len
    return lenv  # type: ignore # noqa: F821


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def kl_divergence_loss(likelihood, target):
    """Exact KL(target ‖ likelihood), in log space.

    PAL returns probabilities, so their logarithm is taken directly and renormalized with
    logsumexp, which removes the rounding in the probabilities' sum. A probability of
    exactly zero -- which float64 reaches once a length's probability falls below about
    1e-308 -- gives an infinite loss."""
    log_likelihood = jnp.log(likelihood)
    log_p = log_likelihood - jax.scipy.special.logsumexp(log_likelihood)
    return jnp.sum(target * (jnp.log(target) - log_p))


def compile_model(n: int, elem: int, dtype: str):
    """The loop model, compiled at the requested size and precision. JAX's 64-bit mode
    must be set before any array is created, so this runs first thing in main()."""
    global T, LEN
    jax.config.update("jax_enable_x64", dtype == "float64")
    T = Fin(elem)
    LEN = Fin(n + 1)
    model = pal(dtype=dtype, print_final_likelihood=False)(gen_natlist)
    return lambda probs: model(probs, n)


def train(call, n: int, *, epochs: int, lr: float, dtype: str, output_csv: Path):
    """Adam on the exact KL, from theta = 0. Returns the loss curve and
    the times at which each epoch ended, with the one-off compile reported separately."""
    real = jnp.float64 if dtype == "float64" else jnp.float32
    target = jnp.full(n + 1, 1 / (n + 1), dtype=real)
    params = jnp.zeros(n, dtype=real)
    optimizer = optax.adam(learning_rate=lr)

    def loss_of(theta):
        return kl_divergence_loss(call(sigmoid(theta)), target)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_of)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    # The initial length distribution: 2^-(k+1), which underflows to zero in float64
    # once N passes about 1073, and the loss is then infinite; `status` reports it.
    initial = jnp.asarray(call(sigmoid(params)))
    smallest = float(jnp.min(initial))
    status = "ok" if smallest > 0 else "underflow"
    print(f"smallest initial probability: {smallest:.3g}")

    jit_start = time.perf_counter()
    _, _, warmup_loss = step(params, optimizer.init(params))
    float(warmup_loss)
    jit_seconds = time.perf_counter() - jit_start

    opt_state = optimizer.init(params)
    losses, times = [], []
    train_start = time.perf_counter()
    prev_epoch_end = 0.0
    for _ in range(epochs):
        params, opt_state, loss = step(params, opt_state)
        loss = float(loss)
        losses.append(loss)
        times.append(prev_epoch_end)
        if loss != loss or loss in (float("inf"), float("-inf")):
            status = "underflow"
            break
        prev_epoch_end = time.perf_counter() - train_start
    train_seconds = time.perf_counter() - train_start

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "time"])
        for epoch, (loss, elapsed) in enumerate(zip(losses, times, strict=True)):
            writer.writerow([epoch, loss, elapsed])

    print(f"JIT compilation: {jit_seconds:.6f}s")
    print(f"Training: {train_seconds:.6f}s  ({train_seconds / max(len(losses), 1) * 1000:.4f}ms per epoch)")
    print(f"Final loss: {losses[-1]:.8f}")
    print(f"status: {status}")
    print(f"Saved training losses to: {output_csv}")
    return losses, times


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="list length (and target values 0..n)")
    parser.add_argument("--elem", type=int, default=ELEM_DEFAULT, help="element domain size")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args(argv)


def default_output_path(filename: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(__file__).resolve().parent / stamp / filename


def log_device_info():
    print("JAX device info:")
    print(jax.devices())
    for device in jax.devices():
        print(f"  Device: {device} == {device.device_kind}")
    print(f"jax.default_backend() == {jax.default_backend()}")
    print()


def main(argv=None):
    args = parse_args(argv)
    call = compile_model(args.n, args.elem, args.dtype)
    log_device_info()
    print(f"n: {args.n}  elem: {args.elem}  epochs: {args.epochs}  lr: {args.lr}  dtype: {args.dtype}")
    train(
        call,
        args.n,
        epochs=args.epochs,
        lr=args.lr,
        dtype=args.dtype,
        output_csv=args.output_csv or default_output_path("pal_loss.csv"),
    )


if __name__ == "__main__":
    main()
