"""natlist trains with Adam on the exact KL, which is 30.7285 at theta = 0 for N = 100."""

import importlib.util
import math
from pathlib import Path

import pytest

NATLIST = Path(__file__).resolve().parent.parent / "natlist" / "natlist.py"


@pytest.fixture(scope="module")
def natlist():
    spec = importlib.util.spec_from_file_location("ae_natlist", NATLIST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines the model; nothing is compiled or run
    return module


def test_first_loss_is_the_exact_kl_and_each_step_is_adams(natlist, tmp_path):
    import jax
    import jax.numpy as jnp
    import optax

    call = natlist.compile_model(100, 100, "float64")
    losses, _ = natlist.train(call, 100, epochs=3, lr=0.3, dtype="float64", output_csv=tmp_path / "loss.csv")
    exact = sum((1 / 101) * math.log((1 / 101) / p) for p in [0.5 ** (k + 1) for k in range(100)] + [0.5 ** 100])
    assert losses[0] == pytest.approx(exact, rel=1e-9)
    assert losses[0] == pytest.approx(30.72852284834407, rel=1e-9)

    # The second loss is the loss after one Adam step from theta = 0.
    target = jnp.full(101, 1 / 101)

    def loss_of(theta):
        return natlist.kl_divergence_loss(call(natlist.sigmoid(theta)), target)

    theta = jnp.zeros(100)
    grads = jax.grad(loss_of)(theta)
    opt = optax.adam(0.3)
    updates, _ = opt.update(grads, opt.init(theta), theta)
    assert float(loss_of(theta + updates)) == pytest.approx(losses[1], rel=1e-9)
