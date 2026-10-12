"""Longer end-to-end tests.

These exercise several syntactic features at once and check that the
inferred distributions match independent NumPy computations.
"""

import itertools

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, Fin, condition, markov, observe, sample

Fin3 = Fin(3)
Fin6 = Fin(6)


def test_cancer_bayes_net_matches_hand_computed_marginal():
    """A Bayesian network for lung cancer (pollution, smoking, cancer, X-ray,
    dyspnoea): jointly samples five variables and returns the (Xray, Dyspnoea)
    marginal."""

    def cancer():
        Pollution: Fin3 = sample(
            0.5 if _ == 0 else 0.4 if _ == 1 else 0.1
        )
        Smoker: Bool = sample(0.3 if _ == 0 else 0.7)
        Cancer: Bool = sample(
            (
                (0.03 if _ == 0 else 0.97)
                if Smoker == 0
                else (0.001 if _ == 0 else 0.999)
            )
            if Pollution == 0
            else (
                (
                    (0.03 if _ == 0 else 0.97)
                    if Smoker == 0
                    else (0.001 if _ == 0 else 0.999)
                )
                if Pollution == 1
                else (
                    (0.05 if _ == 0 else 0.95)
                    if Smoker == 0
                    else (0.02 if _ == 0 else 0.98)
                )
            )
        )
        Xray: Bool = sample(
            (0.9 if _ == 0 else 0.1)
            if Cancer == 0
            else (0.2 if _ == 0 else 0.8)
        )
        Dyspnoea: Bool = sample(
            (0.65 if _ == 0 else 0.35)
            if Cancer == 0
            else (0.3 if _ == 0 else 0.7)
        )
        return Xray, Dyspnoea

    r = compile_pal_fn(cancer)()

    # Independent NumPy reference computation
    P_pol = np.array([0.5, 0.4, 0.1])
    P_smk = np.array([0.3, 0.7])

    def p_cancer(pol, smk):
        if pol in (0, 1):
            return np.array([0.03, 0.97]) if smk == 0 else np.array([0.001, 0.999])
        return np.array([0.05, 0.95]) if smk == 0 else np.array([0.02, 0.98])

    def p_xray(c):
        return np.array([0.9, 0.1]) if c == 0 else np.array([0.2, 0.8])

    def p_dys(c):
        return np.array([0.65, 0.35]) if c == 0 else np.array([0.3, 0.7])

    expected = np.zeros((2, 2))
    for pol, smk, can in itertools.product(range(3), range(2), range(2)):
        w = P_pol[pol] * P_smk[smk] * p_cancer(pol, smk)[can]
        expected += w * np.outer(p_xray(can), p_dys(can))

    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


def test_mixed_coins_via_if_then_else():
    """A mixture model: a fair coin or a biased coin, depending on whether the
    agent decided to cheat."""

    def mixed_coins():
        cheat: Bool = sample([0.5, 0.5])
        with out:
            if cheat == 0:
                fair: Bool = sample([0.5, 0.5])
                yield fair
            else:
                biased: Bool = sample([0.9, 0.1])
                yield biased

        return out

    r = compile_pal_fn(mixed_coins)()
    np.testing.assert_allclose(r, [0.5 * 0.5 + 0.5 * 0.9, 0.5 * 0.5 + 0.5 * 0.1])


def test_natlist_style_chain_of_stop_decisions():
    """Small natlist-style model: at each step we either stop or continue,
    so the length distribution is geometric until the prefix is exhausted."""

    def gen_nat_chain(p=0.5):
        stop_0: Bool = sample([1 - p, p])
        stop_1: Bool = sample(
            (1 if stop_1 == 1 else 0) if stop_0 else [1 - p, p]
        )
        stop_2: Bool = sample(
            (1 if stop_2 == 1 else 0) if stop_1 else [1 - p, p]
        )
        length: Fin3 = 0 if stop_0 else 1 if stop_1 else 2
        return length

    compiled = compile_pal_fn(gen_nat_chain)

    for p in (0.25, 0.5, 0.75):
        r = compiled(p)
        # P(length=0) = p
        # P(length=1) = (1-p) * p
        # P(length=2) = (1-p) * (1-p) — the chain stays "not stopped" thereafter
        expected = np.array([p, (1 - p) * p, (1 - p) * (1 - p)])
        np.testing.assert_allclose(r, expected, atol=1e-7)
        assert pytest.approx(r.sum(), abs=1e-6) == 1.0


def test_markov_with_if_then_else_block_inside():
    """A 2-state markov chain whose body samples a "flip" coin and uses an
    if-then-else block to either copy or invert the current state."""

    def f():
        init: Bool = sample([0.9, 0.1])
        with markov(state := init, repeat=2):
            flip: Bool = sample([0.8, 0.2])
            with new_state:
                if flip == 0:
                    a: Bool = state
                    yield a
                else:
                    b: Bool = 1 - state
                    yield b
            yield new_state

        return state

    r = compile_pal_fn(f)()
    # Transition matrix T[s, s'] is determined by P(no flip)=0.8.
    T = np.array([[0.8, 0.2], [0.2, 0.8]])
    expected = np.array([0.9, 0.1]) @ T @ T
    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


def test_markov_inside_if_then_else_block():
    """One branch of an if-then-else block runs a deterministically-flipping
    chain for two steps (which returns to its starting state); the other
    branch is a constant."""

    def f():
        mode: Bool = sample([0.3, 0.7])
        with out:
            if mode == 0:
                x: Bool = 1
                yield x
            else:
                init: Bool = 0
                with markov(s := init, repeat=2):
                    t: Bool = sample(1 if t != s else 0)
                    yield t
                yield s

        return out

    r = compile_pal_fn(f)()
    # mode==0 (P=0.3): out=1. mode==1 (P=0.7): two deterministic flips → 0.
    np.testing.assert_allclose(r, [0.7, 0.3], atol=1e-7)


def test_nested_if_then_else_blocks():
    """An if-then-else block whose ``if`` branch itself contains an
    if-then-else block."""

    def f():
        a: Fin3 = sample([0.4, 0.4, 0.2])
        b: Bool = sample([0.6, 0.4])
        with out:
            if a == 0:
                with inner:
                    if b == 0:
                        x: Fin3 = 0
                        yield x
                    else:
                        y: Fin3 = 1
                        yield y
                yield inner
            elif a == 1:
                z: Fin3 = 2
                yield z
            else:
                w: Fin3 = 0
                yield w

        return out

    r = compile_pal_fn(f)()
    # a=0 (0.4): out=0 w.p. 0.6, out=1 w.p. 0.4
    # a=1 (0.4): out=2
    # a=2 (0.2): out=0
    expected = np.array([0.4 * 0.6 + 0.2, 0.4 * 0.4, 0.4])
    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


@pytest.mark.parametrize("lva", [True, False])
def test_combined_sample_assign_observe(lva):
    """Exercises every "flat" construct in a single program: ``sample`` for
    the priors, both annotated-assignment forms (single-name and
    expression), and ``observe`` to condition. The intermediate variables are
    marginalized by the compiler (eagerly, or where they go out of scope)."""

    def f():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.7, 0.3])
        z: Bool = x  # (assign single)
        eq: Bool = x == y  # (assign from expression)
        observe(eq, 1)
        return y

    r = compile_pal_fn(f, lva=lva)()
    # observe(eq, 1) forces x == y. observe leaves the tensor unnormalized, so
    # the result is the joint (x=0,y=0)=0.35, (x=1,y=1)=0.15; the total mass
    # 0.5 is P(x == y). Normalizing gives P(y=0)=0.7, P(y=1)=0.3.
    np.testing.assert_allclose(r, [0.35, 0.15], atol=1e-7)
    np.testing.assert_allclose(r / r.sum(), [0.7, 0.3], atol=1e-7)


def test_markov_with_inline_if_then_else_expression_in_sample():
    """The transition kernel inside a markov chain is given by an inline
    if-then-else expression over probability tables."""

    def f():
        init: Bool = sample([1.0, 0.0])
        with markov(s := init, repeat=3):
            new_s: Bool = sample([0.8, 0.2] if s == 0 else [0.3, 0.7])
            yield new_s

        return s

    r = compile_pal_fn(f)()
    T = np.array([[0.8, 0.2], [0.3, 0.7]])
    expected = np.array([1.0, 0.0]) @ T @ T @ T
    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


def test_markov_with_assign_single_inside_body():
    """A markov body that just copies the previous state via an
    assign-single statement; the marginal is therefore stationary."""

    def f():
        init: Fin3 = sample([0.5, 0.3, 0.2])
        with markov(s := init, repeat=2):
            new_s: Fin3 = s  # (assign single)
            yield new_s

        return s

    r = compile_pal_fn(f)()
    np.testing.assert_allclose(r, [0.5, 0.3, 0.2], atol=1e-7)


def test_if_then_else_block_then_observe_on_yielded_variable():
    """An if-then-else block produces an output variable; we then
    ``observe`` it and read off the posterior over the latent condition."""

    def f():
        cond: Bool = sample([0.5, 0.5])
        with out:
            if cond == 0:
                a: Bool = sample([0.3, 0.7])
                yield a
            else:
                b: Bool = sample([0.9, 0.1])
                yield b

        observe(out, 0)
        return cond

    r = compile_pal_fn(f)()
    # P(out=0|cond=0) = 0.3; P(out=0|cond=1) = 0.9. observe leaves the tensor
    # unnormalized, so the result is [0.5*0.3, 0.5*0.9] = [0.15, 0.45], whose
    # mass 0.6 is P(out = 0). Normalizing gives the posterior on cond.
    np.testing.assert_allclose(r, [0.15, 0.45], atol=1e-7)
    np.testing.assert_allclose(r / r.sum(), [0.15 / 0.6, 0.45 / 0.6], atol=1e-7)


def test_condition_dice_sum_constraint():
    """Two uniform Fin6 dice, conditioned on their sum equalling 5. `condition`
    masks without renormalizing, so the surviving (x, y) pairs are an
    anti-diagonal of six cells each still carrying their joint mass 1/36. The
    total 1/6 is P(x + y == 5); normalizing gives 1/6 per surviving cell."""

    def f():
        x: Fin6 = sample([1, 1, 1, 1, 1, 1])
        y: Fin6 = sample([1, 1, 1, 1, 1, 1])
        condition(x + y == 5)
        return x, y

    r = compile_pal_fn(f)()
    anti_diagonal = [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]
    expected = np.zeros((6, 6))
    for i, j in anti_diagonal:
        expected[i, j] = 1 / 36
    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 1 / 6

    posterior = r / r.sum()
    np.testing.assert_allclose([posterior[i, j] for i, j in anti_diagonal], [1 / 6] * 6, atol=1e-7)


def test_condition_equality_then_observe():
    """Combine `condition` with `observe`. The condition restricts the joint of
    (x, y) to the diagonal x == y; `observe(z, 0)` masks z, and z is then
    marginalized, scaling by P(z = 0) = 0.2.

    Also pins order-independence: `--reorder` moves the `observe` before the
    `condition` here, and since both only multiply and sum -- operations that
    commute -- the answer must be identical either way.
    """

    def f():
        x: Fin3 = sample([0.5, 0.3, 0.2])
        y: Fin3 = sample([0.1, 0.6, 0.3])
        z: Fin3 = sample([0.2, 0.5, 0.3])
        condition(x == y)
        observe(z, 0)
        return x, y

    diag = np.array([0.5 * 0.1, 0.3 * 0.6, 0.2 * 0.3])  # [0.05, 0.18, 0.06]

    # z is independent of (x, y), so the joint is the diagonal times P(z = 0).
    # Total mass 0.29 * 0.2 = 0.058 is P(x == y, z == 0).
    expected = np.diag(diag) * 0.2
    for reorder in (True, False):
        r = compile_pal_fn(f, reorder=reorder)()
        np.testing.assert_allclose(r, expected, atol=1e-7)
        assert pytest.approx(float(np.sum(r)), abs=1e-9) == 0.058
        np.testing.assert_allclose(np.diag(r) / np.sum(r), diag / diag.sum(), atol=1e-7)


def test_condition_inside_markov_no_repeats():
    """A 2-step markov chain on Fin3 where each transition samples a uniform
    new state and conditions on it differing from the previous state. The
    likelihood is *not* renormalized between steps, so the residual mass
    halves a la `(2/3)^2`. Returned: marginal over the final state."""

    def f():
        init: Fin3 = sample([0.4, 0.4, 0.2])
        with markov(s := init, repeat=2):
            new_s: Fin3 = sample([1, 1, 1])  # uniform 1/3 each
            condition(new_s != s)
            yield new_s

        return s

    r = compile_pal_fn(f)()

    # NumPy reference: two steps of the kernel
    #   T(s -> s') = (1/3) * [s' != s]
    p = np.array([0.4, 0.4, 0.2])
    T = (np.ones((3, 3)) - np.eye(3)) / 3
    expected = p @ T @ T

    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == (2 / 3) ** 2


def test_markov_with_inner_if_then_else_and_observe_after():
    """A markov chain whose body branches on a sampled coin into two
    different transition kernels, followed by a noisy observation of the
    final state. Exercises markov + nested if-then-else block + sample
    inside both branches + post-chain observe."""

    def f():
        init: Bool = sample([1.0, 0.0])  # start in 0 deterministically
        with markov(s := init, repeat=2):
            coin: Bool = sample([0.5, 0.5])
            with new_s:
                if coin == 0:
                    a: Bool = sample(1 if a == s else 0)  # stay
                    yield a
                else:
                    b: Bool = sample(1 if b != s else 0)  # flip
                    yield b
            yield new_s

        # noisy reading: correct w.p. 0.9
        obs: Bool = sample(0.9 if obs == s else 0.1)
        observe(obs, 0)
        return s

    r = compile_pal_fn(f)()
    # The transition is "stay w.p. 0.5, flip w.p. 0.5" so after 2 steps from
    # state 0 the marginal is [0.5, 0.5]. The observation likelihood is
    # [0.9, 0.1], so the unnormalized result is [0.45, 0.05]; its mass 0.5 is
    # P(obs = 0). Normalizing gives the posterior [0.9, 0.1].
    np.testing.assert_allclose(r, [0.45, 0.05], atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 0.5
    np.testing.assert_allclose(r / r.sum(), [0.9, 0.1], atol=1e-7)


def test_if_then_else_block_with_condition_in_else_branch():
    """An if-then-else block whose else branch contains a `condition`
    statement that prunes part of the Fin3 latent's support."""

    def test2():
        x: Fin3 = sample([1 / 2, 1 / 4, 1 / 4])
        with y:
            if x == 0:
                w: Bool = sample(1 / 5 if _ == 1 else 4 / 5)
                yield w
            else:
                condition(x == 2)
                w: Bool = sample(1)
                yield w
        return y, x

    r = compile_pal_fn(test2)()
    np.testing.assert_allclose(r, [[2 / 5, 0, 1 / 8], [1 / 10, 0, 1 / 8]], atol=1e-7)


# The next three tests share the same prior over (cond, shared, shared2):
#   cond    ~ [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
#   shared  ~ [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
#   shared2 ~ [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
_SHARED_PRIOR = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])


def test_if_elif_else_with_markov_swap_in_else_branch():
    """Third branch enters a 3-step markov chain whose body deterministically
    swaps the two state variables; the post-markov ``yield`` is what
    propagates to the outer ``(y, z)``."""

    def f():
        cond: Fin6 = sample([0.4, 0.2, 0.1])
        shared: Fin6 = sample([0.3, 0.3, 0.1])
        shared2: Fin6 = sample([0.4, 0.2, 0.1])

        with y, z:
            if cond == 0:
                a: Fin6 = shared
                b: Fin6 = cond
                temp = shared2
                temp2 = shared2
                yield a, b
            elif cond == 1:
                c: Fin6 = shared
                d: Fin6 = shared
                yield c, d
            else:
                c: Fin6 = 4
                d: Fin6 = 5

                with markov(cc := c, dd := d, repeat=3):
                    ccc = dd
                    ddd = cc
                    yield ccc, ddd

                yield cc, dd
        return y, z

    r = compile_pal_fn(f)()
    # cond=0 (P=0.4): (y, z) = (shared, 0)        → column 0 += 0.4 * shared
    # cond=1 (P=0.2): (y, z) = (shared, shared)   → diag    += 0.2 * shared
    # cond>=2 (P=0.4): 3 swaps of (4, 5) → (5, 4) → cell [5, 4] += 0.4
    expected = np.zeros((6, 6))
    expected[:, 0] += 0.4 * _SHARED_PRIOR
    expected[np.arange(6), np.arange(6)] += 0.2 * _SHARED_PRIOR
    expected[5, 4] += 0.4
    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


def test_nested_if_then_else_with_markov_in_innermost_branch():
    """An outer ``if cond <= 1`` branch contains a nested if-then-else block;
    its ``cond == 0`` arm runs a 5-step markov chain over (aa, bb) whose
    yields are discarded — the surrounding ``yield a, b`` propagates the
    pre-markov state."""

    def f():
        cond: Fin6 = sample([0.4, 0.2, 0.1])
        shared: Fin6 = sample([0.3, 0.3, 0.1])
        shared2: Fin6 = sample([0.4, 0.2, 0.1])

        with y, z:
            if cond <= 1:
                with a, b:
                    if cond == 0:
                        a: Fin6 = 0
                        b: Fin6 = shared

                        with markov(aa := a, bb := b, repeat=5):
                            aaa: Fin6 = 5 - bb
                            bbb: Fin6 = 5 - a
                            yield aaa, bbb

                        yield a, b
                    else:
                        c: Fin6 = shared
                        d: Fin6 = 3
                        yield c, d
                yield a, b
            else:
                c: Fin6 = 4
                d: Fin6 = 5
                yield c, d
        return y, z

    r = compile_pal_fn(f)()
    # cond=0 (P=0.4):  (y, z) = (0, shared)    → row 0  += 0.4 * shared
    # cond=1 (P=0.2):  (y, z) = (shared, 3)    → col 3  += 0.2 * shared
    # cond>=2 (P=0.4): (y, z) = (4, 5)         → cell [4, 5] += 0.4
    expected = np.zeros((6, 6))
    expected[0, :] += 0.4 * _SHARED_PRIOR
    expected[:, 3] += 0.2 * _SHARED_PRIOR
    expected[4, 5] += 0.4
    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0


def test_markov_wrapping_if_then_else_block():
    """A 2-step markov over (y, z) initialized from (shared, shared2). Each
    step's nested ``with a, b`` if-then-else yields the next (y, z) — which
    is fully determined by ``cond``, so each iteration is a fixed point."""

    def f():
        cond: Fin6 = sample([0.4, 0.2, 0.1])
        shared: Fin6 = sample([0.3, 0.3, 0.1])
        shared2: Fin6 = sample([0.4, 0.2, 0.1])

        with markov(y := shared, z := shared2, repeat=2):
            with a, b:
                if cond == 0:
                    c: Fin6 = 0
                    d: Fin6 = shared
                    yield c, d
                else:
                    c: Fin6 = shared2
                    d: Fin6 = 5
                    yield c, d
            yield a, b
        return y, z

    r = compile_pal_fn(f)()
    # The markov body is a deterministic function of `cond`, so after >=1
    # iterations the final (y, z) is:
    #   cond=0  (P=0.4): (0, shared)   → row 0 += 0.4 * shared
    #   cond!=0 (P=0.6): (shared2, 5)  → col 5 += 0.6 * shared2
    _SHARED2_PRIOR = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
    expected = np.zeros((6, 6))
    expected[0, :] += 0.4 * _SHARED_PRIOR
    expected[:, 5] += 0.6 * _SHARED2_PRIOR
    np.testing.assert_allclose(r, expected, atol=1e-7)
    assert pytest.approx(r.sum(), abs=1e-6) == 1.0
