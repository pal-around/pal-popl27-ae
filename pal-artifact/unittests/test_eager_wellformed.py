"""Well-formedness ("type checking") of the marginalization-free IR, performed by
``check_wellformed`` (pal/src/eager.py) after yield normalization and before
reorder and kill insertion, in both lva modes."""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from pal import Bool, init, markov, sample
from pal.src.eager import IRWellFormednessError

LVA = pytest.mark.parametrize("lva", [True, False])


@LVA
def test_use_before_definition_is_rejected(lva):
    def bad():
        # 'ghost' is read in the weight expression but never defined.
        x: Bool = sample(1 if x == ghost else 0)
        return x

    with pytest.raises(IRWellFormednessError):
        compile_pal_fn(bad, lva=lva)


@LVA
def test_loop_body_rebinding_enclosing_var_is_rejected(lva):
    # The body samples `w`, a name already in scope outside the loop. Names must be
    # fresh, and the closed-form loop liveness relies on it (def(body) = 0 with
    # respect to enclosing variables) -- the inner variable must be renamed.
    def bad():
        w: Bool = sample([0.7, 0.3])
        x: Bool = sample([0.5, 0.5])
        with markov(y := x, repeat=2):
            z: Bool = sample(1 if z == w else 0)
            w: Bool = sample([0.5, 0.5])  # re-declares enclosing w
            yield z
        return y

    with pytest.raises(IRWellFormednessError, match=r"\['w'\] already defined"):
        compile_pal_fn(bad, lva=lva)


@LVA
def test_loop_body_reading_enclosing_var_is_accepted(lva):
    # Reading (not rebinding) an enclosing variable in the body is the legitimate,
    # supported case: `w` stays live across iterations and is marginalized after.
    def good():
        w: Bool = sample([0.7, 0.3])
        x: Bool = sample([0.5, 0.5])
        with markov(y := x, repeat=2):
            z: Bool = sample(0.8 if z == w else 0.2)
            yield z
        return y

    compile_pal_fn(good, lva=lva)()


@LVA
def test_well_formed_program_is_accepted(lva):
    def good():
        x: Bool = sample([0.6, 0.4])
        y: Bool = sample(0.9 if x == 0 else 0.1)
        return y

    # Should compile without raising.
    compile_pal_fn(good, lva=lva)()


# ---------------------------------------------------------------------------
# Fresh names
# ---------------------------------------------------------------------------


@LVA
def test_redefining_a_dead_variable_is_rejected(lva):
    # `x` is no longer read when it is redefined, but it is still in scope: the
    # source language has no marginalization, so the name is not fresh.
    def bad():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample(0.9 if y == x else 0.1)
        x: Bool = sample([0.3, 0.7])
        z: Bool = sample(0.8 if z == x else 0.2)
        return y, z

    with pytest.raises(IRWellFormednessError, match=r"\['x'\] already defined"):
        compile_pal_fn(bad, lva=lva)


@LVA
def test_if_return_name_already_in_scope_is_rejected(lva):
    def bad():
        c: Bool = sample([0.5, 0.5])
        out: Bool = sample([0.5, 0.5])
        with out:
            if c == 0:
                a: Bool = 0
                yield a
            else:
                b: Bool = 1
                yield b
        return out

    with pytest.raises(IRWellFormednessError, match=r"\['out'\] already defined"):
        compile_pal_fn(bad, lva=lva)


@LVA
def test_markov_carry_name_already_in_scope_is_rejected(lva):
    def bad():
        x: Bool = sample([0.5, 0.5])
        y: Bool = sample([0.5, 0.5])
        with markov(y := x, repeat=2):
            z: Bool = sample(1 if z == y else 0)
            yield z
        return y

    with pytest.raises(IRWellFormednessError, match=r"\['y'\] already defined"):
        compile_pal_fn(bad, lva=lva)


@LVA
def test_sibling_branches_may_reuse_a_local_name(lva):
    # Each branch scopes its locals, so both may bind `t`.
    def good():
        c: Bool = sample([0.6, 0.4])
        with out:
            if c == 0:
                t: Bool = sample([0.9, 0.1])
                yield t
            else:
                t: Bool = sample([0.2, 0.8])
                yield t
        return out

    r = compile_pal_fn(good, lva=lva)()
    np.testing.assert_allclose(r, [0.6 * 0.9 + 0.4 * 0.2, 0.6 * 0.1 + 0.4 * 0.8])


@LVA
def test_name_may_be_reused_after_its_block_ends(lva):
    # `t` is local to the first branch; once the if ends the name is free again.
    def good():
        c: Bool = sample([0.6, 0.4])
        with out:
            if c == 0:
                t: Bool = sample([0.9, 0.1])
                yield t
            else:
                u: Bool = sample([0.2, 0.8])
                yield u
        t: Bool = sample(0.7 if t == out else 0.3)
        return t

    r = compile_pal_fn(good, lva=lva)()
    p_out = np.array([0.6 * 0.9 + 0.4 * 0.2, 0.6 * 0.1 + 0.4 * 0.8])
    expected = np.array([p_out[0] * 0.7 + p_out[1] * 0.3, p_out[0] * 0.3 + p_out[1] * 0.7])
    np.testing.assert_allclose(r, expected)


# ---------------------------------------------------------------------------
# Init blocks
# ---------------------------------------------------------------------------


@LVA
def test_init_block_locals_not_visible_in_body(lva):
    def bad():
        with markov(x, repeat=2):
            with init():
                u: Bool = sample([0.5, 0.5])
                x0: Bool = u
                yield x0
            xn: Bool = sample(0.9 if xn == u else 0.1)  # u is local to the init block
            yield xn
        return x

    with pytest.raises(IRWellFormednessError, match=r"\['u'\] used before definition"):
        compile_pal_fn(bad, lva=lva)


@LVA
def test_init_block_use_before_definition_is_rejected(lva):
    def bad():
        with markov(x, repeat=2):
            with init():
                x0: Bool = sample(1 if x0 == ghost else 0)
                yield x0
            xn: Bool = sample([0.5, 0.5])
            yield xn
        return x

    with pytest.raises(IRWellFormednessError, match=r"\['ghost'\] used before definition"):
        compile_pal_fn(bad, lva=lva)


@LVA
def test_init_block_local_shadowing_outer_name_is_rejected(lva):
    def bad():
        u: Bool = sample([0.5, 0.5])
        with markov(x, repeat=2):
            with init():
                u: Bool = sample([0.3, 0.7])
                yield u
            xn: Bool = sample([0.5, 0.5])
            yield xn
        return x

    with pytest.raises(IRWellFormednessError, match=r"\['u'\] already defined"):
        compile_pal_fn(bad, lva=lva)


@LVA
def test_init_block_and_body_may_share_a_local_name(lva):
    # The init block's locals are gone by the time the body runs.
    def good():
        with markov(x, repeat=1):
            with init():
                t: Bool = sample([0.9, 0.1])
                yield t
            t: Bool = sample(0.8 if t == x else 0.2)
            yield t
        return x

    r = compile_pal_fn(good, lva=lva)()
    np.testing.assert_allclose(r, np.array([0.9, 0.1]) @ np.array([[0.8, 0.2], [0.2, 0.8]]))
