"""The JAX+JIT backend must compute the same distributions as the NumPy backend.

Covers the representative program batteries -- in particular loops, whose ``scan``
body is a separate Python function scope for the value tensors it builds, and
branches, which rebuild value tensors whose axes moved."""

import numpy as np
import pytest
from _pal_test_utils import compile_pal_fn

from test_eager_if import PROGRAMS as IF_PROGRAMS
from test_lva_equivalence import PROGRAMS as LVA_PROGRAMS
from test_rescale import PROGRAMS as RESCALE_PROGRAMS

PROGRAMS = LVA_PROGRAMS + IF_PROGRAMS + RESCALE_PROGRAMS


@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.__name__)
@pytest.mark.parametrize("lva", [True, False])
def test_jit_matches_numpy(prog, lva):
    ref = np.asarray(compile_pal_fn(prog, lva=lva)())
    got = np.asarray(compile_pal_fn(prog, lva=lva, numpy=False, jit=True)())
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
