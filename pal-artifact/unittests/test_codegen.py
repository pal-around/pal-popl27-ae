"""Properties of the generated program (pal/src/compile.py) that the returned
tensor does not reveal."""

import contextlib
import io
import re

import pytest
from _pal_test_utils import compile_pal_fn

from test_eager_if import if_sibling_only_read
from test_lva_equivalence import markov_multi_state, markov_with_if_inside

BANNER = "==================== Compiled program ===================="


def _program(prog, **kw) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_pal_fn(prog, print_compiled=True, **kw)
    return buf.getvalue().split(BANNER, 1)[1]


@pytest.mark.parametrize(
    "prog", [if_sibling_only_read, markov_multi_state, markov_with_if_inside], ids=lambda p: p.__name__
)
@pytest.mark.parametrize("lva", [True, False])
def test_enum_tensors_are_only_built_by_c_var(prog, lva):
    """A random variable's value tensor is built where it is read (paper C-Var): every
    assignment to an ``enum_<v>`` local is its build line
    ``enum_<v> = np.array(<domain>).reshape(...)``. ``if_sibling_only_read`` is the
    case where eager marginalization marginalizes outer variables inside branches."""
    src = _program(prog, lva=lva)
    assigned = re.findall(r"^\s*(enum_\w+\s*=[^=].*)$", src, flags=re.M)
    assert assigned
    for line in assigned:
        assert re.fullmatch(r"enum_\w+\s*=\s*np\.array\(.*\)\.reshape\(.*\)", line.strip()), line


def test_generated_code_is_deterministic():
    """The emitted program does not depend on Python's string-hash seed (set iteration
    order), so float summation order is reproducible across runs."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from _pal_test_utils import compile_pal_fn\n"
        "from test_eager_if import if_sibling_only_read\n"
        "from test_lva_equivalence import markov_multi_state, markov_with_if_inside\n"
        "for lva in (True, False):\n"
        "    for p in (if_sibling_only_read, markov_multi_state, markov_with_if_inside):\n"
        "        compile_pal_fn(p, lva=lva, print_compiled=True)\n"
    )
    outs = []
    for seed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outs.append(r.stdout)
    assert BANNER in outs[0]
    assert outs[0] == outs[1]


def test_missing_kill_is_an_internal_error():
    """The code generator marginalizes only at ``StmtKill``: a branch that ends with
    an unmarginalized local raises an internal ``CompilationError``."""
    from _pal_test_utils import parse_ir

    from pal import Bool, sample
    from pal.src.compile import CompilationError, CompileContext, compile_pal
    from pal.src.syntax import StmtIfThenElse, StmtKill

    def prog():
        c: Bool = sample([0.5, 0.5])
        with out:
            if c == 0:
                t: Bool = sample([0.9, 0.1])
                s: Bool = sample(0.8 if s == t else 0.2)
                yield s
            else:
                r: Bool = sample([0.5, 0.5])
                yield r
        return out

    stmts = parse_ir(prog, lva=False, reorder=False)
    branch = next(s for s in stmts if isinstance(s, StmtIfThenElse))
    branch.blocks[0][:] = [s for s in branch.blocks[0] if not isinstance(s, StmtKill)]
    ctxt = CompileContext(
        kwargs={"numpy": True}, non_deterministic_params=set(), scalar_params={}, func_globals=prog.__globals__
    )
    with pytest.raises(CompilationError, match="internal error.*not marginalized"):
        compile_pal(ctxt, stmts)
