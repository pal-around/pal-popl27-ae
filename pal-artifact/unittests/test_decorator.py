"""The ``@pal`` decorator, bare and with options, and the options it takes from the
command line when the program is defined in the script being run."""

import os
import subprocess
import sys
import textwrap

import numpy as np

from pal import Bool, pal, sample

SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np
    from pal import Bool, pal

    @pal
    def bare():
        x: Bool = sample([0.25, 0.75])
        return x

    @pal(numpy=False)
    def pinned():
        y: Bool = sample([0.25, 0.75])
        return y

    for name, f in (("bare", bare), ("pinned", pinned)):
        r = f()
        print(name, type(r).__module__.split(".")[0], np.asarray(r).tolist())
    print("rest", sys.argv[1:])
    """
)


def _run_script(tmp_path, *args):
    script = tmp_path / "program.py"
    script.write_text(SCRIPT)
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, str(script), *args], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return dict(line.split(" ", 1) for line in result.stdout.strip().splitlines())


def test_bare_and_parenthesized_forms_compile():
    @pal
    def f():
        x: Bool = sample([0.25, 0.75])
        return x

    @pal(numpy=True)
    def g():
        x: Bool = sample([0.25, 0.75])
        return x

    np.testing.assert_allclose(np.asarray(f()), [0.25, 0.75])
    np.testing.assert_allclose(g(), [0.25, 0.75])
    assert isinstance(g(), np.ndarray)


def test_imported_programs_ignore_the_command_line(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program.py", "--numpy"])

    @pal
    def f():
        x: Bool = sample([0.25, 0.75])
        return x

    assert not isinstance(f(), np.ndarray)


def test_script_takes_options_from_the_command_line(tmp_path):
    out = _run_script(tmp_path, "--numpy")
    assert out["bare"] == "numpy [0.25, 0.75]"


def test_script_defaults_without_options(tmp_path):
    out = _run_script(tmp_path)
    assert out["bare"] == "jaxlib [0.25, 0.75]"


def test_decorator_options_take_precedence(tmp_path):
    out = _run_script(tmp_path, "--numpy")
    assert out["pinned"] == "jaxlib [0.25, 0.75]"


def test_other_arguments_are_left_to_the_script(tmp_path):
    out = _run_script(tmp_path, "--numpy", "--steps", "3", "data.csv")
    assert out["bare"] == "numpy [0.25, 0.75]"
    assert out["rest"] == "['--numpy', '--steps', '3', 'data.csv']"
