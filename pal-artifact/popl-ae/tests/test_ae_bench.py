"""bench.py tells a process that ran out of memory from one that crashed."""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bench", Path(__file__).resolve().parent.parent / "bench.py")
bench = importlib.util.module_from_spec(spec)
sys.modules["bench"] = bench  # its dataclasses look their module up there
spec.loader.exec_module(bench)


@pytest.mark.parametrize("returncode, stderr", [
    (-9, ""),                                                        # killed by the system
    (-6, "memory allocation of 786432 bytes failed"),                # Dice (Rust BDD library)
    (1, "jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED"),  # JAX on the GPU
    (1, "MemoryError"),                                              # Python
    (-6, "terminate called after throwing an instance of 'std::bad_alloc'"),
    (2, "Fatal error: out of memory."),                              # OCaml
])
def test_out_of_memory(returncode, stderr):
    assert bench.out_of_memory(returncode, stderr)


@pytest.mark.parametrize("returncode, stderr", [
    (-11, ""),                       # segmentation fault
    (-6, "Assertion failed"),
    (1, "ValueError: bad input"),
])
def test_crash_is_not_out_of_memory(returncode, stderr):
    assert not bench.out_of_memory(returncode, stderr)
