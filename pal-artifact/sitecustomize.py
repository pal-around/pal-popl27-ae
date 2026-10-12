import ctypes
import os
import platform

_platforms = os.environ.get("JAX_PLATFORMS", "")
_may_use_cuda = not _platforms or "cuda" in _platforms or "gpu" in _platforms

if platform.system().lower() == "linux" and _may_use_cuda:
    driver_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/run/opengl-driver/lib",  # NixOS
    ]
    for base in driver_paths:
        path = os.path.join(base, "libcuda.so.1")
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                os.environ["JAX_STATUS_LIBCUDA_PRELOADED"] = "1"
            except Exception as exc:
                os.environ["JAX_STATUS_LIBCUDA_PRELOAD_ERROR"] = str(exc)
                pass
            try:
                import subprocess
                subprocess.getoutput("nvidia-smi")
            except Exception:
                pass
            break


def _run_next_sitecustomize() -> None:
    """Run the next ``sitecustomize`` on ``sys.path``, which this file shadows when the
    repository is on PYTHONPATH. A Nix Python environment adds its packages there."""
    import importlib.util
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if os.path.abspath(entry or os.curdir) == here:
            continue
        path = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_next_sitecustomize", path)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return


_run_next_sitecustomize()
