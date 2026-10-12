"""Benchmark driver for the artifact.

Runs one configuration over one suite and writes one CSV row per case:

    python bench.py --config pal-xla --output-dir results/x caesar -L 1,10,100
    python bench.py --config dice --output-dir results/x bayes --networks cancer,survey

Configurations:

    pal-xla          PAL compiled to JAX/XLA (on the GPU with --gpu, else on the CPU)
    pal-xla-nomarg   the same with eager marginalization off (--no-lva)
    pal-numpy        PAL with the NumPy backend (CPU)
    pal-numpy-nomarg the same with eager marginalization off
    dice             Dice, with its default options
    memo             memo (dicefig1 only)

Suites: caesar, diamond, ladder, dicefig1 (-L lengths), bayes (--networks), grid-link and
grid-router (-L grid sizes; the grid is L x L).

**Timing.** Each case runs ``--trials`` times, each in a fresh process; a case whose first
trial takes longer than ``--single-over`` seconds runs once.

* PAL. The generated program is rewritten (``instrument_pal_source``) to compile and call
  a copy of the model ``--warmup`` times, then compile the model itself and call it
  ``--calls`` times. From PAL's ``PAL compilation time`` and ``Run N`` lines:

    compile_s       compiling the measured model
    run_first_s     its first call (for XLA: JAX trace, XLA compilation and the run)
    hot_mean_s      its later calls
    report_cold_s   compile_s + run_first_s
    report_warm_s   compile_s + hot_mean_s

* memo. The program times two calls of the model in one process: run_first_s (with JAX's
  compilation) and hot_mean_s; report_cold_s = run_first_s.
* Dice. The whole process is timed (process_s); report_cold_s = process_s.

Each number is the mean over trials, with a ``_std`` column.

**Memory.** Every PAL case also gets a static estimate of its largest array
(``static_peak_bytes``, from ``--print-static-ndim-peak``, without running the model). A
``-nomarg`` case whose estimate exceeds ``--mem-limit-gb`` is recorded as
``infeasible`` and not run. PAL runs report the measured peak: on the GPU, JAX's
``peak_bytes_in_use`` (``peak_gpu_bytes``); on the CPU, the growth of the resident set
while the measured model is compiled and called (``peak_rss_bytes``, median over trials).
CPU processes run under an address-space limit of ``--mem-limit-gb``.

**Failures.** A case that fails, times out or runs out of memory gets that status; in a
size sweep the larger sizes are then ``skipped``.

**Answers.** Every row records the query's distribution (``answer``, normalized) so
``check.py`` can compare configurations.
"""

from __future__ import annotations

import ast
import copy
import csv
import logging
import os
import re
import resource
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import click

logger = logging.getLogger(__name__)

AE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AE_ROOT.parent

CONFIGS = ("pal-xla", "pal-xla-nomarg", "pal-numpy", "pal-numpy-nomarg", "dice", "memo")
DICE_CMD = ["dice"]

# What a process prints when it runs out of memory: Python, JAX/XLA, Dice's BDD library
# (Rust), C++ and OCaml.
OOM_MESSAGES = ("MemoryError", "RESOURCE_EXHAUSTED", "Out of memory", "out of memory",
                "memory allocation of", "std::bad_alloc")


def out_of_memory(returncode: int, stderr: str) -> bool:
    """Whether a failed process ran out of memory: the system killed it (SIGKILL), or it
    reported running out."""
    return returncode == -9 or any(m in stderr for m in OOM_MESSAGES)


RUN_LINE_RE = re.compile(r"^Run\s+(\d+):\s*([0-9]*\.?[0-9]+)")
PAL_COMPILE_LINE_RE = re.compile(r"^PAL compilation time:\s*([0-9]*\.?[0-9]+)")
STATIC_PEAK_RE = re.compile(r"^Static peak memory bytes:\s*(\d+)")
ANSWER_RE = re.compile(r"^POPL-AE answer:\s*(.*)$")
DTYPE_RE = re.compile(r"^POPL-AE dtype:\s*(\S+)")
GPU_PEAK_RE = re.compile(r"^POPL-AE peak gpu bytes:\s*(\d+)")
RSS_PEAK_RE = re.compile(r"^POPL-AE peak rss bytes:\s*(\d+)")
DEVICE_RE = re.compile(r"^POPL-AE device:\s*(.*)$")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _format_vector(values: list[float]) -> str:
    return " ".join(format(v, ".17g") for v in values)


def _normalize(values: list[float]) -> tuple[list[float], float]:
    mass = sum(values)
    return ([v / mass for v in values], mass) if mass > 0 else (values, mass)


def _setup_logging(log_file: Path) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# PAL: source instrumentation and output parsing
# ---------------------------------------------------------------------------

def _is_time_execution_call(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "time_execution")


def _is_pal_decorated(node: ast.stmt) -> bool:
    if not isinstance(node, ast.FunctionDef):
        return False
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "pal":
            return True
    return False


# Runs just before the measured model is defined (PAL compiles it there): returns the
# allocator's free memory to the system and resets the kernel's high-water mark, so the
# peak read afterwards belongs to the measured model.
_RSS_RESET = """
def _popl_ae_status(key):
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith(key + ':'):
                return int(line.split()[1]) * 1024
    return 0
def _popl_ae_reset_rss():
    import ctypes
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
    except OSError:
        pass
    return _popl_ae_status('VmRSS')
_popl_ae_rss_base = _popl_ae_reset_rss()
"""

_REPORT = """
import numpy as _popl_ae_np
_popl_ae_out = _popl_ae_np.asarray(_popl_ae_ret[0] if isinstance(_popl_ae_ret, tuple) else _popl_ae_ret)
print('POPL-AE peak rss bytes:', max(0, _popl_ae_status('VmHWM') - _popl_ae_rss_base))
try:
    import jax as _popl_ae_jax
    _popl_ae_dev = _popl_ae_jax.devices()[0]
    print('POPL-AE device:', _popl_ae_dev.device_kind)
    _popl_ae_stats = _popl_ae_dev.memory_stats() or {}
    if 'peak_bytes_in_use' in _popl_ae_stats:
        print('POPL-AE peak gpu bytes:', _popl_ae_stats['peak_bytes_in_use'])
except Exception:
    pass
print('POPL-AE dtype:', _popl_ae_out.dtype)
print('POPL-AE answer:', ' '.join(format(float(v), '.17g') for v in _popl_ae_out.ravel()))
# Everything is reported; exit without the interpreter's teardown, in which the CUDA
# runtime occasionally crashes after long runs.
import os as _popl_ae_os, sys as _popl_ae_sys
_popl_ae_sys.stdout.flush()
_popl_ae_os._exit(0)
"""


def instrument_pal_source(source: str, warmup: int, calls: int) -> str:
    """Rewrite a generated PAL program for timing.

    The generators emit a preamble, one ``@pal`` function and trailing
    ``time_execution(f, *args)`` calls. The result is the preamble; a copy of the function
    named ``<f>_warmup``, called ``warmup`` times; the memory reset; the function itself,
    called ``calls`` times; and a report of memory and the last result. ``time_execution``
    numbers its runs across the process, so the warm-up calls are ``Run 1 .. warmup``."""
    tree = ast.parse(source)
    funcs = [i for i, node in enumerate(tree.body) if _is_pal_decorated(node)]
    if len(funcs) != 1:
        raise ValueError(f"expected exactly one @pal function, found {len(funcs)}")
    idx = funcs[0]
    func = tree.body[idx]
    timing = [node for node in tree.body[idx + 1:] if _is_time_execution_call(node)]
    if not timing:
        raise ValueError("no time_execution(...) call after the @pal function")
    extra_args = timing[0].value.args[1:]
    rest = [node for node in tree.body[idx + 1:] if not _is_time_execution_call(node)]

    warm = copy.deepcopy(func)
    warm.name = f"{func.name}_warmup"

    def call(name: str) -> ast.Call:
        return ast.Call(func=ast.Name(id="time_execution", ctx=ast.Load()),
                        args=[ast.Name(id=name, ctx=ast.Load()), *copy.deepcopy(extra_args)], keywords=[])

    body: list[ast.stmt] = [warm]
    body += [ast.Expr(value=call(warm.name)) for _ in range(warmup)]
    body += ast.parse(_RSS_RESET).body
    body.append(func)
    body += [ast.Assign(targets=[ast.Name(id="_popl_ae_ret", ctx=ast.Store())], value=call(func.name))
             for _ in range(calls)]
    tree.body = tree.body[:idx] + body + rest + ast.parse(_REPORT).body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def static_probe_source(source: str) -> str:
    """The generated program without its ``time_execution`` calls: running it compiles
    the model (printing PAL's static estimates) and stops."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not _is_time_execution_call(node)]
    return ast.unparse(tree) + "\n"


def parse_output(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {"runs": [], "compiles": [], "answer": None, "dtype": None, "device": None,
                           "peak_gpu": None, "peak_rss": None, "static_peak": None}
    for line in stdout.splitlines():
        if m := RUN_LINE_RE.match(line):
            out["runs"].append(float(m.group(2)))
        elif m := PAL_COMPILE_LINE_RE.match(line):
            out["compiles"].append(float(m.group(1)))
        elif m := ANSWER_RE.match(line):
            out["answer"] = [float(t) for t in m.group(1).split()]
        elif m := DTYPE_RE.match(line):
            out["dtype"] = m.group(1)
        elif m := DEVICE_RE.match(line):
            out["device"] = m.group(1).strip()
        elif m := GPU_PEAK_RE.match(line):
            out["peak_gpu"] = int(m.group(1))
        elif m := RSS_PEAK_RE.match(line):
            out["peak_rss"] = int(m.group(1))
        elif (m := STATIC_PEAK_RE.match(line)) and out["static_peak"] is None:
            out["static_peak"] = int(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Dice: output parsing
# ---------------------------------------------------------------------------

def parse_dice_output(stdout: str) -> list[float] | None:
    """The query's distribution from Dice's ``Joint Distribution`` table, indexed by value
    (``false``/``true`` are 0/1). Values Dice does not list get probability 0."""
    table: dict[int, float] = {}
    in_block = saw_header = False
    for line in stdout.splitlines():
        if "Joint Distribution" in line:
            in_block, saw_header = True, False
            continue
        if not in_block:
            continue
        parts = line.split()
        if not parts:
            if saw_header:
                break
            continue
        if parts[0].lower() == "value":
            saw_header = True
            continue
        if not saw_header or len(parts) < 2:
            continue
        key = {"false": 0, "true": 1}.get(parts[0].lower())
        if key is None:
            try:
                key = int(parts[0])
            except ValueError:
                return None
        table[key] = float(parts[1])
    if not table:
        return None
    return [table.get(i, 0.0) for i in range(max(table) + 1)]


# ---------------------------------------------------------------------------
# Running cases
# ---------------------------------------------------------------------------

@dataclass
class Bench:
    config: str
    output_dir: Path
    trials: int
    single_over: float
    timeout: int
    warmup: int
    calls: int
    gpu: Optional[str]
    mem_limit_gb: float
    _rows_written: int = field(default=0, init=False)
    _failed: bool = field(default=False, init=False)

    @property
    def tool(self) -> str:
        return self.config.split("-")[0]

    @property
    def nomarg(self) -> bool:
        return self.config.endswith("-nomarg")

    @property
    def on_gpu(self) -> bool:
        return self.gpu is not None and self.config in ("pal-xla", "pal-xla-nomarg", "memo")

    @property
    def csv_path(self) -> Path:
        return self.output_dir / f"{self.config}.csv"

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        # `import pal` resolves to this source tree; sitecustomize.py there loads the
        # host's CUDA driver library for JAX.
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["JAX_PLATFORMS"] = "cuda,cpu" if self.on_gpu else "cpu"
        if self.gpu is not None:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        return env

    def pal_args(self) -> list[str]:
        args = ["--print-pal-compilation-time"]
        if self.config.startswith("pal-numpy"):
            args.append("--numpy")
        if self.nomarg:
            args.append("--no-lva")
        return args

    def _limit_memory(self):
        if self.on_gpu:
            return None  # CUDA reserves far more address space than it uses
        limit = int(self.mem_limit_gb * 2**30)
        return lambda: resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    def _run(self, command: list[str], retries: int = 2) -> subprocess.CompletedProcess:
        """Run a command; a process killed by a signal, unless it ran out of memory, or
        whose CUDA compiler crashed, is run again, up to ``retries`` times, and the retry
        is logged."""
        for attempt in range(retries + 1):
            done = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout,
                                  cwd=AE_ROOT, env=self.env(), preexec_fn=self._limit_memory())
            signalled = done.returncode < 0 and not out_of_memory(done.returncode, done.stderr)
            if not (signalled or "ptxas exited with non-zero error code" in done.stderr) or attempt == retries:
                return done
            logger.warning("  process ended with exit %d; running it again", done.returncode)
        return done

    def run_case(self, *, suite: str, case_id: str, generator_argv: list[str],
                 params: dict[str, Any], sweep: bool) -> dict[str, Any]:
        logger.info("[%s] %s :: %s", suite, case_id, self.config)
        base = {"suite": suite, "case_id": case_id, "config": self.config, **params}
        if sweep and self._failed:
            return self._record({**base, "status": "skipped", "error_message": "a smaller size already failed"})

        sources = self.output_dir / "sources" / self.config
        sources.mkdir(parents=True, exist_ok=True)
        ext = {"dice": "dice", "memo": "args"}.get(self.tool, "py")
        source_path = sources / f"{suite}_{case_id}.{ext}"
        static_peak = None
        if self.tool == "memo":
            command = [*generator_argv]
            source_path.write_text(shlex.join(generator_argv) + "\n")
        else:
            try:
                logger.debug("  $ %s", shlex.join(generator_argv))
                text = subprocess.run(generator_argv, capture_output=True, text=True, timeout=self.timeout,
                                      cwd=AE_ROOT, check=True, env={**self.env(), "JAX_PLATFORMS": "cpu"}).stdout
                if self.tool == "pal":
                    probe = sources / f"{suite}_{case_id}.static.py"
                    probe.write_text(static_probe_source(text))
                    text = instrument_pal_source(text, self.warmup, self.calls)
                source_path.write_text(text)
            except subprocess.CalledProcessError as exc:
                return self._fail(sweep, {**base, "status": "gen_failed",
                                          "error_message": f"exit {exc.returncode}: {exc.stderr.strip()[-300:]}"})
            except subprocess.TimeoutExpired:
                return self._fail(sweep, {**base, "status": "gen_timeout",
                                          "error_message": f"generator timed out after {self.timeout}s"})
            if self.tool == "pal":
                static_peak = self._static_peak(probe)
                if self.nomarg and static_peak is not None and static_peak > self.mem_limit_gb * 2**30:
                    return self._fail(sweep, {**base, "status": "infeasible", "static_peak_bytes": static_peak,
                                              "error_message": f"static estimate {static_peak} B exceeds "
                                                               f"{self.mem_limit_gb:g} GB"})
                command = [sys.executable, str(source_path), *self.pal_args()]
            else:
                command = [*DICE_CMD, str(source_path)]

        per_trial: dict[str, list[float]] = {}
        hot: list[float] = []
        rss: list[int] = []
        answer = mass = dtype = device = peak_gpu = None
        status, error = "ok", None
        for trial in range(1, self.trials + 1):
            logger.debug("  trial %d/%d: $ %s", trial, self.trials, shlex.join(command))
            try:
                started = time.perf_counter()
                done = self._run(command)
                elapsed = time.perf_counter() - started
            except subprocess.TimeoutExpired:
                status, error = "timeout", f"timed out after {self.timeout}s"
                break
            for line in done.stdout.splitlines():
                logger.debug("  | %s", line)
            for line in done.stderr.splitlines():
                logger.debug("  ! %s", line)
            if done.returncode != 0:
                tail = (done.stderr or done.stdout).strip().splitlines()
                last = tail[-1][:300] if tail else ""
                oom = out_of_memory(done.returncode, done.stderr or "")
                status, error = ("oom" if oom else "crash"), f"exit {done.returncode}: {last}"
                break
            per_trial.setdefault("process_s", []).append(elapsed)
            if self.tool == "dice":
                dist = parse_dice_output(done.stdout)
                if dist is None:
                    status, error = "crash", "no Joint Distribution table in Dice's output"
                    break
                answer, mass = _normalize(dist)
                device = "cpu"
            else:
                out = parse_output(done.stdout)
                runs = out["runs"]
                if self.tool == "pal":
                    if len(runs) != self.warmup + self.calls or len(out["compiles"]) != 2 or out["answer"] is None:
                        status, error = "crash", f"unexpected output: {len(runs)} runs, {len(out['compiles'])} compiles"
                        break
                    w = self.warmup
                    compile_s, first = out["compiles"][-1], runs[w]
                    later = runs[w + 1:]
                else:  # memo: two calls of the model
                    if len(runs) < 2 or out["answer"] is None:
                        status, error = "crash", f"unexpected output: {len(runs)} runs"
                        break
                    compile_s, first, later = 0.0, runs[0], runs[1:]
                per_trial.setdefault("compile_s", []).append(compile_s)
                per_trial.setdefault("run_first_s", []).append(first)
                per_trial.setdefault("report_cold_s", []).append(compile_s + first)
                if later:
                    per_trial.setdefault("report_warm_s", []).append(compile_s + statistics.mean(later))
                hot += later
                answer, mass = _normalize(out["answer"])
                dtype, device = out["dtype"], out["device"]
                if out["peak_rss"] is not None:
                    rss.append(out["peak_rss"])
                if out["peak_gpu"] is not None and self.on_gpu:
                    peak_gpu = max(peak_gpu or 0, out["peak_gpu"])
            if elapsed > self.single_over:
                break

        row = {**base, "status": status, "error_message": error, "trials": len(per_trial.get("process_s", [])),
               "static_peak_bytes": static_peak}
        for key in ("compile_s", "run_first_s", "report_cold_s", "report_warm_s", "process_s"):
            row[key], row[key.removesuffix("_s") + "_std"] = _mean_std(per_trial.get(key, []))
        if self.tool == "dice" and row["process_s"] is not None:
            row["report_cold_s"], row["report_cold_std"] = row["process_s"], row["process_std"]
        row["hot_mean_s"], row["hot_std"] = _mean_std(hot)
        row["peak_gpu_bytes"] = peak_gpu
        row["peak_rss_bytes"] = int(statistics.median(rss)) if rss and not self.on_gpu else None
        row["answer"] = _format_vector(answer) if answer is not None else None
        row["answer_mass"] = mass
        row["dtype"] = dtype
        row["device"] = device
        if status != "ok":
            return self._fail(sweep, row)
        return self._record(row)

    def _static_peak(self, probe: Path) -> int | None:
        args = ["--print-static-ndim-peak"] + (["--numpy"] if "numpy" in self.config else [])
        if self.nomarg:
            args.append("--no-lva")
        try:
            done = self._run([sys.executable, str(probe), *args])
        except subprocess.TimeoutExpired:
            return None
        return parse_output(done.stdout)["static_peak"]

    def _fail(self, sweep: bool, row: dict[str, Any]) -> dict[str, Any]:
        logger.error("  %s: %s", row["status"], row.get("error_message"))
        if sweep:
            self._failed = True
        return self._record(row)

    COLUMNS = (
        "suite", "case_id", "config", "L", "nodes", "network", "status", "error_message", "trials",
        "compile_s", "compile_std", "run_first_s", "run_first_std", "hot_mean_s", "hot_std",
        "report_cold_s", "report_cold_std", "report_warm_s", "report_warm_std", "process_s", "process_std",
        "static_peak_bytes", "peak_gpu_bytes", "peak_rss_bytes", "answer", "answer_mass", "dtype", "device",
    )

    def _record(self, row: dict[str, Any]) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w" if self._rows_written == 0 else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="raise")
            if self._rows_written == 0:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in self.COLUMNS})
        self._rows_written += 1
        parts = [f"  => {row['status']}"]
        for key in ("report_cold_s", "report_warm_s", "peak_gpu_bytes", "peak_rss_bytes", "static_peak_bytes"):
            if row.get(key) is not None:
                parts.append(f"{key}={row[key]:.4g}")
        logger.info("%s", "  ".join(parts))
        return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", required=True, type=click.Choice(CONFIGS))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="the suite's directory; the CSV is <output-dir>/<config>.csv")
@click.option("--trials", default=3, type=int, help="fresh processes per case")
@click.option("--single-over", default=20.0, type=float,
              help="run a case once when its first trial takes longer (seconds)")
@click.option("--timeout", default=600, type=int, help="per-process timeout (seconds)")
@click.option("--warmup", default=1, type=int, help="PAL: compiles and calls of a copy of the model first")
@click.option("--calls", default=5, type=int, help="PAL: calls of the measured model")
@click.option("--gpu", default=None, help="run PAL/XLA and memo on this GPU (nvidia-smi's numbering)")
@click.option("--mem-limit-gb", default=64.0, type=float, help="memory limit for CPU processes")
@click.pass_context
def cli(ctx, config, output_dir, trials, single_over, timeout, warmup, calls, gpu, mem_limit_gb):
    ctx.obj = Bench(config=config, output_dir=output_dir.resolve(), trials=trials, single_over=single_over,
                    timeout=timeout, warmup=warmup, calls=calls, gpu=gpu, mem_limit_gb=mem_limit_gb)
    _setup_logging(ctx.obj.output_dir / f"{config}.log")
    logger.info("$ %s", shlex.join(sys.argv))


def _generator(suite_dir: str, tool: str) -> list[str]:
    return [sys.executable, str(AE_ROOT / suite_dir / f"gen_{tool}.py")]


def _markov_suite(ctx, suite: str, lengths: str) -> None:
    bench: Bench = ctx.obj
    for L in _parse_int_list(lengths):
        if bench.tool == "memo":
            if suite != "dicefig1":
                raise click.UsageError("memo runs on the dicefig1 suite only")
            argv = [sys.executable, str(AE_ROOT / "dicefig1" / "memo_dicefig1.py"), "--n", str(L)]
        else:
            argv = [*_generator(suite, bench.tool), "--len", str(L)]
        bench.run_case(suite=suite, case_id=f"l{L}", sweep=True, generator_argv=argv, params={"L": L})


for _suite in ("caesar", "diamond", "ladder", "dicefig1"):
    @cli.command(_suite)
    @click.option("-L", "lengths", required=True, help="comma-separated lengths")
    @click.pass_context
    def _cmd(ctx, lengths, _suite=_suite):
        _markov_suite(ctx, _suite, lengths)


@cli.command("bayes")
@click.option("--networks", default="cancer,survey,alarm,insurance,hepar2,pigs,water,munin,andes,link,diabetes,"
                                    "hailfinder")
@click.pass_context
def bayes_cmd(ctx, networks):
    """Bayesian networks, each queried at networks.NODE_IDS[name]."""
    bench: Bench = ctx.obj
    for name in (n.strip() for n in networks.split(",") if n.strip()):
        bif = AE_ROOT / "bayesian_networks" / "bif" / f"{name}.bif"
        bench.run_case(suite="bayes", case_id=name, sweep=False,
                       generator_argv=[*_generator("bayesian_networks", bench.tool), "--bif", str(bif)],
                       params={"network": name})


@cli.command("grid-link")
@click.option("-L", "sizes", required=True, help="grid sizes (L x L)")
@click.option("-p", "prob", default=0.5, type=float)
@click.pass_context
def grid_link_cmd(ctx, sizes, prob):
    bench: Bench = ctx.obj
    gen = [sys.executable, str(AE_ROOT / "grid_networks" / f"gen_{bench.tool}_link_failure.py")]
    for L in _parse_int_list(sizes):
        bench.run_case(suite="grid-link", case_id=f"l{L}", sweep=True,
                       generator_argv=[*gen, "--mode", "3-flow", "--n", str(L), "--m", str(L), "--p", str(prob)],
                       params={"L": L, "nodes": L * L})


@cli.command("grid-router")
@click.option("-L", "sizes", required=True, help="grid sizes (L x L)")
@click.option("--p-init", default=0.5, type=float)
@click.option("--p-edge", default=(0.51, 0.52), type=(float, float))
@click.option("--p-inner", default=(0.53, 0.54, 0.55, 0.56), type=(float, float, float, float))
@click.pass_context
def grid_router_cmd(ctx, sizes, p_init, p_edge, p_inner):
    bench: Bench = ctx.obj
    gen = [sys.executable, str(AE_ROOT / "grid_networks" / f"gen_{bench.tool}_router_failure.py")]
    for L in _parse_int_list(sizes):
        argv = [*gen, "--n", str(L), "--m", str(L), "--p-init", str(p_init),
                "--p-edge", *map(str, p_edge), "--p-inner", *map(str, p_inner)]
        bench.run_case(suite="grid-router", case_id=f"l{L}", sweep=True, generator_argv=argv,
                       params={"L": L, "nodes": L * L})


if __name__ == "__main__":
    cli()
