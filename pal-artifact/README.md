# PAL

> **POPL 2027 artifact evaluation:** start with [`popl-ae/README.md`](popl-ae/README.md).

PAL is a probabilistic programming language embedded in Python for discrete models.
A PAL program is a decorated Python function; PAL compiles it to array code that computes
the exact distribution of the returned variables, and runs it with JAX (optionally
XLA-compiled, on CPU or GPU) or with NumPy.

```python
from pal import Bool, Fin3, pal, time_execution

@pal
def cancer():
    pollution: Fin3 = sample([0.5, 0.4, 0.1])
    smoker: Bool = sample([0.3, 0.7])
    cancer: Bool = sample(pollution > [smoker > [[0.03, 0.97], [0.001, 0.999]],
                                       smoker > [[0.03, 0.97], [0.001, 0.999]],
                                       smoker > [[0.05, 0.95], [0.02, 0.98]]])
    xray: Bool = sample(cancer > [[0.9, 0.1], [0.2, 0.8]])
    return xray

time_execution(cancer)   # prints the run time; cancer() returns the distribution of xray
```

## Setup

The development environment is a Nix flake:

```sh
nix develop            # Python 3.12 with JAX (CUDA), NumPy, and the benchmark tools
```

`nix develop` puts the repository root on `PYTHONPATH`, so `import pal` uses this
checkout.

## The language

| Statement | Meaning |
|---|---|
| `x: T = sample(w)` | draw `x` from domain `T` with weights `w`, normalized per assignment of the variables `w` reads; `_` in `w` stands for `x` |
| `x: T = e` | deterministic assignment |
| `a = b` | copy a variable |
| `condition(e)` | keep only the outcomes where `e` holds |
| `observe(x, v)` | `condition(x == v)`; `v` may be a list of values |
| `with y, z:` + `if`/`elif`/`else` | branch; each branch ends with `yield` of the values bound to `y, z` |
| `with markov(s := s0, repeat=n):` … `yield s1` | a loop of `n` steps carrying `s` (initialized from `s0`, or from a `with init():` block) |
| `with markov(…, repeat=n, xs=(p := theta)):` | the same loop, with `p` bound to one row of the parameter `theta` per iteration (its leading axis is the iteration) |
| `return x, y` | the variables whose joint distribution is computed |

Domains are `Bool`, `Fin(n)` (values `0 … n-1`; `Fin2` … `Fin9` are predefined) or any
Python list of values. Weights can be written as expressions (`0.9 if x == y else 0.1`),
as a list over the domain (`[0.3, 0.7]`), or as a guarded table (`x > [w0, w1, …]`, which
means `w0 if x == 0 else w1 if x == 1 else …`). A weight row that sums to zero
contributes zero mass. Function parameters that are not random (probabilities, the loop
count) are ordinary arguments.

Names must be fresh within a block. A per-iteration row is data, not a random variable:
a row of numbers used as a whole weight is that step's table over the sampled variable's
domain, and used inside a larger expression it must be a scalar. See
[`examples/`](examples/README.md) for complete programs.

## Running programs

A script's `@pal` functions take the compiler's options from the command line
(`python program.py --numpy`); options passed to the decorator, as in
`@pal(rescale=True)`, take precedence:

| Flag | Default | Effect |
|---|---|---|
| `--numpy` | off | run with NumPy instead of JAX |
| `--jit` / `--no-jit` | on | XLA-compile the program (JAX only) |
| `--dtype float32\|float64` | `float64` | precision of probabilities |
| `--lva` / `--no-lva` | on | marginalize each variable as soon as it is no longer needed |
| `--reorder` / `--no-reorder` | on | reorder each block to lower the peak tensor size |
| `--prune` | off | drop factors that cannot affect the result |
| `--table-lowering` / `--no-table-lowering` | on | turn constant weight tables into arrays |
| `--rescale` | off | rescale inside loops to avoid underflow (`--return-log-scale` returns the scale) |
| `--print-compiled`, `--print-parsed` | off | show the generated array program / the parsed program |
| `--print-final-likelihood` | off | print the result |
| `--print-zero-mass` | off | report where weight rows summing to zero drop mass |
| `--print-pal-compilation-time`, `--print-device`, `--print-static-ndim-peak` | off | diagnostics |

`python program.py --help` prints the full list. Setting `PAL_REORDER_DEBUG=1` prints the
reordering pass's candidate orders.

## Tests

```sh
nix develop .#artifact --command python -m pytest unittests -q
```

`unittests/test_examples.py` runs every program in `examples/` and checks its answer.

## Benchmarks

[`popl-ae/`](popl-ae/README.md) holds the paper's benchmarks: the programs, their Dice
and memo counterparts, and a Makefile that runs them and typesets the results.

## Licence

MIT; see [`LICENSE`](LICENSE).
