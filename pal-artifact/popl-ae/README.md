# Artifact: Tensorizing Probabilistic Programs for Accelerated Exact Inference

This is the artifact of the POPL 2027 submission *Tensorizing Probabilistic Programs for
Accelerated Exact Inference*. It reproduces the paper's evaluation (Section 11): every
experiment, run by one command, with the paper's figures and table typeset from the new
results.

It contains PAL, the compiler the paper describes (`../pal/`); the benchmark programs and
their generators; scripts that run PAL and the baselines (Dice, memo, and Dice in Julia
via Alea.jl); and the LaTeX sources of the paper's figures.

**Badges sought:** Available, Functional and Reusable.

- *Functional:* `make full` runs all the experiments of Section 11 and writes
  `report.pdf` (Figures 10 and 11, Table 1 and the generator-tuning plot, typeset from the
  new results).
  `make check` confirms that PAL, Dice and memo compute the same distributions.
- *Reusable:* the artifact is open source (MIT licence). The environment is pinned by
  `flake.nix` and `flake.lock`. PAL is a Python library that runs any PAL program, and
  the benchmark generators and driver take other sizes, other Bayesian networks
  (`.bif`) and other configurations (section 4).

**At a glance**

| | |
|---|---|
| Platform | Linux on x86-64, with Nix; an NVIDIA GPU for the GPU results |
| Disk | about 7 GB for the environment |
| Memory | 64 GB for the full evaluation (each process is limited to 64 GB) |
| Network | only during installation |
| Installation | about an hour, mostly downloads |
| Sanity test (`make smoke`) | about 5 minutes |
| Full evaluation (`make full`) | about 2 h 15 min with a GPU, about 3 h without, on the verified host |

The artifact runs as an ordinary user. It writes only inside its own directory and, for
Alea.jl's Julia packages, `~/.cache/pal-dev`.

Contents: [1. Claims](#1-claims) · [2. Getting started](#2-getting-started) · [3. Evaluation](#3-evaluation) · [4. Artifact description](#4-artifact-description)

---

## 1. Claims

Paper references are to the submitted version. `report` means `results/<mode>/report.pdf`.

| # | Claim | In the paper | Command | Where to look |
|---|---|---|---|---|
| C1 | On the Markov models, PAL/XLA shows strong scalability compared with Dice. | §11.2 *Markov models*; Fig. 10, top row | `make markov` | report §1, top row |
| C2 | PAL/XLA's JIT compilation adds a roughly constant overhead (cold vs warm time). | Fig. 10, middle row | `make markov` | report §1, middle row |
| C3 | PAL/XLA's peak memory stays constant as the problem sizes grow. | §11.1; Fig. 10, bottom row | `make markov` | report §1, bottom row |
| C4 | On the scaling benchmark, PAL/XLA shows strong scalability compared with memo. | Fig. 10, top row, last panel | `make markov` | report §1 |
| C5 | On the Bayesian networks, PAL/NumPy (compilation plus inference) shows good performance compared with Dice. | §11.2 *Bayesian networks*; Table 1 | `make bayes` | report §2 |
| C6 | On network reliability, PAL is significantly faster than Dice on large grids; eager marginalization reduces PAL/XLA's memory footprint on the GPU. | §11.2 *Network reliability*; Fig. 11 | `make grid` | report §3 |
| C7 | Tuning a random list generator (N = 100), PAL converges in a fraction of the time Dice (Alea.jl) takes. | §11.2 *Tuning random generators*, plot | `make natlist` | report §4 |

**Which claims need a GPU.** C1–C4, the XLA part of C6, and C7 run PAL/XLA on the GPU in
the paper. Without `GPU=<index>`, the same commands run PAL/XLA on the CPU, and the memory
rows show the measured growth of resident memory. C5 runs on the CPU.

---

## 2. Getting started

### Requirements

| Software | Version |
|---|---|
| Linux, x86-64 | tested on Ubuntu 24.04.4 LTS |
| [Nix](https://nixos.org/download), with flakes | tested with Nix 2.21 |
| NVIDIA driver (for the GPU results) | 525 or later (CUDA 12); tested with 580.173.02 |

Nix provides everything else, pinned by `flake.lock` (nixpkgs 25.05):

| Component | Version |
|---|---|
| Python | 3.12, with JAX 0.6.0 (CUDA 12 plugin), NumPy, optax, pytest |
| Dice | [SHoltzen/dice](https://github.com/SHoltzen/dice) at commit `1d8a277` |
| bngen | [SHoltzen/bngen](https://github.com/SHoltzen/bngen) at commit `bee3826` |
| memo | [kach/memo](https://github.com/kach/memo) 1.2.10 |
| Alea.jl | [Tractables/Alea.jl](https://github.com/Tractables/Alea.jl) at commit `9840424`, on Julia 1.9.4 |
| TeX Live | 2024, with pgfplots and latexmk |

No system-wide CUDA toolkit is needed: the environment brings its own CUDA runtime and
needs only the driver. `nvidia-smi` should list the GPU.

**Verified host:** Ubuntu 24.04.4 LTS; Intel Core i7-7820X (8 cores, 16 threads); 128 GB of
memory; NVIDIA GeForce RTX 2080 Ti (11 GB) and RTX 2080 (8 GB); NVIDIA driver 580.173.02.

### Installation (about an hour; needs the network)

1. Install Nix, if the machine does not have it:

   ```
   sh <(curl -L https://nixos.org/nix/install) --daemon
   ```

2. From the artifact's top directory, enter the environment. The first time, this
   downloads and builds it:

   ```
   nix --experimental-features 'nix-command flakes' develop
   ```

3. Inside the environment, download and precompile Alea.jl's Julia packages:

   ```
   cd popl-ae
   make setup
   ```

   It ends with `Alea.jl is ready.` From here on, nothing needs the network.

All later commands run inside this environment, in `popl-ae/`.

### Sanity test (about 5 minutes)

```
make smoke GPU=0      # with an NVIDIA GPU: its index as nvidia-smi shows it
make smoke            # without a GPU
```

`make smoke` runs every experiment at small sizes, typesets the report and compares the
answers. It ends with

```
… cases run by two or more configurations; all agree (tolerance 1e-05).
```

and writes `results/smoke/report.pdf`. The report has four sections, one per experiment:
Figure 1 (Markov models), Table 1 (Bayesian networks; the smoke test runs six of them),
Figure 2 (network reliability) and Figure 3 (generator tuning). At the smoke test's small
sizes the curves are short, and Dice and memo, limited to 10 s per run, stop early.

With `GPU=<index>`, check that PAL/XLA ran on the GPU. The `device` column of the results
names the device each run used, and the first paragraph of `report.pdf` lists them:

```
grep -c NVIDIA results/smoke/caesar/pal-xla.csv     # more than 0
```

### Troubleshooting

- **PAL/XLA runs on the CPU although `GPU=<index>` is given.** Check that `nvidia-smi` works,
  and that the index is the one it shows. `sitecustomize.py` (at the artifact's top
  directory) loads the driver's `libcuda.so.1` from `/usr/lib/x86_64-linux-gnu`, `/usr/lib64`
  or `/run/opengl-driver/lib`; if the driver lives elsewhere, add its directory there.
- **`no kernel image is available for execution on the device`.** Add the GPU's compute
  capability to `cudaCapabilities` in `flake.nix` and enter the environment again.
- **A sweep stops early with status `oom`.** The machine or GPU has less memory than the
  run needs. `MEM_GB=32` (for example) lowers the per-process limit; the sweep then stops
  at a smaller size.
- **`make setup` fails.** It needs the network; run it again when connected.

---

## 3. Evaluation

### Commands and times

Times are on the verified host.

| Command | Runs | Time, CPU only | Time, with `GPU=1` (RTX 2080 Ti) |
|---|---|---|---|
| `make smoke` | every experiment at small sizes; report; check | ~5 min | ~4 min |
| `make full` | `markov`, `bayes`, `grid`, `natlist`, then `report` and `check` | ~3 h | ~2 h 15 min |
| `make markov` | C1–C4 | ~35 min | ~50 min |
| `make bayes` | C5 | ~15 min | ~15 min |
| `make grid` | C6 | ~2 h | ~70 min |
| `make natlist` | C7 | < 1 min | < 1 min |
| `make report` | `report.pdf` from the results | < 1 min | < 1 min |
| `make check` | answers agree across PAL, Dice and memo | < 1 min | < 1 min |

`make markov`, `bayes`, `grid` and `natlist` run the full sizes by default (`MODE=full`)
and write to `results/full/`; `MODE=smoke` runs the small sizes, and `RESULTS=<dir>`
chooses another directory. Rerunning a step overwrites its results.

### How the runs are set up

- **PAL** runs with its default options, in double precision. Each size runs 3 times, each in a fresh process, and the mean is reported.
- **Dice** runs with its default options, 3 runs per size. We found Dice's `-eager-eval` option slows down some benchmarks significantly, so the artifact does not use it. Dice recurses deeply on large programs, so the environment's `dice` runs it with the stack limit raised. The artifact uses the most recent version of Dice. On some machines, Dice's times on the Markov models may be larger than the paper reports.
- **Bayesian networks.** The Dice programs are what [bngen](https://github.com/SHoltzen/bngen) writes, kept to the query's ancestors. The PAL programs sample the same variables, in the same order, with the same weights.
- **Limits.** In `make full`, each run is stopped after 60 s for Dice and memo on the
  Markov models and grids, after 10 minutes for Dice on a Bayesian network, and after 10
  minutes for PAL; each process may use up to 64 GB of memory (`MEM_GB`). These time and
  memory limits are lower than in the paper's evaluation, so that the full evaluation
  finishes in a reasonable time. Each benchmark runs the paper's sizes from smallest to
  largest, and once a size fails (it times out, runs out of memory or crashes), the larger
  sizes are not run and are marked `skipped`. A tool can therefore stop at a smaller size
  than in the paper.

`make check` compares the query's distribution across PAL, Dice, and memo on every case two
or more of them finish, within 10^-5.

### What to look for

For each claim, compare the section of `report.pdf` with the paper's figure or table. A
different machine, GPU or limit gives different numbers, but the trends below stay the
same. Numbers in parentheses are from the verified host with a GPU.

- **C1** (report §1, top row; Fig. 10, top row). Dice's time grows super-linearly with
  the length on diamond, ladder and scaling, and about linearly on cipher; PAL/XLA's grows
  linearly and stays within a few seconds. Within the 60 s limit, Dice stops at smaller
  lengths than in the paper, so the speedups at the largest length both finish are smaller
  than the paper's (cipher: Dice stops at 250 characters, where PAL/XLA is about 80×
  faster; the paper reports 164× at 10^5 characters). On the smallest lengths Dice can be
  faster than PAL/XLA, by less than a second.
- **C2** (report §1, middle row). PAL/XLA's cold and warm curves are parallel: their
  difference, XLA's compilation, stays roughly constant as the length grows (0.1–0.6 s).
- **C3** (report §1, bottom row). PAL/XLA's peak memory is flat across lengths on all four
  benchmarks. On cipher, without eager marginalization (dashed), the joint tensor PAL
  builds at each step is larger than that with eager marginalization. On the CPU, the
  memory row shows the growth of resident memory, which includes Python's own allocations.
- **C4** (report §1, top row, last panel). memo's time grows much faster than PAL/XLA's.
  Within the 60 s limit memo stops before horizon 5·10^4, so the speedup at the largest
  horizon memo finishes is smaller than the paper's (about 100× at 2·10^4; the paper
  reports over 300× at 5·10^4).
- **C5** (report §2, Table 1). PAL/NumPy, counting compilation and inference, is faster
  than Dice on every network Dice finishes. Dice does not finish Diabetes or Hailfinder
  (✗: timeout; ✗ (OOM): out of memory), while PAL/NumPy compiles and runs Diabetes in seconds.
  Without eager marginalization, 7 of the 12 networks exceed the memory limit (●).
- **C6** (report §3, Figure 2; Fig. 11). In panels (a) and (b), Dice's time rises steeply
  and Dice stops at a small grid (12 × 12 for link failure, 14 × 14 for router failure),
  where PAL/NumPy is already over 100× faster; PAL/XLA continues until the GPU's VRAM
  runs out or to the largest grid swept.
  The paper's larger speedups are at larger grids, which Dice reaches
  only with a longer time limit. Without eager marginalization, PAL/XLA stops at a small
  grid. In panels (c) and (d), PAL/XLA's measured GPU peak follows the analytically predicted
  size of the joint tensor (*N*-joint).
- **C7** (report §4). PAL's KL divergence drops below 1 within a fraction of a second,
  Dice's (Alea.jl) within seconds.

Without a GPU, PAL/XLA runs on the CPU and the trends are the same. On the Markov models,
whose arrays are small, PAL/XLA is faster on the CPU, so its speedups over Dice and memo
are larger and XLA's compilation adds less; on the grids, PAL/XLA reaches the time limit
at a smaller grid.

---

## 4. Artifact description

### Files

```
README.md, LICENSE      PAL's README and the licence
pal/                    PAL, the compiler (Python)
examples/               small PAL programs
unittests/              PAL's tests (python -m pytest unittests -q)
flake.nix, flake.lock   the environment; nix/ holds its package definitions
sitecustomize.py        lets JAX find the host's NVIDIA driver
popl-ae/                this artifact
  Makefile              the commands of section 3
  bench.py              runs one configuration over one benchmark suite
  caesar/ diamond/ ladder/ dicefig1/     Markov-model generators (PAL, Dice; memo in dicefig1/)
  bayesian_networks/    the 12 networks (.bif) and their PAL and Dice generators
  grid_networks/        network-reliability generators
  natlist/              generator tuning: natlist.py (PAL), natlist.jl (Dice via Alea.jl)
  report/               make_report.py, check.py; the paper's figure sources
  tests/                tests of the generators, natlist and the report
  make-archive.sh       packs the artifact
```

(The cipher benchmark's files are named `caesar`, and the scaling benchmark's files are named `dicefig1`.)

### How the experiments map to the scripts

| Experiment | Generators (in `popl-ae/`) | Configurations (`bench.py --config`) |
|---|---|---|
| Markov models (C1–C4) | `caesar/`, `diamond/`, `ladder/`, `dicefig1/`: `gen_pal.py`, `gen_dice.py` (`--len N`); `dicefig1/memo_dicefig1.py` | `pal-xla`; `pal-xla-nomarg` (cipher); `dice`; `memo` (scaling) |
| Bayesian networks (C5) | `bayesian_networks/gen_pal.py`, `gen_dice.py` (`--bif FILE [--node N]`); queries in `networks.py` | `pal-numpy`, `pal-numpy-nomarg`, `dice` |
| Network reliability (C6) | `grid_networks/gen_{pal,dice}_{link,router}_failure.py` (`--n L --m L`) | `pal-xla`, `pal-xla-nomarg`, `pal-numpy`, `dice` |
| Generator tuning (C7) | `natlist/natlist.py`, `natlist/natlist.jl` | — |

`bench.py` writes, for each suite and configuration, `results/<mode>/<suite>/<config>.csv`
(times, static and measured peak memory, the answer, and the device), a log with every
command and its output, and every program it ran under `sources/`. `report/make_report.py`
turns the CSVs into the paper's plot data and typesets them with the paper's figure
sources; `report/check.py` compares the answers.

### Trying other inputs

Inside the environment, in `popl-ae/`:

```
python bench.py --config pal-xla --output-dir results/mine caesar -L 7,70,700
python bench.py --config dice --output-dir results/mine bayes --networks alarm
python bayesian_networks/gen_pal.py --bif my-network.bif --node 3 > net.py
python net.py --numpy                               # run a generated PAL program
python natlist/natlist.py --n 50 --epochs 200       # generator tuning at another size
python -m pytest tests -q                           # the artifact's tests
```

A PAL program is a Python module with a decorated function:
`../examples/` and the generated programs under `results/*/*/sources/`
show the syntax, and `python <program> --help` lists PAL's options.

Any `.bif` network works with the Bayesian-network generators; `--node` picks the query
variable by its position in the file. 

To use PAL as a baseline for a new model, write it as a PAL program. PAL's README
(`../README.md`) describes the language and its options, and `../examples/` holds small
programs that each print the distribution they compute: for example,
`python ../examples/hmm.py --numpy`.

### Licence

The artifact is distributed under the MIT licence (`../LICENSE`). Dice, bngen, memo and
Alea.jl are their authors' software, which the Nix environment fetches from their public
repositories under their own licences.
