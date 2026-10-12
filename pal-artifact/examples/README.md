Examples
========

Each program runs on its own and prints the distribution it computes. The default
backend is JAX; `--numpy` selects the NumPy backend (e.g. `python hmm.py --numpy`).
Run them from the repository's development shell (`nix develop`).
`unittests/test_examples.py` checks every answer on both backends.

| Program | Model | Control flow |
| --- | --- | --- |
| [`ladder.py`](ladder.py) | the paper's example: a packet travelling down a ladder of routers with unreliable links | loop |
| [`casino.py`](casino.py) | whether a dealer rolls a fair die or secretly switches to a loaded one: two models of the same rolls | if-then-else whose branches loop over observed data |
| [`delivery.py`](delivery.py) | which of two routes, with different numbers of legs, a parcel took | if-then-else whose branches are loops of different lengths |
| [`hmm.py`](hmm.py) | a hidden Markov model (rain and umbrellas), filtered | loop over observed data |
| [`gamblers_ruin.py`](gamblers_ruin.py) | a random walk with absorbing ends | if-then-else inside a loop |
| [`random_trials.py`](random_trials.py) | successes in a random number of attempts | loop with a random count |
