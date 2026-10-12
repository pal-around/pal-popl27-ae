"""A loop whose number of iterations is random.

A player gets between 0 and 4 attempts, at random, and each attempt succeeds with
probability p. The loop runs once per attempt, so its count is a random variable; the
result is a mixture, over the number of attempts, of binomial distributions. The program
computes the distribution of the number of successes.

    python random_trials.py [--numpy]
"""

import numpy as np

from pal import Bool, Fin, pal, time_execution

Attempts = Fin(5)
Successes = Fin(5)


@pal
def random_trials(p=0.6):
    attempts: Attempts = sample([0.1, 0.2, 0.3, 0.25, 0.15])

    with markov(successes, repeat=attempts):
        with init():
            none_yet: Successes = 0
            yield none_yet
        hit: Bool = sample([1 - p, p])
        more: Successes = successes + hit
        yield more

    return successes


if __name__ == "__main__":
    time_execution(random_trials, 0.6)  # the first call includes compilation
    dist = time_execution(random_trials, 0.6)
    print("distribution of successes:", np.round(np.asarray(dist), 10).tolist())
