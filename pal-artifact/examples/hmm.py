"""A hidden Markov model: is it raining?

Whether it rains on a day depends on whether it rained the day before; we do not see
the rain, only whether the director brings an umbrella. The umbrellas seen on each day
are passed to the loop as data (``xs``) and observed. The program computes the
distribution of rain on the last of n days given all the umbrellas (filtering).

    python hmm.py [--numpy]
"""

import numpy as np

from pal import Bool, pal, time_execution

UMBRELLAS = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1])  # 1 = umbrella seen


@pal
def hmm(umbrellas, n=10):
    rain0: Bool = sample([0.5, 0.5])

    with markov(rain := rain0, repeat=n, xs=(u := umbrellas)):
        rain_today: Bool = sample(rain > [[0.7, 0.3], [0.3, 0.7]])
        umbrella: Bool = sample(rain_today > [[0.8, 0.2], [0.1, 0.9]])
        observe(umbrella, u)
        yield rain_today

    return rain


if __name__ == "__main__":
    time_execution(hmm, UMBRELLAS, len(UMBRELLAS))  # the first call includes compilation
    dist = np.asarray(time_execution(hmm, UMBRELLAS, len(UMBRELLAS)))
    print("P(no rain), P(rain) on the last day:", np.round(dist / dist.sum(), 10).tolist())
