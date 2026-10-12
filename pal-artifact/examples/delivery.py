"""Which route did the parcel take? A mixture of two loops of different lengths.

A parcel goes either directly, in two long legs of 1 to 4 days each, or through a depot,
in six short legs of 0 to 2 days each. Both routes take 5.4 days on average, but the
time through the depot varies more. The two routes are the two branches of an if-then-else,
each a loop over its legs that adds up the days. Given the observed delivery time, the
program computes the posterior over the route.

    python delivery.py [--numpy]
"""

import numpy as np

from pal import Bool, Fin, pal, time_execution

Days = Fin(16)
LongLeg = Fin(5)  # 0..4 days
ShortLeg = Fin(3)  # 0..2 days


@pal
def delivery(observed=8):
    direct: Bool = sample([0.5, 0.5])

    with days:
        if direct:
            start: Days = 0
            with markov(elapsed := start, repeat=2):
                long_leg: LongLeg = sample([0, 0.1, 0.3, 0.4, 0.2])
                after_long: Days = elapsed + long_leg
                yield after_long
            yield elapsed
        else:
            depot_start: Days = 0
            with markov(depot_elapsed := depot_start, repeat=6):
                short_leg: ShortLeg = sample([0.3, 0.5, 0.2])
                after_short: Days = depot_elapsed + short_leg
                yield after_short
            yield depot_elapsed

    observe(days, observed)
    return direct


if __name__ == "__main__":
    time_execution(delivery, 8)  # the first call includes compilation
    dist = np.asarray(time_execution(delivery, 8))
    print("P(via the depot), P(direct) given 8 days:", np.round(dist / dist.sum(), 10).tolist())
