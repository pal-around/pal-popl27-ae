"""The ladder network of the paper's running example.

A packet travels down an n x 2 ladder of routers; between two layers, each of the four
links works with probability a_ij. The program computes the probability that the packet
reaches a router in the last layer.

    python ladder.py [--numpy]
"""

import numpy as np

from pal import Bool, pal, time_execution


@pal
def ladder(a11=0.9, a12=0.9, a21=0.9, a22=0.9, n=10):
    # init: the packet starts at the left router of the first layer
    x1_0: Bool = True
    x2_0: Bool = False

    with markov(x1 := x1_0, x2 := x2_0, repeat=n):
        # links: sample the status s_ij of each link between two layers
        s11: Bool = sample([1 - a11, a11])
        s12: Bool = sample([1 - a12, a12])
        s21: Bool = sample([1 - a21, a21])
        s22: Bool = sample([1 - a22, a22])

        # transmit: a next-layer router gets the packet iff an incoming link works
        x1p: Bool = (x1 and s11) or (x2 and s21)
        x2p: Bool = (x1 and s12) or (x2 and s22)
        yield x1p, x2p

    # success iff either router in the final layer holds the packet
    success: Bool = x1 or x2
    return success


if __name__ == "__main__":
    time_execution(ladder, 0.9, 0.9, 0.9, 0.9, 10)  # the first call includes compilation
    dist = time_execution(ladder, 0.9, 0.9, 0.9, 0.9, 10)
    print("P(success = False), P(success = True):", np.round(np.asarray(dist), 10).tolist())
