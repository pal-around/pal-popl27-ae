"""Gambler's ruin: a random walk with absorbing ends.

A gambler starts with 3 coins and bets one coin per round, winning with probability p.
At 0 coins (ruined) or 6 coins (goal reached) the gambler stops playing. Each round is
a loop iteration whose body branches on whether the game is over. The program computes
the distribution of the gambler's coins after n rounds.

    python gamblers_ruin.py [--numpy]
"""

import numpy as np

from pal import Bool, Fin, pal, time_execution

Coins = Fin(7)


@pal
def gamblers_ruin(p=0.45, n=20):
    coins0: Coins = 3

    with markov(coins := coins0, repeat=n):
        with coins_next:
            if coins == 0 or coins == 6:
                stay: Coins = coins
                yield stay
            else:
                win: Bool = sample([1 - p, p])
                moved: Coins = coins + 1 if win else coins - 1
                yield moved
        yield coins_next

    return coins


if __name__ == "__main__":
    time_execution(gamblers_ruin, 0.45, 20)  # the first call includes compilation
    dist = time_execution(gamblers_ruin, 0.45, 20)
    print("distribution of coins after 20 rounds:", np.round(np.asarray(dist), 10).tolist())
