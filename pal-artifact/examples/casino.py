"""Is the casino cheating? Comparing two models of the same data.

A dealer rolls a die n times. An honest dealer always rolls a fair die. A cheating dealer
secretly switches between the fair die and a die loaded towards six, which makes the
sequence of dice a hidden Markov chain. The two models are the two branches of an
if-then-else, and each branch runs its own loop over the observed rolls, which the loops
receive as data (``xs``). The program computes the posterior over whether the dealer
cheats, jointly with whether the loaded die is in play after the last roll.

    python casino.py [--numpy]
"""

import numpy as np

from pal import Bool, Fin, pal, time_execution

ROLLS = np.array([3, 6, 1, 5, 2, 6, 6, 4, 6, 6, 6, 3, 6, 6, 2, 6, 1, 4, 3, 5])
Face = Fin(6)  # the faces 1..6 as 0..5


@pal
def casino(faces, n=20):
    cheating: Bool = sample([0.5, 0.5])

    with loaded_at_end:
        if cheating:
            # Switch to the loaded die with probability 0.05 per roll, and back to the
            # fair die with probability 0.1; the loaded die rolls a six half the time.
            loaded0: Bool = False
            with markov(loaded := loaded0, repeat=n, xs=(face := faces)):
                loaded_now: Bool = sample(loaded > [[0.95, 0.05], [0.1, 0.9]])
                roll: Face = sample(loaded_now > [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 5]])
                observe(roll, face)
                yield loaded_now
            yield loaded
        else:
            with markov(repeat=n, xs=(fair_face := faces)):
                fair_roll: Face = sample(1)
                observe(fair_roll, fair_face)
                yield
            never: Bool = False
            yield never

    return cheating, loaded_at_end


if __name__ == "__main__":
    time_execution(casino, ROLLS - 1, len(ROLLS))  # the first call includes compilation
    joint = np.asarray(time_execution(casino, ROLLS - 1, len(ROLLS)))
    joint = joint / joint.sum()
    print("P(cheating):", round(float(joint[1].sum()), 10))
    print("P(cheating, loaded die at the end), rows cheating = False/True:", np.round(joint, 10).tolist())
