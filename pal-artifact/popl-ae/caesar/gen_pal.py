import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, required=True)
args = parser.parse_args()

# Setup
print("""from pal import pal, time_execution, Bool, Fin

Alphabet = Fin(32)

# rescale=True is required, not a tuning knob: each iteration multiplies the
# likelihood by ~1e-3, so without rescaling the chain underflows to exactly 0.0
# by length ~100 and this benchmark runs to 100000. Rescaling divides by the max
# once per iteration and accumulates the log divisor, keeping the evidence
# probability recoverable.
@pal(rescale=True)
def caesar():""")

# Function body
print(rf"""    key1: Alphabet = sample(1)

    with markov(key := key1, repeat={args.len}):
        drawnChar: Alphabet = sample([0.04857676, 0.02506715, 0.03412648, 0.04603178, 0.03027796, 0.04411976, 0.04212122, 0.03900026, 0.03839462, 0.03280276, 0.01331712, 0.0291968, 0.04641515, 0.03141848, 0.03729649, 0.04026981, 0.03791304, 0.04819902, 0.02475538, 0.03121783, 0.04465087, 0.02770877, 0.03491384, 0.02128941, 0.0392666, 0.02638507, 0.01410301, 0.01525122, 0.01625581, 0.01364249, 0.01347052, 0.01254452])
        fail: Bool = sample([0.999, 0.001])
        with diff:
            if fail:
                temp: Alphabet = 0
                yield temp
            else:
                temp = key
                yield temp
        condition((drawnChar + diff) % 32 == 23)

        yield key

    return key""")

# Timing
print("time_execution(caesar)")
print("time_execution(caesar)")
