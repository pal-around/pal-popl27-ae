import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, required=True)
args = parser.parse_args()

# Setup
print("""from pal import Bool, pal, time_execution

@pal
def dicefig1():""")

# Function body
print("    x: Bool = sample([0.9, 0.1])")
print(f"    with markov(y := x, repeat={args.len}):")
print("        with z:")
print("            if y:")
print("                temp: Bool = sample([0.7, 0.3])")
print("                yield temp")
print("            else:")
print("                temp: Bool = sample([0.6, 0.4])")
print("                yield temp")
print("        yield z")
print("    return y")

# Timing
print("time_execution(dicefig1)")
print("time_execution(dicefig1)")
