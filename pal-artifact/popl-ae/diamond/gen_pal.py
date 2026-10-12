import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, required=True)
args = parser.parse_args()

# Setup
print("""from pal import Bool, pal, time_execution

@pal
def diamond():""")

# Function body
print("    true: Bool = True")
print(f"    with markov(s1 := true, repeat={args.len}):")
print("        route: Bool = sample(0.5)")
print("        with s2:")
print("            if route:")
print("                temp = s1")
print("                yield temp")
print("            else:")
print("                temp: Bool = False")
print("                yield temp")
print("        with s3:")
print("            if route:")
print("                temp: Bool = False")
print("                yield temp")
print("            else:")
print("                temp = s1")
print("                yield temp")
print("        keep: Bool = sample([0.001, 0.999])")
print("        s4: Bool = s2 or (s3 and keep)")
print("        yield s4")
print("    return s1")

# Timing
print("time_execution(diamond)")
print("time_execution(diamond)")
