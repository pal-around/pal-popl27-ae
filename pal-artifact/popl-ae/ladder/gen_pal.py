import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, required=True)
args = parser.parse_args()

# Setup
print("""from pal import Bool, pal, time_execution

@pal
def ladder():""")

# Function body
print("    true: Bool = True")
print("    false: Bool = False")
print("")
print(f"    with markov(s1 := true, s2 := false, repeat={args.len}):")
print("        with yield1, yield2:")
print("            if s1:")
print("                flip: Bool = sample([0.999, 0.001])")
print("                with yield_inner1, yield_inner2:")
print("                    if flip:")
print("                        temp1: Bool = False")
print("                        temp2: Bool = False")
print("                        yield temp1, temp2")
print("                    else:")
print("                        f: Bool = sample(0.5)")
print("                        not_f: Bool = not f")
print("                        yield f, not_f")
print("                yield yield_inner1, yield_inner2")
print("            elif s2:")
print("                f: Bool = sample(0.5)")
print("                not_f: Bool = not f")
print("                yield f, not_f")
print("            else:")
print("                temp1: Bool = False")
print("                temp2: Bool = False")
print("                yield temp1, temp2")
print("        yield yield1, yield2")
print("")
print("    ret: Bool = s1 and not s2")
print("    return ret")
print("")

# Timing
print("time_execution(ladder)")
print("time_execution(ladder)")
