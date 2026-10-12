import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, required=True)
args = parser.parse_args()

header = """let x = flip 0.1 in"""

def gen():
    return f"let x = if x then flip 0.3 else flip 0.4 in"

print(header)
for i in range(args.len):
    print(gen())
print(f"x")
