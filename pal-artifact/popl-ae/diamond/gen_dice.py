import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, required=True)
args = parser.parse_args()

header = """fun diamond(s1: bool) : bool {
    let route = flip 0.5 in
    let s2 = if route then s1 else false in
    let s3 = if route then false else s1 in
    let keep = flip 0.999 in
    s2 || (s3 && keep)
}
let net1 = diamond(true) in"""

def gen(num):
    return f"let net{num + 1} = diamond(net{num}) in"

print(header)
for i in range(1, args.len):
    print(gen(i))
print(f"net{args.len}")
