import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, required=True)
args = parser.parse_args()

header = """fun ladder(s1: bool, s2: bool) {
    if s1 then
    if flip 0.001 then
        (false, false)
    else
        let f = flip 0.5 in (f, !f)
    else if s2 then
        let f = flip 0.5 in (f, !f)
    else (false, false)
}
let x = (true, false) in"""

def gen():
    return "let x = ladder(fst x, snd x) in"

print(header)
for i in range(1, args.len + 1):
    print(gen())
print("(fst x) && !(snd x)")
