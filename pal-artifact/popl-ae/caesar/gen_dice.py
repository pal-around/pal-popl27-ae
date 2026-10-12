import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, required=True)
args = parser.parse_args()

header = """fun drawChar(key: int(5), observation: int(5)) {
    let drawnChar = discrete(0.04857676, 0.02506715, 0.03412648, 0.04603178, 0.03027796, 0.04411976, 0.04212122, 0.03900026, 0.03839462, 0.03280276, 0.01331712, 0.0291968, 0.04641515, 0.03141848, 0.03729649, 0.04026981, 0.03791304, 0.04819902, 0.02475538, 0.03121783, 0.04465087, 0.02770877, 0.03491384, 0.02128941, 0.0392666, 0.02638507, 0.01410301, 0.01525122, 0.01625581, 0.01364249, 0.01347052, 0.01254452) in
    let encrypted = key + drawnChar in
    let fail = flip 0.001 in
    if fail then (observe observation == drawnChar)
    else (observe observation == encrypted)
}
let key1 = discrete(0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125) in"""


def gen():
    return "let tmp = drawChar(key1, int(5, 23)) in"


print(header)
for i in range(args.len):
    print(gen())
print("key1")
