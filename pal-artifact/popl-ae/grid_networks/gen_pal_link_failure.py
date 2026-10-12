import argparse


def create_program(args: argparse.Namespace) -> str:
    if args.n <= 0 or args.m <= 0:
        raise ValueError("n and m must be positive integers")

    if args.p < 0 or args.p > 1:
        raise ValueError("p must be between 0 and 1")

    lines = []

    # Setup
    lines.append("from pal import pal, time_execution, Bool")
    lines.append("")
    lines.append("@pal")
    lines.append("def grid_network():")

    # Function body
    lines.append("    init_conn_1: Bool = True")

    # Every coin gets its own name: PAL variable names must be fresh within a scope.
    for j in range(2, args.m + 1):
        lines.append("")
        lines.append(f"    init_flip_{j}: Bool = sample([1 - {args.p}, {args.p}])")
        lines.append(f"    init_conn_{j}: Bool = init_conn_{j - 1} and init_flip_{j}")

    lines.append("")
    if args.n == 1:
        lines.append(f"    return init_conn_{args.m}")
    else:
        lines.append(
            f"    with markov({', '.join(f'prev_conn_{j} := init_conn_{j}' for j in range(1, args.m + 1))}, repeat={args.n - 1}):"
        )
        lines.append(f"        flip_1: Bool = sample([1 - {args.p}, {args.p}])")
        lines.append("        conn_1: Bool = prev_conn_1 and flip_1")

        for j in range(2, args.m + 1):
            lines.append("")
            lines.append(f"        flip1_{j}: Bool = sample([1 - {args.p}, {args.p}])")
            lines.append(f"        flip2_{j}: Bool = sample([1 - {args.p}, {args.p}])")

            if args.mode == "2-flow":
                lines.append(f"        conn_{j}: Bool = (conn_{j - 1} and flip1_{j}) or (prev_conn_{j} and flip2_{j})")
            else:
                lines.append(f"        flip3_{j}: Bool = sample([1 - {args.p}, {args.p}])")
                lines.append(
                    f"        conn_{j}: Bool = (conn_{j - 1} and flip1_{j}) or (prev_conn_{j} and flip2_{j}) or (prev_conn_{j - 1} and flip3_{j})"
                )

        lines.append("")
        lines.append(f"        yield {', '.join(f'conn_{j}' for j in range(1, args.m + 1))}")

        lines.append("")
        lines.append(f"    return prev_conn_{args.m}")

    # Timing
    lines.append("")
    lines.append("time_execution(grid_network)")
    lines.append("time_execution(grid_network)")

    return "\n".join(lines)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, choices=["3-flow", "2-flow"])
parser.add_argument("--n", type=int, required=True, help="Number of rows")
parser.add_argument("--m", type=int, required=True, help="Number of columns")
parser.add_argument("--p", type=float, required=True, help="Probability that a link works")
args = parser.parse_args()

print(create_program(args))
