import argparse


def create_program(args: argparse.Namespace) -> str:
    if args.n <= 0 or args.m <= 0:
        raise ValueError("n and m must be positive integers")

    if not all(0 <= p <= 1 for p in args.p_init):
        raise ValueError("p_init values must be between 0 and 1")
    if not all(0 <= p <= 1 for p in args.p_edge):
        raise ValueError("p_edge values must be between 0 and 1")
    if not all(0 <= p <= 1 for p in args.p_inner):
        raise ValueError("p_inner values must be between 0 and 1")

    lines = []

    # Setup
    lines.append("from pal import pal, time_execution, Bool")
    lines.append("")
    lines.append("@pal")
    lines.append("def grid_network():")

    # Function body
    lines.append(f"    init_conn_1: Bool = sample([1 - {args.p_init[0]}, {args.p_init[0]}])")

    for j in range(2, args.m + 1):
        lines.append(f"    with init_conn_{j}:")
        lines.append(f"        if init_conn_{j - 1}:")
        lines.append(f"            temp: Bool = sample([1 - {args.p_edge[0]}, {args.p_edge[0]}])")
        lines.append("            yield temp")
        lines.append("        else:")
        lines.append(f"            temp: Bool = sample([1 - {args.p_edge[1]}, {args.p_edge[1]}])")
        lines.append("            yield temp")

    lines.append("")
    if args.n == 1:
        lines.append(f"    return init_conn_{args.m}")
    else:
        lines.append(
            f"    with markov({', '.join(f'prev_conn_{j} := init_conn_{j}' for j in range(1, args.m + 1))}, repeat={args.n - 1}):"
        )
        lines.append("        with conn_1:")
        lines.append("            if prev_conn_1:")
        lines.append(f"                temp: Bool = sample([1 - {args.p_edge[0]}, {args.p_edge[0]}])")
        lines.append("                yield temp")
        lines.append("            else:")
        lines.append(f"                temp: Bool = sample([1 - {args.p_edge[1]}, {args.p_edge[1]}])")
        lines.append("                yield temp")

        for j in range(2, args.m + 1):
            lines.append(f"        with conn_{j}:")
            lines.append(f"            if conn_{j - 1}:")
            lines.append("                with temp:")
            lines.append(f"                    if prev_conn_{j}:")
            lines.append(f"                        temp2: Bool = sample([1 - {args.p_inner[0]}, {args.p_inner[0]}])")
            lines.append("                        yield temp2")
            lines.append("                    else:")
            lines.append(f"                        temp2: Bool = sample([1 - {args.p_inner[1]}, {args.p_inner[1]}])")
            lines.append("                        yield temp2")
            lines.append("                yield temp")
            lines.append("            else:")
            lines.append("                with temp:")
            lines.append(f"                    if prev_conn_{j}:")
            lines.append(f"                        temp2: Bool = sample([1 - {args.p_inner[2]}, {args.p_inner[2]}])")
            lines.append("                        yield temp2")
            lines.append("                    else:")
            lines.append(f"                        temp2: Bool = sample([1 - {args.p_inner[3]}, {args.p_inner[3]}])")
            lines.append("                        yield temp2")
            lines.append("                yield temp")

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
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--m", type=int, required=True)
parser.add_argument("--p-init", type=float, nargs=1, required=True)
parser.add_argument("--p-edge", type=float, nargs=2, required=True)
parser.add_argument("--p-inner", type=float, nargs=4, required=True)
args = parser.parse_args()

print(create_program(args))
