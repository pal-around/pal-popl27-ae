"""Turn a results directory into the paper's figures and table, typeset with LaTeX.

    python report/make_report.py RESULTS

Writes RESULTS/report/: the measurements in the CSV formats the paper's figure sources
read (data/), each figure drawn from them, the Bayesian-network table, and report.tex,
which includes them all. `make report` then runs latexmk on it.
"""

import csv
import math
import re
import statistics
import sys
from pathlib import Path

AE_ROOT = Path(__file__).resolve().parent.parent
PAPER_SRC = AE_ROOT / "report" / "paper"
MIB = 2**20
TIMES = "\\times"
NETWORKS = ["cancer", "survey", "alarm", "insurance", "hepar2", "pigs", "water", "munin", "andes", "link",
            "diabetes", "hailfinder"]


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------

def read(results: Path, suite: str, config: str) -> list[dict]:
    path = results / suite / f"{config}.csv"
    return list(csv.DictReader(path.open())) if path.exists() else []


def ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["status"] == "ok"]


def num(row: dict | None, key: str) -> float | None:
    if row is None or row.get(key) in (None, ""):
        return None
    return float(row[key])


def by_length(rows: list[dict]) -> dict[int, dict]:
    return {int(r["L"]): r for r in rows}


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(["" if v is None else (f"{v:.10g}" if isinstance(v, float) else v) for v in row])


def peak_mib(row: dict) -> float | None:
    """The measured peak: GPU memory when the run used a GPU, else resident memory."""
    value = num(row, "peak_gpu_bytes")
    if value is None:
        value = num(row, "peak_rss_bytes")
    return None if value is None else value / MIB


# ---------------------------------------------------------------------------
# The paper's plot CSVs, from the measurements
# ---------------------------------------------------------------------------

MARKOV_FILES = {"caesar": "caesar", "diamond": "diamond", "ladder": "ladder", "dicefig1": "dicefig1"}


def markov_data(results: Path, data: Path) -> dict:
    limits = {}
    for suite in MARKOV_FILES:
        dice = ok(read(results, suite, "dice"))
        write_csv(data / f"figures/{suite}/{suite}-dice.csv", ["length", "mean"],
                  [[int(r["L"]), num(r, "report_cold_s")] for r in dice])
        xla = by_length(ok(read(results, suite, "pal-xla")))
        nm = by_length(ok(read(results, suite, "pal-xla-nomarg")))
        lengths = sorted(set(xla) | set(nm))
        rows = [[L, num(xla.get(L), "report_cold_s"), num(nm.get(L), "report_cold_s"),
                 num(xla.get(L), "report_warm_s")] for L in lengths]
        name = "markov-caesar-ablation-plot.csv" if suite == "caesar" else f"markov-{suite}-plot.csv"
        write_csv(data / "popl2027-data" / name, ["length", "xlareport", "xlanmreport", "xlawarmreport"], rows)
        times = [v for r in rows for v in (r[1], r[3]) if v is not None]
        limits[f"{suite}-time"] = max(times) * 1.1 if times else 1.0
        mem = [[L, peak_mib(xla[L])] for L in sorted(xla)]
        write_csv(data / f"popl2027-data/markov-{suite}-xla-mem-plot.csv", ["length", "peakmemMB"], mem)
        values = [v for _, v in mem if v]
        if suite == "caesar":
            mem_nm = [[L, peak_mib(nm[L])] for L in sorted(nm)]
            write_csv(data / "popl2027-data/markov-caesar-xla-mem-nm-plot.csv", ["length", "peaknmmemMB"], mem_nm)
            values += [v for _, v in mem_nm if v]
        # The log axis spans the whole decades around the data and is labelled at its two
        # ends, so its labels are powers of ten that do not crowd a short axis.
        lo = 10 ** math.floor(math.log10(min(values))) if values else 0.01
        hi = 10 ** math.ceil(math.log10(max(values))) if values else 1.0
        limits[f"{suite}-mem-min"], limits[f"{suite}-mem"] = lo, hi if hi > lo else lo * 10
    memo = ok(read(results, "dicefig1", "memo"))
    write_csv(data / "figures/dicefig1/dicefig1-memo.csv", ["length", "run1mean"],
              [[int(r["L"]), num(r, "report_cold_s")] for r in memo])
    return limits


GRID_FILES = {"grid-link": ("link_failure_3flow", "link3flow"), "grid-router": ("router_failure", "router")}


def grid_data(results: Path, data: Path) -> None:
    for suite, (dice_name, name) in GRID_FILES.items():
        dice = ok(read(results, suite, "dice"))
        write_csv(data / f"figures/grid_networks/{dice_name}-dice.csv", ["length", "mean"],
                  [[int(r["L"]), num(r, "report_cold_s")] for r in dice])
        xla_all = by_length(read(results, suite, "pal-xla"))
        nm_all = by_length(read(results, suite, "pal-xla-nomarg"))
        xla, nm = by_length(ok(list(xla_all.values()))), by_length(ok(list(nm_all.values())))
        lengths = sorted(set(xla) | set(nm))
        write_csv(data / f"popl2027-data/grid-{name}-xla-plot.csv", ["length", "report", "reportnm"],
                  [[L, num(xla.get(L), "report_cold_s"), num(nm.get(L), "report_cold_s")] for L in lengths])
        numpy = ok(read(results, suite, "pal-numpy"))
        write_csv(data / f"popl2027-data/grid-{name}-numpy-plot.csv", ["length", "report"],
                  [[int(r["L"]), num(r, "report_cold_s")] for r in numpy])
        # The static estimates exist for every size the compiler reached, run or not.
        static = sorted(set(xla_all) | set(nm_all))

        def static_mib(row):
            v = num(row, "static_peak_bytes")
            return None if v is None else v / MIB
        write_csv(data / f"popl2027-data/grid-{name}-njoint-plot.csv", ["length", "peakmemMB", "peaknmMB"],
                  [[L, static_mib(xla_all.get(L)), static_mib(nm_all.get(L))] for L in static])
        write_csv(data / f"popl2027-data/grid-{name}-xla-mem-plot.csv", ["length", "peakmemMB", "peaknmMB"],
                  [[L, peak_mib(xla[L]) if L in xla else None, peak_mib(nm[L]) if L in nm else None]
                   for L in lengths])


def has_rows(path: Path) -> bool:
    """Whether a CSV file exists and has a row below its header."""
    return path.exists() and len(path.read_text().strip().splitlines()) > 1


def natlist_data(results: Path, data: Path) -> None:
    for name in ("pal_loss", "dice_loss"):
        src = results / "natlist" / f"{name}.csv"
        dst = data / f"figures/natlist/{name}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text() if src.exists() else "epoch,loss,time\n")


# ---------------------------------------------------------------------------
# Figures: the paper's sources, pointed at a data directory
# ---------------------------------------------------------------------------

def figure(name: str, data_dir: Path, limits: dict | None = None) -> str:
    tex = (PAPER_SRC / name).read_text()
    tex = tex.replace("\\newcommand\\MkSubfigLabel", "\\providecommand\\MkSubfigLabel")
    tex = re.sub(r"\\detokenize\{((?:figures|popl2027-data)/[^}]*)\}",
                 lambda m: "\\detokenize{" + (data_dir / m.group(1)).as_posix() + "}", tex)
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)
    tex = re.sub(r"~?\\cite\{[^}]*\}", "", tex)
    tex = tex.replace("\\citet{holtzen2020dice}", "Holtzen et al.\\ (2020)")
    # An axis with no data is cleared, and the coordinates named inside it with it; the
    # labels placed at them then need a fallback, defined before any axis.
    names = sorted(set(re.findall(r"\\coordinate \(([^)]*)\)", tex)))
    fallback = "".join(f"\\coordinate ({n}) at (0,0);" for n in names)
    tex = re.sub(r"(\\begin\{tikzpicture\}(?:\[[^\]]*\])?)", lambda m: m.group(1) + fallback, tex, count=1)
    tex = tex.replace("\\begin{figure}\n", "\\begin{figure}[h!]\n", 1)
    if limits is not None:
        tex = legend_images(tex)
    if name == "eval-natlist.tex":
        tex = tex.replace("\\begin{wrapfigure}{r}{0.30\\columnwidth}",
                          "\\begin{figure}[h]\\centering\\begin{minipage}{0.4\\linewidth}")
        tex = tex.replace("\\end{wrapfigure}", "\\end{minipage}\n\\caption{Tuning a random list generator "
                          "(N = 100): KL divergence over time.}\\end{figure}")
    if limits and name == "eval-dice.tex":
        # The second and third rows' ranges follow the data plotted in them.
        tex = tex.replace("ymax = 0.68,", f"ymax = {limits['ladder-time']:.4g},")
        tex = tex.replace("ymax = 0.50,", f"ymax = {limits['dicefig1-time']:.4g},")
        for suite in ("caesar", "diamond", "ladder", "dicefig1"):
            tex = re.sub(r"ymin=0(?:\.0)?,(\s*)ymax=(?:40|0\.05),",
                         lambda m: f"ymin={limits[suite + '-mem-min']:.4g},{m.group(1)}"
                                   f"ymax={limits[suite + '-mem']:.4g},{m.group(1)}"
                                   f"ytick={{{limits[suite + '-mem-min']:.4g},{limits[suite + '-mem']:.4g}}},",
                         tex, count=1)
    return tex


_ADDPLOT = re.compile(r"\\addplot\[([^\]]*)\]\s*table\s*\[([^\]]*)\]\s*\{[^;]*\};")
_DOMAIN = re.compile(r",?\s*restrict expr to domain=\{x\}\{([0-9.e+]+):[^}]*\}")


def legend_images(tex: str) -> str:
    """Some plots in the paper's sources draw nothing visible and exist for their legend
    entry: they keep only sizes from a threshold up (``restrict expr to domain={x}{A:B}``
    with A > 0). Such a plot becomes a legend image in its style, so the legend stays
    aligned when a run has no data above the threshold."""
    def replace(m: re.Match) -> str:
        opts, table_opts = m.group(1), m.group(2)
        dom = _DOMAIN.search(opts) or _DOMAIN.search(table_opts)
        if not dom or float(dom.group(1)) <= 0:
            return m.group(0)
        return "\\addlegendimage{" + _DOMAIN.sub("", opts).strip(", ") + "}"
    return _ADDPLOT.sub(replace, tex)


# ---------------------------------------------------------------------------
# The Bayesian-network table
# ---------------------------------------------------------------------------

def fmt_time(v: float | None) -> str:
    if v is None:
        return "{--}"
    return f"{v:.3g}" if v < 1000 else f"{v:.0f}"


def fmt_mem(b: float | None) -> str:
    if b is None:
        return "{--}"
    if b >= MIB:
        return f"{b / MIB:.1f}\\,MB"
    return f"{max(1, round(b / 1024))}\\,KB"


def status_text(row: dict) -> str:
    """A run's status as table text (``gen_failed`` -> ``gen failed``)."""
    return row["status"].replace("_", " ")


def bn_table(results: Path) -> str:
    pal = {r["network"]: r for r in read(results, "bayes", "pal-numpy")}
    nm = {r["network"]: r for r in read(results, "bayes", "pal-numpy-nomarg")}
    dice = {r["network"]: r for r in read(results, "bayes", "dice")}
    timeout = None
    log = results / "bayes" / "dice.log"
    if log.exists() and (m := re.search(r"--timeout (\d+)", log.read_text())):
        timeout = int(m.group(1))
    lines, speedups = [], []
    for net in NETWORKS:
        if net not in pal:
            continue
        p = pal[net]
        comp, inf = num(p, "compile_s"), num(p, "run_first_s")
        if p["status"] != "ok":
            cells = f"\\multicolumn{{3}}{{c|}}{{{status_text(p)}}}"
        else:
            cells = f"{fmt_time(comp)} & {fmt_time(inf)} & {fmt_mem(num(p, 'peak_rss_bytes'))}"
        n = nm.get(net)
        if n is None:
            nm_cells = "{--} & {--}"
        elif n["status"] in ("infeasible", "oom"):
            nm_cells = "\\multicolumn{2}{c|}{\\OOM}"
        elif n["status"] != "ok":
            nm_cells = f"\\multicolumn{{2}}{{c|}}{{{status_text(n)}}}"
        else:
            nm_cells = f"{fmt_time(num(n, 'run_first_s'))} & {fmt_mem(num(n, 'peak_rss_bytes'))}"
        d = dice.get(net)
        if d is not None and d["status"] == "ok" and comp is not None and inf is not None:
            dice_s = num(d, "report_cold_s")
            speedup = dice_s / (comp + inf)
            speedups.append(speedup)
            dice_cell, speed_cell = fmt_time(dice_s), f"{speedup:.1f}$\\times$"
        else:
            status = d["status"] if d is not None else None
            dice_cell = {"timeout": "{\\TO}", "oom": "{\\TO\\,(OOM)}"}.get(status, "{--}")
            speed_cell = "{--}"
        lines.append(f"{net.capitalize():<11}& {cells} & {nm_cells} & {dice_cell} & {speed_cell} \\\\")
    geo = math.exp(statistics.mean(math.log(s) for s in speedups)) if speedups else None
    to_text = f"{timeout}\\,s" if timeout else "the timeout"
    return rf"""
\begin{{table}}[h]
\fontsize{{8.5}}{{9.0}}\selectfont
\newcommand{{\TO}}{{\faTimes\xspace}}
\newcommand{{\OOM}}{{\scalebox{{.8}}{{\faCircle}}\xspace}}
\caption{{Performance of \PLang and Dice on Bayesian networks. For \PLang/NumPy,
compilation (\emph{{comp.}}) and inference (\emph{{inf.}}) time, and \emph{{peak mem.}}, the
measured growth of resident memory while the model is compiled and run; the \emph{{no marg.}}
columns give inference time and peak memory with eager marginalization disabled. \TO: did not finish within {to_text}. \TO\,(OOM): ran out of
memory. \OOM: without eager marginalization, the static estimate of the largest array
exceeds the memory limit (or the run ran out of memory). Speedup is Dice over total \PLang/NumPy time.
Geometric-mean speedup over the {len(speedups)} networks Dice finishes:
{'--' if geo is None else f'{geo:.1f}$' + TIMES + '$'}.}}
\centering
\begin{{adjustbox}}{{max width=\linewidth}}
\begin{{tabular}}{{l|rrr|rr|rr}}
\toprule
\multirow{{2}}{{*}}{{benchmark}} & \multicolumn{{3}}{{c|}}{{\PLang/NumPy}} & \multicolumn{{2}}{{c|}}{{no marg.}}
 & \multirow{{2}}{{*}}{{Dice (s)}} & \multirow{{2}}{{*}}{{speedup}} \\
 & comp.\ (s) & inf.\ (s) & peak mem. & inf.\ (s) & peak mem. & & \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# report.tex
# ---------------------------------------------------------------------------

NOT_RUN = "\\par\\medskip\\noindent\\emph{This experiment has no results in this run.}\\par\n"


def tex_escape(text: str) -> str:
    """Text with LaTeX's special characters escaped."""
    special = {"\\": r"\textbackslash{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    special.update({c: "\\" + c for c in "&%$#_{}"})
    return "".join(special.get(c, c) for c in text)


def devices(results: Path) -> str:
    seen = set()
    for csv_path in results.glob("*/*.csv"):
        for r in csv.DictReader(csv_path.open()):
            if r.get("device") and r["config"] != "dice":
                seen.add(r["device"])
    return ", ".join(sorted(seen)) or "--"


def main() -> None:
    results = Path(sys.argv[1]).resolve()
    out = results / "report"
    data = out / "data"
    limits = markov_data(results, data)
    grid_data(results, data)
    natlist_data(results, data)

    ran = {
        "eval-dice.tex": any((results / s).is_dir() for s in MARKOV_FILES),
        "eval-grid-networks.tex": any((results / s).is_dir() for s in GRID_FILES),
        "eval-natlist.tex": all(has_rows(results / "natlist" / f"{n}.csv") for n in ("pal_loss", "dice_loss")),
    }
    parts = {}
    for name in ("eval-dice.tex", "eval-grid-networks.tex", "eval-natlist.tex"):
        stem = name.removesuffix(".tex")
        parts[stem] = figure(name, data, limits) if ran[name] else NOT_RUN
    parts["eval-bn"] = bn_table(results) if (results / "bayes").is_dir() else NOT_RUN
    for stem, tex in parts.items():
        (out / f"{stem}.tex").write_text(tex)

    (out / "report.tex").write_text(rf"""\documentclass[10pt]{{article}}
\input{{{(AE_ROOT / 'report' / 'preamble.tex').as_posix()}}}
\begin{{document}}
\begin{{center}}\Large Reproduced evaluation\end{{center}}
\noindent Results: \texttt{{{tex_escape(results.name)}}}. Devices used by PAL and memo: {tex_escape(devices(results))}.
In the figures, the memory row shows the measured GPU peak for runs on a GPU
and the measured growth of resident memory for runs on the CPU.

\section{{Markov models}}
\input{{eval-dice}}
\clearpage
\section{{Bayesian networks}}
\input{{eval-bn}}
\clearpage
\section{{Network reliability}}
\input{{eval-grid-networks}}
\clearpage
\section{{Tuning a random generator}}
\input{{eval-natlist}}
\end{{document}}
""")
    print(f"wrote {out / 'report.tex'}")


if __name__ == "__main__":
    main()
