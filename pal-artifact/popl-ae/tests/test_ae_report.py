"""`make report` and `make check` work on partial results: a run that was cut short, or that
ran only some experiments, still gives a report.pdf."""

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

AE = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk is not on PATH")

COLUMNS = ("suite,case_id,config,L,nodes,network,status,error_message,trials,compile_s,compile_std,"
           "run_first_s,run_first_std,hot_mean_s,hot_std,report_cold_s,report_cold_std,report_warm_s,"
           "report_warm_std,process_s,process_std,static_peak_bytes,peak_gpu_bytes,peak_rss_bytes,"
           "answer,answer_mass,dtype,device").split(",")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make(target: str, results: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["make", target, f"RESULTS={results}"], cwd=AE, capture_output=True, text=True)


def test_empty_results(tmp_path):
    assert make("report", tmp_path).returncode == 0
    assert (tmp_path / "report" / "report.pdf").stat().st_size > 0
    check = make("check", tmp_path)
    assert check.returncode == 0 and "nothing to compare" in check.stdout


def test_failed_cases_and_header_only_files(tmp_path):
    base = {"suite": "bayes", "trials": 1}
    write_rows(tmp_path / "bayes" / "pal-numpy.csv", [
        {**base, "case_id": "cancer", "config": "pal-numpy", "network": "cancer", "status": "ok",
         "compile_s": 0.001, "run_first_s": 0.0002, "report_cold_s": 0.0012, "peak_rss_bytes": 40960,
         "answer": "0.2 0.8"},
        {**base, "case_id": "survey", "config": "pal-numpy", "network": "survey", "status": "gen_failed",
         "error_message": "exit -11"},
    ])
    write_rows(tmp_path / "bayes" / "dice.csv", [
        {**base, "case_id": "cancer", "config": "dice", "network": "cancer", "status": "ok",
         "report_cold_s": 0.03, "answer": "0.2 0.8"},
        {**base, "case_id": "survey", "config": "dice", "network": "survey", "status": "oom",
         "error_message": "exit -6: memory allocation of 786432 bytes failed"},
    ])
    write_rows(tmp_path / "caesar" / "dice.csv", [])
    for name in ("pal_loss", "dice_loss"):
        (tmp_path / "natlist").mkdir(exist_ok=True)
        (tmp_path / "natlist" / f"{name}.csv").write_text("epoch,loss,time\n")

    report = make("report", tmp_path)
    assert report.returncode == 0, report.stdout[-2000:] + report.stderr[-2000:]
    assert (tmp_path / "report" / "report.pdf").stat().st_size > 0
    check = make("check", tmp_path)
    assert check.returncode == 0 and "1 cases run by two or more configurations; all agree" in check.stdout
