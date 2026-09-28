#!/usr/bin/env python3
"""Run the reference demo, capture its runtime, and record the tested environment."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("$", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)


def main():
    report_lines = ["DeepTPACS validation report", "=" * 38, ""]

    verify = run([sys.executable, "tools/verify_release.py"])
    print(verify.stdout, end="")
    report_lines += ["Release verification:", verify.stdout.strip(), ""]

    env = run([sys.executable, "tools/collect_environment.py"])
    print(env.stdout, end="")

    demo_output = ROOT / "demo" / "demo_output.csv"
    cache = ROOT / "results" / "demo_graphs.bin"
    start = time.perf_counter()
    demo = run([
        sys.executable,
        "predict_csv.py",
        "--input", "demo/input_smiles.csv",
        "--output", str(demo_output.relative_to(ROOT)),
        "--cache", str(cache.relative_to(ROOT)),
        "--device", "auto",
    ])
    runtime = time.perf_counter() - start
    print(demo.stdout, end="")

    reference = ROOT / "demo" / "reference_output.csv"
    shutil.copyfile(demo_output, reference)
    report_lines += [
        "Demo command:",
        "python predict_csv.py --input demo/input_smiles.csv --output demo/demo_output.csv --device auto",
        "",
        f"Demo runtime: {runtime:.3f} s",
        f"Demo molecules: 3",
        f"Reference output: {reference.relative_to(ROOT)}",
        "",
        "Program output:",
        demo.stdout.strip(),
        "",
    ]

    report = ROOT / "environment" / "VALIDATION_REPORT.txt"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Validation report: {report}")
    print(f"Reference output: {reference}")


if __name__ == "__main__":
    main()
