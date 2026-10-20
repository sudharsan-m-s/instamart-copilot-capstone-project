"""Run all four parts from the project root, stopping at the first failed step."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["generate_data.py"],
    ["-m", "src.part_a.audit"],
    ["-m", "src.part_a.clean_data"],
    ["-m", "src.part_a.database"],
    ["-m", "src.part_a.validate"],
    ["-m", "src.part_a.queries"],
    ["-m", "src.part_a.analysis"],
    ["-m", "src.part_b.segmentation"],
    ["-m", "src.part_b.lapse_model"],
    ["-m", "src.part_b.evaluation"],
    ["-m", "src.part_c.privacy"],
    ["-m", "src.part_c.triage"],
    ["-m", "src.part_d.policy_assistant"],
    ["-m", "src.part_d.refund_agent"],
    ["-m", "unittest", "discover", "-s", "tests", "-v"],
]


def main():
    logs = ROOT / "outputs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    # Small deterministic workloads benefit from avoiding large BLAS thread pools.
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    for number, command in enumerate(STEPS, start=1):
        print(f"[{number}/{len(STEPS)}] python {' '.join(command)}", flush=True)
        result = subprocess.run(
            [sys.executable, *command], cwd=ROOT, env=environment,
            text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (logs / f"{number:02d}.txt").write_text(result.stdout, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    print("ALL FOUR PARTS REPRODUCED SUCCESSFULLY")


if __name__ == "__main__":
    main()
