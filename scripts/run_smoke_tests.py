from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "python")

    build_cmd = [
        sys.executable,
        "-c",
        (
            "from extra_boost_py.go_lib import build_shared, build_poisson_legacy_shared; "
            "build_shared(); build_poisson_legacy_shared()"
        ),
    ]
    run(build_cmd, env)

    run([sys.executable, "python/examples/full_pipeline.py"], env)
    run([sys.executable, "python/examples/poisson_legacy_quickcheck.py"], env)


if __name__ == "__main__":
    main()
