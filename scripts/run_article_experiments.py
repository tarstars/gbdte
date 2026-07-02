#!/usr/bin/env python3
"""Run article experiment suites.

Usage:
    PYTHONPATH=python .venv/bin/python scripts/run_article_experiments.py --suite all --seeds 10
"""
import argparse
from pathlib import Path

from extra_boost_py.experiments.suites import SUITES, run_suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["all", *SUITES])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-scale run")
    ap.add_argument("--out", default="reports/article_experiments")
    args = ap.parse_args()

    names = list(SUITES) if args.suite == "all" else [args.suite]
    for name in names:
        out = run_suite(name, Path(args.out), seeds=args.seeds, quick=args.quick)
        print(f"[done] {name} -> {out}")


if __name__ == "__main__":
    main()
