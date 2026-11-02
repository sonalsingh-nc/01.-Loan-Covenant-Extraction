"""Thin wrapper so `python evals/run_eval.py ...` keeps working without
`pip install -e .`. All logic lives in covenant_extraction.evals.cli; run
with --help for usage, or use the installed `covenant-eval` command."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from covenant_extraction.evals.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
