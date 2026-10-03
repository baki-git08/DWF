"""pytest entry point for the golden-master check (see regression.py).  Run:  python -m pytest tests"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression  # noqa: E402


def test_matches_golden_master():
    diffs = regression.check()
    assert not diffs, "\n".join(diffs)
