"""Data locations for the analysis scripts.

Result files live in 02-data/results/ (one folder per analysis, plus the preserved generations in interventions/) and
the evaluation inputs in 02-data/inputs/. Set RESULTS_DIR to read a copy laid out the same way.
"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(CODE, "..", ".."))
RESULTS = os.path.abspath(os.environ.get("RESULTS_DIR") or os.path.join(REPO, "02-data", "results"))
DATA = os.path.dirname(RESULTS)


def result_path(*parts):
    return os.path.join(RESULTS, *parts)
