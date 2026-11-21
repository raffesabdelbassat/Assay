"""Scorer plugin interface and built-in scorers."""

from assay.scorers.base import Scorer
from assay.scorers.builtins import build_scorers
from assay.scorers.judge import JudgeScorer

__all__ = ["Scorer", "JudgeScorer", "build_scorers"]
