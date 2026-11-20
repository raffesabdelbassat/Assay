"""Scorer plugin interface and built-in scorers."""

from assay.scorers.base import Scorer
from assay.scorers.builtins import build_scorers

__all__ = ["Scorer", "build_scorers"]
