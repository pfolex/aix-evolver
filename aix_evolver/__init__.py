"""
AIX-Evolver: Elastic Reasoning Self-Improvement Engine
Created by Alex Pfeil (@pfolex)
"""

from aix_evolver.__version__ import __version__
from aix_evolver.dataset import Problem, ProblemManager
from aix_evolver.generator import RolloutGenerator, RolloutResult
from aix_evolver.verifier import ReasoningVerifier, VerificationScore
from aix_evolver.elastic_brain import ElasticBrain, ExpansionEvent
from aix_evolver.trainer import ElasticTrainer

__all__ = [
    "__version__",
    "Problem",
    "ProblemManager",
    "RolloutGenerator",
    "RolloutResult",
    "ReasoningVerifier",
    "VerificationScore",
    "ElasticBrain",
    "ExpansionEvent",
    "ElasticTrainer",
]
