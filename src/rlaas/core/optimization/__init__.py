"""
Multi-Objective Optimization Engine for RLaaS.

This module provides multi-objective optimization capabilities including:
- NSGA-III algorithm implementation
- MOEA/D algorithm implementation
- Pareto frontier generation
- Conflict resolution using TOPSIS
- Dynamic weight adjustment
"""

from .engine import OptimizationEngine
from .algorithms import NSGAIIIAlgorithm, MOEADAlgorithm
from .conflict_resolver import ConflictResolver, TOPSISResolver
from .objectives import ObjectiveFunction, MultiObjectiveProblem
from .pareto import ParetoFrontier, ParetoSolution

__all__ = [
    "OptimizationEngine",
    "NSGAIIIAlgorithm",
    "MOEADAlgorithm", 
    "ConflictResolver",
    "TOPSISResolver",
    "ObjectiveFunction",
    "MultiObjectiveProblem",
    "ParetoFrontier",
    "ParetoSolution",
]
