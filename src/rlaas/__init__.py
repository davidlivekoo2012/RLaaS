"""
RLaaS - Reinforcement Learning as a Service Platform

A modern, cloud-native platform for Reinforcement Learning as a Service,
specifically designed for multi-objective optimization scenarios.
"""

__version__ = "0.1.0"
__author__ = "RLaaS Team"
__email__ = "team@rlaas.ai"

from .core import *
from .training import *
from .inference import *
from .data import *

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
