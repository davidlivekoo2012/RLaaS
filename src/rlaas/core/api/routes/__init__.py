"""
API routes for RLaaS platform.
"""

from .health import router as health_router
from .auth import router as auth_router
from .optimization import router as optimization_router
from .training import router as training_router
from .inference import router as inference_router
from .data import router as data_router

__all__ = [
    "health_router",
    "auth_router", 
    "optimization_router",
    "training_router",
    "inference_router",
    "data_router",
]
