"""
API Gateway module for RLaaS platform.

This module provides the main API gateway functionality including:
- FastAPI application setup
- Route management
- Authentication and authorization
- Request/response middleware
- API documentation
"""

from .main import app
from .routes import *
from .middleware import *
from .auth import *

__all__ = [
    "app",
    "routes",
    "middleware", 
    "auth",
]
