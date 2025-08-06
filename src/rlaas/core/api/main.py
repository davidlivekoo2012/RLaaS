"""
Main FastAPI application for RLaaS API Gateway.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time
from typing import Dict, Any

from rlaas.config import get_config
from .routes import (
    optimization_router,
    training_router,
    inference_router,
    data_router,
    auth_router,
    health_router,
)
from .middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Create FastAPI application
app = FastAPI(
    title="RLaaS API Gateway",
    description="Reinforcement Learning as a Service Platform API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Setup Prometheus metrics
if config.environment == "production":
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)

# Include routers
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(optimization_router, prefix="/api/v1/optimization", tags=["Optimization"])
app.include_router(training_router, prefix="/api/v1/training", tags=["Training"])
app.include_router(inference_router, prefix="/api/v1/inference", tags=["Inference"])
app.include_router(data_router, prefix="/api/v1/data", tags=["Data"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"Global exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None),
        }
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting RLaaS API Gateway...")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Debug mode: {config.debug}")
    
    # Initialize database connections, caches, etc.
    # This will be implemented when we add the database layer


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down RLaaS API Gateway...")
    
    # Clean up resources
    # This will be implemented when we add the database layer


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "Welcome to RLaaS - Reinforcement Learning as a Service",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/info")
async def info() -> Dict[str, Any]:
    """Get API information."""
    return {
        "name": "RLaaS API Gateway",
        "version": "0.1.0",
        "environment": config.environment,
        "features": [
            "Multi-Objective Optimization",
            "Reinforcement Learning Training",
            "Model Serving",
            "A/B Testing",
            "Feature Store",
            "Real-time Inference",
        ],
        "supported_algorithms": [
            "NSGA-III",
            "MOEA/D", 
            "SAC",
            "PPO",
            "TOPSIS",
        ],
        "use_cases": [
            "5G Network Optimization",
            "Recommendation Systems",
            "Multi-objective Decision Making",
        ],
    }
