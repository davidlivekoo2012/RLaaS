"""
Health check endpoints for RLaaS API Gateway.
"""

import time
import psutil
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import redis
import asyncpg
from rlaas.config import get_config

router = APIRouter()
config = get_config()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: float
    version: str
    environment: str
    uptime: float
    checks: Dict[str, Any]


class ComponentHealth(BaseModel):
    """Individual component health model."""
    status: str
    response_time: float
    details: Dict[str, Any] = {}


# Track application start time
start_time = time.time()


async def check_database() -> ComponentHealth:
    """Check database connectivity."""
    start = time.time()
    
    try:
        # Try to connect to PostgreSQL
        conn = await asyncpg.connect(config.database.url)
        await conn.execute("SELECT 1")
        await conn.close()
        
        response_time = time.time() - start
        return ComponentHealth(
            status="healthy",
            response_time=response_time,
            details={"type": "postgresql"}
        )
    
    except Exception as e:
        response_time = time.time() - start
        return ComponentHealth(
            status="unhealthy",
            response_time=response_time,
            details={"error": str(e), "type": "postgresql"}
        )


async def check_redis() -> ComponentHealth:
    """Check Redis connectivity."""
    start = time.time()
    
    try:
        redis_client = redis.from_url(config.redis.url)
        redis_client.ping()
        redis_client.close()
        
        response_time = time.time() - start
        return ComponentHealth(
            status="healthy",
            response_time=response_time,
            details={"type": "redis"}
        )
    
    except Exception as e:
        response_time = time.time() - start
        return ComponentHealth(
            status="unhealthy",
            response_time=response_time,
            details={"error": str(e), "type": "redis"}
        )


async def check_system_resources() -> ComponentHealth:
    """Check system resource usage."""
    start = time.time()
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        response_time = time.time() - start
        
        # Determine health based on resource usage
        status = "healthy"
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "degraded"
        if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
            status = "unhealthy"
        
        return ComponentHealth(
            status=status,
            response_time=response_time,
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_available": memory.available,
                "disk_free": disk.free
            }
        )
    
    except Exception as e:
        response_time = time.time() - start
        return ComponentHealth(
            status="unhealthy",
            response_time=response_time,
            details={"error": str(e)}
        )


@router.get("/", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Comprehensive health check endpoint.
    
    Returns the overall health status of the RLaaS platform including:
    - Application status
    - Database connectivity
    - Redis connectivity
    - System resource usage
    """
    current_time = time.time()
    uptime = current_time - start_time
    
    # Perform health checks
    checks = {}
    
    # Check database
    checks["database"] = await check_database()
    
    # Check Redis
    checks["redis"] = await check_redis()
    
    # Check system resources
    checks["system"] = await check_system_resources()
    
    # Determine overall status
    overall_status = "healthy"
    for check in checks.values():
        if check.status == "unhealthy":
            overall_status = "unhealthy"
            break
        elif check.status == "degraded":
            overall_status = "degraded"
    
    return HealthStatus(
        status=overall_status,
        timestamp=current_time,
        version="0.1.0",
        environment=config.environment,
        uptime=uptime,
        checks=checks
    )


@router.get("/liveness")
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe endpoint.
    
    Returns a simple status indicating the application is running.
    """
    return {"status": "alive"}


@router.get("/readiness")
async def readiness_probe() -> Dict[str, str]:
    """
    Kubernetes readiness probe endpoint.
    
    Returns status indicating the application is ready to serve traffic.
    """
    # Check critical dependencies
    try:
        # Quick database check
        db_health = await check_database()
        if db_health.status == "unhealthy":
            raise HTTPException(status_code=503, detail="Database not ready")
        
        return {"status": "ready"}
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")


@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """
    Basic metrics endpoint for monitoring.
    
    Returns application metrics for monitoring systems.
    """
    current_time = time.time()
    uptime = current_time - start_time
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "uptime_seconds": uptime,
        "cpu_usage_percent": cpu_percent,
        "memory_usage_percent": memory.percent,
        "memory_used_bytes": memory.used,
        "memory_total_bytes": memory.total,
        "timestamp": current_time,
    }
