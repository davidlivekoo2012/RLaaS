"""
Middleware components for the RLaaS API Gateway.
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis
from rlaas.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request started - ID: {request_id}, "
            f"Method: {request.method}, "
            f"URL: {request.url}, "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Request completed - ID: {request_id}, "
            f"Status: {response.status_code}, "
            f"Duration: {process_time:.3f}s"
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.redis_client = None
        
        try:
            self.redis_client = redis.from_url(config.redis.url)
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting: {e}")
    
    async def dispatch(self, request: Request, call_next):
        if not self.redis_client:
            # Skip rate limiting if Redis is not available
            return await call_next(request)
        
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        client_key = f"rate_limit:{client_ip}"
        
        try:
            # Check current request count
            current_requests = self.redis_client.get(client_key)
            
            if current_requests is None:
                # First request from this client
                self.redis_client.setex(client_key, 60, 1)
            else:
                current_count = int(current_requests)
                if current_count >= self.requests_per_minute:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                            "retry_after": 60,
                        },
                        headers={"Retry-After": "60"}
                    )
                else:
                    self.redis_client.incr(client_key)
        
        except Exception as e:
            logger.warning(f"Rate limiting error: {e}")
            # Continue without rate limiting if Redis fails
        
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'"
        )
        
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for handling authentication."""
    
    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/auth/login",
            "/auth/register",
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Check for authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication required",
                    "message": "Please provide a valid authentication token",
                }
            )
        
        # Extract and validate token
        token = auth_header.split(" ")[1]
        
        # TODO: Implement token validation
        # For now, we'll just check if token is not empty
        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid token",
                    "message": "Authentication token is invalid",
                }
            )
        
        # Add user information to request state
        # TODO: Extract user info from validated token
        request.state.user_id = "user_123"  # Placeholder
        request.state.user_roles = ["user"]  # Placeholder
        
        return await call_next(request)


class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for caching responses."""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.redis_client = None
        
        try:
            self.redis_client = redis.from_url(config.redis.url)
        except Exception as e:
            logger.warning(f"Redis not available for caching: {e}")
    
    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests
        if request.method != "GET" or not self.redis_client:
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"cache:{request.url.path}:{request.url.query}"
        
        try:
            # Check if response is cached
            cached_response = self.redis_client.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {cache_key}")
                return Response(
                    content=cached_response,
                    media_type="application/json",
                    headers={"X-Cache": "HIT"}
                )
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            try:
                # Read response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                # Cache the response
                self.redis_client.setex(cache_key, self.cache_ttl, body)
                
                # Return response with cache headers
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers, **{"X-Cache": "MISS"}),
                    media_type=response.media_type
                )
            
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        return response
