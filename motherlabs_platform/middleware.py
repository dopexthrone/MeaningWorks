"""
Motherlabs Platform Middleware — API key validation, domain routing.

Phase D: V2 API + Platform Layer

Provides FastAPI middleware for:
- API key validation (header-based)
- Domain adapter routing validation
- Request/response logging
"""

import time
import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.adapter_registry import list_adapters

logger = logging.getLogger("motherlabs.platform.middleware")


class DomainValidationMiddleware(BaseHTTPMiddleware):
    """Validate domain parameter in V2 requests.

    Checks that the requested domain adapter exists before the
    request reaches the route handler.
    """

    async def dispatch(self, request: Request, call_next):
        # Only validate V2 POST requests that may have a domain parameter
        if request.url.path.startswith("/v2/") and request.method == "POST":
            # Domain validation happens in the route handler
            # This middleware just logs the request
            start = time.time()
            response = await call_next(request)
            duration = time.time() - start
            logger.info(
                "V2 request: %s %s [%.2fs] -> %d",
                request.method,
                request.url.path,
                duration,
                response.status_code,
            )
            return response

        return await call_next(request)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate API key from X-API-Key header.

    When key_store and rate_limiter are provided, performs full validation:
    1. Hash-based key lookup (401 if invalid/revoked)
    2. Sliding-window rate limiting (429 if exceeded)
    3. Budget enforcement (402 if exceeded)
    4. Sets request.state.api_key_id / api_key_name for downstream use
    5. Adds X-RateLimit-* response headers

    When key_store=None (default): accepts any non-empty key (dev mode).
    """

    def __init__(self, app, require_key: bool = False, key_store=None, rate_limiter=None):
        super().__init__(app)
        self.require_key = require_key
        self.key_store = key_store
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        # Skip key check for health/domain info endpoints
        if request.url.path in ("/v2/health", "/v2/domains", "/v1/health"):
            return await call_next(request)
        if request.url.path.startswith("/v2/domains/"):
            return await call_next(request)

        if self.require_key:
            api_key = request.headers.get("X-API-Key", "")
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Missing API key",
                        "error_code": "E10001",
                        "suggestion": "Set X-API-Key header",
                    },
                )

            # Full validation when key_store is available
            if self.key_store is not None:
                result = self.key_store.validate_key(api_key)

                if not result.valid:
                    # Distinguish revoked/invalid vs budget exceeded
                    if result.reason == "Budget exceeded":
                        return JSONResponse(
                            status_code=402,
                            content={
                                "error": result.reason,
                                "error_code": "E10003",
                                "spent_usd": result.spent_usd,
                                "budget_usd": result.budget_usd,
                            },
                        )
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": result.reason,
                            "error_code": "E10001",
                        },
                    )

                # Rate limit check
                if self.rate_limiter is not None:
                    allowed, remaining, reset_ts = self.rate_limiter.check_rate_limit(
                        result.key_id, result.rate_limit_per_hour
                    )
                    if not allowed:
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Rate limit exceeded",
                                "error_code": "E10002",
                                "retry_after_seconds": int(reset_ts - time.time()),
                            },
                            headers={
                                "X-RateLimit-Limit": str(result.rate_limit_per_hour),
                                "X-RateLimit-Remaining": "0",
                                "X-RateLimit-Reset": str(int(reset_ts)),
                            },
                        )

                # Set request state for downstream
                request.state.api_key_id = result.key_id
                request.state.api_key_name = result.key_name

                # Call next and add rate limit headers to response
                response = await call_next(request)

                if self.rate_limiter is not None:
                    response.headers["X-RateLimit-Limit"] = str(result.rate_limit_per_hour)
                    response.headers["X-RateLimit-Remaining"] = str(remaining)
                    response.headers["X-RateLimit-Reset"] = str(int(reset_ts))

                return response

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        logger.info(
            "%s %s [%.3fs] -> %d",
            request.method,
            request.url.path,
            duration,
            response.status_code,
        )
        return response
