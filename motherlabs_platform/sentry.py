"""Sentry integration — exception tracking for production.

No-op if SENTRY_DSN is not set. Filters InputQualityError (user error, not system).
"""

import os
import logging

logger = logging.getLogger("motherlabs.sentry")


def init_sentry() -> None:
    """Initialize Sentry SDK if SENTRY_DSN is configured."""
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        logger.info("SENTRY_DSN not set — Sentry disabled")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        def _before_send(event, hint):
            """Filter out user-caused errors."""
            if "exc_info" in hint:
                exc_type = hint["exc_info"][0]
                exc_name = getattr(exc_type, "__name__", "")
                # InputQualityError = user sent garbage input, not a system error
                if exc_name in ("InputQualityError", "ValidationError"):
                    return None
            return event

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("MOTHERLABS_ENV", "development"),
            traces_sample_rate=0.1,
            before_send=_before_send,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
            ],
        )
        logger.info("Sentry initialized")

    except ImportError:
        logger.warning("sentry-sdk not installed — Sentry disabled")
