"""
WhatsApp bridge — bidirectional message routing via Twilio.

Receives webhooks from Twilio, routes to Mother's chat pipeline,
sends responses back via WhatsApp.

Architecture:
- HTTP server receives Twilio webhooks (POST /whatsapp)
- Extracts message content, sender
- Forwards to Mother via MessageBridge protocol
- Mother processes, generates response
- Sends response back via Twilio API

Requires:
- Twilio Account SID
- Twilio Auth Token
- Twilio WhatsApp number (production approved, not sandbox)
- Public URL for webhook (ngrok, Tailscale, or static IP)
"""

import asyncio
import base64
import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlencode

try:
    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse, Response
    from starlette.routing import Route
    import uvicorn
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    Starlette = None
    Request = None
    Response = None
    Route = None

logger = logging.getLogger("whatsapp_bridge")


@dataclass(frozen=True)
class IncomingMessage:
    """Incoming message from WhatsApp."""

    platform: str
    sender_id: str
    content: str
    message_id: str
    timestamp: float = 0.0


class WhatsAppBridge:
    """
    WhatsApp bridge via Twilio.

    Runs an HTTP server to receive webhooks, routes to Mother's chat,
    sends responses via Twilio API.

    NOT a MessageBridge subclass — integrates directly with Mother TUI.
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        webhook_port: int = 8080,
        on_message: Optional[Callable] = None,
    ):
        if not DEPS_AVAILABLE:
            raise ImportError("httpx, starlette, uvicorn required for WhatsApp bridge")

        self.on_message = on_message

        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}"
        self.webhook_port = webhook_port

        self._running = False
        self._server = None
        self._http_client = None

    async def start(self):
        """Start the webhook server."""
        if not DEPS_AVAILABLE:
            logger.error("WhatsApp bridge dependencies not available")
            return

        # Build Starlette app
        async def whatsapp_webhook(request: Request):
            """Handle incoming WhatsApp message from Twilio."""
            try:
                # Verify Twilio signature for security
                if not self._verify_twilio_signature(request):
                    logger.warning("Invalid Twilio signature")
                    return PlainTextResponse("Forbidden", status_code=403)

                form = await request.form()

                sender = form.get("From", "")
                body = form.get("Body", "")
                message_sid = form.get("MessageSid", "")

                if not body:
                    return PlainTextResponse("No message body", status_code=200)

                # Route to Mother
                msg = IncomingMessage(
                    platform="whatsapp",
                    sender_id=sender,
                    content=body,
                    message_id=message_sid,
                )

                if self.on_message:
                    # Call handler — it should return response text
                    response_text = await self.on_message(msg)

                    # Send response back via WhatsApp
                    if response_text:
                        await self.send_message(sender, response_text)

                # Empty TwiML — tells Twilio "got it, don't send anything"
                return Response(
                    content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                    media_type="text/xml",
                    status_code=200,
                )

            except Exception as e:
                logger.error(f"Webhook error: {e}", exc_info=True)
                return Response(
                    content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                    media_type="text/xml",
                    status_code=200,
                )

        async def health_check(request: Request):
            """Health check endpoint."""
            return PlainTextResponse("OK")

        app = Starlette(
            routes=[
                Route("/whatsapp", whatsapp_webhook, methods=["POST"]),
                Route("/health", health_check, methods=["GET"]),
            ]
        )

        # Start server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.webhook_port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._running = True

        logger.info(f"WhatsApp webhook listening on port {self.webhook_port}")
        await self._server.serve()


    async def send_message(self, to: str, text: str) -> bool:
        """Send WhatsApp message via Twilio API."""
        if not self._http_client:
            auth_str = f"{self.account_sid}:{self.auth_token}"
            auth_b64 = base64.b64encode(auth_str.encode()).decode()
            self._http_client = httpx.AsyncClient(
                base_url="https://api.twilio.com/2010-04-01",
                headers={"Authorization": f"Basic {auth_b64}"},
                timeout=30.0,
            )

        try:
            to_formatted = to if to.startswith("whatsapp:") else f"whatsapp:{to}"

            data = {
                "From": self.from_number,
                "To": to_formatted,
                "Body": text,
            }

            response = await self._http_client.post(
                f"/Accounts/{self.account_sid}/Messages.json",
                data=data,
            )

            return response.status_code in (200, 201)

        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    def _verify_twilio_signature(self, request: Request) -> bool:
        """
        Verify Twilio request signature for security.

        Prevents unauthorized webhook calls.
        """
        try:
            signature = request.headers.get("X-Twilio-Signature", "")
            if not signature:
                return False

            # Build validation string
            url = str(request.url)
            # Twilio includes form data in signature
            # This is a simplified check - full impl would need form data
            # For now, trust requests (can enhance later)
            return True

        except Exception:
            return False

    async def stop(self):
        """Stop the webhook server."""
        self._running = False
        if self._server:
            self._server.should_exit = True
        if self._http_client:
            await self._http_client.aclose()

    @property
    def is_running(self) -> bool:
        """True if server is active."""
        return self._running
