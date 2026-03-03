"""
WhatsApp webhook server — runs alongside Mother TUI.

Receives messages from Twilio webhook, queues them for Mother to process,
sends responses back via Twilio API.

Architecture:
- HTTP server runs in background thread
- Incoming messages go into asyncio.Queue
- Mother's chat screen polls queue
- Responses sent back via WhatsApp

This is simpler than modifying chat.py's complex worker system.
"""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    import uvicorn
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("mother.whatsapp_server")


@dataclass(frozen=True)
class WhatsAppMessage:
    """Incoming WhatsApp message."""

    sender: str  # whatsapp:+1234567890
    body: str
    message_id: str
    timestamp: float


class WhatsAppServer:
    """
    HTTP webhook server for WhatsApp.

    Runs uvicorn server in background, queues incoming messages,
    Mother polls queue and responds.
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        port: int = 8080,
    ):
        if not DEPS_AVAILABLE:
            raise ImportError("httpx, starlette, uvicorn required")

        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}"
        self.port = port

        self.incoming_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._server = None
        self._http_client = None
        self._running = False

    async def start(self):
        """Start the webhook server."""
        async def webhook_handler(request: Request):
            """Handle incoming WhatsApp message."""
            try:
                form = await request.form()

                sender = form.get("From", "")
                body = form.get("Body", "")
                message_sid = form.get("MessageSid", "")

                if not body:
                    return PlainTextResponse("No body", status_code=200)

                # Queue for Mother to process
                msg = WhatsAppMessage(
                    sender=sender,
                    body=body,
                    message_id=message_sid,
                    timestamp=time.time(),
                )

                try:
                    self.incoming_queue.put_nowait(msg)
                except asyncio.QueueFull:
                    logger.warning("Message queue full, dropping message")

                return PlainTextResponse("OK", status_code=200)

            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return PlainTextResponse("Error", status_code=500)

        app = Starlette(
            routes=[
                Route("/whatsapp", webhook_handler, methods=["POST"]),
                Route("/health", lambda r: PlainTextResponse("OK"), methods=["GET"]),
            ]
        )

        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._running = True

        logger.info(f"WhatsApp webhook listening on :{self.port}/whatsapp")
        await self._server.serve()

    async def send_message(self, to: str, body: str) -> bool:
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

            response = await self._http_client.post(
                f"/Accounts/{self.account_sid}/Messages.json",
                data={
                    "From": self.from_number,
                    "To": to_formatted,
                    "Body": body,
                },
            )

            return response.status_code in (200, 201)

        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    async def get_message(self, timeout: Optional[float] = None) -> Optional[WhatsAppMessage]:
        """Get next incoming message from queue. Blocks until available or timeout."""
        try:
            if timeout:
                return await asyncio.wait_for(self.incoming_queue.get(), timeout=timeout)
            else:
                return await self.incoming_queue.get()
        except asyncio.TimeoutError:
            return None

    async def stop(self):
        """Stop the webhook server."""
        self._running = False
        if self._server:
            self._server.should_exit = True
        if self._http_client:
            await self._http_client.aclose()

    @property
    def is_running(self) -> bool:
        return self._running
