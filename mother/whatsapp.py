"""
WhatsApp integration — send and receive messages via Twilio API.

LEAF module. Uses httpx for Twilio REST API.

Enables Mother to:
- Send WhatsApp messages
- Receive incoming messages (webhook receiver)
- Build conversational interfaces via WhatsApp
- Send media (images, documents)

Requires:
- Twilio Account SID
- Twilio Auth Token
- Twilio WhatsApp number (sandbox or approved)
"""

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass(frozen=True)
class WhatsAppResult:
    """Outcome of a WhatsApp operation."""

    success: bool
    operation: str = ""
    message_sid: str = ""
    to: str = ""
    from_: str = ""
    body: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0


class WhatsAppClient:
    """
    Twilio WhatsApp API client.

    Sends messages via Twilio's WhatsApp API.
    """

    API_BASE = "https://api.twilio.com/2010-04-01"

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
    ):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required for WhatsApp integration")

        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID", "")
        self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.from_number = from_number or os.environ.get("TWILIO_WHATSAPP_NUMBER", "")

        if not self.account_sid or not self.auth_token:
            raise ValueError("Twilio Account SID and Auth Token required")

        if not self.from_number:
            # Default to Twilio sandbox number
            self.from_number = "whatsapp:+REDACTED"

        # Basic auth for Twilio API
        auth_str = f"{self.account_sid}:{self.auth_token}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()

        self.client = httpx.Client(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=30.0,
        )

    def send_message(self, to: str, body: str, media_url: str = "") -> WhatsAppResult:
        """
        Send a WhatsApp message.

        Args:
            to: Recipient number in format '+1234567890' (will be prefixed with 'whatsapp:')
            body: Message text
            media_url: Optional media URL (image, PDF, etc.)

        Returns:
            WhatsAppResult with message SID and status
        """
        if not body:
            return WhatsAppResult(
                success=False,
                operation="send",
                error="Message body cannot be empty",
            )

        # Ensure numbers have whatsapp: prefix
        to_formatted = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
        from_formatted = self.from_number if self.from_number.startswith("whatsapp:") else f"whatsapp:{self.from_number}"

        start = time.time()

        try:
            data = {
                "From": from_formatted,
                "To": to_formatted,
                "Body": body,
            }

            if media_url:
                data["MediaUrl"] = media_url

            response = self.client.post(
                f"/Accounts/{self.account_sid}/Messages.json",
                data=data,
            )

            elapsed = time.time() - start

            if response.status_code not in (200, 201):
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("message", response.text or f"HTTP {response.status_code}")
                return WhatsAppResult(
                    success=False,
                    operation="send",
                    to=to,
                    from_=from_formatted,
                    error=error_msg,
                    duration_seconds=elapsed,
                )

            data = response.json()
            message_sid = data.get("sid", "")

            return WhatsAppResult(
                success=True,
                operation="send",
                message_sid=message_sid,
                to=to_formatted,
                from_=from_formatted,
                body=body,
                duration_seconds=elapsed,
            )

        except httpx.TimeoutException:
            return WhatsAppResult(
                success=False,
                operation="send",
                to=to,
                error="Request timed out",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return WhatsAppResult(
                success=False,
                operation="send",
                to=to,
                error=str(e),
                duration_seconds=time.time() - start,
            )

    def get_messages(self, limit: int = 20) -> List[Dict[str, str]]:
        """
        Get recent messages (sent and received).

        Returns list of {sid, from, to, body, date_sent, status}
        """
        try:
            response = self.client.get(
                f"/Accounts/{self.account_sid}/Messages.json",
                params={"PageSize": min(limit, 100)},
            )

            if response.status_code != 200:
                return []

            data = response.json()
            messages = data.get("messages", [])

            return [
                {
                    "sid": m.get("sid", ""),
                    "from": m.get("from", ""),
                    "to": m.get("to", ""),
                    "body": m.get("body", ""),
                    "date_sent": m.get("date_sent", ""),
                    "status": m.get("status", ""),
                }
                for m in messages
            ]

        except Exception:
            return []

    def close(self):
        """Close the HTTP client."""
        if self.client:
            self.client.close()
