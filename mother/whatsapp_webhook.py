"""
WhatsApp webhook setup helper.

Sets up ngrok tunnel and WhatsApp webhook receiver so Mother can receive
incoming WhatsApp messages.

Usage:
    python -m mother.whatsapp_webhook --ngrok-token TOKEN --account-sid SID --auth-token TOKEN --from-number +1234567890
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("whatsapp_webhook")


def setup_ngrok(auth_token: str, port: int = 8080) -> Optional[str]:
    """
    Start ngrok tunnel and return public URL.

    Returns:
        Public HTTPS URL or None if failed
    """
    try:
        # Configure ngrok auth
        subprocess.run(
            ["ngrok", "config", "add-authtoken", auth_token],
            capture_output=True,
            check=True,
        )

        # Start ngrok tunnel in background
        proc = subprocess.Popen(
            ["ngrok", "http", str(port), "--log=stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for tunnel to be ready and extract URL
        # ngrok prints JSON on startup with public_url
        time.sleep(3)  # Give ngrok time to start

        # Get ngrok API status
        import httpx
        try:
            resp = httpx.get("http://localhost:4040/api/tunnels", timeout=5.0)
            data = resp.json()
            tunnels = data.get("tunnels", [])
            if tunnels:
                public_url = tunnels[0].get("public_url", "")
                if public_url.startswith("https://"):
                    logger.info(f"ngrok tunnel: {public_url}")
                    return public_url
        except Exception as e:
            logger.error(f"Failed to get ngrok URL: {e}")

        return None

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"ngrok setup failed: {e}")
        return None


def configure_twilio_webhook(
    account_sid: str,
    auth_token: str,
    phone_number_sid: str,
    webhook_url: str,
) -> bool:
    """
    Configure Twilio to send incoming WhatsApp messages to webhook URL.

    Returns:
        True if successful
    """
    import base64
    import httpx

    try:
        # Format webhook URL
        if not webhook_url.endswith("/whatsapp"):
            webhook_url = f"{webhook_url}/whatsapp"

        # Twilio API authentication
        auth_str = f"{account_sid}:{auth_token}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()

        # Update phone number webhook
        client = httpx.Client(
            base_url="https://api.twilio.com/2010-04-01",
            headers={"Authorization": f"Basic {auth_b64}"},
            timeout=30.0,
        )

        resp = client.post(
            f"/Accounts/{account_sid}/IncomingPhoneNumbers/{phone_number_sid}.json",
            data={"SmsUrl": webhook_url, "SmsMethod": "POST"},
        )

        if resp.status_code == 200:
            logger.info(f"Twilio webhook configured: {webhook_url}")
            return True
        else:
            logger.error(f"Twilio API error: {resp.status_code} {resp.text}")
            return False

    except Exception as e:
        logger.error(f"Failed to configure Twilio webhook: {e}")
        return False


async def run_webhook_server(
    account_sid: str,
    auth_token: str,
    from_number: str,
    port: int = 8080,
):
    """Start WhatsApp webhook receiver."""
    from messaging.whatsapp_bridge import WhatsAppBridge

    async def handle_message(msg):
        """Handle incoming WhatsApp message."""
        logger.info(f"Received from {msg.sender_id}: {msg.content}")
        # For now, just echo back
        return f"You said: {msg.content}"

    bridge = WhatsAppBridge(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        webhook_port=port,
        on_message=handle_message,
    )

    logger.info(f"Starting WhatsApp webhook on port {port}")
    await bridge.start()


def main():
    parser = argparse.ArgumentParser(description="Set up WhatsApp webhook with ngrok")
    parser.add_argument("--ngrok-token", required=True, help="ngrok auth token")
    parser.add_argument("--account-sid", required=True, help="Twilio Account SID")
    parser.add_argument("--auth-token", required=True, help="Twilio Auth Token")
    parser.add_argument("--from-number", required=True, help="Twilio WhatsApp number")
    parser.add_argument("--port", type=int, default=8080, help="Webhook port (default 8080)")
    parser.add_argument("--skip-twilio-config", action="store_true", help="Skip Twilio webhook configuration")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Step 1: Start ngrok
    logger.info("Setting up ngrok tunnel...")
    public_url = setup_ngrok(args.ngrok_token, args.port)

    if not public_url:
        logger.error("Failed to start ngrok tunnel")
        sys.exit(1)

    logger.info(f"✓ ngrok tunnel active: {public_url}")
    logger.info(f"  Webhook URL: {public_url}/whatsapp")

    # Step 2: Configure Twilio (if not skipped)
    if not args.skip_twilio_config:
        logger.info("\nConfiguring Twilio webhook...")
        logger.info("Note: You'll need to manually configure the webhook URL in Twilio console:")
        logger.info(f"  1. Go to https://console.twilio.com/")
        logger.info(f"  2. Navigate to Messaging > WhatsApp Senders")
        logger.info(f"  3. Click on your WhatsApp number")
        logger.info(f"  4. Set 'When a message comes in' to: {public_url}/whatsapp")
        logger.info(f"  5. Save")

    # Step 3: Start webhook server
    logger.info(f"\nStarting WhatsApp webhook server on port {args.port}...")
    logger.info("Press Ctrl+C to stop\n")

    try:
        asyncio.run(run_webhook_server(
            account_sid=args.account_sid,
            auth_token=args.auth_token,
            from_number=args.from_number,
            port=args.port,
        ))
    except KeyboardInterrupt:
        logger.info("\nStopping webhook server...")
        sys.exit(0)


if __name__ == "__main__":
    main()
