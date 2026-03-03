"""
Tests for mother/whatsapp.py -- LEAF module.

Covers: WhatsAppResult frozen dataclass, WhatsAppClient with mocked httpx,
send_message, get_messages.
"""

from unittest.mock import MagicMock, patch

import pytest

# httpx is optional
pytest.importorskip("httpx")

from mother.whatsapp import WhatsAppResult, WhatsAppClient


class TestWhatsAppResult:
    def test_frozen(self):
        r = WhatsAppResult(success=True, operation="send")
        with pytest.raises(AttributeError):
            r.success = False

    def test_defaults(self):
        r = WhatsAppResult(success=True)
        assert r.operation == ""
        assert r.message_sid == ""
        assert r.to == ""


class TestWhatsAppClient:
    @patch("mother.whatsapp.httpx.Client")
    def test_requires_credentials(self, mock_client_cls):
        with pytest.raises(ValueError, match="SID and Auth Token required"):
            WhatsAppClient(account_sid="", auth_token="")

    @patch("mother.whatsapp.httpx.Client")
    def test_send_message_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"sid": "SM123456"}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WhatsAppClient(
            account_sid="AC123",
            auth_token="token123",
            from_number="whatsapp:+REDACTED"
        )
        result = client.send_message("+12345678900", "Hello")

        assert result.success
        assert result.message_sid == "SM123456"
        assert "whatsapp:" in result.to

    @patch("mother.whatsapp.httpx.Client")
    def test_send_message_empty_body(self, mock_client_cls):
        client = WhatsAppClient(
            account_sid="AC123",
            auth_token="token123",
        )
        result = client.send_message("+12345678900", "")

        assert not result.success
        assert "empty" in result.error

    @patch("mother.whatsapp.httpx.Client")
    def test_send_message_api_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = '{"message": "Invalid credentials"}'
        mock_response.json.return_value = {"message": "Invalid credentials"}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WhatsAppClient(
            account_sid="AC123",
            auth_token="bad_token",
        )
        result = client.send_message("+12345678900", "Test")

        assert not result.success
        assert "Invalid credentials" in result.error

    @patch("mother.whatsapp.httpx.Client")
    def test_get_messages(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [
                {
                    "sid": "SM1",
                    "from": "whatsapp:+11234567890",
                    "to": "whatsapp:+REDACTED",
                    "body": "Hello",
                    "date_sent": "2026-02-15",
                    "status": "delivered",
                }
            ]
        }
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WhatsAppClient(account_sid="AC123", auth_token="token123")
        messages = client.get_messages(limit=20)

        assert len(messages) == 1
        assert messages[0]["sid"] == "SM1"
        assert messages[0]["body"] == "Hello"

    @patch("mother.whatsapp.httpx.Client")
    def test_auto_prefixes_whatsapp(self, mock_client_cls):
        """Phone numbers get whatsapp: prefix automatically."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"sid": "SM123"}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WhatsAppClient(
            account_sid="AC123",
            auth_token="token123",
            from_number="+14155238886"  # No prefix
        )
        result = client.send_message("+REDACTED_PHONE", "Test")

        # Verify both got prefixed
        call_data = mock_client.post.call_args[1]["data"]
        assert call_data["From"].startswith("whatsapp:")
        assert call_data["To"].startswith("whatsapp:")
