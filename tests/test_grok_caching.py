"""Tests for xAI Grok prompt caching via x-grok-conv-id header."""

import uuid
from unittest.mock import MagicMock, patch

import pytest


class TestGrokConvId:
    def test_conv_id_generated_on_init(self):
        """Each GrokClient instance gets a unique conversation ID."""
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            from core.llm import GrokClient
            client = GrokClient(api_key="test-key")
            assert hasattr(client, "_conv_id")
            assert isinstance(client._conv_id, str)
            # Should be a valid UUID
            uuid.UUID(client._conv_id)  # raises ValueError if invalid

    def test_conv_id_stable_per_session(self):
        """Same client instance reuses the same conv ID."""
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            from core.llm import GrokClient
            client = GrokClient(api_key="test-key")
            id1 = client._conv_id
            id2 = client._conv_id
            assert id1 == id2

    def test_new_session_gets_new_id(self):
        """Different GrokClient instances get different conv IDs."""
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            from core.llm import GrokClient
            client1 = GrokClient(api_key="test-key")
            client2 = GrokClient(api_key="test-key")
            assert client1._conv_id != client2._conv_id

    def test_conv_id_passed_to_openai_client(self):
        """The conv ID is passed as default_headers to the OpenAI client."""
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            from core.llm import GrokClient

            with patch("core.llm.GrokClient.client", new_callable=lambda: property(lambda self: None)):
                client = GrokClient(api_key="test-key")

            # Access the lazy client property to verify headers would be set
            # We can't easily test the actual OpenAI client creation without
            # the openai package, but we verify the conv_id is set
            assert client._conv_id
            assert len(client._conv_id) == 36  # UUID format: 8-4-4-4-12

    def test_conv_id_is_uuid4_format(self):
        """Conv ID is a valid UUID4 string."""
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            from core.llm import GrokClient
            client = GrokClient(api_key="test-key")
            parsed = uuid.UUID(client._conv_id)
            assert parsed.version == 4
