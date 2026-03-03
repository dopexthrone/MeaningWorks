"""
Tests for OllamaClient — local LLM via Ollama native API.

All tests mock requests — no actual Ollama server needed.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# --- Availability ---

class TestOllamaAvailability:
    """Test OllamaClient.is_available() static method."""

    @patch("requests.get")
    def test_available_when_server_responds(self, mock_get):
        from core.llm import OllamaClient
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        assert OllamaClient.is_available() is True
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags", timeout=2.0
        )

    @patch("requests.get")
    def test_unavailable_when_server_down(self, mock_get):
        from core.llm import OllamaClient
        mock_get.side_effect = Exception("Connection refused")

        assert OllamaClient.is_available() is False

    @patch("requests.get")
    def test_unavailable_on_timeout(self, mock_get):
        from core.llm import OllamaClient
        mock_get.side_effect = Exception("timeout")

        assert OllamaClient.is_available() is False

    @patch("requests.get")
    def test_unavailable_on_bad_status(self, mock_get):
        from core.llm import OllamaClient
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        assert OllamaClient.is_available() is False

    @patch("requests.get")
    def test_custom_base_url(self, mock_get):
        from core.llm import OllamaClient
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        OllamaClient.is_available(base_url="http://192.168.1.50:11434")
        mock_get.assert_called_once_with(
            "http://192.168.1.50:11434/api/tags", timeout=2.0
        )


# --- Complete ---

class TestOllamaComplete:
    """Test OllamaClient.complete() method."""

    @patch("requests.post")
    def test_basic_complete(self, mock_post):
        from core.llm import OllamaClient
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Hello there!"},
            "prompt_eval_count": 25,
            "eval_count": 10,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient(model="llama3:8b")
        result = client.complete([{"role": "user", "content": "hi"}])

        assert result == "Hello there!"
        assert client.last_usage["input_tokens"] == 25
        assert client.last_usage["output_tokens"] == 10
        assert client.last_usage["total_tokens"] == 35

    @patch("requests.post")
    def test_complete_with_system_prompt(self, mock_post):
        from core.llm import OllamaClient
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "I am helpful."},
            "prompt_eval_count": 50,
            "eval_count": 5,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient()
        result = client.complete(
            [{"role": "user", "content": "hi"}],
            system="You are helpful."
        )

        assert result == "I am helpful."
        # Verify system message was prepended
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."

    @patch("requests.post")
    def test_complete_deterministic_temperature(self, mock_post):
        from core.llm import OllamaClient
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "ok"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient(deterministic=True)
        client.complete([{"role": "user", "content": "hi"}])

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.0

    @patch("requests.post")
    def test_complete_non_deterministic_temperature(self, mock_post):
        from core.llm import OllamaClient
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "ok"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient(deterministic=False)
        client.complete([{"role": "user", "content": "hi"}])

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.7

    @patch("requests.post")
    def test_complete_with_system_delegates(self, mock_post):
        from core.llm import OllamaClient
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "delegated"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient()
        result = client.complete_with_system("sys prompt", "user msg")

        assert result == "delegated"
        payload = mock_post.call_args[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"


# --- Stream ---

class TestOllamaStream:
    """Test OllamaClient.stream() method."""

    @patch("requests.post")
    def test_stream_basic(self, mock_post):
        from core.llm import OllamaClient

        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}).encode(),
            json.dumps({"message": {"content": " world"}, "done": False}).encode(),
            json.dumps({
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 20,
                "eval_count": 8,
            }).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_resp

        client = OllamaClient()
        tokens = list(client.stream([{"role": "user", "content": "hi"}]))

        assert tokens == ["Hello", " world"]
        assert client.last_usage["input_tokens"] == 20
        assert client.last_usage["output_tokens"] == 8
        assert client.last_usage["total_tokens"] == 28

    @patch("requests.post")
    def test_stream_with_system(self, mock_post):
        from core.llm import OllamaClient

        lines = [
            json.dumps({"message": {"content": "ok"}, "done": True,
                         "prompt_eval_count": 5, "eval_count": 1}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_resp

        client = OllamaClient()
        list(client.stream(
            [{"role": "user", "content": "hi"}],
            system="Be brief."
        ))

        payload = mock_post.call_args[1]["json"]
        assert payload["stream"] is True
        assert payload["messages"][0]["role"] == "system"

    @patch("requests.post")
    def test_stream_skips_empty_lines(self, mock_post):
        from core.llm import OllamaClient

        lines = [
            b"",  # empty line
            json.dumps({"message": {"content": "data"}, "done": True,
                         "prompt_eval_count": 5, "eval_count": 1}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_resp

        client = OllamaClient()
        tokens = list(client.stream([{"role": "user", "content": "hi"}]))
        assert "data" in tokens


# --- Vision ---

class TestOllamaVision:
    """Test Ollama vision image injection via bridge._inject_images()."""

    def test_inject_images_ollama_format(self):
        from mother.bridge import EngineBridge

        bridge = EngineBridge(provider="local")
        messages = [
            {"role": "user", "content": "What is this?"},
        ]
        images = ["base64data1", "base64data2"]

        result = bridge._inject_images(messages, images)

        # Ollama format: images field on message dict
        last = result[-1]
        assert last["role"] == "user"
        assert last["content"] == "What is this?"
        assert last["images"] == ["base64data1", "base64data2"]

    def test_inject_images_claude_format_unchanged(self):
        from mother.bridge import EngineBridge

        bridge = EngineBridge(provider="claude")
        messages = [
            {"role": "user", "content": "What is this?"},
        ]
        images = ["base64data"]

        result = bridge._inject_images(messages, images)

        last = result[-1]
        assert isinstance(last["content"], list)
        assert last["content"][0]["type"] == "text"
        assert last["content"][1]["type"] == "image"


# --- Failover ---

class TestOllamaInFailover:
    """Test OllamaClient as primary in FailoverClient."""

    @patch("requests.post")
    def test_failover_uses_ollama_first(self, mock_post):
        from core.llm import OllamaClient, FailoverClient, MockClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "local response"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        ollama = OllamaClient()
        mock = MockClient()
        failover = FailoverClient([ollama, mock])

        result = failover.complete([{"role": "user", "content": "hi"}])
        assert result == "local response"
        assert failover.last_usage["input_tokens"] == 10

    @patch("requests.post")
    def test_failover_falls_back_on_ollama_error(self, mock_post):
        from core.llm import OllamaClient, FailoverClient, MockClient

        mock_post.side_effect = Exception("Connection refused")

        ollama = OllamaClient()
        mock = MockClient()
        failover = FailoverClient([ollama, mock])

        result = failover.complete([{"role": "user", "content": "hi"}])
        assert "[Mock response" in result


# --- create_client ---

class TestCreateClientLocal:
    """Test create_client() with local/ollama provider."""

    def test_create_local_client(self):
        from core.llm import create_client, OllamaClient
        client = create_client(provider="local")
        assert isinstance(client, OllamaClient)
        assert client.model == "llama3:8b"

    def test_create_ollama_client(self):
        from core.llm import create_client, OllamaClient
        client = create_client(provider="ollama", model="mistral")
        assert isinstance(client, OllamaClient)
        assert client.model == "mistral"

    def test_create_local_with_custom_url(self):
        from core.llm import create_client, OllamaClient
        client = create_client(
            provider="local",
            base_url="http://192.168.1.50:11434"
        )
        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://192.168.1.50:11434"

    @patch("core.llm.OllamaClient.is_available", return_value=True)
    def test_auto_detect_falls_back_to_ollama(self, mock_avail):
        """Auto-detect picks Ollama when no API keys and Ollama is running."""
        from core.llm import create_client, OllamaClient
        import os

        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                                  "XAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")}
        with patch.dict(os.environ, env_clean, clear=True):
            client = create_client(provider="auto")
            assert isinstance(client, OllamaClient)

    @patch("core.llm.OllamaClient.is_available", return_value=False)
    def test_auto_detect_raises_when_nothing_available(self, mock_avail):
        from core.llm import create_client
        import os

        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                                  "XAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")}
        with patch.dict(os.environ, env_clean, clear=True):
            with pytest.raises(ValueError, match="No API key found"):
                create_client(provider="auto")


# --- Errors ---

class TestOllamaErrors:
    """Test OllamaClient error handling."""

    @patch("requests.post")
    def test_http_error_propagates(self, mock_post):
        from core.llm import OllamaClient
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_post.return_value = mock_resp

        client = OllamaClient()
        with pytest.raises(requests.HTTPError):
            client.complete([{"role": "user", "content": "hi"}])

    @patch("requests.post")
    def test_connection_error_propagates(self, mock_post):
        from core.llm import OllamaClient
        import requests

        mock_post.side_effect = requests.ConnectionError("Connection refused")

        client = OllamaClient()
        with pytest.raises(requests.ConnectionError):
            client.complete([{"role": "user", "content": "hi"}])

    @patch("requests.post")
    def test_timeout_error_propagates(self, mock_post):
        from core.llm import OllamaClient
        import requests

        mock_post.side_effect = requests.Timeout("Request timed out")

        client = OllamaClient()
        with pytest.raises(requests.Timeout):
            client.complete([{"role": "user", "content": "hi"}])

    @patch("requests.post")
    def test_empty_response_content(self, mock_post):
        from core.llm import OllamaClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": ""},
            "prompt_eval_count": 5,
            "eval_count": 0,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient()
        result = client.complete([{"role": "user", "content": "hi"}])
        assert result == ""


# --- List Models ---

class TestOllamaListModels:
    """Test OllamaClient.list_models() static method."""

    @patch("requests.get")
    def test_list_models_success(self, mock_get):
        from core.llm import OllamaClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3:8b", "size": 4700000000},
                {"name": "mistral:latest", "size": 4100000000},
                {"name": "codellama:13b", "size": 7400000000},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        models = OllamaClient.list_models()
        assert models == ["llama3:8b", "mistral:latest", "codellama:13b"]

    @patch("requests.get")
    def test_list_models_empty(self, mock_get):
        from core.llm import OllamaClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        models = OllamaClient.list_models()
        assert models == []

    @patch("requests.get")
    def test_list_models_server_down(self, mock_get):
        from core.llm import OllamaClient

        mock_get.side_effect = Exception("Connection refused")

        models = OllamaClient.list_models()
        assert models == []

    @patch("requests.get")
    def test_list_models_custom_url(self, mock_get):
        from core.llm import OllamaClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "phi3:mini"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        models = OllamaClient.list_models(base_url="http://10.0.0.5:11434")
        mock_get.assert_called_with(
            "http://10.0.0.5:11434/api/tags", timeout=5.0
        )
        assert models == ["phi3:mini"]


# --- Config integration ---

class TestOllamaConfig:
    """Test config.py local provider integration."""

    def test_providers_includes_local(self):
        from mother.config import PROVIDERS
        assert "local" in PROVIDERS

    def test_default_models_includes_local(self):
        from mother.config import DEFAULT_MODELS
        assert "local" in DEFAULT_MODELS
        assert DEFAULT_MODELS["local"] == "llama3:8b"

    def test_env_vars_includes_local(self):
        from mother.config import ENV_VARS
        assert "local" in ENV_VARS
        assert ENV_VARS["local"] == "LOCAL_MODEL_URL"

    def test_mother_config_local_fields(self):
        from mother.config import MotherConfig
        config = MotherConfig()
        assert config.local_base_url == "http://localhost:11434"
        assert config.local_model == "llama3:8b"


# --- Bridge cost rates ---

class TestBridgeCostRates:
    """Test bridge.py local provider cost rates."""

    def test_local_cost_is_zero(self):
        from mother.bridge import COST_RATES
        assert "local" in COST_RATES
        assert COST_RATES["local"]["input"] == 0.0
        assert COST_RATES["local"]["output"] == 0.0
