"""Tests for Huey swarm task integration."""

import os
import tempfile
from unittest.mock import patch, MagicMock

# Must set data dir and immediate mode BEFORE importing worker modules
_test_dir = tempfile.mkdtemp(prefix="motherlabs_test_")
os.environ.setdefault("MOTHERLABS_DATA_DIR", _test_dir)
os.environ["MOTHERLABS_HUEY_IMMEDIATE"] = "1"


class TestSwarmExecuteTask:
    """swarm_execute_task Huey task."""

    @patch("worker.keystore.restore_llm_key")
    @patch("worker.keystore.apply_llm_key", return_value={})
    @patch("swarm.conductor.SwarmConductor")
    @patch("worker.keystore.decrypt_if_present")
    def test_basic_execution(self, mock_decrypt, mock_conductor_cls, mock_apply, mock_restore):
        """Task creates state and calls conductor.execute()."""
        from worker.swarm_tasks import swarm_execute_task
        from swarm.state import SwarmState, SwarmResult

        mock_decrypt.return_value = None

        mock_result = SwarmResult(
            success=True,
            state=SwarmState(intent="test", swarm_id="test-id"),
            agent_timings={"compile": 1.0},
            total_duration_s=1.5,
        )
        mock_conductor = MagicMock()
        mock_conductor.execute.return_value = mock_result
        mock_conductor_cls.return_value = mock_conductor

        result = swarm_execute_task.call_local(
            intent="Build a todo app",
            domain="software",
            request_type="compile_only",
        )

        assert result["success"] is True
        mock_conductor.execute.assert_called_once()

    @patch("worker.keystore.restore_llm_key")
    @patch("worker.keystore.apply_llm_key", return_value={})
    @patch("swarm.conductor.SwarmConductor")
    @patch("worker.keystore.decrypt_if_present")
    def test_with_byo_key(self, mock_decrypt, mock_conductor_cls, mock_apply, mock_restore):
        """BYO key is decrypted and passed to state."""
        from worker.swarm_tasks import swarm_execute_task
        from swarm.state import SwarmState, SwarmResult

        mock_decrypt.return_value = "sk-ant-test-key"

        mock_result = SwarmResult(
            success=True,
            state=SwarmState(intent="test"),
        )
        mock_conductor = MagicMock()
        mock_conductor.execute.return_value = mock_result
        mock_conductor_cls.return_value = mock_conductor

        result = swarm_execute_task.call_local(
            intent="test",
            llm_api_key_enc="encrypted-key",
        )

        assert result["success"] is True
        mock_decrypt.assert_called_once_with("encrypted-key")

    @patch("worker.keystore.restore_llm_key")
    @patch("worker.keystore.apply_llm_key", return_value={})
    @patch("swarm.conductor.SwarmConductor")
    @patch("worker.keystore.decrypt_if_present")
    def test_exception_handling(self, mock_decrypt, mock_conductor_cls, mock_apply, mock_restore):
        """Exceptions are caught and returned as error dict."""
        from worker.swarm_tasks import swarm_execute_task

        mock_decrypt.return_value = None
        mock_conductor = MagicMock()
        mock_conductor.execute.side_effect = RuntimeError("Boom")
        mock_conductor_cls.return_value = mock_conductor

        result = swarm_execute_task.call_local(
            intent="test",
        )

        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert "Boom" in result["errors"][0]["message"]

    @patch("worker.keystore.restore_llm_key")
    @patch("worker.keystore.apply_llm_key", return_value={})
    @patch("swarm.conductor.SwarmConductor")
    @patch("worker.keystore.decrypt_if_present")
    def test_provider_detection_openai(self, mock_decrypt, mock_conductor_cls, mock_apply, mock_restore):
        """OpenAI key prefix is detected."""
        from worker.swarm_tasks import swarm_execute_task
        from swarm.state import SwarmState, SwarmResult

        mock_decrypt.return_value = "sk-test-openai-key"

        mock_result = SwarmResult(
            success=True,
            state=SwarmState(intent="test"),
        )
        mock_conductor = MagicMock()
        mock_conductor.execute.return_value = mock_result
        mock_conductor_cls.return_value = mock_conductor

        result = swarm_execute_task.call_local(
            intent="test",
            llm_api_key_enc="encrypted",
        )

        # Verify state was created with detected provider
        call_args = mock_conductor.execute.call_args
        state = call_args[0][0]
        assert state.provider == "openai"

    @patch("worker.keystore.restore_llm_key")
    @patch("worker.keystore.apply_llm_key", return_value={})
    @patch("swarm.conductor.SwarmConductor")
    @patch("worker.keystore.decrypt_if_present")
    def test_progress_callback_wired(self, mock_decrypt, mock_conductor_cls, mock_apply, mock_restore):
        """Progress callback is passed to conductor."""
        from worker.swarm_tasks import swarm_execute_task
        from swarm.state import SwarmState, SwarmResult

        mock_decrypt.return_value = None

        mock_result = SwarmResult(
            success=True,
            state=SwarmState(intent="test"),
        )
        mock_conductor = MagicMock()
        mock_conductor.execute.return_value = mock_result
        mock_conductor_cls.return_value = mock_conductor

        swarm_execute_task.call_local(
            intent="test",
        )

        # Verify progress_callback was passed
        call_args = mock_conductor.execute.call_args
        assert "progress_callback" in call_args[1] or len(call_args[0]) > 1
