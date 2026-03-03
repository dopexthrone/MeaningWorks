"""
Phase 3: Tests for conversation memory store.
"""

import time
import pytest
from pathlib import Path

from mother.memory import ConversationStore, ChatMessage


class TestConversationStoreCreation:
    """Test store initialization."""

    def test_creates_db(self, tmp_path):
        db = tmp_path / "history.db"
        store = ConversationStore(path=db)
        assert db.exists()
        store.close()

    def test_has_session_id(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        assert len(store.session_id) > 10
        store.close()


class TestMessageCRUD:
    """Test adding and retrieving messages."""

    def test_add_and_get(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "hello")
        store.add_message("assistant", "hi there")
        history = store.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        store.close()

    def test_message_has_timestamp(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        before = time.time()
        store.add_message("user", "test")
        history = store.get_history()
        assert history[0].timestamp >= before
        store.close()

    def test_limit_returns_recent(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        for i in range(10):
            store.add_message("user", f"msg {i}")
        history = store.get_history(limit=3)
        assert len(history) == 3
        assert history[-1].content == "msg 9"
        store.close()

    def test_message_id_returned(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        mid = store.add_message("user", "test")
        assert isinstance(mid, int)
        assert mid > 0
        store.close()

    def test_metadata_roundtrip(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "test", metadata={"cost": 0.01})
        history = store.get_history()
        assert history[0].metadata["cost"] == 0.01
        store.close()


class TestContextWindow:
    """Test context window retrieval."""

    def test_fits_within_budget(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        # Add a message with known length
        store.add_message("user", "a" * 1000)
        store.add_message("assistant", "b" * 1000)
        store.add_message("user", "c" * 1000)
        # Budget of 500 tokens ~ 2000 chars — should fit ~2 messages
        msgs = store.get_context_window(max_tokens=500)
        assert len(msgs) <= 3
        assert all("role" in m for m in msgs)
        store.close()

    def test_returns_llm_format(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "hello")
        msgs = store.get_context_window()
        assert msgs[0] == {"role": "user", "content": "hello"}
        store.close()


class TestSessions:
    """Test session management."""

    def test_list_sessions(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "hello")
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["message_count"] == 1
        store.close()

    def test_clear_session(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "hello")
        store.add_message("assistant", "hi")
        deleted = store.clear_session()
        assert deleted == 2
        assert store.get_history() == []
        store.close()


class TestContextWindowBudget:
    """Test conservative token estimation."""

    def test_budget_uses_3_chars_per_token(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        # 300 chars per message. Budget of 100 tokens = 300 chars = fits 1 message
        store.add_message("user", "a" * 300)
        store.add_message("assistant", "b" * 300)
        msgs = store.get_context_window(max_tokens=100)
        assert len(msgs) == 1
        store.close()


class TestCrossSessionSummary:
    """Test cross-session summary for memory injection."""

    def test_empty_store_summary(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        summary = store.get_cross_session_summary()
        assert summary["total_sessions"] == 0
        assert summary["topics"] == []
        assert summary["days_since_last"] is None
        store.close()

    def test_summary_with_messages(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "Build me a todo app")
        store.add_message("assistant", "Sure thing")
        summary = store.get_cross_session_summary()
        assert summary["total_sessions"] == 1
        assert summary["total_messages"] == 2
        assert len(summary["topics"]) == 1
        assert "todo" in summary["topics"][0].lower()
        assert summary["days_since_last"] is not None
        store.close()

    def test_session_topics(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "First user message in session")
        store.add_message("assistant", "Response")
        store.add_message("user", "Second user message")
        topics = store.get_session_topics(limit=2)
        assert len(topics) == 2
        assert topics[0] == "First user message in session"
        store.close()

    def test_message_count_all(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "hello")
        store.add_message("assistant", "hi")
        store.add_message("user", "another", session_id="other-session")
        assert store.get_message_count_all() == 3
        store.close()


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_to_llm_message(self):
        msg = ChatMessage(role="user", content="test")
        assert msg.to_llm_message() == {"role": "user", "content": "test"}


class TestGetLastUserMessageTime:
    """Tests for get_last_user_message_time."""

    def test_returns_timestamp(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("user", "hello")
        result = store.get_last_user_message_time()
        assert result is not None
        assert result > 0
        store.close()

    def test_returns_none_when_empty(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        result = store.get_last_user_message_time()
        assert result is None
        store.close()

    def test_ignores_assistant_messages(self, tmp_path):
        store = ConversationStore(path=tmp_path / "h.db")
        store.add_message("assistant", "hi there")
        result = store.get_last_user_message_time()
        assert result is None
        store.close()
