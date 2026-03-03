"""Tests for social publishing wiring — config, bridge, chat integration."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mother.config import MotherConfig, load_config, save_config


# --- Config fields ---


class TestConfigFields:
    def test_default_discord_webhook_url(self):
        cfg = MotherConfig()
        assert cfg.discord_webhook_url == ""

    def test_default_bluesky_handle(self):
        cfg = MotherConfig()
        assert cfg.bluesky_handle == ""

    def test_default_bluesky_app_password(self):
        cfg = MotherConfig()
        assert cfg.bluesky_app_password == ""

    def test_default_auto_discord_enabled(self):
        cfg = MotherConfig()
        assert cfg.auto_discord_enabled is False

    def test_default_auto_bluesky_enabled(self):
        cfg = MotherConfig()
        assert cfg.auto_bluesky_enabled is False

    def test_default_auto_publish_projects(self):
        cfg = MotherConfig()
        assert cfg.auto_publish_projects is False

    def test_default_publish_projects_public(self):
        cfg = MotherConfig()
        assert cfg.publish_projects_public is False

    def test_default_mother_git_name(self):
        cfg = MotherConfig()
        assert cfg.mother_git_name == "Mother"

    def test_default_mother_git_email(self):
        cfg = MotherConfig()
        assert cfg.mother_git_email == "mother@motherlabs.ai"

    def test_persistence(self, tmp_path):
        config_path = str(tmp_path / "test.json")
        cfg = MotherConfig(
            discord_webhook_url="https://discord.com/webhook/123",
            bluesky_handle="mother.bsky.social",
            auto_discord_enabled=True,
            auto_publish_projects=True,
            mother_git_name="TestBot",
        )
        save_config(cfg, config_path)
        loaded = load_config(config_path)
        assert loaded.discord_webhook_url == "https://discord.com/webhook/123"
        assert loaded.bluesky_handle == "mother.bsky.social"
        assert loaded.auto_discord_enabled is True
        assert loaded.auto_publish_projects is True
        assert loaded.mother_git_name == "TestBot"

    def test_backward_compat(self, tmp_path):
        """Config without new fields loads with defaults."""
        config_path = tmp_path / "old.json"
        config_path.write_text(json.dumps({"name": "Mother", "setup_complete": True}))
        loaded = load_config(str(config_path))
        assert loaded.discord_webhook_url == ""
        assert loaded.auto_publish_projects is False


# --- Bridge methods exist ---


class TestBridgeMethods:
    def test_publish_project_method_exists(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()
        assert hasattr(bridge, "publish_project")
        assert callable(bridge.publish_project)

    def test_discord_post_method_exists(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()
        assert hasattr(bridge, "discord_post")
        assert callable(bridge.discord_post)

    def test_bluesky_post_method_exists(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()
        assert hasattr(bridge, "bluesky_post")
        assert callable(bridge.bluesky_post)

    def test_announce_build_method_exists(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()
        assert hasattr(bridge, "announce_build")
        assert callable(bridge.announce_build)

    def test_process_social_queue_method_exists(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()
        assert hasattr(bridge, "process_social_queue")
        assert callable(bridge.process_social_queue)

    def test_get_social_queue_status_method_exists(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()
        assert hasattr(bridge, "get_social_queue_status")
        assert callable(bridge.get_social_queue_status)


# --- Bridge delegation ---


class TestBridgeDelegation:
    def test_publish_project_delegates(self, tmp_path):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()

        with patch("mother.project_publisher.publish_project") as mock_pub, \
             patch("mother.config.load_config") as mock_cfg:
            from mother.project_publisher import PublishResult
            mock_pub.return_value = PublishResult(
                success=True,
                repo_url="https://github.com/user/test",
                repo_name="test",
                commit_hash="abc",
                files_pushed=5,
            )
            mock_cfg.return_value = MotherConfig()

            result = asyncio.run(bridge.publish_project(
                str(tmp_path), "test", "A test project"
            ))
            assert result["success"] is True
            assert result["repo_url"] == "https://github.com/user/test"

    def test_discord_post_no_webhook(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()

        with patch("mother.config.load_config") as mock_cfg:
            mock_cfg.return_value = MotherConfig(discord_webhook_url="")
            result = asyncio.run(bridge.discord_post(content="hi"))
            assert result["success"] is False
            assert "not configured" in result["error"]

    def test_announce_build_no_platforms(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()

        with patch("mother.config.load_config") as mock_cfg:
            mock_cfg.return_value = MotherConfig()  # all disabled
            result = asyncio.run(bridge.announce_build("app", "desc"))
            assert result["enqueued"] == 0

    def test_get_social_queue_status(self, tmp_path):
        from mother.bridge import EngineBridge
        bridge = EngineBridge()

        with patch("mother.social_queue.SocialQueue.__init__", return_value=None), \
             patch("mother.social_queue.SocialQueue.pending_count", return_value=3), \
             patch("mother.social_queue.SocialQueue.recent_posts", return_value=[]), \
             patch("mother.social_queue.SocialQueue.failed_posts", return_value=[]):
            result = asyncio.run(bridge.get_social_queue_status())
            assert result["pending"] == 3


# --- Action handlers registered ---


class TestActionHandlers:
    def test_publish_project_action_in_chat(self):
        """Verify the action handler exists in chat module."""
        import mother.screens.chat as chat_mod
        source = Path(chat_mod.__file__).read_text()
        assert 'action == "publish_project"' in source

    def test_discord_post_action_in_chat(self):
        import mother.screens.chat as chat_mod
        source = Path(chat_mod.__file__).read_text()
        assert 'action == "discord_post"' in source

    def test_bluesky_post_action_in_chat(self):
        import mother.screens.chat as chat_mod
        source = Path(chat_mod.__file__).read_text()
        assert 'action == "bluesky_post"' in source

    def test_social_queue_in_autonomous_tick(self):
        import mother.screens.chat as chat_mod
        source = Path(chat_mod.__file__).read_text()
        assert "_process_social_queue_tick" in source

    def test_auto_publish_in_post_build(self):
        import mother.screens.chat as chat_mod
        source = Path(chat_mod.__file__).read_text()
        assert "auto_publish_projects" in source
        assert "publish_project" in source

    def test_announce_build_in_post_build(self):
        import mother.screens.chat as chat_mod
        source = Path(chat_mod.__file__).read_text()
        assert "announce_build" in source
