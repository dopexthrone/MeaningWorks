"""
Tests for WhatsApp setup automation scripts.

Tests that execute scripts require the repo working directory and
~/.motherlabs/mother.json to be present. They skip gracefully in CI.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Resolve script paths relative to repo root (works regardless of cwd)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETUP_SCRIPT = _REPO_ROOT / "scripts" / "setup_whatsapp_webhook.py"
_VERIFY_SCRIPT = _REPO_ROOT / "scripts" / "verify_whatsapp.py"
_INTEGRATION_GUIDE = _REPO_ROOT / "WHATSAPP_INTEGRATION.md"
_MOTHER_CONFIG = Path.home() / ".motherlabs" / "mother.json"


def test_setup_script_exists():
    """Setup script exists and is executable."""
    if not _SETUP_SCRIPT.exists():
        pytest.skip("Setup script not found (CI or non-standard layout)")
    assert _SETUP_SCRIPT.stat().st_mode & 0o111  # Executable


@pytest.mark.skipif(
    not _MOTHER_CONFIG.exists(),
    reason="Mother config not initialized (~/.motherlabs/mother.json missing)",
)
def test_setup_script_verify_only():
    """Setup script --verify-only runs without starting ngrok."""
    if not _SETUP_SCRIPT.exists():
        pytest.skip("Setup script not found")
    result = subprocess.run(
        [sys.executable, str(_SETUP_SCRIPT), "--verify-only"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "Config loaded" in result.stdout
    assert "ngrok auth token" in result.stdout


def test_verify_script_exists():
    """Verification script exists and is executable."""
    if not _VERIFY_SCRIPT.exists():
        pytest.skip("Verify script not found (CI or non-standard layout)")
    assert _VERIFY_SCRIPT.stat().st_mode & 0o111  # Executable


@pytest.mark.skipif(
    not _MOTHER_CONFIG.exists(),
    reason="Mother config not initialized (~/.motherlabs/mother.json missing)",
)
def test_verify_script_runs():
    """Verification script runs and checks config."""
    if not _VERIFY_SCRIPT.exists():
        pytest.skip("Verify script not found")
    result = subprocess.run(
        [sys.executable, str(_VERIFY_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "Configuration:" in result.stdout
    assert "Dependencies:" in result.stdout
    assert "Services:" in result.stdout


def test_integration_guide_exists():
    """WhatsApp integration documentation exists."""
    if not _INTEGRATION_GUIDE.exists():
        pytest.skip("Integration guide not found (CI or non-standard layout)")

    content = _INTEGRATION_GUIDE.read_text()
    assert "WhatsApp Integration for Mother" in content
    assert "Quick Setup" in content
    assert "Troubleshooting" in content


def test_config_has_whatsapp_fields():
    """Mother config includes WhatsApp fields."""
    if not _MOTHER_CONFIG.exists():
        pytest.skip("Mother config not initialized")

    config = json.loads(_MOTHER_CONFIG.read_text())

    # Check required fields exist
    assert "whatsapp_enabled" in config
    assert "whatsapp_webhook_enabled" in config
    assert "whatsapp_webhook_port" in config
    assert "twilio_account_sid" in config
    assert "twilio_auth_token" in config
    assert "twilio_whatsapp_number" in config
    assert "ngrok_auth_token" in config


def test_setup_script_help():
    """Setup script --help shows usage."""
    if not _SETUP_SCRIPT.exists():
        pytest.skip("Setup script not found")
    result = subprocess.run(
        [sys.executable, str(_SETUP_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "Setup WhatsApp webhook for Mother" in result.stdout
    assert "--port" in result.stdout
    assert "--verify-only" in result.stdout
