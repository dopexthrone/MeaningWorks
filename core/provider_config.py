"""
Provider-Specific Configuration Registry.

Derived from: NEXT-STEPS.md - Provider-Specific Tuning

Allows per-provider and per-stage configuration for:
- Max retries
- Max tokens
- Timeout settings
- Prompt style variants
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ProviderStageConfig:
    """Configuration for a specific provider and stage combination."""
    max_retries: int = 2
    max_tokens: int = 4096
    timeout_seconds: int = 60
    prompt_style: str = "default"  # "default" or "concise"


# Default configuration for unknown providers/stages
DEFAULT_CONFIG = ProviderStageConfig()


# Provider-specific configurations
# Structure: provider -> stage -> config
PROVIDER_CONFIGS: Dict[str, Dict[str, ProviderStageConfig]] = {
    "grok": {
        "default": ProviderStageConfig(),
        "synthesis": ProviderStageConfig(max_retries=2),
    },
    "openai": {
        "default": ProviderStageConfig(),
        # OpenAI GPT-5.1 works better with concise prompts
        "synthesis": ProviderStageConfig(max_retries=3, prompt_style="concise"),
    },
    "claude": {
        "default": ProviderStageConfig(),
        "synthesis": ProviderStageConfig(max_retries=2),
    },
    "gemini": {
        "default": ProviderStageConfig(),
        # Gemini may need more retries due to different system prompt handling
        "synthesis": ProviderStageConfig(max_retries=3),
    },
}


def get_config(provider: str, stage: str = "default") -> ProviderStageConfig:
    """
    Get stage-specific configuration for a provider.

    Args:
        provider: Provider name (e.g., "grok", "openai", "claude", "gemini")
        stage: Stage name (e.g., "intent", "synthesis", "default")

    Returns:
        ProviderStageConfig for the provider/stage combination.
        Falls back to provider default, then global default.
    """
    # Normalize provider name
    provider = provider.lower()

    # Check for provider-specific config
    if provider in PROVIDER_CONFIGS:
        provider_stages = PROVIDER_CONFIGS[provider]
        # Check for stage-specific config
        if stage in provider_stages:
            return provider_stages[stage]
        # Fall back to provider default
        if "default" in provider_stages:
            return provider_stages["default"]

    # Fall back to global default
    return DEFAULT_CONFIG


def get_all_providers() -> list:
    """Get list of all configured providers."""
    return list(PROVIDER_CONFIGS.keys())


def update_provider_config(
    provider: str,
    stage: str,
    **kwargs
) -> None:
    """
    Update provider configuration at runtime.

    Args:
        provider: Provider name
        stage: Stage name
        **kwargs: Config fields to update (max_retries, max_tokens, etc.)
    """
    provider = provider.lower()

    if provider not in PROVIDER_CONFIGS:
        PROVIDER_CONFIGS[provider] = {"default": ProviderStageConfig()}

    if stage not in PROVIDER_CONFIGS[provider]:
        PROVIDER_CONFIGS[provider][stage] = ProviderStageConfig()

    config = PROVIDER_CONFIGS[provider][stage]
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
