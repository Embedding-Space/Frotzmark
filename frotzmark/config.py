"""
Configuration loading from environment variables.

This module handles settings that come from environment variables only.
CLI-configurable settings (story file, manual file, model, seed) are
handled by the CLI layer.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_DIR = Path(__file__).parent / "prompts"

# =============================================================================
# Environment-only settings (security-sensitive, not exposed via CLI)
# =============================================================================

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY must be set in .env file")

# Provider configuration (defaults to OpenRouter)
PROVIDER_TYPE = os.getenv("PROVIDER_TYPE", "openrouter")
BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")

# Optional: Logfire configuration for observability
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

# =============================================================================
# CLI-configurable settings (with env var fallbacks)
# These are exported but not validated here - the CLI layer handles them
# =============================================================================

def get_model_name() -> str:
    """Get model name from env var, or None if not set."""
    return os.getenv("MODEL_NAME")

def get_random_seed() -> int:
    """Get random seed from env var, defaults to 0."""
    return int(os.getenv("RANDOM_SEED", "0"))

def get_story_file() -> str:
    """Get story file path from env var, or None if not set."""
    return os.getenv("STORY_FILE")

def get_manual_file() -> str:
    """Get manual file path from env var, or None if not set."""
    return os.getenv("MANUAL_FILE")
