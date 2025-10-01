"""
Configuration loading from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# API Configuration
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY must be set in .env file")

# Provider configuration (defaults to OpenRouter)
PROVIDER_TYPE = os.getenv("PROVIDER_TYPE", "openrouter")
BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")

# Model selection
MODEL_NAME = os.getenv("MODEL_NAME")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME must be set in .env file")

# Game configuration
STORY_FILE = os.getenv("STORY_FILE")
if not STORY_FILE:
    raise ValueError("STORY_FILE must be set in .env file (path to .z3/.z5/.z8 game file)")

MANUAL_FILE = os.getenv("MANUAL_FILE")
if not MANUAL_FILE:
    raise ValueError("MANUAL_FILE must be set in .env file (path to game manual markdown)")

# Optional: Random seed for reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "0"))

# Optional: Logfire configuration for observability
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_DIR = Path(__file__).parent / "prompts"

# Resolve story and manual paths (support both absolute and relative)
STORY_PATH = Path(STORY_FILE)
if not STORY_PATH.is_absolute():
    STORY_PATH = PROJECT_ROOT / STORY_FILE

MANUAL_PATH = Path(MANUAL_FILE)
if not MANUAL_PATH.is_absolute():
    MANUAL_PATH = PROJECT_ROOT / MANUAL_FILE

# Validation
if not STORY_PATH.exists():
    raise ValueError(f"Story file not found: {STORY_PATH}")

if not MANUAL_PATH.exists():
    raise ValueError(f"Manual file not found: {MANUAL_PATH}")
