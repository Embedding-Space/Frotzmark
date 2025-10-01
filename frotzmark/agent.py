"""
PydanticAI agent setup for interactive fiction gameplay.

Defaults to OpenRouter for maximum model flexibility, but can be
configured to use other providers via environment variables.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from .config import (
    API_KEY,
    BASE_URL,
    MODEL_NAME,
    MANUAL_PATH,
    PROMPT_DIR,
    PROVIDER_TYPE
)


def load_system_prompt() -> str:
    """
    Load and assemble the system prompt from files.

    Structure:
    - prompts/preamble.md (optional): Context and instructions
    - [MANUAL_FILE]: Game-specific documentation
    - prompts/postamble.md (optional): Additional guidance
    """
    parts = []

    # Load preamble
    preamble_file = PROMPT_DIR / "preamble.md"
    if preamble_file.exists():
        parts.append(preamble_file.read_text().strip())

    # Load game manual
    if MANUAL_PATH.exists():
        parts.append(MANUAL_PATH.read_text().strip())

    # Load postamble
    postamble_file = PROMPT_DIR / "postamble.md"
    if postamble_file.exists():
        parts.append(postamble_file.read_text().strip())

    return "\n\n".join(parts)


def create_agent() -> Agent:
    """
    Create and configure the PydanticAI agent.

    By default, uses OpenRouter as the provider for maximum model
    availability. Advanced users can override PROVIDER_TYPE and BASE_URL
    to use other OpenAI-compatible endpoints (vLLM, etc.).
    """

    if PROVIDER_TYPE == "openrouter":
        # OpenRouter via OpenAI-compatible provider
        provider = OpenAIProvider(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        model = OpenAIChatModel(MODEL_NAME, provider=provider)

    else:
        # Generic OpenAI-compatible endpoint
        # (for vLLM servers, custom deployments, etc.)
        provider = OpenAIProvider(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        model = OpenAIChatModel(MODEL_NAME, provider=provider)

    system_prompt = load_system_prompt()

    agent = Agent(
        model=model,
        system_prompt=system_prompt
    )

    return agent
