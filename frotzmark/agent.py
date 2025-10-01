"""
PydanticAI agent setup for interactive fiction gameplay.

Defaults to OpenRouter for maximum model flexibility, but can be
configured to use other providers via environment variables.
"""

from pathlib import Path
from typing import Optional
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from .config import (
    API_KEY,
    BASE_URL,
    PROMPT_DIR,
    PROVIDER_TYPE
)


def load_system_prompt(manual_path: Optional[Path] = None) -> str:
    """
    Load and assemble the system prompt from files.

    Structure:
    - prompts/preamble.md (optional): Context and instructions
    - [manual_path]: Game-specific documentation (optional)
    - prompts/postamble.md (optional): Additional guidance

    Args:
        manual_path: Path to game manual markdown file (optional)
    """
    parts = []

    # Load preamble
    preamble_file = PROMPT_DIR / "preamble.md"
    if preamble_file.exists():
        parts.append(preamble_file.read_text().strip())

    # Load game manual if provided
    if manual_path and manual_path.exists():
        parts.append(manual_path.read_text().strip())

    # Load postamble
    postamble_file = PROMPT_DIR / "postamble.md"
    if postamble_file.exists():
        parts.append(postamble_file.read_text().strip())

    return "\n\n".join(parts)


def create_agent(
    model_name: str,
    manual_path: Optional[Path] = None,
    reasoning_effort: Optional[str] = None
) -> Agent:
    """
    Create and configure the PydanticAI agent.

    By default, uses OpenRouter as the provider for maximum model
    availability. Advanced users can override PROVIDER_TYPE and BASE_URL
    to use other OpenAI-compatible endpoints (vLLM, etc.).

    Args:
        model_name: Name of the model to use (e.g., 'google/gemini-2.5-flash-lite')
        manual_path: Path to game manual markdown file (optional)
        reasoning_effort: Reasoning effort level for OpenRouter ('low', 'medium', 'high') (optional)
    """

    if PROVIDER_TYPE == "openrouter":
        # OpenRouter via OpenAI-compatible provider
        provider = OpenAIProvider(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        model = OpenAIChatModel(model_name, provider=provider)

    else:
        # Generic OpenAI-compatible endpoint
        # (for vLLM servers, custom deployments, etc.)
        provider = OpenAIProvider(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        model = OpenAIChatModel(model_name, provider=provider)

    system_prompt = load_system_prompt(manual_path)

    # Configure model settings with reasoning if requested
    model_settings = None
    if reasoning_effort and PROVIDER_TYPE == "openrouter":
        model_settings = ModelSettings(
            extra_body={
                'reasoning': {
                    'effort': reasoning_effort,
                    'exclude': False  # Include reasoning output in response
                }
            }
        )

    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        model_settings=model_settings
    )

    return agent
