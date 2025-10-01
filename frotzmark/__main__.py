"""
Main entry point for Frotzmark.
"""

import sys
import signal
import re
from typing import Any

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

from .agent import create_agent
from .game_session import GameSession
from .config import (
    LOGFIRE_TOKEN,
    MODEL_NAME,
    STORY_PATH,
    RANDOM_SEED
)


def signal_handler(sig: int, frame: Any) -> None:
    """Handle Ctrl-C gracefully."""
    print("\n\nFrotzmark session ended.")
    sys.exit(0)


def strip_thinking_tags(text: str) -> str:
    """
    Remove <thinking>...</thinking> tags and their contents from text.

    Returns only the command that should be sent to the game.
    """
    # Remove thinking tags and everything between them
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    # Strip whitespace
    cleaned = cleaned.strip()
    # If there are multiple lines, take only the first one
    if '\n' in cleaned:
        cleaned = cleaned.split('\n')[0]
    return cleaned.strip()


def extract_thinking(text: str) -> str:
    """Extract the content inside <thinking> tags."""
    match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def main() -> None:
    """Main Frotzmark loop."""

    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    print("🎮 Frotzmark: LLMs vs Interactive Fiction")
    print(f"Model: {MODEL_NAME}")
    print(f"Game: {STORY_PATH.name}")
    print("Press Ctrl-C to exit\n")

    # Configure Logfire if available and token is set
    if LOGFIRE_AVAILABLE and LOGFIRE_TOKEN:
        print("Configuring Logfire observability...")
        logfire.configure(
            token=LOGFIRE_TOKEN,
            service_name="frotzmark",
            console=False
        )
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx()
        print("Logfire instrumentation enabled.\n")

    try:
        # Initialize game session and agent
        print("Initializing game...")
        session = GameSession(str(STORY_PATH), random_seed=RANDOM_SEED)
        agent = create_agent()

        print("Starting game...\n")

        # Start the game and get initial output
        game_output = session.start()
        print(game_output)
        print()

        # Message history for conversation context
        message_history = []
        turn_number = 0

        # Main game loop
        span_context = (
            logfire.span('frotzmark_game_session', model=MODEL_NAME)
            if LOGFIRE_AVAILABLE and LOGFIRE_TOKEN
            else None
        )

        with span_context if span_context else _nullcontext():
            try:
                while not session.is_finished():
                    turn_number += 1

                    # Give the game output to the model as the user prompt
                    result = agent.run_sync(game_output, message_history=message_history)

                    # Get the model's full output
                    model_output = result.output

                    # Extract and display thinking
                    thinking = extract_thinking(model_output)
                    if thinking:
                        print(f"<thinking>\n{thinking}\n</thinking>\n")

                    # Strip thinking tags to get just the command
                    command = strip_thinking_tags(model_output)

                    if not command:
                        print("[Game ended - no command provided]")
                        break

                    # Execute the command
                    print(f">{command}")
                    game_output = session.send_command(command)
                    print(game_output)
                    print()

                    # Update message history for next turn
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\nFrotzmark session ended.")
                # Exit cleanly from the span without re-raising
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


class _nullcontext:
    """Dummy context manager for when Logfire is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
