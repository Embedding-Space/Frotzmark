"""
Main entry point for Frotzmark.
"""

import sys
import signal
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
from pydantic_core import to_json, to_jsonable_python

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

from .agent import create_agent
from .game_session import GameSession
from .config import (
    LOGFIRE_TOKEN,
    get_model_name,
    get_random_seed,
)


def signal_handler(sig: int, frame: Any) -> None:
    """Handle Ctrl-C gracefully."""
    click.echo("\n\nFrotzmark session ended.")
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


def find_manual(story_path: Path) -> Optional[Path]:
    """
    Auto-discover manual file next to story file.

    Looks for a .md file with the same name as the story file.
    E.g., zork1.z5 → zork1.md

    Args:
        story_path: Path to the story file

    Returns:
        Path to manual file if found, None otherwise
    """
    manual_path = story_path.with_suffix('.md')
    if manual_path.exists():
        return manual_path
    return None


def wrap_text(text: str, width: Optional[int] = None) -> str:
    """
    Wrap text to terminal width for better readability.

    Args:
        text: Text to wrap
        width: Terminal width (auto-detected if None)

    Returns:
        Wrapped text
    """
    if width is None:
        try:
            width, _ = click.get_terminal_size()  # type: ignore
        except:
            width = 80  # fallback

    # Don't wrap if text is already short enough
    if len(text) <= width:
        return text

    # Simple word-wrapping (doesn't break words)
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph.strip():
            lines.append('')
            continue

        words = paragraph.split()
        current_line = []
        current_length = 0

        for word in words:
            word_len = len(word) + (1 if current_line else 0)
            if current_length + word_len <= width:
                current_line.append(word)
                current_length += word_len
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

    return '\n'.join(lines)


def save_checkpoint(
    checkpoint_path: Path,
    turn: int,
    message_history: list,
    next_prompt: str,
    model_name: str,
    story_path: Path,
) -> None:
    """
    Save a checkpoint to resume later.

    Creates two files:
    - checkpoint.json: Metadata and message history
    - checkpoint.sav: Z-machine game state (Quetzal format)
    """
    # Save game state to Quetzal file
    save_file = checkpoint_path.with_suffix('.sav')

    # Serialize message history
    serialized_history = to_jsonable_python(message_history)

    # Create checkpoint metadata
    checkpoint_data = {
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'turn': turn,
        'model': model_name,
        'story_file': str(story_path),
        'save_file': str(save_file),
        'next_prompt': next_prompt,
        'message_history': serialized_history,
    }

    # Write checkpoint JSON
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """
    Load a checkpoint file.

    Returns:
        Checkpoint data dict, or None if file doesn't exist
    """
    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path, 'r') as f:
        return json.load(f)


@click.command()
@click.argument('story', type=click.Path(exists=True, path_type=Path), required=False)
@click.argument('manual', type=click.Path(exists=True, path_type=Path), required=False)
@click.option('--model', '-m', help='Model to use (e.g., google/gemini-2.5-flash-lite)')
@click.option('--seed', '-s', type=int, help='Random seed for reproducibility')
@click.option('--wrap/--no-wrap', default=True, help='Wrap output to terminal width')
@click.option('--resume', '-r', 'resume_file', type=click.Path(exists=True, path_type=Path), help='Resume from checkpoint file')
@click.option('--checkpoint', '-c', 'checkpoint_file', type=click.Path(path_type=Path), default='checkpoint.json', help='Checkpoint file path (default: checkpoint.json)')
def main(
    story: Optional[Path],
    manual: Optional[Path],
    model: Optional[str],
    seed: Optional[int],
    wrap: bool,
    resume_file: Optional[Path],
    checkpoint_file: Path,
) -> None:
    """
    Frotzmark: LLMs vs Interactive Fiction

    STORY: Path to Z-machine story file (.z3, .z5, .z8) [required unless --resume]

    MANUAL: Path to game manual (markdown). If omitted, Frotzmark will
    look for a .md file with the same name as the story file.

    Use --resume to continue from a checkpoint.
    """

    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    # Handle resume mode
    checkpoint_data = None
    if resume_file:
        click.echo(f"📂 Resuming from checkpoint: {resume_file.name}\n")
        checkpoint_data = load_checkpoint(resume_file)
        if not checkpoint_data:
            click.echo("Error: Could not load checkpoint file", err=True)
            sys.exit(1)

        # Override story and model from checkpoint
        story = Path(checkpoint_data['story_file'])
        model_name = checkpoint_data['model']
        random_seed = seed if seed is not None else get_random_seed()

        # Manual still needs to be discovered/specified
        if manual is None:
            manual = find_manual(story)
    else:
        # Normal mode - story is required
        if story is None:
            click.echo("Error: STORY argument required (unless using --resume)", err=True)
            sys.exit(1)

        # Resolve configuration with CLI overrides
        model_name = model or get_model_name()
        if not model_name:
            click.echo("Error: Model must be specified via --model or MODEL_NAME env var", err=True)
            sys.exit(1)

        random_seed = seed if seed is not None else get_random_seed()

        # Auto-discover manual if not provided
        if manual is None:
            manual = find_manual(story)
            if manual:
                click.echo(f"📖 Auto-discovered manual: {manual.name}")

    # Display startup info
    click.echo("🎮 Frotzmark: LLMs vs Interactive Fiction")
    click.echo(f"Model: {model_name}")
    click.echo(f"Game: {story.name}")
    if manual:
        click.echo(f"Manual: {manual.name}")
    click.echo("Press Ctrl-C to exit\n")

    # Configure Logfire if available and token is set
    if LOGFIRE_AVAILABLE and LOGFIRE_TOKEN:
        click.echo("Configuring Logfire observability...")
        logfire.configure(
            token=LOGFIRE_TOKEN,
            service_name="frotzmark",
            console=False
        )
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx()
        click.echo("Logfire instrumentation enabled.\n")

    try:
        # Initialize game session and agent
        click.echo("Initializing game...")
        session = GameSession(str(story), random_seed=random_seed)
        agent = create_agent(model_name, manual_path=manual if manual else None)

        # Handle checkpoint restore or fresh start
        if checkpoint_data:
            # Restore from checkpoint
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            save_file = Path(checkpoint_data['save_file'])
            if not session.restore_state(str(save_file)):
                click.echo("Error: Failed to restore game state", err=True)
                sys.exit(1)

            # Restore message history
            message_history = ModelMessagesTypeAdapter.validate_python(
                checkpoint_data['message_history']
            )
            turn_number = checkpoint_data['turn']
            game_output = checkpoint_data['next_prompt']

            click.echo(f"Resumed at turn {turn_number}\n")
            output_text = wrap_text(game_output) if wrap else game_output
            click.echo(output_text)
            click.echo()
        else:
            # Fresh start
            click.echo("Starting game...\n")

            # Start the game and get initial output
            game_output = session.start()
            output_text = wrap_text(game_output) if wrap else game_output
            click.echo(output_text)
            click.echo()

            # Message history for conversation context
            message_history = []
            turn_number = 0

        # Main game loop
        span_context = (
            logfire.span('frotzmark_game_session', model=model_name)
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
                        thinking_text = wrap_text(thinking) if wrap else thinking
                        click.echo(f"<thinking>\n{thinking_text}\n</thinking>\n")

                    # Strip thinking tags to get just the command
                    command = strip_thinking_tags(model_output)

                    if not command:
                        click.echo("[Game ended - no command provided]")
                        break

                    # Execute the command
                    click.echo(f">{command}")
                    game_output = session.send_command(command)
                    output_text = wrap_text(game_output) if wrap else game_output
                    click.echo(output_text)
                    click.echo()

                    # Display turn count
                    click.secho(f"[Turn {turn_number}]", fg='cyan', dim=True)
                    click.echo()

                    # Update message history for next turn
                    message_history = result.all_messages()

                    # Save checkpoint after each turn
                    session.save_state(str(checkpoint_file.with_suffix('.sav')))
                    save_checkpoint(
                        checkpoint_path=checkpoint_file,
                        turn=turn_number,
                        message_history=message_history,
                        next_prompt=game_output,
                        model_name=model_name,
                        story_path=story,
                    )

            except KeyboardInterrupt:
                click.echo("\n\nFrotzmark session ended.")
                # Exit cleanly from the span without re-raising

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
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
