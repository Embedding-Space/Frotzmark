"""
Tests for GameSession using canonical Zork 1 transcript.

This validates that our Z-machine wrapper produces correct output
by comparing against a known-good transcript from sample.txt.
"""

import pytest
from pathlib import Path
from frotzmark.game_session import GameSession


def parse_transcript(transcript_path: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Parse a transcript file into initial output and command/response pairs.

    Format:
        Initial game output
        >command
        response
        >command
        response
        ...

    Returns:
        (initial_output, [(command, response), ...])
    """
    with open(transcript_path, 'r') as f:
        content = f.read()

    # Split on lines starting with ">"
    parts = content.split('\n>')

    # First part is initial output
    initial = parts[0].strip()

    # Rest are command/response pairs
    pairs = []
    for part in parts[1:]:
        lines = part.split('\n', 1)
        command = lines[0]
        response = lines[1].strip() if len(lines) > 1 else ""
        pairs.append((command, response))

    return initial, pairs


@pytest.fixture
def transcript():
    """Load and parse the canonical Zork 1 transcript."""
    transcript_path = Path(__file__).parent.parent / "sample.txt"
    return parse_transcript(str(transcript_path))


@pytest.fixture
def session():
    """Create a fresh GameSession for each test."""
    story_path = Path(__file__).parent.parent / "games" / "zork1.z5"
    return GameSession(str(story_path), random_seed=0)


def test_initial_output(session, transcript):
    """Test that game initialization produces correct output."""
    expected_initial, _ = transcript
    actual_initial = session.start()

    # Strip the trailing ">" prompt for comparison since that's UI chrome
    actual_clean = actual_initial.rstrip('\n>').strip()
    expected_clean = expected_initial.strip()

    assert actual_clean == expected_clean


def test_each_command(session, transcript):
    """Test that each command produces the expected response."""
    _, pairs = transcript

    # Start the game
    session.start()

    for i, (command, expected_response) in enumerate(pairs):
        actual_output = session.send_command(command)

        # Clean up the output:
        # - Remove trailing prompt ">"
        # - Strip whitespace
        actual_clean = actual_output.rstrip('\n>').strip()
        expected_clean = expected_response.strip()

        assert actual_clean == expected_clean, (
            f"Command {i+1} ('{command}') produced incorrect output.\n"
            f"Expected:\n{expected_clean}\n\n"
            f"Got:\n{actual_clean}"
        )


def test_turn_count(session, transcript):
    """Test that turn counting works correctly."""
    _, pairs = transcript

    session.start()
    assert session.get_turn_count() == 0

    for i, (command, _) in enumerate(pairs):
        session.send_command(command)
        assert session.get_turn_count() == i + 1
