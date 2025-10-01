"""
GameSession: Clean interface for running Z-machine games programmatically.

This module provides a high-level API for:
- Loading Z-machine story files
- Running games turn-by-turn
- Controlling randomness for reproducibility
- Managing game state
"""

import random
import sys
from pathlib import Path

# Add vendored xyppy to path
vendor_path = Path(__file__).parent / "vendor"
sys.path.insert(0, str(vendor_path))

from xyppy import zenv, ops

from .screen import ProgrammaticScreen


class GameSession:
    """
    High-level interface for programmatic Z-machine control.

    Example usage:
        session = GameSession("games/zork1.z5", random_seed=0)
        output = session.start()
        print(output)

        while not session.is_finished():
            command = get_next_command()
            output = session.send_command(command)
            print(output)
    """

    def __init__(self, story_path: str, random_seed: int = 0):
        """
        Initialize a game session.

        Args:
            story_path: Path to the Z-machine story file (.z3, .z5, etc.)
            random_seed: Random seed for deterministic behavior (default: 0)
        """
        # Set random seed for deterministic behavior
        random.seed(random_seed)

        # Load story file
        with open(story_path, 'rb') as f:
            story_data = f.read()

        # Create a minimal args object (xyppy expects this)
        class Args:
            no_slow_scroll = True

        # Create environment
        self.env = zenv.Env(story_data, Args())

        # Replace the screen with our programmatic version
        self.screen = ProgrammaticScreen(self.env)
        self.env.screen = self.screen

        # Setup opcodes
        ops.setup_opcodes(self.env)

        self._started = False
        self._finished = False
        self._turn_count = 0

    def start(self) -> str:
        """
        Start the game and return initial output.

        Returns:
            The game's initial text (intro, location description, etc.)

        Raises:
            RuntimeError: If the game has already been started
        """
        if self._started:
            raise RuntimeError("Game already started")

        self._started = True

        # Run until we hit an input request
        try:
            while True:
                zenv.step(self.env)
        except StopIteration:
            # Game is waiting for input
            return self.screen.get_output()

    def send_command(self, command: str) -> str:
        """
        Send a command to the game and return the resulting output.

        Args:
            command: The command string (e.g., "open mailbox", "north")

        Returns:
            The game's response to the command

        Raises:
            RuntimeError: If the game hasn't been started yet
        """
        if not self._started:
            raise RuntimeError("Game not started - call start() first")

        if self._finished:
            raise RuntimeError("Game has finished")

        # Queue the command
        self.screen.queue_command(command)
        self._turn_count += 1

        # Resume execution until next input request
        try:
            while True:
                zenv.step(self.env)
        except StopIteration:
            # Game is waiting for input
            return self.screen.get_output()
        except SystemExit:
            # Game has quit
            self._finished = True
            return self.screen.get_output()

    def is_finished(self) -> bool:
        """
        Check if the game has finished (quit/died).

        Returns:
            True if the game is over, False otherwise
        """
        return self._finished

    def get_turn_count(self) -> int:
        """
        Get the number of commands sent so far.

        Returns:
            The turn count (number of times send_command() was called)
        """
        return self._turn_count

    @property
    def memory(self):
        """
        Direct access to Z-machine memory (for advanced use cases).

        Returns:
            The memory array from the xyppy environment
        """
        return self.env.mem

    @property
    def program_counter(self) -> int:
        """
        Current program counter value (for debugging/analysis).

        Returns:
            The current PC value
        """
        return self.env.pc
