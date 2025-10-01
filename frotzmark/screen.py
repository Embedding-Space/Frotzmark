"""
Custom Screen implementation for programmatic control of Z-machine I/O.

This replaces xyppy's terminal-based screen with one that:
- Captures output to a buffer
- Reads input from a command queue
- Raises StopIteration when waiting for input
"""


class ProgrammaticScreen:
    """
    Screen implementation for xyppy that enables programmatic control.

    Instead of reading from stdin and writing to stdout, this implementation:
    - Accumulates game output in a buffer
    - Pulls commands from a queue
    - Signals when the game is waiting for input
    """

    def __init__(self, env):
        """
        Initialize the screen.

        Args:
            env: The xyppy Env object (required by xyppy's screen interface)
        """
        self.env = env
        self.output_buffer = []
        self.command_queue = []
        self.waiting_for_input = False
        self.commands_dispensed = 0  # Track commands given out this turn

    def queue_command(self, command: str):
        """
        Queue a command for the game to process.

        Args:
            command: The command string (e.g., "open mailbox")
        """
        self.command_queue.append(command)
        self.waiting_for_input = False

    def get_output(self) -> str:
        """
        Get accumulated output and clear the buffer.

        Returns:
            All text output since the last call to get_output(),
            with trailing prompt ('>') and whitespace removed.
        """
        result = ''.join(self.output_buffer)
        self.output_buffer.clear()

        # Strip trailing prompt and whitespace for cleaner output
        result = result.rstrip('\n>')
        result = result.rstrip()

        return result

    def write(self, text: str):
        """
        Capture output to buffer.

        Args:
            text: Text output from the game
        """
        self.output_buffer.append(text)

    def get_line_of_input(self, prompt: str = '', prefilled: str = '') -> str:
        """
        Called by the game when it needs input.

        Instead of blocking on stdin, pulls from the command queue.
        If no commands are queued, returns empty string (which the game
        will reject and ask again).

        Args:
            prompt: Optional prompt to display (written to output)
            prefilled: Pre-filled text (may contain leftover text from buffer)

        Returns:
            The next command from the queue, or empty string if queue is empty
        """
        if prompt:
            self.write(prompt)

        if not self.command_queue:
            self.waiting_for_input = True
            # Always raise StopIteration when we have no commands
            # This prevents the game from processing empty input at any point
            raise StopIteration("Waiting for input")

        command = self.command_queue.pop(0)
        self.waiting_for_input = False
        self.commands_dispensed += 1  # Increment counter

        # Don't echo the command ourselves - the game handles that via
        # its own input buffering/display mechanism

        return command

    # No-op methods required by xyppy's screen interface

    def flush(self):
        """No-op for programmatic control."""
        pass

    def first_draw(self):
        """No-op for programmatic control."""
        pass

    def update_seen_lines(self):
        """No-op for programmatic control."""
        pass
