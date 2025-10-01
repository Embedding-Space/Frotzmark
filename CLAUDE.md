# Frotzmark

**10/1/2025**

Frotzmark pits large language models against vintage computer games to illuminate gaps between learned knowledge and apply-able knowledge.

(We're working on the language.)

Here's the program in pseudocode:

intro = game.initialize()
output = model.run(intro)
print(output)
BEGIN LOOP
    result = game.run(output)
    print(result)
    output = model.run(result)
END LOOP

## Technical details

- Use `uv` exclusively
- Use vendored `xyppy` to run Z-machine games (see Dependencies below)
- Use PydanticAI to abstract away all the LLM interface stuff

## Dependencies

### Vendored Code

This project vendors [xyppy](https://github.com/theinternetftw/xyppy), a Python Z-machine interpreter, located in `frotzmark/vendor/xyppy/`. We vendor it (rather than using it as a dependency) to ensure long-term reproducibility of research results. See `frotzmark/vendor/README.md` for details.

## How it works (from Zorkmark)

PydanticAI makes this stupid simple:

1. **Create an agent** with a system prompt and model config:
   ```python
   from pydantic_ai import Agent
   from pydantic_ai.models.openai import OpenAIChatModel

   agent = Agent(
       model=OpenAIChatModel('some-model'),
       system_prompt="You are playing Zork..."
   )
   ```

2. **Run the game loop** - just pass game output as the user prompt:
   ```python
   message_history = []
   game_output = game.initial_text

   while True:
       # Model sees game output as user message
       result = agent.run_sync(game_output, message_history=message_history)

       # Model's response is the command
       command = result.output

       # Execute command, get new game output
       game_output = game.send_command(command)

       # Update history for next turn
       message_history = result.all_messages()
   ```

That's it. PydanticAI handles:
- Message history management
- Provider abstraction (OpenAI, Anthropic, etc.)
- Serialization for checkpointing
- All the async/streaming complexity

The game output literally becomes the user prompt. The model's output literally becomes the game command. No tools, no complex state management, just a conversation.
