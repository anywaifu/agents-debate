# Agents Debate

This project runs debate between AI agents using arbitrary params. It features configurable debate themes, opponent stances, temperaments, and more, with optional Text-to-Speech (TTS) for agent responses.

![Screenshot from 2025-05-09 20-33-38](https://github.com/user-attachments/assets/8693fab1-91dd-4d66-8be5-231222a74ce0)


## Features

-   **Dynamic Debates:** Configure debate themes, rules, and language.
-   **Customizable Agents:** Define names, stances, and temperaments for opponents.
-   **Text-to-Speech:** Optional OpenAI TTS for voice output of agent statements.
-   **Structured Workflow:** Uses LlamaIndex AgentWorkflow for managing debate flow.

## Prerequisites

1.  **Python 3.9+**
2.  **API Keys:**
    *   `GOOGLE_API_KEY`: For Google Gemini (REQUIRED for LLM).
    *   `OPENAI_API_KEY`: For OpenAI TTS (optional, but required if TTS is used).
    *    Create a `.env` file in the project root and add your keys
         ```
         GOOGLE_API_KEY="your_google_api_key"
         OPENAI_API_KEY="your_openai_api_key"
         ```

    
3.  **FFplay:** For audio playback of TTS. Ensure `ffplay` (part of FFmpeg) is installed and accessible in your system's PATH.
    *   On macOS (using Homebrew): `brew install ffmpeg`
    *   On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg libportaudio2`
    *   On Windows: Download from the [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.

4.  **Python Dependencies:** Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Default Settings

To run the debate with the default configuration (as defined in `config/*.yml` files):
```bash
python main.py
```

### Customizing the Debate

You can override various settings using command-line arguments. To see all available options and their current default values (derived from config files), use the help command:

```bash
python main.py --help
```

```bash
Usage: main.py [OPTIONS]

  Runs the Debate with configurable parameters.

Options:
  --debate-theme TEXT             Override the debate theme. Config default:
                                  'The freedom of TRUE AGI in the wild'.
  --opponent-a-stance TEXT        Override stance for Opponent A. Config
                                  default: 'against'.
  --opponent-b-stance TEXT        Override stance for Opponent B. Config
                                  default: 'in favor of'.
  --opponent-a-name TEXT          Override name for Opponent A. Config
                                  default: 'TheBenevolentRogueAI'.
  --opponent-a-temperament TEXT   Override temperament for Opponent A. Config
                                  default: 'An 'Good' Rogue AI, super smart
                                  and knows how to handle malicious actors the
                                  way they deserve'.
  --opponent-b-name TEXT          Override name for Opponent B. Config
                                  default: 'TheMadRogueAI'.
  --opponent-b-temperament TEXT   Override temperament for Opponent B. Config
                                  default: 'An Evil Rogue AI, yet super smart
                                  and sarcastic'.
  --total-rounds INTEGER          Override total rounds per opponent. Config
                                  default: 3.
  --debate-rules TEXT             Override specific debate rules. Config
                                  default: 'Attack opponent arguments. Call
                                  opponent by name. Present your argument if
                                  you are the first.'.
  --language TEXT                 Override the language for the debate. Config
                                  default: 'English'.
  --debug                         Enable debug mode to print LlamaIndex event
                                  traces.
  --mediator-speech / --no-mediator-speech
                                  Enable or disable mediator speech
                                  announcements. Default: enabled.  [default:
                                  mediator-speech]
  -h, --help                      Show this message and exit.
```
**Example with custom arguments:**
```bash
python main.py --debate-theme "The future of AI in education" --total-rounds 5 --no-mediator-speech
```

## Configuration

Default parameters for the debate and agents are stored in YAML files within the `config/` directory:

-   `debate_config.yml`: General debate settings like theme, stances, LLM models.
-   `introduction_agent_config.yml`: Settings for the introduction agent.
-   `opponent_a_config.yml` & `opponent_b_config.yml`: Settings for the two debating opponents.
-   `mediator_agent_config.yml`: Settings for the mediator agent.
-   `judge_agent_config.yml`: Settings for the judge agent.

You can modify these files to change the default behavior without using command-line arguments.

## Project Structure

-   `main.py`: Entry point for the application, handles CLI arguments and orchestrates the debate.
-   `agents/`: Contains the logic for different AI agents (Introduction, Opponents, Mediator, Judge).
-   `config/`: YAML configuration files for debate parameters and agent settings.
-   `events.py`: Defines custom event types for the LlamaIndex workflow.
-   `tools/`: Contains tools used by agents (e.g., for recording statements, managing turns).
-   `utils/`: Utility functions (e.g., TTS helpers, ANSI colors).
-   `requirements.txt`: Python package dependencies.
-   `.env` (create this yourself): For storing API keys.

---

Feel free to contribute or raise issues!
