import asyncio
import os

from typing import Optional

import click
import yaml
from pathlib import Path
from llama_index.core.workflow import Context # type: ignore

from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput # type: ignore
from llama_index.llms.google_genai import GoogleGenAI # type: ignore

from dotenv import load_dotenv

from events import OpponentStatementEvent, IntroductionCompleteEvent, JudgmentDeliveredEvent, MediatorAnnouncementEvent
from agents.introduction_agent import create_introduction_agent
from agents.opponent_agents import create_opponent_agent
from agents.mediator_agent import create_mediator_agent
from agents.judge_agent import create_judge_agent
from utils.ansi_colors import RESET, RED, YELLOW, BLUE, MAGENTA, CYAN


CONFIG_PATH = Path(__file__).parent / "config"

def load_config(file_name: str) -> dict:
    """Loads a YAML configuration file."""
    with open(CONFIG_PATH / file_name, 'r') as f:
        return yaml.safe_load(f)

load_dotenv()


if not os.getenv("GOOGLE_API_KEY"): # type: ignore
    raise ValueError("GOOGLE_API_KEY environment variable not set for Gemini.")


async def setup_and_run_debate(
    debate_theme_override: Optional[str],
    opponent_a_stance_override: Optional[str],
    opponent_b_stance_override: Optional[str],
    opponent_a_name_override: Optional[str],
    opponent_a_temperament_override: Optional[str],
    opponent_b_name_override: Optional[str],
    opponent_b_temperament_override: Optional[str],
    total_rounds_override: Optional[int],
    debate_rules_override: Optional[str],
    language_override: Optional[str],
    debug_enabled: bool,
    mediator_speech_enabled: bool,
):
    if debug_enabled:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("requests").setLevel(logging.INFO)
        logging.getLogger("urllib3").setLevel(logging.INFO)


    # Load configurations
    debate_cfg = load_config("debate_config.yml")
    intro_agent_cfg = load_config("introduction_agent_config.yml")
    opponent_a_cfg = load_config("opponent_a_config.yml")
    opponent_b_cfg = load_config("opponent_b_config.yml")
    mediator_agent_cfg = load_config("mediator_agent_config.yml")
    judge_agent_cfg = load_config("judge_agent_config.yml")

    # Initialize with defaults from config files, overridden by CLI arguments if provided
    debate_theme = debate_theme_override or debate_cfg["debate_theme"]
    opponent_a_stance = opponent_a_stance_override or debate_cfg["opponent_a_stance"]
    opponent_b_stance = opponent_b_stance_override or debate_cfg["opponent_b_stance"]
    opponent_a_name_idea = opponent_a_name_override or opponent_a_cfg["default_name_idea"]
    opponent_a_temperament = opponent_a_temperament_override or opponent_a_cfg["default_temperament"]
    opponent_b_name_idea = opponent_b_name_override or opponent_b_cfg["default_name_idea"]
    opponent_b_temperament = opponent_b_temperament_override or opponent_b_cfg["default_temperament"]
    total_rounds = total_rounds_override if total_rounds_override is not None else debate_cfg["total_rounds"]
    debate_rules_input = debate_rules_override or debate_cfg["debate_rules"]
    language = language_override or debate_cfg["language"]

    llm = GoogleGenAI(model=debate_cfg["llm_model_gemini"], api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"{CYAN}--- Debate Setup ---{RESET}")
    print(f"Effective Debate Configuration:")
    print(f"  Debate Theme: {debate_theme}")
    print(f"  Opponent A ('{opponent_a_name_idea}') Stance: {opponent_a_stance}")
    print(f"  Opponent A Temperament: {opponent_a_temperament}")
    print(f"  Opponent B ('{opponent_b_name_idea}') Stance: {opponent_b_stance}")
    print(f"  Opponent B Temperament: {opponent_b_temperament}")
    print(f"  Total Rounds: {total_rounds}")
    print(f"  Debate Rules: {debate_rules_input}")
    print(f"  Language: {language}")
    print(f"  LLM Model: {debate_cfg['llm_model_gemini']}")
    print(f"  TTS Model: {debate_cfg['tts_model_openai']}")
    print("---")

    introduction_agent = create_introduction_agent(
        llm=llm,
        config=intro_agent_cfg,
        debate_theme=debate_theme,
        language=language,
        debate_rules=debate_rules_input
    )

    opponent_a_agent = create_opponent_agent(
        llm=llm,
        config=opponent_a_cfg,
        name=opponent_a_name_idea,
        role_description=f"You argue persuasively {opponent_a_stance} the debate theme: '{debate_theme}'.",
        temperament=opponent_a_temperament,
        debate_theme=debate_theme,
        language=language,
        debate_rules=debate_rules_input,
    )
    opponent_b_agent = create_opponent_agent(
        llm=llm,
        config=opponent_b_cfg,
        name=opponent_b_name_idea,
        role_description=f"You argue persuasively {opponent_b_stance} the debate theme: '{debate_theme}'.",
        temperament=opponent_b_temperament,
        debate_theme=debate_theme,
        language=language,
        debate_rules=debate_rules_input,
    )
    judge_agent = create_judge_agent(
        llm=llm,
        config=judge_agent_cfg,
        language=language
    )

    mediator_agent = create_mediator_agent(
        llm=llm,
        config=mediator_agent_cfg,
        opponent_a_name=opponent_a_agent.name,
        opponent_b_name=opponent_b_agent.name,
        judge_name=judge_agent.name,
        language=language,
        total_rounds=total_rounds,
        debate_rules=debate_rules_input,
        mediator_speech_enabled=mediator_speech_enabled,
    )

    initial_state = {
        f"{opponent_a_agent.name}_turns": 0,
        f"{opponent_b_agent.name}_turns": 0,
        "current_speaker": "none",
        "total_rounds": total_rounds,
        "debate_theme": debate_theme,
        "opponent_a_name": opponent_a_agent.name,
        "opponent_b_name": opponent_b_agent.name,
        "debate_rules": debate_rules_input,
        "tts_config": {
            "model": debate_cfg["tts_model_openai"],
            "voices": {
                introduction_agent.name: intro_agent_cfg["tts_voice"],
                opponent_a_agent.name: opponent_a_cfg["tts_voice"],
                opponent_b_agent.name: opponent_b_cfg["tts_voice"],
                judge_agent.name: judge_agent_cfg["tts_voice"],
                mediator_agent.name: mediator_agent_cfg["tts_voice"],
            }
        }
    }

    debate_workflow = AgentWorkflow(
        agents=[introduction_agent, opponent_a_agent, opponent_b_agent, mediator_agent, judge_agent],
        root_agent=introduction_agent.name,
        initial_state=initial_state,
    )

    print(f"{CYAN}--- Starting Debate ---{RESET}")

    ctx = Context(debate_workflow)

    handler = debate_workflow.run(
        user_msg="Please start and manage the political debate according to the rules.",
        ctx=ctx
    )

    async for event in handler.stream_events():
        if isinstance(event, IntroductionCompleteEvent):
            print(f"\n{MAGENTA}📜 Introduction ({event.agent_name} on '{debate_theme}'):{RESET}\n  {event.introduction_message}")
        elif isinstance(event, OpponentStatementEvent):
            color = BLUE if event.speaker_name == opponent_a_agent.name else RED
            print(f"\n{color}💬 {event.speaker_name}:{RESET}\n  {event.statement}") # type: ignore
        elif isinstance(event, MediatorAnnouncementEvent):
            print(f"\n{CYAN}🗣️  {event.agent_name} (Mediator):{RESET}\n  {event.announcement_text}")
        elif isinstance(event, JudgmentDeliveredEvent):
            print(f"\n{YELLOW}⚖️ Judge ({event.judge_name}):{RESET}\n  {event.judgment_text}")
            print(f"{YELLOW}🏆 Declared Winner:{RESET} {event.winner}")
            print()


    print(f"\n{CYAN}--- End of Debate ---{RESET}")

    try:
        final_state = await ctx.get("state")
        print(f"\n{CYAN}--- Final Debate State ---{RESET}")
        for key, value in final_state.items(): # type: ignore
            print(f"{key}: {value}")
    except ValueError:
        print(f"\n{CYAN}--- Could not retrieve final debate state. ---{RESET}")

if __name__ == "__main__":
    # Load configs here to use in help messages for click
    _debate_cfg_defaults = load_config("debate_config.yml")
    _opponent_a_cfg_defaults = load_config("opponent_a_config.yml")
    _opponent_b_cfg_defaults = load_config("opponent_b_config.yml")

    @click.command(context_settings=dict(help_option_names=['-h', '--help']))
    @click.option(
        "--debate-theme", "debate_theme_override",
        default=None, type=str,
        help=f"Override the debate theme. Config default: '{_debate_cfg_defaults['debate_theme']}'.",
        show_default=False, # Show custom help instead
    )
    @click.option(
        "--opponent-a-stance", "opponent_a_stance_override",
        default=None, type=str,
        help=f"Override stance for Opponent A. Config default: '{_debate_cfg_defaults['opponent_a_stance']}'.",
        show_default=False,
    )
    @click.option(
        "--opponent-b-stance", "opponent_b_stance_override",
        default=None, type=str,
        help=f"Override stance for Opponent B. Config default: '{_debate_cfg_defaults['opponent_b_stance']}'.",
        show_default=False,
    )
    @click.option(
        "--opponent-a-name", "opponent_a_name_override",
        default=None, type=str,
        help=f"Override name for Opponent A. Config default: '{_opponent_a_cfg_defaults['default_name_idea']}'.",
        show_default=False,
    )
    @click.option(
        "--opponent-a-temperament", "opponent_a_temperament_override",
        default=None, type=str,
        help=f"Override temperament for Opponent A. Config default: '{_opponent_a_cfg_defaults['default_temperament']}'.",
        show_default=False,
    )
    @click.option(
        "--opponent-b-name", "opponent_b_name_override",
        default=None, type=str,
        help=f"Override name for Opponent B. Config default: '{_opponent_b_cfg_defaults['default_name_idea']}'.",
        show_default=False,
    )
    @click.option(
        "--opponent-b-temperament", "opponent_b_temperament_override",
        default=None, type=str,
        help=f"Override temperament for Opponent B. Config default: '{_opponent_b_cfg_defaults['default_temperament']}'.",
        show_default=False,
    )
    @click.option(
        "--total-rounds", "total_rounds_override",
        default=None, type=int,
        help=f"Override total rounds per opponent. Config default: {_debate_cfg_defaults['total_rounds']}.",
        show_default=False,
    )
    @click.option(
        "--debate-rules", "debate_rules_override",
        default=None, type=str,
        help=f"Override specific debate rules. Config default: '{_debate_cfg_defaults['debate_rules']}'.",
        show_default=False,
    )
    @click.option(
        "--language", "language_override",
        default=None, type=str,
        help=f"Override the language for the debate. Config default: '{_debate_cfg_defaults['language']}'.",
        show_default=False,
    )
    @click.option(
        "--debug",
        "debug_enabled",
        is_flag=True,
        help="Enable debug mode to print request traces.",
        show_default=True,
    )
    @click.option(
        "--mediator-speech/--no-mediator-speech",
        "mediator_speech_enabled",
        default=True,
        help="Enable or disable mediator speech announcements. Default: enabled.",
        show_default=True,
    )
    def cli_main(
        debate_theme_override: Optional[str],
        opponent_a_stance_override: Optional[str],
        opponent_b_stance_override: Optional[str],
        opponent_a_name_override: Optional[str],
        opponent_a_temperament_override: Optional[str],
        opponent_b_name_override: Optional[str],
        opponent_b_temperament_override: Optional[str],
        total_rounds_override: Optional[int],
        debate_rules_override: Optional[str],
        language_override: Optional[str],
        debug_enabled: bool,
        mediator_speech_enabled: bool,
    ):
        """Runs the Debate with configurable parameters."""
        asyncio.run(setup_and_run_debate(
            debate_theme_override=debate_theme_override,
            opponent_a_stance_override=opponent_a_stance_override,
            opponent_b_stance_override=opponent_b_stance_override,
            opponent_a_name_override=opponent_a_name_override,
            opponent_a_temperament_override=opponent_a_temperament_override,
            opponent_b_name_override=opponent_b_name_override,
            opponent_b_temperament_override=opponent_b_temperament_override,
            total_rounds_override=total_rounds_override,
            debate_rules_override=debate_rules_override,
            language_override=language_override,
            debug_enabled=debug_enabled,
            mediator_speech_enabled=mediator_speech_enabled,
        ))

    cli_main()