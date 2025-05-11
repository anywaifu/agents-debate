"""Opponent Agents for the debate workflow."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from tools.recording_tools import record_statement_tool


def create_opponent_agent(
    llm: LLM, config: dict, name: str, role_description: str, temperament: str, debate_theme: str, language: str, debate_rules: str
) -> FunctionAgent:
    """Creates a generic opponent agent with a specific role, temperament, and debate theme."""
    system_prompt = config["system_prompt_template"].format(
        name=name,
        temperament=temperament,
        role_description=role_description,
        debate_theme=debate_theme,
        debate_rules=debate_rules,
        language=language
    )

    return FunctionAgent(
        name=name,
        description=f"Opponent {name} arguing about '{debate_theme}'. Stance: {role_description.split(' ')[2]}. Speaks in {language}.", # Extracts stance
        system_prompt=system_prompt,
        llm=llm,
        tools=[record_statement_tool],
        can_handoff_to=["MediatorAgent"],
    )