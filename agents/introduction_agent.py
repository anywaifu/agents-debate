"""Introduction Agent for the debate workflow."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from tools.recording_tools import record_introduction_tool


def create_introduction_agent(llm: LLM, config: dict, debate_theme: str, language: str, debate_rules: str) -> FunctionAgent:
    """Creates the IntroductionAgent."""
    agent_name = config["default_name"]
    system_prompt = config["system_prompt_template"].format(
        agent_name=agent_name,
        debate_theme=debate_theme,
        debate_rules=debate_rules,
        language=language
    )

    return FunctionAgent(
        name=agent_name,
        description=f"Introduces the debate on '{debate_theme}' and its rules. Speaks in {language}.", # Description can be more dynamic if needed
        system_prompt=system_prompt,
        llm=llm,
        tools=[record_introduction_tool],
        can_handoff_to=["MediatorAgent"]
    )