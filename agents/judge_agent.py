"""Judge Agent for the debate workflow."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from tools.recording_tools import record_judgment_tool


def create_judge_agent(llm: LLM, config: dict, language: str) -> FunctionAgent:
    """Creates the JudgeAgent."""
    agent_name = config["default_name"]
    system_prompt = config["system_prompt_template"].format(
        agent_name=agent_name,
        language=language
    )

    return FunctionAgent(
        name=agent_name,
        description=f"The debate judge. Declares a winner and provides reasoning based on arguments. Speaks in {language}.",
        system_prompt=system_prompt,
        llm=llm,
        tools=[record_judgment_tool],
        can_handoff_to=[],
    )
