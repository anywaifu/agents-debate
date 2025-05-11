"""Mediator Agent for the debate workflow."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from tools.debate_tools import track_turn_tool, check_debate_status_tool
from tools.recording_tools import record_mediator_announcement_tool


def create_mediator_agent(
    llm: LLM,
    config: dict,
    opponent_a_name: str,
    opponent_b_name: str,
    judge_name: str,
    language: str,
    total_rounds: int,
    debate_rules: str,
    mediator_speech_enabled: bool,
) -> FunctionAgent:
    """
    Creates the MediatorAgent.
    It requires the names of the two opponent agents to manage turns and handoffs.
    Mediator speech can be disabled.
    """
    agent_name = config["default_name"]
    system_prompt = config["system_prompt_template"].format(
        opponent_a_name=opponent_a_name,
        opponent_b_name=opponent_b_name,
        judge_name=judge_name,
        language=language,
        total_rounds=total_rounds,
        debate_rules=debate_rules,
    )

    agent_tools = [track_turn_tool, check_debate_status_tool]
    if mediator_speech_enabled:
        agent_tools.append(record_mediator_announcement_tool)

    return FunctionAgent(
        name=agent_name,
        description="Mediates the debate, manages turns, and decides handoffs between speakers or to the judge.",
        system_prompt=system_prompt,
        llm=llm,
        tools=agent_tools,
        can_handoff_to=[opponent_a_name, opponent_b_name, judge_name],
    )