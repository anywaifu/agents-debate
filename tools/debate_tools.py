"""Tools for managing debate flow, turns, and status."""

import asyncio
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool


async def wait_tool_func(wait_time: int) -> str:
    """Waits for a specified number of seconds."""
    await asyncio.sleep(wait_time)
    return f"Waited for {wait_time} seconds. Continue."


async def track_turn_func(ctx: Context, speaker_name: str) -> str:
    """
    Tracks the turn for the given speaker (e.g., 'LeftWingAgent' or 'RightWingAgent').
    Updates the turn count for the speaker and sets them as the current speaker.
    """
    current_workflow_state = await ctx.get("state") # type: ignore
    speaker_turn_key = f"{speaker_name}_turns"

    if speaker_turn_key not in current_workflow_state:
        # This should ideally be initialized in the workflow's initial_state
        current_workflow_state[speaker_turn_key] = 0
    
    current_workflow_state[speaker_turn_key] += 1
    current_workflow_state["current_speaker"] = speaker_name
    await ctx.set("state", current_workflow_state)

    turn_count = current_workflow_state.get(speaker_turn_key, 0)
    return f"Tracked turn for {speaker_name}. They have had {turn_count} turns. Current speaker is {speaker_name}."


async def check_debate_status_func(ctx: Context) -> str:
    """
    Checks if the debate should end based on total_rounds per side.
    Returns a directive string for the MediatorAgent.
    """
    current_workflow_state = await ctx.get("state") # type: ignore
    total_rounds = current_workflow_state.get("total_rounds", 1) 
    opponent_a_name = str(current_workflow_state.get("opponent_a_name", ""))
    opponent_b_name = str(current_workflow_state.get("opponent_b_name", ""))
    
    # This is the speaker whose turn was just tracked by track_turn_tool
    # They are the candidate for the current speaking turn.
    designated_speaker = str(current_workflow_state.get("current_speaker", "")) # type: ignore

    if not opponent_a_name or not opponent_b_name or not designated_speaker or designated_speaker == "none":
        return "ACTION: ERROR_CRITICAL_STATE_MISSING_FOR_DEBATE_STATUS_CHECK"

    opponent_a_turns = current_workflow_state.get(f"{opponent_a_name}_turns", 0)
    opponent_b_turns = current_workflow_state.get(f"{opponent_b_name}_turns", 0)

    # The debate ends if either opponent's turn count *exceeds* total_rounds when they are selected.
    # This means track_turn_tool has been called for their (total_rounds + 1)th turn,
    # implying their (total_rounds)th statement has already been completed in their previous turn.
    if opponent_a_turns > total_rounds or opponent_b_turns > total_rounds:
        return "ACTION: HANDOFF_TO_JUDGE_AGENT"
    
    # Otherwise, the debate continues, handoff to the designated speaker.
    return f"ACTION: HANDOFF_TO_SPEAKER:{designated_speaker}"


check_debate_status_tool = FunctionTool.from_defaults(
    fn=check_debate_status_func,
    name="check_debate_status_tool",
    description="Checks if debate rounds are complete to end or continue."
)
track_turn_tool = FunctionTool.from_defaults(
    fn=track_turn_func,
    name="track_turn_tool",
    description="Updates turn count for the current speaker."
)