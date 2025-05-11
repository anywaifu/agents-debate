"""Tools for recording debate events like statements, introductions, and judgments."""

from llama_index.core.workflow import Context, StopEvent
from llama_index.core.tools import FunctionTool

from events import OpponentStatementEvent, IntroductionCompleteEvent, JudgmentDeliveredEvent, MediatorAnnouncementEvent
from utils.tts_utils import get_tts_params_from_state, speak_text

async def record_statement_tool_func(ctx: Context, agent_name: str, statement: str) -> str:
    """Records the speaker's statement to a custom event stream."""
    tts_model, tts_voice = await get_tts_params_from_state(ctx, agent_name)
    statement_event = OpponentStatementEvent(speaker_name=agent_name, statement=statement)

    ctx.write_event_to_stream(statement_event)
    await speak_text(text_to_speak=statement, model=tts_model, voice=tts_voice)
    return f"Statement from {agent_name} recorded successfully and spoken."


async def record_introduction_tool_func(ctx: Context, agent_name: str, introduction_message: str) -> str:
    """Records the IntroductionAgent's opening message."""
    tts_model, tts_voice = await get_tts_params_from_state(ctx, agent_name)
    intro_event = IntroductionCompleteEvent(agent_name=agent_name, introduction_message=introduction_message)
    ctx.write_event_to_stream(intro_event)
    await speak_text(text_to_speak=introduction_message, model=tts_model, voice=tts_voice)
    return f"Introduction from {agent_name} recorded successfully and spoken."


async def record_judgment_tool_func(ctx: Context, agent_name: str, judgment_text: str, declared_winner: str):
    """Records the JudgeAgent's final judgment."""
    full_judgment_speech = f"The judgment is as follows: {judgment_text}. The declared winner is: {declared_winner}."
    judgment_event = JudgmentDeliveredEvent(judge_name=agent_name, judgment_text=judgment_text, winner=declared_winner)
    tts_model, tts_voice = await get_tts_params_from_state(ctx, agent_name)
    ctx.write_event_to_stream(judgment_event)
    await speak_text(text_to_speak=full_judgment_speech, model=tts_model, voice=tts_voice)
    ctx.write_event_to_stream(StopEvent(result="Debate is over!"))


async def record_mediator_announcement_tool_func(ctx: Context, agent_name: str, announcement_text: str) -> str:
    """Records the Mediator's announcement to the event stream and speaks it."""
    tts_model, tts_voice = await get_tts_params_from_state(ctx, agent_name)
    announcement_event = MediatorAnnouncementEvent(agent_name=agent_name, announcement_text=announcement_text)

    ctx.write_event_to_stream(announcement_event)
    await speak_text(text_to_speak=announcement_text, model=tts_model, voice=tts_voice)
    return f"Announcement from {agent_name} recorded successfully and spoken: '{announcement_text}'"


record_mediator_announcement_tool = FunctionTool.from_defaults(
    fn=record_mediator_announcement_tool_func,
    name="record_mediator_announcement_tool",
    description="Records and speaks an announcement from the Mediator. Use for necessary clarifications or transitions."
)


record_judgment_tool = FunctionTool.from_defaults(
    fn=record_judgment_tool_func,
    name="record_judgment_tool",
    description="Records judge's final verdict, winner, and reasoning."
)

record_statement_tool = FunctionTool.from_defaults(
    fn=record_statement_tool_func,
    name="record_statement_tool",
    description="Records a debater's official statement."
)

record_introduction_tool = FunctionTool.from_defaults(
    fn=record_introduction_tool_func,
    name="record_introduction_tool",
    description="Records the debate's opening introduction."
)