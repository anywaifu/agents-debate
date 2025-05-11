"""Tool for custom logging within the workflow."""

from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from events import CustomLogEvent


async def log_message_tool_func(ctx: Context, message: str, log_level: str = "INFO") -> str:
    """Logs a custom message to the event stream with a specified log level."""
    log_event = CustomLogEvent(message=message, log_level=log_level)
    ctx.write_event_to_stream(log_event)
    return f"Message logged: '{message}' with level {log_level}. Proceed with your next action."


log_message_tool = FunctionTool.from_defaults(
    fn=log_message_tool_func,
    name="log_message_tool",
    description="Logs a custom message to the event stream.")