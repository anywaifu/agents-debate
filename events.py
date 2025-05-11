"""Custom event definitions for the LlamaIndex debate workflow."""

from llama_index.core.workflow import Event


class OpponentStatementEvent(Event):
    """Event to capture an opponent's debate statement."""
    speaker_name: str
    statement: str
    event_type: str = "opponent_statement_event"


class IntroductionCompleteEvent(Event):
    """Event to capture the IntroductionAgent's opening message."""
    agent_name: str
    introduction_message: str
    event_type: str = "introduction_complete_event"


class JudgmentDeliveredEvent(Event):
    """Event to capture the JudgeAgent's final judgment."""
    judge_name: str
    judgment_text: str
    winner: str
    event_type: str = "judgment_delivered_event"


class CustomLogEvent(Event):
    """Event for custom logging messages within the workflow."""
    message: str
    log_level: str = "INFO"
    event_type: str = "custom_log_event"


class MediatorAnnouncementEvent(Event):
    """Event for Mediator's announcements."""
    agent_name: str
    announcement_text: str
    event_type: str = "mediator_announcement_event"