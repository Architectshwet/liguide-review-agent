from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver

from liquide.agents.review_analyst import create_review_analyst_agent
from liquide.prompts.review_analyst_prompt import REVIEW_ANALYST_SYSTEM_PROMPT_TEMPLATE
from liquide.state.checkpointer import get_checkpointer
from liquide.state.store import get_store
from liquide.utils.logger import get_logger

logger = get_logger(__name__)

_liquide_agent = None
_use_memory_checkpointer = False

def _get_current_date_string() -> str:
    """Get current date as formatted string for prompts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)")


def create_liquide_agent(use_memory_checkpointer: bool = False, system_prompt: str | None = None):
    if use_memory_checkpointer:
        checkpointer = MemorySaver()
        logger.info("Using in-memory checkpointer for development")
    else:
        checkpointer = get_checkpointer()

    store = get_store()
    review_agent = create_review_analyst_agent(
        checkpointer=checkpointer, store=store, system_prompt=system_prompt
    )
    liquide_agent = review_agent
    logger.info("Liquide Agent compiled successfully")
    return liquide_agent


def initialize_liquide_agent(use_memory_checkpointer: bool = False):
    """Store checkpointer mode for on-the-fly agent creation. No agent created at startup."""
    global _use_memory_checkpointer
    _use_memory_checkpointer = use_memory_checkpointer
    logger.info("Liquide Agent config initialized (agents created on-the-fly)")
    return None


def create_liquide_agent_for_request():
    """Create a fresh Liquide agent for each request, with current date in system prompt."""
    current_date = _get_current_date_string()
    system_prompt = REVIEW_ANALYST_SYSTEM_PROMPT_TEMPLATE.format(current_date=current_date)
    return create_liquide_agent(
        use_memory_checkpointer=_use_memory_checkpointer, system_prompt=system_prompt
    )


def get_liquide_agent():
    if _liquide_agent is None:
        raise RuntimeError(
            "Liquide agent has not been initialized. "
            "Call initialize_liquide_agent() after checkpointer.setup() in application lifespan."
        )
    return _liquide_agent


def reset_liquide_agent():
    global _liquide_agent
    _liquide_agent = None
    logger.info("Liquide Agent reset")


def create_liquide_agent_dev():
    return create_liquide_agent(use_memory_checkpointer=True)
