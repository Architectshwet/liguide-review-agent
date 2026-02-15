import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from liquide.prompts.review_analyst_prompt import (
    REVIEW_ANALYST_SYSTEM_PROMPT,
    REVIEW_ANALYST_SYSTEM_PROMPT_TEMPLATE,
)
from liquide.tools.review_rag_tools import query_review_rag
from liquide.utils.logger import get_logger

logger = get_logger(__name__)


# Initialize LLM for Review Analyst Agent
review_analyst_llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_REVIEW_ANALYST", "gpt-5.1"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE_REVIEW_ANALYST", 0.2)),
)


def create_review_analyst_agent(checkpointer=None, store=None, system_prompt: str | None = None):
    """
    Create the Review Analyst Agent with review RAG tool.

    Args:
        checkpointer: Optional checkpointer for conversation memory
        store: Optional store for session data
        system_prompt: Optional system prompt override (e.g. with current date). If None, uses default.

    Returns:
        Compiled Review Analyst Agent graph
    """
    prompt = system_prompt if system_prompt is not None else REVIEW_ANALYST_SYSTEM_PROMPT
    # Create Review Analyst Agent with review query tool
    agent = create_agent(
        model=review_analyst_llm,
        tools=[query_review_rag],
        system_prompt=prompt,
        name="LiquideReviewAnalyst",
        checkpointer=checkpointer,
        store=store,
    )

    logger.info("Review Analyst Agent created successfully")
    return agent
