from typing import Literal, Optional

from langchain_core.tools import tool

from liquide.services.review_rag_service import review_rag_service
from liquide.utils.logger import get_logger

logger = get_logger(__name__)


@tool("QueryReviewRAG")
async def query_review_rag(
    question: str,
    start_date: Optional[str] = "",
    end_date: Optional[str] = "",
    device: Literal["", "Android", "iOS"] = "",
    rating: Optional[list[int]] = None,
    country: Optional[str] = "",
    version: Literal["", "v1", "v2", "v3"] = "",
    mobile_model: Optional[str] = "",
) -> dict:
    """
    Use this tool when a user asks any question about Liquide app reviews.
    It retrieves matching review context with metadata so you can write the final answer naturally.

    Args:
        question: Question about Liquide app reviews.
        start_date: Start date in `YYYY-MM-DD` format.
        end_date: End date in `YYYY-MM-DD` format.
        device: Infer platform from intent and context, not only explicit OS words.
            Treat Apple/iPhone/App Store cues as `iOS`, and Google Play or non-Apple phone-brand cues as `Android`; keep empty if unclear.
        rating: List of rating values between 1 and 5. Examples: `[5]`, `[3, 5]`, `[1, 2, 3, 4]`.
        country: Country name for filtering reviews.
        version: Version bucket. Allowed values: `v1`, `v2`, `v3`.
        mobile_model: Mobile model name when the user asks for model-specific analysis.

    Returns:
        A dictionary with retrieved review documents, metadata, applied filters, notes, and next_action.
    """
    logger.info(
        "QueryReviewRAG args: %s",
        {
            "question": question,
            "start_date": start_date,
            "end_date": end_date,
            "device": device,
            "rating": rating,
            "country": country,
            "version": version,
            "mobile_model": mobile_model,
        },
    )
    tool_output = await review_rag_service.query_async(
        question=question,
        start_date=start_date,
        end_date=end_date,
        device=device,
        rating=rating,
        country=country,
        version=version,
        mobile_model=mobile_model,
    )
    logger.info("QueryReviewRAG output: %s", tool_output)
    return tool_output
