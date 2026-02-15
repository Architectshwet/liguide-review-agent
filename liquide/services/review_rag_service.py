from typing import Any, Optional

from liquide.services.embedding_service import ReviewEmbeddingService
from liquide.services.search_review_service import SearchReviewService


class ReviewRAGService:
    """Facade service combining review embedding and review search services."""

    def __init__(self):
        """Initialize embedding and search services used by the API and tools."""
        self.embedding_service = ReviewEmbeddingService()
        self.search_service = SearchReviewService()

    def clear_vector_store(self) -> None:
        """Clear the Qdrant vector store (delete and recreate empty collection). Use before ingest to start fresh."""
        self.embedding_service.qdrant.clear_collection()

    async def clear_vector_store_async(self) -> None:
        """Async interface wrapper for clear_vector_store."""
        self.clear_vector_store()

    def index_reviews_from_payload(self, payload: Any) -> dict[str, Any]:
        """Ingest reviews from provided payload into vector storage."""
        return self.embedding_service.ingest_reviews_from_payload(payload)

    async def index_reviews_from_payload_async(self, payload: Any) -> dict[str, Any]:
        """Async interface wrapper for index_reviews_from_payload."""
        return self.index_reviews_from_payload(payload)

    def fetch_live_reviews(
        self,
        google_play_app_id: str,
        apple_product_id: str,
        apple_country: str = "in",
        max_pages: int = 2,
    ) -> list[dict[str, Any]]:
        """Fetch raw live reviews from both app stores without indexing."""
        return self.embedding_service.collect_reviews_from_store_apis(
            google_play_app_id=google_play_app_id,
            apple_product_id=apple_product_id,
            apple_country=apple_country,
            max_pages=max_pages,
        )

    async def fetch_live_reviews_async(
        self,
        google_play_app_id: str,
        apple_product_id: str,
        apple_country: str = "in",
        max_pages: int = 2,
    ) -> list[dict[str, Any]]:
        """Async interface wrapper for fetch_live_reviews."""
        return self.fetch_live_reviews(
            google_play_app_id=google_play_app_id,
            apple_product_id=apple_product_id,
            apple_country=apple_country,
            max_pages=max_pages,
        )

    def ingest_live(self, google_play_app_id: str, apple_product_id: str, apple_country: str = "in", max_pages: int = 2) -> dict[str, Any]:
        """Fetch live reviews and index embeddings in one pipeline call."""
        return self.embedding_service.ingest_live_reviews(
            google_play_app_id=google_play_app_id,
            apple_product_id=apple_product_id,
            apple_country=apple_country,
            max_pages=max_pages,
        )

    async def ingest_live_async(
        self, google_play_app_id: str, apple_product_id: str, apple_country: str = "in", max_pages: int = 2
    ) -> dict[str, Any]:
        """Async interface wrapper for ingest_live."""
        return self.ingest_live(
            google_play_app_id=google_play_app_id,
            apple_product_id=apple_product_id,
            apple_country=apple_country,
            max_pages=max_pages,
        )

    def preview_reviews_from_payload(self, payload: Any, limit: int = 20) -> dict[str, Any]:
        """Preview generated documents and metadata for payload data."""
        return self.embedding_service.preview_payload(payload=payload, limit=limit)

    async def preview_reviews_from_payload_async(self, payload: Any, limit: int = 20) -> dict[str, Any]:
        """Async interface wrapper for preview_reviews_from_payload."""
        return self.preview_reviews_from_payload(payload=payload, limit=limit)

    # Backward-compatible wrappers.
    def ingest_payload(self, payload: Any) -> dict[str, Any]:
        """Backward-compatible alias for index_reviews_from_payload."""
        return self.index_reviews_from_payload(payload)

    def preview_payload(self, payload: Any, limit: int = 20) -> dict[str, Any]:
        """Backward-compatible alias for preview_reviews_from_payload."""
        return self.preview_reviews_from_payload(payload=payload, limit=limit)

    def query(
        self,
        question: str,
        start_date: Optional[str] = "",
        end_date: Optional[str] = "",
        device: Optional[str] = "",
        rating: Optional[list[int]] = None,
        country: Optional[str] = "",
        version: Optional[str] = "",
        mobile_model: Optional[str] = "",
    ) -> dict[str, Any]:
        """Run filtered vector search and return retrieved context for LLM answering."""
        return self.search_service.query(
            question=question,
            start_date=start_date,
            end_date=end_date,
            device=device,
            rating=rating,
            country=country,
            version=version,
            mobile_model=mobile_model,
        )

    async def query_async(
        self,
        question: str,
        start_date: Optional[str] = "",
        end_date: Optional[str] = "",
        device: Optional[str] = "",
        rating: Optional[list[int]] = None,
        country: Optional[str] = "",
        version: Optional[str] = "",
        mobile_model: Optional[str] = "",
    ) -> dict[str, Any]:
        """Async interface wrapper for query."""
        return self.query(
            question=question,
            start_date=start_date,
            end_date=end_date,
            device=device,
            rating=rating,
            country=country,
            version=version,
            mobile_model=mobile_model,
        )

    def get_data(self, limit: int = 20) -> dict[str, Any]:
        """Read back indexed review rows from Qdrant for inspection."""
        return self.embedding_service.qdrant.get_points(limit=limit)

    async def get_data_async(self, limit: int = 20) -> dict[str, Any]:
        """Async interface wrapper for get_data."""
        return self.get_data(limit=limit)


review_rag_service = ReviewRAGService()
