import hashlib
import os
from datetime import UTC, datetime
from typing import Any, Optional

from serpapi import GoogleSearch

from liquide.services.qdrant_service import get_qdrant_service
from liquide.utils.logger import get_logger

logger = get_logger(__name__)


class ReviewEmbeddingService:
    """
    Collects Liquide reviews, normalizes records, prepares metadata/documents, and indexes embeddings.

    Main pipeline order:
    1) collect_reviews_from_store_apis
    2) process_raw_reviews
    3) create_documents_and_metadata + index_reviews
    """

    def __init__(self):
        """Initialize the embedding service with a shared Qdrant client."""
        self.qdrant = get_qdrant_service()
        self.max_document_chars = 600
        # Version buckets inferred from sample timeline:
        # v3: >= 2025-10-04, v2: 2025-05-25..2025-10-03, v1: <= 2025-05-24
        self.v3_start = datetime(2025, 10, 4, tzinfo=UTC)
        self.v2_start = datetime(2025, 5, 25, tzinfo=UTC)

    @staticmethod
    def _parse_any_date(value: Optional[str]) -> datetime | None:
        """Parse supported date formats from SerpAPI responses into UTC datetime."""
        if not value:
            return None
        for parser in (
            lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")),
            lambda v: datetime.strptime(v, "%B %d, %Y").replace(tzinfo=UTC),
            lambda v: datetime.strptime(v, "%d %B %Y").replace(tzinfo=UTC),
            lambda v: datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=UTC),
        ):
            try:
                return parser(value)
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_raw_reviews(payload: Any) -> list[dict[str, Any]]:
        """Extract raw review list from accepted payload shapes."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("reviews"), list):
            return payload["reviews"]
        if isinstance(payload, dict):
            return payload.get("google_play", []) + payload.get("apple", [])
        raise ValueError("Payload must be a review list or object containing reviews")

    @staticmethod
    def _stable_review_id(item: dict[str, Any]) -> str:
        """Create a stable fallback id when source id is missing."""
        raw = f"{item.get('title','')}|{item.get('snippet') or item.get('text') or ''}|{item.get('date') or item.get('review_date') or ''}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"review-{digest}"

    def _normalize_review_record(self, item: dict[str, Any]) -> dict[str, Any]:
        """Normalize one raw review to a consistent internal shape."""
        dt = self._parse_any_date(item.get("iso_date") or item.get("date") or item.get("review_date"))
        rating_val = int(float(item.get("rating") or 0))
        rating_val = max(1, min(5, rating_val))

        review_text = (item.get("snippet") or item.get("text") or "").strip()
        if len(review_text) > self.max_document_chars:
            review_text = review_text[: self.max_document_chars].strip()

        raw_version = str(item.get("version") or "").strip().lower()
        if raw_version in {"v1", "v2", "v3"}:
            version = raw_version
        elif dt:
            if dt >= self.v3_start:
                version = "v3"
            elif dt >= self.v2_start:
                version = "v2"
            else:
                version = "v1"
        else:
            version = "v1"

        return {
            "id": item.get("id") or self._stable_review_id(item),
            "title": item.get("title") or (item.get("author") or {}).get("name") or "Unknown",
            "rating": rating_val,
            "date": dt.strftime("%Y-%m-%d") if dt else "",
            "date_ts": int(dt.timestamp()) if dt else None,
            "version": version,
            "device": item.get("device") or "Android",
            "country": item.get("country") or "India",
            "review_text": review_text,
        }

    def collect_reviews_from_store_apis(
        self,
        google_play_app_id: str,
        apple_product_id: str,
        apple_country: str = "in",
        max_pages: int = 2,
    ) -> list[dict[str, Any]]:
        """Fetch raw reviews from Google Play and Apple App Store and return one combined list."""
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("SERPAPI_API_KEY is required for live ingestion")

        combined_reviews: list[dict[str, Any]] = []
        play_params = {
            "engine": "google_play_product",
            "product_id": google_play_app_id,
            "store": "apps",
            "all_reviews": "true",
            "num": 199, # Max reviews per page to save API credits
            "sort_by": 1, # 1 for 'Most Relevant', 2 for 'Newest'
            "api_key": api_key,
        }

        android_count = 0
        page_count = 0
        while page_count < max_pages:
            results = GoogleSearch(play_params).get_dict()
            if "error" in results:
                logger.warning("Google Play fetch stopped: %s", results["error"])
                break
            reviews = results.get("reviews", [])
            for review in reviews:
                review["device"] = "Android"
                review["country"] = "India"
                combined_reviews.append(review)
            android_count += len(reviews)
            page_count += 1
            next_token = (results.get("serpapi_pagination") or {}).get("next_page_token")
            if not next_token:
                break
            play_params["next_page_token"] = next_token

        ios_count = 0
        for page in range(1, max_pages + 1):
            apple_params = {
                "engine": "apple_reviews",
                "product_id": apple_product_id,
                "country": apple_country,
                "sort": "mosthelpful",
                "page": page,
                "api_key": api_key,
            }
            results = GoogleSearch(apple_params).get_dict()
            if "error" in results:
                logger.warning("Apple fetch stopped: %s", results["error"])
                break
            reviews = results.get("reviews", [])
            if not reviews:
                break
            for review in reviews:
                review["device"] = "iOS"
                review["country"] = "India"
                combined_reviews.append(review)
            ios_count += len(reviews)

        logger.info(
            "\n%s\nAndroid reviews collected: %s\niOS reviews collected: %s\nTotal live reviews collected: %s\n%s",
            "/" * 60,
            android_count,
            ios_count,
            len(combined_reviews),
            "/" * 60,
        )

        logger.info("Collected %s live reviews from store APIs", len(combined_reviews))
        return combined_reviews

    def process_raw_reviews(self, raw_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert raw API reviews to normalized review records used for indexing."""
        processed = [self._normalize_review_record(item) for item in raw_reviews]
        logger.info("Processed %s reviews into normalized records", len(processed))
        return processed

    def create_documents_and_metadata(self, processed_reviews: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """
        Build ids, documents, and metadata for indexing.

        Metadata fields are intentionally limited to:
        id, title, rating, date, date_ts, version, device, country.
        """
        ids = [review["id"] for review in processed_reviews]
        documents = [review["review_text"] for review in processed_reviews]
        metadatas = [
            {
                "id": review["id"],
                "title": review["title"],
                "rating": review["rating"],
                "date": review["date"],
                "date_ts": review["date_ts"],
                "version": review["version"],
                "device": str(review["device"]).lower(),
                "country": str(review["country"]).lower(),
            }
            for review in processed_reviews
        ]
        logger.info("Created %s documents and metadata rows", len(ids))
        return ids, documents, metadatas

    def index_reviews(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> dict[str, Any]:
        """Index review embeddings in Qdrant and return ingestion summary."""
        self.qdrant.add_embeddings(ids=ids, documents=documents, metadatas=metadatas)
        total_docs = self.qdrant.get_points(limit=1).get("total_docs", 0)
        return {
            "status": "ok",
            "inserted": len(ids),
            "updated": 0,
            "total_docs": total_docs,
        }

    def ingest_reviews_from_payload(self, payload: Any) -> dict[str, Any]:
        """Run full indexing pipeline for already-available payload data."""
        raw_reviews = self._extract_raw_reviews(payload)
        processed_reviews = self.process_raw_reviews(raw_reviews)
        ids, documents, metadatas = self.create_documents_and_metadata(processed_reviews)
        return self.index_reviews(ids=ids, documents=documents, metadatas=metadatas)

    def ingest_live_reviews(
        self,
        google_play_app_id: str,
        apple_product_id: str,
        apple_country: str = "in",
        max_pages: int = 2,
    ) -> dict[str, Any]:
        """Run full indexing pipeline by fetching fresh data from both app stores."""
        raw_reviews = self.collect_reviews_from_store_apis(
            google_play_app_id=google_play_app_id,
            apple_product_id=apple_product_id,
            apple_country=apple_country,
            max_pages=max_pages,
        )
        if not raw_reviews:
            raise ValueError("No live reviews fetched from Google Play/App Store")
        processed_reviews = self.process_raw_reviews(raw_reviews)
        ids, documents, metadatas = self.create_documents_and_metadata(processed_reviews)
        return self.index_reviews(ids=ids, documents=documents, metadatas=metadatas)

    def preview_payload(self, payload: Any, limit: int = 20) -> dict[str, Any]:
        """Preview normalized documents and metadata (without indexing) for the given payload."""
        raw_reviews = self._extract_raw_reviews(payload)
        processed_reviews = self.process_raw_reviews(raw_reviews)
        ids, documents, metadatas = self.create_documents_and_metadata(processed_reviews)
        return {
            "count": len(ids),
            "documents": documents,
            "metadatas": metadatas,
        }
