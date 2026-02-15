from datetime import UTC, datetime
from typing import Optional

from liquide.services.qdrant_service import get_qdrant_service


class SearchReviewService:
    """Search/query service over review vectors and metadata filters."""

    def __init__(self):
        """Initialize search service with shared Qdrant client."""
        self.qdrant = get_qdrant_service()

    @staticmethod
    def _parse_date(value: Optional[str]) -> datetime | None:
        """Parse supported date formats into UTC datetime."""
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

    def _is_date_in_range(self, value: str, start_dt: datetime | None, end_dt: datetime | None) -> bool:
        """Check if review date string is within requested date range."""
        if not start_dt and not end_dt:
            return True
        dt = self._parse_date(value)
        if not dt:
            return False
        if start_dt and dt < start_dt:
            return False
        if end_dt and dt > end_dt:
            return False
        return True

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
    ) -> dict:
        """
        Run vector search with metadata filters and return context for final LLM response generation.

        Date range filtering is applied after retrieval because indexed metadata keeps only human-readable date.
        """
        notes: list[str] = []

        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)
        start_ts = int(start_dt.timestamp()) if start_dt else None
        end_ts = int(end_dt.timestamp()) if end_dt else None
        if end_ts is not None:
            # Make end-date filtering inclusive for whole-day inputs.
            end_ts += 86399

        normalized_device = device.lower() if device else ""
        normalized_country = country.strip().lower() if country else ""
        normalized_version = version.strip().lower() if version else ""

        rating_values = [int(r) for r in (rating or []) if int(r) in [1, 2, 3, 4, 5]]

        if mobile_model:
            effective_device = "iOS" if normalized_device == "ios" else "Android"
            notes.append(
                f"The mobile_model filter is not supported in our current system and was ignored. "
                f"We used the device filter instead: {effective_device}."
            )

        applied_filters: dict = {}
        if start_date:
            applied_filters["start_date"] = start_date
        if end_date:
            applied_filters["end_date"] = end_date
        if normalized_device:
            applied_filters["device"] = "iOS" if normalized_device == "ios" else "Android"
        if rating_values:
            applied_filters["rating"] = rating_values
        if normalized_country:
            applied_filters["country"] = normalized_country
        if normalized_version:
            applied_filters["version"] = normalized_version

        qdrant_filter = self.qdrant.build_filter(
            device=normalized_device,
            rating_values=rating_values,
            country=normalized_country,
            version=normalized_version,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        result_limit = 12
        results = self.qdrant.query(query_text=question, n_results=result_limit, filters=qdrant_filter)
        retrieved_documents = []
        for i in range(len(results.get("ids", []))):
            metadata = results["metadatas"][i]
            text = results["documents"][i]

            if not self._is_date_in_range(str(metadata.get("date") or ""), start_dt, end_dt):
                continue

            retrieved_documents.append(
                {
                    "text": text,
                    "metadata": {
                        "id": metadata.get("id") or results["ids"][i],
                        "rating": metadata.get("rating"),
                        "date": metadata.get("date"),
                        "version": metadata.get("version"),
                        "device": str(metadata.get("device") or ""),
                        "country": str(metadata.get("country") or ""),
                    },
                }
            )
            if len(retrieved_documents) >= 12:
                break

        return {
            "question": question,
            "retrieved_documents": retrieved_documents,
            "applied_filters": {"qdrant_filter": applied_filters},
            "notes": notes,
            "next_action": """
Use only question and retrieved_documents for the final response.
Do not invent facts outside retrieved context.

If answerable, return sections in this order:
1. Summary (3-5 bullets)
2. Stats
3. Evidence
4. Applied filters

Notes section is optional:
- Add Notes only when notes is non-empty.
- If Notes is included, use only the tool notes content and do not add extra commentary.

Applied filters rule:
- Mention only keys present in applied_filters.qdrant_filter.
- Do not mention unapplied or unsupported filters.
- If none were applied, say: No explicit filters applied.

Evidence rule:
- Provide 3-5 short quotes.
- For each quote include: id, device, date.
- Use format like: id=<id>, device=<device>, date=<date>.

If not answerable from retrieved context:
- Do not guess.
- Respond in 1-2 lines that retrieved reviews do not contain enough information.
""".strip(),
        }
