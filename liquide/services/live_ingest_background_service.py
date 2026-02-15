import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from liquide.services.review_rag_service import review_rag_service
from liquide.utils.logger import get_logger

logger = get_logger(__name__)


class LiveIngestBackgroundService:
    """Background-job service for long-running live review ingestion."""

    def __init__(self, max_jobs: int = 200):
        self._jobs: dict[str, dict[str, Any]] = {}
        self._max_jobs = max_jobs

    @staticmethod
    def load_sample_reviews_payload() -> list[dict[str, Any]]:
        sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "reviews_sample.json"
        return json.loads(sample_path.read_text(encoding="utf-8"))

    def _prune_jobs(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        removable = sorted(self._jobs.items(), key=lambda item: item[1].get("created_at", 0.0))
        overflow = len(self._jobs) - self._max_jobs
        for job_id, _ in removable[:overflow]:
            self._jobs.pop(job_id, None)

    def create_job(self, max_pages: int, fallback_to_sample: bool) -> str:
        job_id = f"live-ingest-{uuid.uuid4()}"
        now = time.time()
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "max_pages": max_pages,
            "fallback_to_sample": fallback_to_sample,
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }
        self._prune_jobs()
        return job_id

    def _update_job(self, job_id: str, **updates: Any) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    async def execute_live_ingest(self, max_pages: int, fallback_to_sample: bool) -> dict[str, Any]:
        google_play_app_id = os.getenv("GOOGLE_PLAY_APP_ID", "life.liquide.app")
        apple_product_id = os.getenv("APPLE_PRODUCT_ID", "1624726081")
        apple_country = os.getenv("APPLE_COUNTRY", "in")
        try:
            await review_rag_service.clear_vector_store_async()
            live_payload = await review_rag_service.fetch_live_reviews_async(
                google_play_app_id=google_play_app_id,
                apple_product_id=apple_product_id,
                apple_country=apple_country,
                max_pages=max_pages,
            )
            if not live_payload:
                raise ValueError("No live reviews fetched from Google Play/App Store")

            sample_payload = self.load_sample_reviews_payload()
            merged_payload = live_payload + sample_payload
            merged_result = await review_rag_service.index_reviews_from_payload_async(merged_payload)
            return {
                "status": "ok",
                "live_count": len(live_payload),
                "sample_count": len(sample_payload),
                "merged_count": len(merged_payload),
                "ingest_result": merged_result,
            }
        except Exception as e:
            if not fallback_to_sample:
                raise RuntimeError(str(e)) from e
            try:
                payload = self.load_sample_reviews_payload()
                fallback_result = await review_rag_service.index_reviews_from_payload_async(payload)
                return {
                    "status": "fallback_to_sample",
                    "reason": str(e),
                    "ingest_result": fallback_result,
                }
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Live ingestion failed: {e}. Sample fallback failed: {fallback_error}"
                ) from fallback_error

    async def run_job(self, job_id: str, max_pages: int, fallback_to_sample: bool) -> None:
        self._update_job(job_id, status="running")
        try:
            result = await self.execute_live_ingest(max_pages=max_pages, fallback_to_sample=fallback_to_sample)
            result_status = str(result.get("status", "ok"))
            self._update_job(job_id, status=result_status, result=result, error=None)
        except Exception as e:
            logger.error("Live ingest background job failed for %s: %s", job_id, e)
            self._update_job(job_id, status="failed", error=str(e), result=None)


live_ingest_background_service = LiveIngestBackgroundService()
