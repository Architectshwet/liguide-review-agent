"""Qdrant service for storing and searching Liquide review embeddings."""

import hashlib
import logging
import os
from functools import lru_cache
from typing import Any

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, Range, VectorParams

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database"""

    def __init__(self):
        self.path = os.getenv("QDRANT_PATH", "./qdrant_vectordb")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "liquide_reviews")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        self.client = QdrantClient(path=self.path)
        self.embedder = OpenAIEmbeddings(model=self.embedding_model)
        self._initialize_collection()

    def _point_id(self, value: str) -> int:
        digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:15]
        return int(digest, 16)

    def _initialize_collection(self):
        collections = self.client.get_collections().collections
        names = [col.name for col in collections]
        if self.collection_name not in names:
            vector_size = len(self.embedder.embed_query("liquide review"))
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def add_embeddings(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        embeddings = self.embedder.embed_documents(documents)
        points: list[PointStruct] = []
        for i in range(len(ids)):
            payload = metadatas[i].copy()
            payload["document"] = documents[i]
            points.append(
                PointStruct(
                    id=self._point_id(ids[i]),
                    vector=embeddings[i],
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_text: str, n_results: int = 8, filters: Filter | None = None) -> dict[str, Any]:
        query_vector = self.embedder.embed_query(query_text)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=n_results,
            query_filter=filters,
        ).points

        ids, documents, metadatas, distances = [], [], [], []
        for result in results:
            payload = result.payload or {}
            ids.append(payload.get("id") or str(result.id))
            documents.append(payload.get("document", ""))
            metadatas.append({k: v for k, v in payload.items() if k != "document"})
            distances.append(result.score)

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }

    def build_filter(
        self,
        device: str = "",
        rating_values: list[int] | None = None,
        country: str = "",
        version: str = "",
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> Filter | None:
        must_conditions: list[FieldCondition] = []

        if rating_values:
            clean = [int(r) for r in rating_values if int(r) in [1, 2, 3, 4, 5]]
            if clean:
                must_conditions.append(FieldCondition(key="rating", match=MatchAny(any=clean)))

        normalized_device = device.strip().lower() if device else ""
        normalized_country = country.strip().lower() if country else ""
        normalized_version = version.strip().lower() if version else ""
        if normalized_device:
            must_conditions.append(FieldCondition(key="device", match=MatchValue(value=normalized_device)))
        if normalized_country:
            must_conditions.append(FieldCondition(key="country", match=MatchValue(value=normalized_country)))
        if normalized_version:
            must_conditions.append(FieldCondition(key="version", match=MatchValue(value=normalized_version)))

        if start_ts or end_ts:
            must_conditions.append(
                FieldCondition(
                    key="date_ts",
                    range=Range(gte=start_ts, lte=end_ts),
                )
            )

        return Filter(must=must_conditions) if must_conditions else None

    def clear_collection(self) -> None:
        """Delete the collection and recreate it empty. Use before re-ingesting to start fresh."""
        try:
            collections = self.client.get_collections().collections
            names = [col.name for col in collections]
            if self.collection_name in names:
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection not found, skipping delete: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Collection delete check failed: {e}")
        self._initialize_collection()

    def get_points(self, limit: int = 20) -> dict[str, Any]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        items = []
        for p in points:
            payload = p.payload or {}
            items.append(
                {
                    "id": payload.get("id"),
                    "title": payload.get("title"),
                    "rating": payload.get("rating"),
                    "snippet": (payload.get("document") or "")[:220],
                    "date": payload.get("date"),
                    "version": payload.get("version"),
                    "device": payload.get("device"),
                    "country": payload.get("country"),
                }
            )

        total = self.client.get_collection(self.collection_name).points_count
        return {"total_docs": total, "items": items}


@lru_cache
def get_qdrant_service() -> QdrantService:
    return QdrantService()
