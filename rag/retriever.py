"""Scored retrieval facade over LangChain's standard Retriever interface."""

import time
from collections import deque
from typing import Any, Deque, Dict, List

from rag.models import SearchHit
from rag.vector_store import LangChainVectorStore


class KnowledgeRetriever:
    def __init__(self, vector_store: LangChainVectorStore, default_k: int = 5) -> None:
        self._vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": default_k})
        self._latencies_ms: Deque[float] = deque(maxlen=500)
        self._total = 0
        self._errors = 0

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if not query.strip() or top_k <= 0:
            return []
        started = time.monotonic()
        self._total += 1
        try:
            results = self._vector_store.similarity_search_with_score(query, k=top_k)
        except Exception:
            self._errors += 1
            raise
        finally:
            self._latencies_ms.append((time.monotonic() - started) * 1000)
        hits: List[SearchHit] = []
        for document, distance in results:
            metadata = dict(document.metadata)
            hits.append(
                SearchHit(
                    title=str(metadata.get("title", "")),
                    content=document.page_content,
                    score=max(0.0, min(1.0, 1.0 - float(distance))),
                    chunk=int(metadata.get("chunk_index", 0)),
                    source=str(metadata.get("source", "")),
                    metadata=metadata,
                )
            )
        return hits
    def stats(self) -> Dict[str, Any]:
        values = sorted(self._latencies_ms)

        def percentile(value: float) -> float:
            if not values:
                return 0.0
            index = min(len(values) - 1, max(0, int((len(values) - 1) * value)))
            return round(values[index], 1)

        return {
            "total": self._total,
            "errors": self._errors,
            "p50_latency_ms": percentile(0.50),
            "p95_latency_ms": percentile(0.95),
        }
