"""Feature rollout router for legacy, shadow, canary, and LangChain RAG."""

import asyncio
import hashlib
import logging
import threading
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from evaluation.rag_evaluator import DEFAULT_RAG_CASES, RAGEvaluator, RAGTestCase

logger = logging.getLogger(__name__)


class RAGRolloutMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    CANARY = "canary"
    LANGCHAIN = "langchain"

    @classmethod
    def from_value(cls, value: str, *, langchain_enabled: bool = False) -> "RAGRolloutMode":
        raw = (value or "").strip().lower()
        if not raw:
            return cls.LANGCHAIN if langchain_enabled else cls.LEGACY
        try:
            return cls(raw)
        except ValueError:
            logger.warning("未知 RAG 灰度模式 %r，使用 legacy", value)
            return cls.LEGACY


@dataclass
class RAGRolloutStats:
    total_searches: int = 0
    legacy_responses: int = 0
    langchain_responses: int = 0
    dual_read_attempts: int = 0
    shadow_comparisons: int = 0
    legacy_read_failures: int = 0
    langchain_read_failures: int = 0
    top1_matches: int = 0
    overlap_total: float = 0.0
    primary_failures: int = 0
    fallback_successes: int = 0
    fallback_failures: int = 0
    dual_write_failures: int = 0


class KnowledgeBaseRollout:
    """Present one knowledge-base interface while old and new indexes coexist."""

    def __init__(
        self,
        legacy_backend: Any,
        langchain_backend: Optional[Any] = None,
        *,
        mode: RAGRolloutMode = RAGRolloutMode.LEGACY,
        canary_percent: float = 0.0,
    ) -> None:
        if mode != RAGRolloutMode.LEGACY and langchain_backend is None:
            raise ValueError(f"RAG rollout mode {mode.value} requires a LangChain backend")
        self._legacy = legacy_backend
        self._langchain = langchain_backend
        self._mode = mode
        self._canary_percent = max(0.0, min(float(canary_percent), 100.0))
        self._stats = RAGRolloutStats()
        self._stats_lock = threading.Lock()

    async def search_handler(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query = str(params.get("query") or "")
        top_k = max(1, min(int(params.get("top_k", 5)), 50))
        with self._stats_lock:
            self._stats.total_searches += 1

        if self._mode == RAGRolloutMode.LEGACY:
            result = await self._search_backend(self._legacy, query, top_k, context)
            self._record_response(False)
            return result

        if self._mode == RAGRolloutMode.LANGCHAIN:
            return await self._primary_with_fallback(
                self._langchain,
                self._legacy,
                query,
                top_k,
                context,
                langchain_primary=True,
            )

        legacy_result, langchain_result = await asyncio.gather(
            self._search_backend(self._legacy, query, top_k, context),
            self._search_backend(self._langchain, query, top_k, context),
            return_exceptions=True,
        )
        self._record_comparison(legacy_result, langchain_result, top_k)

        use_langchain = (
            self._mode == RAGRolloutMode.CANARY and self._is_canary_query(query)
        )
        primary = langchain_result if use_langchain else legacy_result
        secondary = legacy_result if use_langchain else langchain_result
        return self._choose_result(primary, secondary, langchain_primary=use_langchain)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Synchronous compatibility path with the same rollout semantics."""
        top_k = max(1, min(int(top_k), 50))
        with self._stats_lock:
            self._stats.total_searches += 1

        if self._mode == RAGRolloutMode.LEGACY:
            try:
                result = self._legacy.search(query, top_k)
            except Exception:
                with self._stats_lock:
                    self._stats.primary_failures += 1
                raise
            self._record_response(False)
            return result

        if self._mode == RAGRolloutMode.LANGCHAIN:
            try:
                primary: Any = self._langchain.search(query, top_k)
            except Exception as ex:
                primary = ex
            if not isinstance(primary, BaseException):
                self._record_response(True)
                return primary
            try:
                secondary: Any = self._legacy.search(query, top_k)
            except Exception as ex:
                secondary = ex
            return self._choose_result(primary, secondary, langchain_primary=True)

        try:
            legacy_result: Any = self._legacy.search(query, top_k)
        except Exception as ex:
            legacy_result = ex
        try:
            langchain_result: Any = self._langchain.search(query, top_k)
        except Exception as ex:
            langchain_result = ex
        self._record_comparison(legacy_result, langchain_result, top_k)
        use_langchain = (
            self._mode == RAGRolloutMode.CANARY and self._is_canary_query(query)
        )
        primary = langchain_result if use_langchain else legacy_result
        secondary = legacy_result if use_langchain else langchain_result
        return self._choose_result(primary, secondary, langchain_primary=use_langchain)

    async def aadd_documents(self, documents: Sequence[Dict[str, Any]]) -> int:
        if self._mode == RAGRolloutMode.LEGACY:
            return await self._add_backend(self._legacy, documents)

        legacy_count, langchain_count = await asyncio.gather(
            self._add_backend(self._legacy, documents),
            self._add_backend(self._langchain, documents),
            return_exceptions=True,
        )
        primary_is_langchain = self._mode in {
            RAGRolloutMode.CANARY,
            RAGRolloutMode.LANGCHAIN,
        }
        primary = langchain_count if primary_is_langchain else legacy_count
        secondary = legacy_count if primary_is_langchain else langchain_count
        return self._resolve_write_result(primary, secondary)

    def add_documents(self, documents: Sequence[Dict[str, Any]]) -> int:
        if self._mode == RAGRolloutMode.LEGACY:
            return int(self._legacy.add_documents(documents))

        try:
            legacy_count: Any = self._legacy.add_documents(documents)
        except Exception as ex:
            legacy_count = ex
        try:
            langchain_count: Any = self._langchain.add_documents(documents)
        except Exception as ex:
            langchain_count = ex
        primary_is_langchain = self._mode in {
            RAGRolloutMode.CANARY,
            RAGRolloutMode.LANGCHAIN,
        }
        primary = langchain_count if primary_is_langchain else legacy_count
        secondary = legacy_count if primary_is_langchain else langchain_count
        return self._resolve_write_result(primary, secondary)

    async def evaluate_backends(
        self,
        cases: Sequence[RAGTestCase] = DEFAULT_RAG_CASES,
    ) -> Dict[str, Any]:
        if self._langchain is None:
            raise RuntimeError("LangChain RAG backend is not initialized")
        evaluator = RAGEvaluator()

        async def legacy_search(query: str, top_k: int) -> List[Dict[str, Any]]:
            return await self._search_backend(self._legacy, query, top_k, None)

        async def langchain_search(query: str, top_k: int) -> List[Dict[str, Any]]:
            return await self._search_backend(self._langchain, query, top_k, None)

        return await evaluator.compare(legacy_search, langchain_search, cases)

    @property
    def doc_count(self) -> int:
        if self._mode in {RAGRolloutMode.CANARY, RAGRolloutMode.LANGCHAIN}:
            return int(self._langchain.doc_count)
        return int(self._legacy.doc_count)

    def rag_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = asdict(self._stats)
        comparisons = stats["shadow_comparisons"]
        stats.update(
            {
                "mode": self._mode.value,
                "canary_percent": self._canary_percent,
                "top1_match_rate": (
                    stats["top1_matches"] / comparisons if comparisons else 0.0
                ),
                "average_overlap_at_k": (
                    stats["overlap_total"] / comparisons if comparisons else 0.0
                ),
            }
        )
        result: Dict[str, Any] = {"rollout": stats}
        runtime_stats = getattr(self._langchain, "rag_stats", None)
        if callable(runtime_stats):
            result["langchain"] = runtime_stats()
        result["legacy_total_chunks"] = int(self._legacy.doc_count)
        if self._langchain is not None:
            result["langchain_total_chunks"] = int(self._langchain.doc_count)
        return result

    async def _primary_with_fallback(
        self,
        primary_backend: Any,
        secondary_backend: Any,
        query: str,
        top_k: int,
        context: Optional[Dict[str, Any]],
        *,
        langchain_primary: bool,
    ) -> List[Dict[str, Any]]:
        try:
            result = await self._search_backend(primary_backend, query, top_k, context)
            self._record_response(langchain_primary)
            return result
        except Exception:
            with self._stats_lock:
                self._stats.primary_failures += 1
            try:
                result = await self._search_backend(secondary_backend, query, top_k, context)
            except Exception:
                with self._stats_lock:
                    self._stats.fallback_failures += 1
                raise
            with self._stats_lock:
                self._stats.fallback_successes += 1
            self._record_response(not langchain_primary)
            return result

    async def _search_backend(
        self,
        backend: Any,
        query: str,
        top_k: int,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        handler = getattr(backend, "search_handler", None)
        if callable(handler):
            value = handler({"query": query, "top_k": top_k}, context)
            if asyncio.iscoroutine(value):
                value = await value
        else:
            value = await asyncio.to_thread(backend.search, query, top_k)
        if not isinstance(value, list):
            raise TypeError("knowledge backend must return a list")
        return value

    @staticmethod
    async def _add_backend(backend: Any, documents: Sequence[Dict[str, Any]]) -> int:
        async_add = getattr(backend, "aadd_documents", None)
        if callable(async_add):
            return int(await async_add(documents))
        return int(await asyncio.to_thread(backend.add_documents, documents))

    def _choose_result(
        self,
        primary: Any,
        secondary: Any,
        *,
        langchain_primary: bool,
    ) -> List[Dict[str, Any]]:
        if not isinstance(primary, BaseException):
            self._record_response(langchain_primary)
            return primary
        with self._stats_lock:
            self._stats.primary_failures += 1
        if not isinstance(secondary, BaseException):
            with self._stats_lock:
                self._stats.fallback_successes += 1
            self._record_response(not langchain_primary)
            return secondary
        with self._stats_lock:
            self._stats.fallback_failures += 1
        raise primary

    def _resolve_write_result(self, primary: Any, secondary: Any) -> int:
        if isinstance(primary, BaseException):
            with self._stats_lock:
                self._stats.primary_failures += 1
            if isinstance(secondary, BaseException):
                with self._stats_lock:
                    self._stats.fallback_failures += 1
                raise primary
            with self._stats_lock:
                self._stats.fallback_successes += 1
                self._stats.dual_write_failures += 1
            return int(secondary)
        if isinstance(secondary, BaseException):
            with self._stats_lock:
                self._stats.dual_write_failures += 1
            logger.warning("RAG secondary index write failed: %s", type(secondary).__name__)
        return int(primary)

    def _record_response(self, langchain: bool) -> None:
        with self._stats_lock:
            if langchain:
                self._stats.langchain_responses += 1
            else:
                self._stats.legacy_responses += 1

    def _record_comparison(self, legacy: Any, langchain: Any, top_k: int) -> None:
        legacy_failed = isinstance(legacy, BaseException)
        langchain_failed = isinstance(langchain, BaseException)
        with self._stats_lock:
            self._stats.dual_read_attempts += 1
            if legacy_failed:
                self._stats.legacy_read_failures += 1
            if langchain_failed:
                self._stats.langchain_read_failures += 1
        if legacy_failed or langchain_failed:
            return
        legacy_keys = self._result_keys(legacy[:top_k])
        langchain_keys = self._result_keys(langchain[:top_k])
        union = legacy_keys | langchain_keys
        overlap = len(legacy_keys & langchain_keys) / len(union) if union else 1.0
        legacy_top = next(iter(self._result_keys(legacy[:1])), "")
        langchain_top = next(iter(self._result_keys(langchain[:1])), "")
        with self._stats_lock:
            self._stats.shadow_comparisons += 1
            self._stats.overlap_total += overlap
            if legacy_top == langchain_top:
                self._stats.top1_matches += 1

    @staticmethod
    def _result_keys(results: Sequence[Any]) -> set[str]:
        keys = set()
        for item in results:
            if not isinstance(item, dict):
                keys.add(str(item))
                continue
            chunk_id = item.get("_chunk_id") or item.get("chunk_id")
            if chunk_id:
                keys.add(f"chunk:{chunk_id}")
                continue
            keys.add(
                f"display:{item.get('title', '')}:{item.get('chunk', item.get('chunk_index', 0))}"
            )
        return keys

    def _is_canary_query(self, query: str) -> bool:
        bucket = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "big")
        return (bucket % 10_000) < int(self._canary_percent * 100)
