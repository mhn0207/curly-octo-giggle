"""LangChain RAG knowledge base with idempotent document replacement."""

import asyncio
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar

from rag.default_documents import DEFAULT_DOCUMENTS
from rag.document_processor import DocumentProcessor
from rag.retriever import KnowledgeRetriever
from rag.vector_store import LangChainVectorStore

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _ReadWriteLock:
    """Allow concurrent searches while document replacement remains exclusive."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._condition:
            while self._writer or self._waiting_writers:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


class LangChainKnowledgeBase:
    """Compatibility facade for the existing MCP knowledge-search tool."""

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        collection_name: str = "knowledge_base_langchain",
        embedding_provider: str = "chroma_default",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimensions: int = 384,
        embedding_cache_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        max_concurrency: int = 4,
        vector_store: Optional[LangChainVectorStore] = None,
        load_defaults: bool = True,
    ) -> None:
        self._processor = DocumentProcessor(chunk_size, chunk_overlap)
        self._vector_store = vector_store or LangChainVectorStore(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_cache_dir=embedding_cache_dir,
        )
        self._retriever = KnowledgeRetriever(self._vector_store)
        self._index_lock = _ReadWriteLock()
        self._max_concurrency = max(1, max_concurrency)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._counter_lock = threading.Lock()
        self._active_operations = 0
        self._peak_active_operations = 0
        if load_defaults and self.doc_count == 0:
            self.add_documents(DEFAULT_DOCUMENTS)

    def add_documents(self, documents: Sequence[Dict[str, Any]]) -> int:
        """Insert new documents or replace all chunks for changed content."""
        added = 0
        with self._index_lock.write():
            for raw in documents:
                chunks = self._processor.process(dict(raw))
                if not chunks:
                    continue
                document_id = str(chunks[0].metadata["document_id"])
                content_hash = str(chunks[0].metadata["content_hash"])
                existing = self._vector_store.get_document_chunks(document_id)
                old_ids = [str(value) for value in existing.get("ids", [])]
                old_metadatas = existing.get("metadatas", []) or []
                if (
                    old_ids
                    and len(old_metadatas) == len(old_ids)
                    and all(
                        metadata and metadata.get("content_hash") == content_hash
                        for metadata in old_metadatas
                    )
                ):
                    continue

                new_ids: List[str] = []
                try:
                    new_ids = self._vector_store.add_documents(chunks)
                    if old_ids:
                        self._vector_store.delete(old_ids)
                except Exception:
                    if new_ids:
                        try:
                            self._vector_store.delete(new_ids)
                        except Exception as rollback_error:
                            logger.error("RAG rollback failed: %s", rollback_error)
                    raise
                added += len(new_ids)
        if added:
            logger.info("RAG imported %s chunks", added)
        return added

    async def aadd_documents(self, documents: Sequence[Dict[str, Any]]) -> int:
        return await self._run_bounded(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        with self._index_lock.read():
            return [hit.to_tool_dict() for hit in self._retriever.search(query, top_k)]

    async def search_handler(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        del context
        query = str(params.get("query") or "")
        top_k = max(1, min(int(params.get("top_k", 5)), 50))
        return await self._run_bounded(self.search, query, top_k)

    async def _run_bounded(self, function: Callable[..., T], *args: Any) -> T:
        async with self._semaphore:
            with self._counter_lock:
                self._active_operations += 1
                self._peak_active_operations = max(
                    self._peak_active_operations,
                    self._active_operations,
                )
            try:
                return await asyncio.to_thread(function, *args)
            finally:
                with self._counter_lock:
                    self._active_operations -= 1

    def rag_stats(self) -> Dict[str, Any]:
        with self._counter_lock:
            concurrency = {
                "active_operations": self._active_operations,
                "peak_active_operations": self._peak_active_operations,
                "max_concurrency": self._max_concurrency,
            }
        return {
            "total_chunks": self.doc_count,
            "retrieval": self._retriever.stats(),
            "concurrency": concurrency,
        }

    @property
    def doc_count(self) -> int:
        return self._vector_store.count
