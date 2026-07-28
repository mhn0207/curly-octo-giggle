"""LangChain Chroma adapter with explicit client-side embeddings."""

import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class ChromaDefaultEmbeddings(Embeddings):
    """Expose Chroma's all-MiniLM-L6-v2 function through LangChain."""

    model_name = "all-MiniLM-L6-v2"
    dimensions = 384

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._function = DefaultEmbeddingFunction()
        if cache_dir:
            self._function.DOWNLOAD_PATH = str(
                pathlib.Path(cache_dir) / "onnx_models" / self.model_name
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            [float(value) for value in vector]
            for vector in self._function(texts)
        ]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class LangChainVectorStore:
    """Own the LangChain vector store and the Chroma connection lifecycle."""

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
        client: Optional[Any] = None,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        if embeddings is None and (
            embedding_provider != "chroma_default"
            or embedding_model != ChromaDefaultEmbeddings.model_name
            or embedding_dimensions != ChromaDefaultEmbeddings.dimensions
        ):
            raise ValueError(
                "当前实现仅支持 chroma_default/all-MiniLM-L6-v2/384；更换模型必须重建索引"
            )
        self._client = client or self._connect(chroma_host, chroma_port, chroma_path)
        self._embeddings = embeddings or ChromaDefaultEmbeddings(embedding_cache_dir)
        expected_metadata = {
            "description": "知应 AI LangChain RAG knowledge base",
            "hnsw:space": "cosine",
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
        }
        self._store = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self._embeddings,
            collection_metadata=expected_metadata,
        )
        actual_metadata = self._store._collection.metadata or {}
        for key in ("hnsw:space", "embedding_provider", "embedding_model", "embedding_dimensions"):
            if actual_metadata.get(key) != expected_metadata[key]:
                raise RuntimeError(
                    f"collection {collection_name!r} 的 {key} 与当前 embedding 配置不兼容"
                )

    @staticmethod
    def _connect(host: str, port: int, path: str) -> Any:
        try:
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            client.heartbeat()
            logger.info("RAG ChromaDB connected: %s:%s", host, port)
            return client
        except Exception as ex:
            logger.info("RAG ChromaDB unavailable (%s); using local path %s", ex, path)
            # HttpClient mutates its Settings object to the REST implementation in Chroma 0.5.
            # A fresh Settings instance is therefore required for a real embedded fallback.
            return chromadb.PersistentClient(
                path=path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

    @property
    def raw_store(self) -> Chroma:
        return self._store

    @property
    def count(self) -> int:
        return self._store._collection.count()

    def add_documents(self, documents: Sequence[Document]) -> List[str]:
        docs = list(documents)
        ids = [str(doc.metadata["chunk_id"]) for doc in docs]
        if not docs:
            return []
        return self._store.add_documents(docs, ids=ids)

    def delete(self, ids: Sequence[str]) -> None:
        values = list(ids)
        if values:
            self._store.delete(ids=values)

    def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        return self._store.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )

    def similarity_search_with_score(
        self, query: str, k: int
    ) -> List[Tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k)

    def as_retriever(self, *, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return self._store.as_retriever(search_kwargs=search_kwargs or {})
