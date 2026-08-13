"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
import pathlib
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from rag.default_documents import DEFAULT_DOCUMENTS

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        embedding_cache_dir: Optional[str] = None,
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 服务端模式不传 embedding_function；本地模式仍使用 Chroma 默认模型，
        # 但允许把下载目录固定到应用可写的持久卷。
        collection_options: Dict[str, Any] = {
            "name": self.COLLECTION_NAME,
            "metadata": {"description": "知应 AI RAG 知识库"},
        }
        if not self._use_server and embedding_cache_dir:
            embedding_function = DefaultEmbeddingFunction()
            embedding_function.DOWNLOAD_PATH = str(
                pathlib.Path(embedding_cache_dir)
                / "onnx_models"
                / "all-MiniLM-L6-v2"
            )
            collection_options["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(**collection_options)

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。重复内容跳过；同一文档内容变化时替换旧分片。
        """
        changed_chunks = 0

        for raw in documents:
            title = str(raw.get("title") or "").strip()
            content = str(raw.get("content") or "").strip()
            source = str(raw.get("source") or "").strip()
            chunks = self._chunk_text(content, chunk_size=500)
            if not chunks:
                continue

            content_hash = self._digest(content)
            explicit_id = str(raw.get("document_id") or "").strip()
            identity = f"{source}\n{title}" if source or title else content_hash
            document_id = explicit_id or self._digest(identity)
            ids = [
                self._digest(f"{document_id}:{content_hash}:{index}")
                for index in range(len(chunks))
            ]
            metadatas = [
                {
                    "title": title,
                    "source": source,
                    "document_id": document_id,
                    "content_hash": content_hash,
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                }
                for index in range(len(chunks))
            ]

            current = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            current_ids = [str(value) for value in current.get("ids", [])]
            current_metadatas = current.get("metadatas") or []
            legacy_ids = self._legacy_ids_for_title(title)
            unchanged = (
                not legacy_ids
                and len(current_ids) == len(ids)
                and set(current_ids) == set(ids)
                and len(current_metadatas) == len(current_ids)
                and all(
                    metadata and metadata.get("content_hash") == content_hash
                    for metadata in current_metadatas
                )
            )
            if unchanged:
                continue

            # 先 upsert 新版本，再删除旧分片；写入失败时旧版本仍然保留。
            self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
            obsolete_ids = [
                value
                for value in dict.fromkeys(current_ids + legacy_ids)
                if value not in ids
            ]
            if obsolete_ids:
                self._collection.delete(ids=obsolete_ids)
            changed_chunks += len(ids)

        if changed_chunks:
            logger.info(f"知识库导入或替换 {changed_chunks} 个文档片段")

        return changed_chunks

    def _legacy_ids_for_title(self, title: str) -> List[str]:
        """查找旧版缺少 document_id 元数据的同标题分片，供首次导入时迁移。"""
        if not title:
            return []
        result = self._collection.get(where={"title": title}, include=["metadatas"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        return [
            str(record_id)
            for index, record_id in enumerate(ids)
            if not (
                index < len(metadatas)
                and metadatas[index]
                and metadatas[index].get("document_id")
            )
        ]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await asyncio.to_thread(self.search, query, top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（企业服务场景常见问题）。"""
        self.add_documents(DEFAULT_DOCUMENTS)
        logger.info(f"已导入默认知识库: {len(DEFAULT_DOCUMENTS)} 篇文档")
