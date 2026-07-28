"""Document normalization, recursive splitting, and stable chunk identity."""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Convert knowledge-base documents into metadata-rich LangChain chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", ". ", " ", ""],
        )

    def process(self, raw: Dict[str, Any]) -> List[Document]:
        content = str(raw.get("content") or "").strip()
        if not content:
            return []

        title = str(raw.get("title") or "").strip()
        source = str(raw.get("source") or "").strip()
        content_hash = self._digest(content)
        explicit_id = str(raw.get("document_id") or "").strip()
        identity = f"{source}\n{title}" if source or title else content_hash
        document_id = explicit_id or self._digest(identity)
        updated_at = datetime.now(timezone.utc).isoformat()

        base_metadata: Dict[str, Any] = {
            "document_id": document_id,
            "content_hash": content_hash,
            "title": title,
            "source": source,
            "updated_at": updated_at,
        }
        chunks = self._splitter.split_documents(
            [Document(page_content=content, metadata=base_metadata)]
        )
        total = len(chunks)
        for index, chunk in enumerate(chunks):
            start_index = int(chunk.metadata.get("start_index", 0))
            chunk.metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": total,
                    "start_index": start_index,
                    "chunk_id": self._digest(
                        f"{document_id}:{content_hash}:{index}:{start_index}"
                    ),
                }
            )
        return chunks

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
