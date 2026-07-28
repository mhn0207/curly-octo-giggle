"""Stable data contracts used by the RAG layer and MCP adapter."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class SearchHit:
    """One scored chunk returned by the retriever."""

    title: str
    content: str
    score: float
    chunk: int = 0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tool_dict(self) -> Dict[str, Any]:
        """Return the existing public API shape plus private deduplication keys."""
        result: Dict[str, Any] = {
            "title": self.title,
            "content": self.content,
            "score": round(max(0.0, min(1.0, self.score)), 4),
            "chunk": self.chunk,
        }
        chunk_id = self.metadata.get("chunk_id")
        document_id = self.metadata.get("document_id")
        if chunk_id:
            result["_chunk_id"] = chunk_id
        if document_id:
            result["_document_id"] = document_id
        return result
