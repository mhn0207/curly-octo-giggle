import asyncio
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from mcp.knowledge_base import KnowledgeBase as LegacyKnowledgeBase

from rag.document_processor import DocumentProcessor
from rag.knowledge_base import LangChainKnowledgeBase
from rag.models import SearchHit
from rag.retriever import KnowledgeRetriever
from rag.vector_store import ChromaDefaultEmbeddings, LangChainVectorStore


class InMemoryVectorStore:
    def __init__(self):
        self.records = {}
        self.search_results = []
        self.fail_delete_ids = set()
        self.search_delay = 0.0
        self.active_searches = 0
        self.peak_searches = 0
        self.search_lock = threading.Lock()

    @property
    def count(self):
        return len(self.records)

    def as_retriever(self, search_kwargs=None):
        return {"search_kwargs": search_kwargs or {}}

    def get_document_chunks(self, document_id):
        matches = [
            (chunk_id, document)
            for chunk_id, document in self.records.items()
            if document.metadata.get("document_id") == document_id
        ]
        return {
            "ids": [chunk_id for chunk_id, _ in matches],
            "documents": [document.page_content for _, document in matches],
            "metadatas": [document.metadata for _, document in matches],
        }

    def add_documents(self, documents):
        ids = []
        for document in documents:
            chunk_id = document.metadata["chunk_id"]
            self.records[chunk_id] = document
            ids.append(chunk_id)
        return ids

    def delete(self, ids):
        if self.fail_delete_ids.intersection(ids):
            raise RuntimeError("delete failed")
        for chunk_id in ids:
            self.records.pop(chunk_id, None)

    def similarity_search_with_score(self, query, k):
        del query
        with self.search_lock:
            self.active_searches += 1
            self.peak_searches = max(self.peak_searches, self.active_searches)
        try:
            if self.search_delay:
                time.sleep(self.search_delay)
            return self.search_results[:k]
        finally:
            with self.search_lock:
                self.active_searches -= 1


class InMemoryLegacyCollection:
    def __init__(self):
        self.records = {}

    def count(self):
        return len(self.records)

    def get(self, where=None, include=None):
        del include
        conditions = where or {}
        matches = [
            (record_id, record)
            for record_id, record in self.records.items()
            if all(
                record["metadata"].get(key) == value
                for key, value in conditions.items()
            )
        ]
        return {
            "ids": [record_id for record_id, _ in matches],
            "documents": [record["document"] for _, record in matches],
            "metadatas": [record["metadata"] for _, record in matches],
        }

    def upsert(self, ids, documents, metadatas):
        for record_id, document, metadata in zip(ids, documents, metadatas):
            self.records[str(record_id)] = {
                "document": document,
                "metadata": dict(metadata),
            }

    def delete(self, ids):
        for record_id in ids:
            self.records.pop(str(record_id), None)


class DeterministicEmbeddings(Embeddings):
    def _vector(self, text):
        vector = [0.0] * 384
        if "退款" in text:
            vector[0] = 1.0
        elif "配送" in text:
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        return vector

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class LegacyKnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.collection = InMemoryLegacyCollection()
        self.kb = LegacyKnowledgeBase.__new__(LegacyKnowledgeBase)
        self.kb._collection = self.collection

    def test_duplicate_import_is_noop_and_changed_content_replaces_old_chunks(self):
        document = {"title": "退款政策", "content": "退款会退回原支付账户。"}

        first_count = self.kb.add_documents([document])
        first_ids = set(self.collection.records)
        duplicate_count = self.kb.add_documents([document])
        updated_count = self.kb.add_documents(
            [{"title": "退款政策", "content": "退款通常在三个工作日内原路退回。"}]
        )

        self.assertEqual(first_count, 1)
        self.assertEqual(duplicate_count, 0)
        self.assertEqual(updated_count, 1)
        self.assertEqual(len(self.collection.records), 1)
        self.assertTrue(first_ids.isdisjoint(self.collection.records))
        record = next(iter(self.collection.records.values()))
        self.assertIn("三个工作日", record["document"])
        self.assertIn("document_id", record["metadata"])
        self.assertIn("content_hash", record["metadata"])

    def test_first_reimport_migrates_same_title_legacy_chunks(self):
        self.collection.records["legacy-chunk"] = {
            "document": "旧版退款内容",
            "metadata": {"title": "退款政策", "chunk_index": 0, "total_chunks": 1},
        }

        changed_count = self.kb.add_documents(
            [{"title": "退款政策", "content": "新版退款内容"}]
        )

        self.assertEqual(changed_count, 1)
        self.assertNotIn("legacy-chunk", self.collection.records)
        self.assertEqual(len(self.collection.records), 1)
        metadata = next(iter(self.collection.records.values()))["metadata"]
        self.assertTrue(metadata["document_id"])
        self.assertTrue(metadata["content_hash"])


class DocumentProcessorTests(unittest.TestCase):
    def test_short_text_has_one_metadata_rich_chunk(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.process(
            {"title": "退款", "content": "七天内可以退款。", "source": "faq.md"}
        )

        self.assertEqual(len(chunks), 1)
        metadata = chunks[0].metadata
        for key in (
            "document_id",
            "content_hash",
            "title",
            "source",
            "chunk_index",
            "total_chunks",
            "start_index",
            "updated_at",
            "chunk_id",
        ):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["source"], "faq.md")
        self.assertEqual(metadata["chunk_index"], 0)

    def test_long_text_has_overlap_and_stable_chunk_ids(self):
        processor = DocumentProcessor(chunk_size=40, chunk_overlap=10)
        raw = {"title": "长文", "content": "abcdefghijklmnopqrstuvwxyz" * 5}
        first = processor.process(raw)
        second = processor.process(raw)

        self.assertGreater(len(first), 1)
        self.assertEqual(first[0].page_content[-10:], first[1].page_content[:10])
        self.assertEqual(
            [chunk.metadata["chunk_id"] for chunk in first],
            [chunk.metadata["chunk_id"] for chunk in second],
        )
        self.assertEqual(
            [chunk.metadata["start_index"] for chunk in first],
            sorted(chunk.metadata["start_index"] for chunk in first),
        )

    def test_chinese_punctuation_and_markdown_are_split(self):
        processor = DocumentProcessor(chunk_size=24, chunk_overlap=4)
        content = "# 标题\n\n第一段介绍退款政策；包含细节，便于处理。\n第二段介绍配送规则！"
        chunks = processor.process({"title": "帮助", "content": content})
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))

    def test_same_source_with_distinct_titles_has_distinct_document_ids(self):
        processor = DocumentProcessor()
        first = processor.process({"title": "A", "content": "one", "source": "batch.json"})
        second = processor.process({"title": "B", "content": "two", "source": "batch.json"})
        self.assertNotEqual(
            first[0].metadata["document_id"],
            second[0].metadata["document_id"],
        )

    def test_empty_document_is_ignored(self):
        self.assertEqual(DocumentProcessor().process({"title": "empty", "content": "  "}), [])


class KnowledgeBaseTests(unittest.TestCase):
    def test_duplicate_import_is_idempotent_and_update_removes_old_chunks(self):
        store = InMemoryVectorStore()
        kb = LangChainKnowledgeBase(vector_store=store, load_defaults=False, chunk_size=30, chunk_overlap=5)

        first_count = kb.add_documents([{"title": "政策", "content": "旧内容" * 30}])
        first_ids = set(store.records)
        duplicate_count = kb.add_documents([{"title": "政策", "content": "旧内容" * 30}])
        updated_count = kb.add_documents([{"title": "政策", "content": "新内容" * 30}])

        self.assertGreater(first_count, 0)
        self.assertEqual(duplicate_count, 0)
        self.assertGreater(updated_count, 0)
        self.assertTrue(first_ids.isdisjoint(store.records))
        self.assertTrue(all("旧内容" not in doc.page_content for doc in store.records.values()))

    def test_failed_version_switch_rolls_back_new_chunks(self):
        store = InMemoryVectorStore()
        kb = LangChainKnowledgeBase(vector_store=store, load_defaults=False)
        kb.add_documents([{"title": "政策", "content": "旧版本"}])
        old_ids = set(store.records)
        store.fail_delete_ids = set(old_ids)

        with self.assertRaises(RuntimeError):
            kb.add_documents([{"title": "政策", "content": "新版本"}])

        self.assertEqual(set(store.records), old_ids)
        self.assertTrue(all(doc.page_content == "旧版本" for doc in store.records.values()))

    def test_search_hit_score_and_private_identity_mapping(self):
        store = InMemoryVectorStore()
        store.search_results = [
            (
                Document(
                    page_content="审核通过后退款",
                    metadata={
                        "title": "退款政策",
                        "source": "faq.md",
                        "chunk_index": 2,
                        "chunk_id": "chunk-2",
                        "document_id": "doc-1",
                    },
                ),
                0.2,
            )
        ]
        retriever = KnowledgeRetriever(store)
        hit = retriever.search("退款", top_k=1)[0]
        mapped = hit.to_tool_dict()

        self.assertAlmostEqual(hit.score, 0.8)
        self.assertEqual(mapped["score"], 0.8)
        self.assertEqual(mapped["chunk"], 2)
        self.assertEqual(mapped["_chunk_id"], "chunk-2")
        self.assertEqual(mapped["_document_id"], "doc-1")
        self.assertNotIn("source", mapped)

    def test_retriever_empty_query_and_empty_result(self):
        store = InMemoryVectorStore()
        retriever = KnowledgeRetriever(store)
        self.assertEqual(retriever.search("", 5), [])
        self.assertEqual(retriever.search("anything", 5), [])

    def test_search_hit_clamps_public_score(self):
        hit = SearchHit(title="x", content="y", score=1.5)
        self.assertEqual(hit.to_tool_dict()["score"], 1.0)


class KnowledgeBaseConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_searches_run_concurrently_with_bounded_thread_usage(self):
        store = InMemoryVectorStore()
        store.search_delay = 0.05
        kb = LangChainKnowledgeBase(
            vector_store=store,
            load_defaults=False,
            max_concurrency=3,
        )

        await asyncio.gather(
            *[
                kb.search_handler({"query": f"q-{index}", "top_k": 2}, None)
                for index in range(3)
            ]
        )

        self.assertGreaterEqual(store.peak_searches, 2)
        stats = kb.rag_stats()
        self.assertGreaterEqual(stats["concurrency"]["peak_active_operations"], 2)
        self.assertEqual(stats["concurrency"]["max_concurrency"], 3)
        self.assertEqual(stats["retrieval"]["total"], 3)
        self.assertGreaterEqual(stats["retrieval"]["p95_latency_ms"], 40)


class ChromaIntegrationTests(unittest.TestCase):
    def test_legacy_local_fallback_uses_configured_embedding_cache(self):
        class FakeCollection:
            def count(self):
                return 1

        class FakeClient:
            def __init__(self):
                self.collection_options = None

            def get_or_create_collection(self, **kwargs):
                self.collection_options = kwargs
                return FakeCollection()

        client = FakeClient()
        cache_dir = str(Path("data") / "embedding-cache-test")
        with (
            patch(
                "mcp.knowledge_base.chromadb.HttpClient",
                side_effect=RuntimeError("server unavailable"),
            ),
            patch(
                "mcp.knowledge_base.chromadb.PersistentClient",
                return_value=client,
            ),
        ):
            LegacyKnowledgeBase(
                chroma_host="127.0.0.1",
                chroma_port=9,
                chroma_path="./data/test",
                embedding_cache_dir=cache_dir,
            )

        embedding = client.collection_options["embedding_function"]
        self.assertEqual(
            embedding.DOWNLOAD_PATH,
            str(Path(cache_dir) / "onnx_models" / "all-MiniLM-L6-v2"),
        )

    def test_default_embeddings_convert_numpy_scalars_to_python_floats(self):
        embeddings = ChromaDefaultEmbeddings.__new__(ChromaDefaultEmbeddings)
        embeddings._function = lambda texts: [
            np.asarray([index, index + 0.5], dtype=np.float32)
            for index, _ in enumerate(texts)
        ]

        result = embeddings.embed_documents(["first", "second"])

        self.assertEqual(result, [[0.0, 0.5], [1.0, 1.5]])
        self.assertTrue(all(type(value) is float for vector in result for value in vector))

    def test_http_failure_uses_fresh_settings_for_persistent_fallback(self):
        fallback_client = object()

        def fail_http_client(*, host, port, settings):
            del host, port
            settings.chroma_api_impl = "chromadb.api.fastapi.FastAPI"
            raise RuntimeError("server unavailable")

        with (
            patch("rag.vector_store.chromadb.HttpClient", side_effect=fail_http_client),
            patch(
                "rag.vector_store.chromadb.PersistentClient",
                return_value=fallback_client,
            ) as persistent_client,
        ):
            result = LangChainVectorStore._connect("127.0.0.1", 9, "./data/test")

        self.assertIs(result, fallback_client)
        fallback_settings = persistent_client.call_args.kwargs["settings"]
        self.assertNotEqual(
            fallback_settings.chroma_api_impl,
            "chromadb.api.fastapi.FastAPI",
        )

    def test_explicit_embeddings_collection_metadata_and_scored_search(self):
        client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        collection_name = f"test_{uuid.uuid4().hex}"
        store = LangChainVectorStore(
            client=client,
            collection_name=collection_name,
            embeddings=DeterministicEmbeddings(),
        )
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        store.add_documents(
            processor.process(
                {"title": "退款政策", "content": "退款会退回原支付账户", "source": "faq"}
            )
            + processor.process(
                {"title": "配送说明", "content": "配送需要三到五天", "source": "delivery"}
            )
        )

        results = store.similarity_search_with_score("退款多久到账", k=1)
        metadata = store.raw_store._collection.metadata

        self.assertEqual(results[0][0].metadata["title"], "退款政策")
        self.assertEqual(metadata["embedding_provider"], "chroma_default")
        self.assertEqual(metadata["embedding_model"], "all-MiniLM-L6-v2")
        self.assertEqual(metadata["embedding_dimensions"], 384)


if __name__ == "__main__":
    unittest.main()
