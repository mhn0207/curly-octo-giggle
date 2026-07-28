import unittest

from evaluation.rag_evaluator import RAGEvaluator, RAGTestCase
from rag.rollout import KnowledgeBaseRollout, RAGRolloutMode


class FakeBackend:
    def __init__(self, results=None, *, doc_count=3):
        self.results = results or {}
        self.doc_count = doc_count
        self.fail_search = False
        self.fail_add = False
        self.search_calls = []
        self.sync_search_calls = []
        self.add_calls = []

    async def search_handler(self, params, context):
        del context
        self.search_calls.append(params["query"])
        if self.fail_search:
            raise RuntimeError("search failed")
        value = self.results.get(params["query"], self.results.get("*", []))
        return list(value)[: params["top_k"]]

    async def aadd_documents(self, documents):
        self.add_calls.append(list(documents))
        if self.fail_add:
            raise RuntimeError("add failed")
        return len(documents)

    def search(self, query, top_k):
        self.sync_search_calls.append(query)
        if self.fail_search:
            raise RuntimeError("search failed")
        return list(self.results.get(query, self.results.get("*", [])))[:top_k]

    def add_documents(self, documents):
        self.add_calls.append(list(documents))
        if self.fail_add:
            raise RuntimeError("add failed")
        return len(documents)

    def rag_stats(self):
        return {"retrieval": {"total": len(self.search_calls)}}


class RAGEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_recall_mrr_and_comparison_gate(self):
        cases = [
            RAGTestCase("q1", ("A",)),
            RAGTestCase("q2", ("B",)),
        ]

        async def legacy(query, top_k):
            values = {
                "q1": [{"title": "A"}],
                "q2": [{"title": "X"}, {"title": "B"}],
            }
            return values[query][:top_k]

        async def langchain(query, top_k):
            values = {
                "q1": [{"title": "A"}],
                "q2": [{"title": "B"}],
            }
            return values[query][:top_k]

        report = await RAGEvaluator(recall_ks=(1, 3)).compare(
            legacy,
            langchain,
            cases,
        )

        self.assertEqual(report["legacy"]["recall"]["recall_at_1"], 0.5)
        self.assertEqual(report["legacy"]["mrr"], 0.75)
        self.assertEqual(report["langchain"]["recall"]["recall_at_1"], 1.0)
        self.assertTrue(report["recall_gate_passed"])


class RAGRolloutTests(unittest.IsolatedAsyncioTestCase):
    def make_backends(self):
        legacy = FakeBackend({"*": [{"title": "legacy", "chunk": 0}]})
        langchain = FakeBackend({"*": [{"title": "langchain", "chunk": 0}]})
        return legacy, langchain

    async def test_shadow_queries_both_but_returns_legacy(self):
        legacy, langchain = self.make_backends()
        router = KnowledgeBaseRollout(
            legacy,
            langchain,
            mode=RAGRolloutMode.SHADOW,
        )

        result = await router.search_handler({"query": "q", "top_k": 3}, None)
        stats = router.rag_stats()["rollout"]

        self.assertEqual(result[0]["title"], "legacy")
        self.assertEqual(legacy.search_calls, ["q"])
        self.assertEqual(langchain.search_calls, ["q"])
        self.assertEqual(stats["shadow_comparisons"], 1)
        self.assertEqual(stats["legacy_responses"], 1)

    async def test_shadow_records_langchain_read_failure(self):
        legacy, langchain = self.make_backends()
        langchain.fail_search = True
        router = KnowledgeBaseRollout(
            legacy,
            langchain,
            mode=RAGRolloutMode.SHADOW,
        )

        result = await router.search_handler({"query": "q", "top_k": 3}, None)
        stats = router.rag_stats()["rollout"]

        self.assertEqual(result[0]["title"], "legacy")
        self.assertEqual(stats["dual_read_attempts"], 1)
        self.assertEqual(stats["langchain_read_failures"], 1)
        self.assertEqual(stats["shadow_comparisons"], 0)

    async def test_sync_shadow_dual_reads_and_sync_import_dual_writes(self):
        legacy, langchain = self.make_backends()
        router = KnowledgeBaseRollout(
            legacy,
            langchain,
            mode=RAGRolloutMode.SHADOW,
        )

        result = router.search("q", 1)
        count = router.add_documents([{"title": "A", "content": "B"}])

        self.assertEqual(result[0]["title"], "legacy")
        self.assertEqual(legacy.sync_search_calls, ["q"])
        self.assertEqual(langchain.sync_search_calls, ["q"])
        self.assertEqual(count, 1)
        self.assertEqual(len(legacy.add_calls), 1)
        self.assertEqual(len(langchain.add_calls), 1)

    async def test_canary_percentage_selects_deterministically(self):
        legacy, langchain = self.make_backends()
        all_new = KnowledgeBaseRollout(
            legacy,
            langchain,
            mode=RAGRolloutMode.CANARY,
            canary_percent=100,
        )
        no_new = KnowledgeBaseRollout(
            legacy,
            langchain,
            mode=RAGRolloutMode.CANARY,
            canary_percent=0,
        )

        new_result = await all_new.search_handler({"query": "same", "top_k": 1}, None)
        old_result = await no_new.search_handler({"query": "same", "top_k": 1}, None)

        self.assertEqual(new_result[0]["title"], "langchain")
        self.assertEqual(old_result[0]["title"], "legacy")

    async def test_langchain_failure_falls_back_to_legacy(self):
        legacy, langchain = self.make_backends()
        langchain.fail_search = True
        router = KnowledgeBaseRollout(
            legacy,
            langchain,
            mode=RAGRolloutMode.LANGCHAIN,
        )

        result = await router.search_handler({"query": "q", "top_k": 1}, None)
        stats = router.rag_stats()["rollout"]

        self.assertEqual(result[0]["title"], "legacy")
        self.assertEqual(stats["primary_failures"], 1)
        self.assertEqual(stats["fallback_successes"], 1)

    async def test_document_import_dual_writes_and_tracks_secondary_failure(self):
        legacy, langchain = self.make_backends()
        router = KnowledgeBaseRollout(
            legacy,
            langchain,
            mode=RAGRolloutMode.LANGCHAIN,
        )
        documents = [{"title": "A", "content": "B"}]

        count = await router.aadd_documents(documents)
        self.assertEqual(count, 1)
        self.assertEqual(len(legacy.add_calls), 1)
        self.assertEqual(len(langchain.add_calls), 1)

        legacy.fail_add = True
        count = await router.aadd_documents(documents)
        self.assertEqual(count, 1)
        self.assertEqual(router.rag_stats()["rollout"]["dual_write_failures"], 1)

    async def test_backend_evaluation_is_available_in_legacy_rollout_mode(self):
        cases = [RAGTestCase("q", ("langchain",))]
        legacy, langchain = self.make_backends()
        router = KnowledgeBaseRollout(
            legacy,
            langchain,
            mode=RAGRolloutMode.LEGACY,
        )
        report = await router.evaluate_backends(cases)
        self.assertFalse(report["recall_gate_passed"] is None)
        self.assertEqual(report["langchain"]["recall"]["recall_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
