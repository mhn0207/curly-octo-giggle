import unittest
from types import SimpleNamespace

import api.main as api_main


class FakeKnowledgeBase:
    def __init__(self):
        self.doc_count = 7
        self.documents = None

    async def aadd_documents(self, documents):
        self.documents = documents
        return 2

    async def evaluate_backends(self):
        return {"recall_gate_passed": True}


class FakeToolManager:
    def __init__(self):
        self.invalidated = []
        self.search_data = [
            {"title": "????", "content": "???????", "score": 0.9, "chunk": 0}
        ]

    def invalidate_cache(self, tool_name=None):
        self.invalidated.append(tool_name)

    async def search_with_rewrite(self, tool_name, query, top_k):
        del tool_name, query
        return SimpleNamespace(
            success=True,
            data=self.search_data[:top_k],
            reranked=True,
        )


class EncodingSafeOutput:
    encoding = "gbk"

    def __init__(self):
        self.value = ""

    def write(self, value):
        value.encode(self.encoding)
        self.value += value
        return len(value)

    def flush(self):
        return None


class BannerTests(unittest.TestCase):
    def test_banner_is_safe_on_gbk_console(self):
        output = EncodingSafeOutput()
        original = api_main.sys.stdout
        try:
            api_main.sys.stdout = output
            api_main._print_banner()
        finally:
            api_main.sys.stdout = original

        self.assertIn("知应 AI", output.value)


class KnowledgeApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_kb = api_main._knowledge_base
        self.previous_manager = api_main._tool_manager
        self.kb = FakeKnowledgeBase()
        self.manager = FakeToolManager()
        api_main._knowledge_base = self.kb
        api_main._tool_manager = self.manager

    async def asyncTearDown(self):
        api_main._knowledge_base = self.previous_kb
        api_main._tool_manager = self.previous_manager

    async def test_add_contract_and_cache_invalidation(self):
        response = await api_main.add_knowledge(
            api_main.BatchDocInput(
                documents=[api_main.DocInput(title="退款", content="七天内退款")]
            )
        )

        self.assertEqual(
            response,
            {
                "message": "成功导入 2 个文档片段",
                "added_chunks": 2,
                "total_chunks": 7,
            },
        )
        self.assertEqual(self.kb.documents, [{"title": "退款", "content": "七天内退款"}])
        self.assertEqual(self.manager.invalidated, ["knowledge_search"])

    async def test_stats_contract_does_not_depend_on_bound_tool_handler(self):
        self.assertEqual(await api_main.knowledge_stats(), {"total_chunks": 7})

    async def test_search_contract_and_chat_knowledge_context_are_unchanged(self):
        response = await api_main.search("??????", top_k=3)
        context, used = await api_main._build_knowledge_context("??????", top_k=3)

        self.assertEqual(set(response), {"query", "results", "reranked"})
        self.assertEqual(response["results"][0]["title"], "????")
        self.assertTrue(response["reranked"])
        self.assertTrue(used)
        self.assertIn("????", context)

        self.manager.search_data = []
        empty_context, empty_used = await api_main._build_knowledge_context("??????")
        self.assertEqual(empty_context, "")
        self.assertFalse(empty_used)

    async def test_rag_evaluation_endpoint_uses_dual_read_backend(self):
        self.assertEqual(
            await api_main.run_rag_eval(),
            {"recall_gate_passed": True},
        )


if __name__ == "__main__":
    unittest.main()
