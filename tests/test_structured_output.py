import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from core.llm_factory import create_chat_model
from core.structured_invoker import (
    StructuredInvoker,
    StructuredOutputMode,
    parse_json_array,
    parse_json_object,
)
from core.structured_schemas import (
    IntentLLMOutput,
    JudgeOutput,
    QueryRewriteOutput,
    RerankOutput,
    UserProfileOutput,
)
from evaluation.evaluator import LLMJudge
from mcp.tool_manager import MCPToolManager, Tool


class FakeRunnable:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    async def ainvoke(self, messages):
        del messages
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeModel:
    def __init__(self, runnable):
        self.runnable = runnable
        self.bind_calls = 0

    def with_structured_output(self, schema, **kwargs):
        del schema, kwargs
        self.bind_calls += 1
        return self.runnable


class SchemaOutputInvoker:
    def __init__(self, outputs):
        self.outputs = outputs

    async def ainvoke(self, schema, messages, legacy_fallback=None):
        del messages, legacy_fallback
        return self.outputs[schema]


class StructuredSchemaTests(unittest.TestCase):
    def test_valid_and_optional_outputs(self):
        intent = IntentLLMOutput(intent="query", confidence=0.8)
        profile = UserProfileOutput()
        self.assertEqual(intent.entities.order_id, [])
        self.assertEqual(profile.entities.products, [])

    def test_invalid_extra_field_and_range_are_rejected(self):
        with self.assertRaises(ValidationError):
            IntentLLMOutput(intent="query", confidence=1.1)
        with self.assertRaises(ValidationError):
            JudgeOutput(
                relevance=0.8,
                accuracy=0.8,
                completeness=0.8,
                helpfulness=0.8,
                invented=True,
            )

    def test_invalid_intent_and_scalar_entity_are_rejected(self):
        with self.assertRaises(ValidationError):
            IntentLLMOutput(intent="invented", confidence=0.8)
        with self.assertRaises(ValidationError):
            IntentLLMOutput(
                intent="billing",
                confidence=0.8,
                entities={"order_id": "A1234"},
            )

    def test_json_fragment_parser_handles_surrounding_text(self):
        self.assertEqual(parse_json_object('说明 {"ok": true} 尾注'), {"ok": True})
        self.assertEqual(parse_json_array('```json\n[2, 0, 1]\n```'), [2, 0, 1])
        with self.assertRaises(ValueError):
            parse_json_object("没有结构化内容")

    def test_target_modules_do_not_parse_json_fragments_themselves(self):
        root = Path(__file__).resolve().parent.parent
        for relative in (
            "core/intent_recognizer.py",
            "mcp/tool_manager.py",
            "memory/conversation_memory.py",
            "evaluation/evaluator.py",
        ):
            content = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn("raw.find(", content)
            self.assertNotIn("raw.rfind(", content)


class ChatModelFactoryTests(unittest.TestCase):
    def test_official_and_custom_base_url_bind_structured_tools(self):
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_URL": "https://api.anthropic.com"},
        ):
            official = create_chat_model(
                api_key="official-secret",
                model="claude-test",
                max_retries=0,
            )
        custom = create_chat_model(
            api_key="third-party-secret",
            model="claude-test",
            base_url="https://provider.example.invalid",
            max_retries=0,
        )

        self.assertEqual(official.anthropic_api_url, "https://api.anthropic.com")
        self.assertEqual(custom.anthropic_api_url, "https://provider.example.invalid")
        self.assertNotIn("third-party-secret", repr(custom))
        self.assertIsNotNone(custom.with_structured_output(IntentLLMOutput))


class StructuredInvokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_failure_retries_then_succeeds(self):
        runnable = FakeRunnable(
            [
                {"intent": "query", "confidence": 3.0},
                {"intent": "query", "confidence": 0.9},
            ]
        )
        invoker = StructuredInvoker(
            FakeModel(runnable),
            mode=StructuredOutputMode.TOOL,
            validation_retries=1,
            component="test",
        )

        result = await invoker.ainvoke(IntentLLMOutput, [{"role": "user", "content": "x"}])

        self.assertEqual(result.intent, "query")
        self.assertEqual(runnable.calls, 2)
        self.assertEqual(invoker.stats.validation_failures, 1)
        self.assertEqual(invoker.stats.retries, 1)
        self.assertEqual(invoker.stats.success, 1)

    async def test_model_failure_does_not_retry_and_uses_fallback(self):
        runnable = FakeRunnable([RuntimeError("provider unavailable")])
        invoker = StructuredInvoker(
            FakeModel(runnable),
            mode=StructuredOutputMode.TOOL,
            validation_retries=2,
            component="test",
        )

        async def fallback():
            return QueryRewriteOutput(queries=["fallback query"])

        result = await invoker.ainvoke(
            QueryRewriteOutput,
            [{"role": "user", "content": "x"}],
            legacy_fallback=fallback,
        )

        self.assertEqual(result.queries, ["fallback query"])
        self.assertEqual(runnable.calls, 1)
        self.assertEqual(invoker.stats.model_failures, 1)
        self.assertEqual(invoker.stats.fallbacks, 1)
        self.assertEqual(invoker.stats.retries, 0)

    async def test_legacy_mode_is_primary_even_when_fallback_is_disabled(self):
        invoker = StructuredInvoker(
            None,
            mode=StructuredOutputMode.LEGACY_JSON,
            fallback_enabled=False,
            component="test",
        )

        async def legacy_call():
            return QueryRewriteOutput(queries=["legacy"])

        result = await invoker.ainvoke(
            QueryRewriteOutput,
            [{"role": "user", "content": "x"}],
            legacy_fallback=legacy_call,
        )
        self.assertEqual(result.queries, ["legacy"])
        self.assertEqual(invoker.stats.success, 1)
        self.assertEqual(invoker.stats.fallbacks, 0)
        self.assertEqual(invoker.stats_snapshot()["mode"], "legacy_json")

    async def test_shadow_mode_compares_both_and_returns_legacy_result(self):
        runnable = FakeRunnable([{"queries": ["structured"]}])
        invoker = StructuredInvoker(
            FakeModel(runnable),
            mode=StructuredOutputMode.TOOL,
            component="test",
            shadow_enabled=True,
        )

        async def legacy_call():
            return QueryRewriteOutput(queries=["legacy"])

        result = await invoker.ainvoke(
            QueryRewriteOutput,
            [{"role": "user", "content": "x"}],
            legacy_fallback=legacy_call,
        )
        stats = invoker.stats_snapshot()
        self.assertEqual(result.queries, ["legacy"])
        self.assertEqual(stats["shadow_calls"], 1)
        self.assertEqual(stats["shadow_mismatches"], 1)
        self.assertEqual(stats["success"], 1)

    async def test_shadow_mode_uses_structured_result_if_legacy_fails(self):
        runnable = FakeRunnable([{"queries": ["structured"]}])
        invoker = StructuredInvoker(
            FakeModel(runnable),
            mode=StructuredOutputMode.TOOL,
            component="test",
            shadow_enabled=True,
        )

        async def legacy_call():
            raise RuntimeError("legacy failed")

        result = await invoker.ainvoke(
            QueryRewriteOutput,
            [{"role": "user", "content": "x"}],
            legacy_fallback=legacy_call,
        )
        self.assertEqual(result.queries, ["structured"])
        self.assertEqual(invoker.stats.shadow_legacy_failures, 1)
        self.assertEqual(invoker.stats.fallbacks, 1)

    async def test_structured_and_legacy_failures_raise_without_sensitive_logs(self):
        secret = "customer-card-4111111111111111"
        invoker = StructuredInvoker(
            FakeModel(FakeRunnable([RuntimeError(secret)])),
            mode=StructuredOutputMode.TOOL,
            component="test",
        )

        async def failing_fallback():
            raise RuntimeError(secret)

        with self.assertLogs("core.structured_invoker", level="WARNING") as captured:
            with self.assertRaises(RuntimeError):
                await invoker.ainvoke(
                    QueryRewriteOutput,
                    [{"role": "user", "content": secret}],
                    legacy_fallback=failing_fallback,
                )

        self.assertNotIn(secret, "\n".join(captured.output))
        self.assertEqual(invoker.stats.fallback_failures, 1)

    async def test_unsupported_native_mode_uses_fallback(self):
        invoker = StructuredInvoker(
            FakeModel(FakeRunnable([])),
            mode=StructuredOutputMode.NATIVE,
            component="test",
        )

        async def fallback():
            return RerankOutput(ordered_indexes=[1, 0])

        with patch.object(StructuredInvoker, "_supports_native_json_schema", return_value=False):
            result = await invoker.ainvoke(
                RerankOutput,
                [{"role": "user", "content": "x"}],
                legacy_fallback=fallback,
            )
        self.assertEqual(result.ordered_indexes, [1, 0])
        self.assertEqual(invoker.stats.fallbacks, 1)


class JudgeFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_failure_keeps_failure_marker(self):
        class FailingMessages:
            async def create(self, **kwargs):
                del kwargs
                raise RuntimeError("judge offline")

        client = SimpleNamespace(messages=FailingMessages())
        scores = await LLMJudge(client, "test-model").judge("question", "answer")

        self.assertTrue(scores.judge_failed)
        self.assertEqual(scores.overall, 0.5)
        self.assertIn("judge offline", scores.error)


class ToolManagerStructuredTests(unittest.IsolatedAsyncioTestCase):
    def make_manager(self, rewrite, rerank):
        invoker = SchemaOutputInvoker(
            {
                QueryRewriteOutput: QueryRewriteOutput(queries=rewrite),
                RerankOutput: RerankOutput(ordered_indexes=rerank),
            }
        )
        return MCPToolManager(
            api_key="test-key",
            rewrite_invoker=invoker,
            rerank_invoker=invoker,
        )

    async def test_rewrite_normalizes_and_deduplicates(self):
        manager = self.make_manager(["原问题", "  子查询  ", "子查询", "另一个"], [])
        result = await manager.rewrite_query("原问题", n=3)
        self.assertEqual(result, ["原问题", "子查询", "另一个"])

    async def test_rerank_filters_duplicate_out_of_range_and_missing_indexes(self):
        manager = self.make_manager(["unused"], [2, 2, 99, -1])
        items = ["a", "b", "c", "d"]
        result = await manager._rerank("query", items, top_k=3)
        self.assertEqual(result, ["c", "a", "b"])

    async def test_multi_query_deduplicates_by_chunk_id_and_hides_internal_metadata(self):
        manager = self.make_manager(["第二问法"], [0, 1, 2])

        async def handler(params, context):
            del context
            query = params["query"]
            return [
                {
                    "title": "common",
                    "content": "same",
                    "score": 0.9,
                    "chunk": 0,
                    "_chunk_id": "shared",
                    "_document_id": "doc-shared",
                },
                {
                    "title": query,
                    "content": query,
                    "score": 0.8,
                    "chunk": 0,
                    "_chunk_id": f"chunk-{query}",
                },
            ]

        manager.register(
            Tool(
                name="knowledge_search",
                description="test",
                handler=handler,
                schema={"type": "object"},
                cache_ttl=60,
            )
        )
        result = await manager.search_with_rewrite("knowledge_search", "原问题", top_k=3)

        self.assertTrue(result.success)
        self.assertEqual(len(result.data), 3)
        self.assertTrue(all(not any(key.startswith("_") for key in item) for item in result.data))

    def test_cache_invalidation_is_scoped(self):
        manager = self.make_manager(["unused"], [])
        manager._cache = {
            "knowledge_search:a": ([], 99),
            "other:b": ([], 99),
        }
        manager.invalidate_cache("knowledge_search")
        self.assertEqual(list(manager._cache), ["other:b"])


if __name__ == "__main__":
    unittest.main()
