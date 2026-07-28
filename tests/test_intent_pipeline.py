import sys
import types
import unittest
from types import SimpleNamespace


# The unit tests exercise local orchestration logic and never call a real API.
# Prefer the installed SDK so this module does not poison later LangChain imports.
try:  # pragma: no cover - the validated environment always has the SDK
    import anthropic  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for isolated runs
    anthropic_stub = types.ModuleType("anthropic")

    class AsyncAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    anthropic_stub.AsyncAnthropic = AsyncAnthropic
    sys.modules["anthropic"] = anthropic_stub


from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    GeneralAgent,
    Request,
)
from core.intent_recognizer import (
    IntentCategory,
    IntentRecognizer,
    IntentResult,
    UrgencyLevel,
)


def make_recognizer() -> IntentRecognizer:
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer.model = "test-model"
    recognizer.threshold = 0.5
    recognizer._embedding_enabled = False
    recognizer._tpl_embeddings = {}
    recognizer._cache = {}
    recognizer.cache_hits = 0
    recognizer.cache_misses = 0
    return recognizer


class IntentPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_key_includes_recent_history(self):
        recognizer = make_recognizer()
        calls = []

        async def fake_llm(self, message, history):
            calls.append(history)
            history_text = " ".join(item.get("content", "") for item in (history or []))
            intent = (
                IntentCategory.BILLING
                if "membership" in history_text
                else IntentCategory.REQUEST
            )
            return {
                "intent": intent,
                "confidence": 0.9,
                "reasoning": "context-aware test",
                "entities": self._empty_entities(),
            }

        recognizer._llm_recognize = types.MethodType(fake_llm, recognizer)
        member_history = [{"role": "user", "content": "membership purchase"}]
        order_history = [{"role": "user", "content": "change shipping address"}]

        first = await recognizer.recognize("how to undo it", member_history)
        second = await recognizer.recognize("how to undo it", order_history)
        cached = await recognizer.recognize("how to undo it", member_history)

        self.assertEqual(first.intent, IntentCategory.BILLING)
        self.assertEqual(second.intent, IntentCategory.REQUEST)
        self.assertIs(cached, first)
        self.assertEqual(len(calls), 2)
        self.assertEqual(recognizer.cache_hits, 1)
        self.assertEqual(recognizer.cache_misses, 2)

    async def test_recognize_uses_fused_confidence(self):
        recognizer = make_recognizer()

        async def fake_llm(self, message, history):
            return {
                "intent": IntentCategory.BILLING,
                "confidence": 0.8,
                "reasoning": "billing",
                "entities": self._empty_entities(),
            }

        recognizer._llm_recognize = types.MethodType(fake_llm, recognizer)
        result = await recognizer.recognize("refund", None)

        # LLM: 0.8 * 0.85, pattern: 0.25 * 0.15.
        self.assertAlmostEqual(result.confidence, 0.7175)
        self.assertEqual(result.intent, IntentCategory.BILLING)
        self.assertNotEqual(result.confidence, 0.8)

    async def test_entities_are_merged_without_second_llm_call(self):
        recognizer = make_recognizer()
        calls = 0

        async def fake_llm(self, message, history):
            nonlocal calls
            calls += 1
            return {
                "intent": IntentCategory.BILLING,
                "confidence": 0.9,
                "reasoning": "duplicate charge",
                "entities": {
                    "order_id": ["A1234"],
                    "product": ["membership"],
                    "date": [],
                    "amount": [],
                    "error_code": [],
                },
            }

        recognizer._llm_recognize = types.MethodType(fake_llm, recognizer)
        result = await recognizer.recognize(
            "订单号 A1234 重复扣款 299元，页面报错401",
            None,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(result.entities["order_id"], ["A1234"])
        self.assertEqual(result.entities["product"], ["membership"])
        self.assertEqual(result.entities["amount"], ["299元"])
        self.assertEqual(result.entities["error_code"], ["401"])

    async def test_orchestrator_copies_entities_to_request(self):
        entities = {
            "order_id": ["A1234"],
            "product": [],
            "date": [],
            "amount": ["299元"],
            "error_code": [],
        }

        class FakeRecognizer:
            async def recognize(self, message, history=None):
                return IntentResult(
                    intent=IntentCategory.BILLING,
                    confidence=0.9,
                    urgency=UrgencyLevel.LOW,
                    entities=entities,
                    reasoning="billing",
                    latency_ms=1.0,
                )

        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._intent_recognizer = FakeRecognizer()
        orchestrator._pool = {}
        captured = {}

        async def fake_execute(self, req, agent_type):
            captured.update(req.entities)
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="ok",
                success=True,
            )

        orchestrator._execute = types.MethodType(fake_execute, orchestrator)
        request = Request(message="billing question", user_id="u1", conv_id="c1")
        await orchestrator.run(request)

        self.assertEqual(captured, entities)
        self.assertEqual(request.entities, entities)

    async def test_agent_prompt_contains_structured_entities(self):
        class FakeMessages:
            def __init__(self):
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(content=[SimpleNamespace(text="ok")])

        fake_messages = FakeMessages()
        fake_client = SimpleNamespace(messages=fake_messages)
        agent = GeneralAgent(fake_client, "test-model")
        request = Request(
            message="please check it",
            user_id="u1",
            conv_id="c1",
            entities={"order_id": ["A1234"], "amount": ["299元"]},
        )

        response = await agent._call_llm(request)
        background = fake_messages.kwargs["messages"][0]["content"]

        self.assertEqual(response, "ok")
        self.assertIn("A1234", background)
        self.assertIn("299元", background)
        self.assertIn("[已提取实体]", background)


if __name__ == "__main__":
    unittest.main()
