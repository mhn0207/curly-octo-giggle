import types
import unittest

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    Request,
)
from agents.result_synthesizer import (
    AgentWorkResult,
    FactRecord,
    Recommendation,
    SynthesisAgent,
)
from core.intent_recognizer import IntentCategory


def _tool_call(
    tool_name,
    result,
    *,
    tool_use_id,
    success=True,
    error=None,
):
    return {
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "input": {"order_id": result.get("order_id", "A123")},
        "success": success,
        "cached": False,
        "latency_ms": 1.0,
        "error": error,
        "result": result,
    }


class ResultSynthesizerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.synthesizer = SynthesisAgent()

    def test_tool_facts_and_recommendations_are_deduplicated(self):
        order = {
            "order_id": "A123",
            "product": "知应 AI 专业版月度订阅",
            "status": "paid",
            "amount": "99.00",
            "currency": "CNY",
        }
        technical = AgentResponse(
            agent_type=AgentType.TECHNICAL,
            content="请重新登录后再试。",
            success=True,
            tool_calls=[
                _tool_call(
                    "get_order_status",
                    order,
                    tool_use_id="tool-1",
                )
            ],
        )
        billing = AgentResponse(
            agent_type=AgentType.BILLING,
            content="请重新登录后再试。",
            success=True,
            tool_calls=[
                _tool_call(
                    "get_order_status",
                    order,
                    tool_use_id="tool-2",
                )
            ],
        )

        outcome = self.synthesizer.synthesize([technical, billing])

        self.assertEqual(len(outcome.confirmed_facts), 4)
        self.assertEqual(outcome.response.count("请重新登录后再试。"), 1)
        self.assertEqual(len(outcome.tool_calls), 2)
        self.assertFalse(outcome.conflicts)

    def test_same_priority_business_conflict_requires_human(self):
        first = AgentResponse(
            agent_type=AgentType.TECHNICAL,
            content="技术侧检查完成。",
            success=True,
        )
        first.work_result = AgentWorkResult(
            agent_type="technical",
            success=True,
            confirmed_facts=[
                FactRecord(
                    key="order:A123:status",
                    label="订单 A123 状态",
                    value="paid",
                    source="tool:order_primary",
                    source_kind="tool",
                    agent_type="technical",
                )
            ],
        )
        second = AgentResponse(
            agent_type=AgentType.BILLING,
            content="账单侧检查完成。",
            success=True,
        )
        second.work_result = AgentWorkResult(
            agent_type="billing",
            success=True,
            confirmed_facts=[
                FactRecord(
                    key="order:A123:status",
                    label="订单 A123 状态",
                    value="cancelled",
                    source="tool:order_replica",
                    source_kind="tool",
                    agent_type="billing",
                )
            ],
        )

        outcome = self.synthesizer.synthesize([first, second])

        self.assertTrue(outcome.requires_human)
        self.assertTrue(outcome.conflicts[0].blocking)
        self.assertFalse(outcome.conflicts[0].resolved)
        self.assertNotIn(
            "order:A123:status",
            [fact.key for fact in outcome.confirmed_facts],
        )
        self.assertIn("高风险操作不应继续自动执行", outcome.response)

    def test_tool_fact_wins_over_knowledge_base_assumption(self):
        response = AgentResponse(
            agent_type=AgentType.BILLING,
            content="已核对订单。",
            success=True,
        )
        response.work_result = AgentWorkResult(
            agent_type="billing",
            success=True,
            confirmed_facts=[
                FactRecord(
                    key="order:A123:status",
                    label="订单 A123 状态",
                    value="paid",
                    source="tool:get_order_status",
                    source_kind="tool",
                ),
                FactRecord(
                    key="order:A123:status",
                    label="订单 A123 状态",
                    value="pending",
                    source="knowledge_base:article-1",
                    source_kind="knowledge_base",
                ),
            ],
            recommended_actions=[
                Recommendation(
                    agent_type="billing",
                    content="已核对订单。",
                )
            ],
        )

        outcome = self.synthesizer.synthesize([response])

        self.assertEqual(outcome.confirmed_facts[0].value, "paid")
        self.assertTrue(outcome.conflicts[0].resolved)
        self.assertFalse(outcome.requires_human)

    def test_partial_failure_keeps_available_result_and_marks_degraded(self):
        successful = AgentResponse(
            agent_type=AgentType.TECHNICAL,
            content="请清理缓存并重新登录。",
            success=True,
        )
        failed = AgentResponse(
            agent_type=AgentType.BILLING,
            content="账单服务暂时不可用。",
            success=False,
        )

        outcome = self.synthesizer.synthesize([successful, failed])

        self.assertTrue(outcome.partial_failure)
        self.assertIn("请清理缓存并重新登录", outcome.response)
        self.assertIn("部分 Agent 未完成处理", outcome.response)
        self.assertNotIn("[technical]", outcome.response)

    def test_pending_refund_is_exposed_as_approval_required(self):
        response = AgentResponse(
            agent_type=AgentType.BILLING,
            content="退款申请已提交。",
            success=True,
            tool_calls=[
                _tool_call(
                    "create_refund_request",
                    {
                        "request_id": "RR-001",
                        "order_id": "A123",
                        "payment_id": "PAY-A123-002",
                        "amount": "99.00",
                        "currency": "CNY",
                        "status": "pending_review",
                    },
                    tool_use_id="refund-1",
                )
            ],
        )

        outcome = self.synthesizer.synthesize([response])

        self.assertTrue(outcome.requires_approval)
        self.assertIn("尚未发生实际退款", outcome.response)

    async def test_parallel_orchestrator_uses_structured_synthesis(self):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._synthesis_agent = SynthesisAgent()
        responses = {
            AgentType.TECHNICAL: AgentResponse(
                agent_type=AgentType.TECHNICAL,
                content="技术侧建议重新登录。",
                success=True,
            ),
            AgentType.BILLING: AgentResponse(
                agent_type=AgentType.BILLING,
                content="账单侧未发现退款完成记录。",
                success=True,
            ),
        }

        async def fake_execute(self, req, agent_type):
            return responses[agent_type]

        orchestrator._execute = types.MethodType(fake_execute, orchestrator)
        request = Request(
            message="登录报错并且退款没到账",
            user_id="u1",
            conv_id="c1",
            intent=IntentCategory.BILLING,
        )

        result = await orchestrator.run_parallel(
            request,
            [AgentType.TECHNICAL, AgentType.BILLING],
        )

        self.assertIn("综合各专业 Agent", result.response)
        self.assertNotIn("[technical]", result.response)
        self.assertIsNotNone(result.synthesis)
        self.assertEqual(len(result.synthesis["agents"]), 2)

    async def test_parallel_exception_becomes_partial_failure(self):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)

        async def fake_execute(self, req, agent_type):
            if agent_type == AgentType.BILLING:
                raise RuntimeError("billing unavailable")
            return AgentResponse(
                agent_type=AgentType.TECHNICAL,
                content="技术侧结果仍然可用。",
                success=True,
            )

        orchestrator._execute = types.MethodType(fake_execute, orchestrator)
        request = Request(
            message="登录失败并且支付异常",
            user_id="u1",
            conv_id="c1",
            intent=IntentCategory.BILLING,
        )

        result = await orchestrator.run_parallel(
            request,
            [AgentType.TECHNICAL, AgentType.BILLING],
        )

        self.assertIn("技术侧结果仍然可用", result.response)
        self.assertTrue(result.synthesis["partial_failure"])
        self.assertIn("部分 Agent 未完成处理", result.response)


if __name__ == "__main__":
    unittest.main()
