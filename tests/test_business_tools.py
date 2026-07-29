import asyncio
import copy
import unittest
from types import SimpleNamespace

from agents.agent_orchestrator import BillingAgent, Request
from mcp.business_tools import register_business_tools
from mcp.tool_manager import MCPToolManager, Tool


def make_manager() -> MCPToolManager:
    manager = MCPToolManager(
        api_key="test-key",
        rewrite_invoker=object(),
        rerank_invoker=object(),
    )
    register_business_tools(manager)
    return manager


def billing_context(request_id: str = "req-1") -> dict:
    return {
        "agent_type": "billing",
        "request_id": request_id,
        "user_id": "u1001",
        "conv_id": "conv-1",
    }


class BusinessToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_payment_query_detects_duplicate_charge(self):
        manager = make_manager()

        result = await manager.call(
            "query_payment",
            {"order_id": "a123"},
            context=billing_context(),
        )

        self.assertTrue(result.success)
        self.assertTrue(result.data["duplicate_payment_detected"])
        self.assertEqual(
            result.data["duplicate_payment_ids"],
            ["PAY-A123-001", "PAY-A123-002"],
        )
        self.assertEqual(len(result.data["payments"]), 2)

    async def test_permissions_and_pydantic_validation_are_enforced(self):
        manager = make_manager()

        denied = await manager.call(
            "query_payment",
            {"order_id": "A123"},
            context={"agent_type": "technical", "request_id": "req-denied"},
        )
        invalid = await manager.call(
            "get_order_status",
            {"order_id": "A!"},
            context={"agent_type": "general", "request_id": "req-invalid"},
        )

        self.assertFalse(denied.success)
        self.assertIn("cannot call", denied.error)
        self.assertFalse(invalid.success)
        self.assertIn("Parameter validation failed", invalid.error)

        statuses = [
            item["status"]
            for item in manager.get_recent_executions(limit=10)
        ]
        self.assertEqual(statuses, ["failed", "failed"])

    async def test_refund_request_is_pending_review_and_idempotent(self):
        manager = make_manager()
        params = {
            "order_id": "A123",
            "payment_id": "PAY-A123-002",
            "amount": "99.00",
            "reason": "duplicate charge",
            "idempotency_key": "refund:A123:duplicate",
        }

        first = await manager.call(
            "create_refund_request",
            params,
            context=billing_context("req-refund-1"),
        )
        replay = await manager.call(
            "create_refund_request",
            params,
            context=billing_context("req-refund-2"),
        )

        self.assertTrue(first.success)
        self.assertEqual(first.data["status"], "pending_review")
        self.assertFalse(first.data["idempotent_replay"])
        self.assertEqual(first.data["request_id"], replay.data["request_id"])
        self.assertTrue(replay.data["idempotent_replay"])
        self.assertIn("尚未执行真实退款", first.data["message"])

    async def test_refund_amount_cannot_exceed_payment(self):
        manager = make_manager()

        result = await manager.call(
            "create_refund_request",
            {
                "order_id": "A123",
                "payment_id": "PAY-A123-002",
                "amount": "199.00",
                "reason": "duplicate charge",
            },
            context=billing_context(),
        )

        self.assertFalse(result.success)
        self.assertIn("超过支付记录可退金额", result.error)

    async def test_timeout_is_bounded_and_audited(self):
        manager = make_manager()

        async def slow_handler(params, context):
            del params, context
            await asyncio.sleep(0.05)
            return {"ok": True}

        manager.register(
            Tool(
                name="slow_tool",
                description="test timeout",
                handler=slow_handler,
                schema={"type": "object"},
                allowed_agents={"billing"},
                timeout_s=0.001,
            )
        )

        result = await manager.call(
            "slow_tool",
            {},
            context=billing_context("req-timeout"),
        )

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)
        execution = manager.get_recent_executions(
            limit=10,
            request_id="req-timeout",
        )[0]
        self.assertEqual(execution["status"], "failed")
        self.assertGreaterEqual(execution["latency_ms"], 0.0)


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise AssertionError("unexpected model call")
        return SimpleNamespace(content=self._responses.pop(0))


class AgentToolCallingTests(unittest.IsolatedAsyncioTestCase):
    async def test_billing_agent_runs_multi_step_native_tool_loop(self):
        manager = make_manager()
        messages = FakeMessages(
            [
                [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "get_order_status",
                        "input": {"order_id": "A123"},
                    }
                ],
                [
                    {
                        "type": "tool_use",
                        "id": "tool-2",
                        "name": "query_payment",
                        "input": {"order_id": "A123"},
                    }
                ],
                [
                    {
                        "type": "tool_use",
                        "id": "tool-3",
                        "name": "create_refund_request",
                        "input": {
                            "order_id": "A123",
                            "payment_id": "PAY-A123-002",
                            "amount": "99.00",
                            "reason": "duplicate charge",
                            "idempotency_key": "refund:A123:agent-loop",
                        },
                    }
                ],
                [
                    {
                        "type": "text",
                        "text": (
                            "已确认订单 A123 存在两笔成功扣款；退款申请已创建，"
                            "当前状态为 pending_review，尚未执行真实退款。"
                        ),
                    }
                ],
            ]
        )
        agent = BillingAgent(
            SimpleNamespace(messages=messages),
            "test-model",
            tool_manager=manager,
            max_tool_steps=4,
        )
        request = Request(
            message="订单 A123 被重复扣款了，请退款",
            user_id="u1001",
            conv_id="conv-agent",
        )

        response = await agent.handle(request)

        self.assertTrue(response.success)
        self.assertEqual(
            [item["tool_name"] for item in response.tool_calls],
            ["get_order_status", "query_payment", "create_refund_request"],
        )
        self.assertIn("pending_review", response.content)
        self.assertEqual(len(messages.calls), 4)
        self.assertEqual(
            {tool["name"] for tool in messages.calls[0]["tools"]},
            {"get_order_status", "query_payment", "create_refund_request"},
        )
        execution_names = [
            item["tool_name"]
            for item in manager.get_recent_executions(
                limit=10,
                request_id=request.request_id,
            )
        ]
        self.assertEqual(
            execution_names,
            ["get_order_status", "query_payment", "create_refund_request"],
        )

    async def test_tool_step_limit_forces_final_answer_without_more_tools(self):
        manager = make_manager()
        messages = FakeMessages(
            [
                [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "get_order_status",
                        "input": {"order_id": "A123"},
                    }
                ],
                [
                    {
                        "type": "tool_use",
                        "id": "tool-2",
                        "name": "query_payment",
                        "input": {"order_id": "A123"},
                    }
                ],
                [{"type": "text", "text": "Reached the tool limit safely."}],
            ]
        )
        agent = BillingAgent(
            SimpleNamespace(messages=messages),
            "test-model",
            tool_manager=manager,
            max_tool_steps=2,
        )

        response = await agent.handle(
            Request(
                message="查询订单 A123 的支付问题",
                user_id="u1001",
                conv_id="conv-limit",
            )
        )

        self.assertTrue(response.success)
        self.assertEqual(response.content, "Reached the tool limit safely.")
        self.assertEqual(len(response.tool_calls), 2)
        self.assertIn("tools", messages.calls[0])
        self.assertIn("tools", messages.calls[1])
        self.assertNotIn("tools", messages.calls[2])


if __name__ == "__main__":
    unittest.main()
