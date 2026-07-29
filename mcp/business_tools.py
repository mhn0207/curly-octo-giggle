"""可独立运行的模拟订单、支付和退款申请工具。"""

from __future__ import annotations

import hashlib
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mcp.tool_manager import MCPToolManager, Tool


class BusinessToolError(ValueError):
    """可安全返回给 Agent 的业务错误，不包含内部堆栈或敏感信息。"""


class StrictToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _normalize_order_id(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise ValueError("订单号不能为空")
    if not (3 <= len(normalized) <= 32):
        raise ValueError("订单号长度必须在 3 到 32 个字符之间")
    if not all(character.isalnum() or character == "-" for character in normalized):
        raise ValueError("订单号只能包含字母、数字和连字符")
    return normalized


class GetOrderStatusInput(StrictToolModel):
    order_id: str = Field(description="需要查询的订单号，例如 A123")

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, value: str) -> str:
        return _normalize_order_id(value)


class GetOrderStatusOutput(StrictToolModel):
    order_id: str
    product: str
    status: Literal["pending", "paid", "shipped", "completed", "cancelled"]
    amount: Decimal
    currency: str
    created_at: str
    updated_at: str


class QueryPaymentInput(StrictToolModel):
    order_id: str = Field(description="需要查询支付记录的订单号")

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, value: str) -> str:
        return _normalize_order_id(value)


class PaymentRecord(StrictToolModel):
    payment_id: str
    amount: Decimal
    currency: str
    channel: str
    status: Literal["pending", "captured", "failed", "refunded"]
    paid_at: str


class QueryPaymentOutput(StrictToolModel):
    order_id: str
    payments: List[PaymentRecord]
    duplicate_payment_detected: bool
    duplicate_payment_ids: List[str] = Field(default_factory=list)


class CreateRefundRequestInput(StrictToolModel):
    order_id: str = Field(description="需要申请退款的订单号")
    payment_id: Optional[str] = Field(
        default=None,
        description="指定退款的支付记录；重复扣款时应选择需要退回的那一笔",
    )
    amount: Optional[Decimal] = Field(
        default=None,
        gt=0,
        max_digits=12,
        decimal_places=2,
        description="申请退款金额；省略时使用所选支付记录的金额",
    )
    reason: str = Field(min_length=3, max_length=300, description="退款原因")
    idempotency_key: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
        description="可选幂等键；相同键重复调用不会创建第二条申请",
    )

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, value: str) -> str:
        return _normalize_order_id(value)


class CreateRefundRequestOutput(StrictToolModel):
    request_id: str
    order_id: str
    payment_id: str
    amount: Decimal
    currency: str
    reason: str
    status: Literal["pending_review"]
    idempotency_key: str
    idempotent_replay: bool = False
    created_at: str
    message: str


class _Order(StrictToolModel):
    order_id: str
    customer_id: str
    product: str
    status: Literal["pending", "paid", "shipped", "completed", "cancelled"]
    amount: Decimal
    currency: str
    created_at: str
    updated_at: str


class MockBusinessRepository:
    """线程安全、可重复初始化的内存业务仓库。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._orders: Dict[str, _Order] = {}
        self._payments: Dict[str, List[PaymentRecord]] = {}
        self._refunds_by_key: Dict[str, CreateRefundRequestOutput] = {}
        self._seed()

    def _seed(self) -> None:
        self._orders = {
            "A123": _Order(
                order_id="A123",
                customer_id="u1001",
                product="知应 AI 专业版月度订阅",
                status="paid",
                amount=Decimal("99.00"),
                currency="CNY",
                created_at="2026-07-20T09:30:00+00:00",
                updated_at="2026-07-20T09:31:00+00:00",
            ),
            "12345": _Order(
                order_id="12345",
                customer_id="demo-user",
                product="客服知识库扩展包",
                status="shipped",
                amount=Decimal("299.00"),
                currency="CNY",
                created_at="2026-07-18T03:20:00+00:00",
                updated_at="2026-07-21T08:10:00+00:00",
            ),
        }
        self._payments = {
            "A123": [
                PaymentRecord(
                    payment_id="PAY-A123-001",
                    amount=Decimal("99.00"),
                    currency="CNY",
                    channel="alipay",
                    status="captured",
                    paid_at="2026-07-20T09:30:20+00:00",
                ),
                PaymentRecord(
                    payment_id="PAY-A123-002",
                    amount=Decimal("99.00"),
                    currency="CNY",
                    channel="alipay",
                    status="captured",
                    paid_at="2026-07-20T09:30:42+00:00",
                ),
            ],
            "12345": [
                PaymentRecord(
                    payment_id="PAY-12345-001",
                    amount=Decimal("299.00"),
                    currency="CNY",
                    channel="wechat",
                    status="captured",
                    paid_at="2026-07-18T03:20:30+00:00",
                )
            ],
        }

    def get_order_status(self, order_id: str) -> GetOrderStatusOutput:
        normalized = _normalize_order_id(order_id)
        with self._lock:
            order = self._orders.get(normalized)
            if order is None:
                raise BusinessToolError(f"未找到订单 {normalized}")
            return GetOrderStatusOutput(
                order_id=order.order_id,
                product=order.product,
                status=order.status,
                amount=order.amount,
                currency=order.currency,
                created_at=order.created_at,
                updated_at=order.updated_at,
            )

    def query_payment(self, order_id: str) -> QueryPaymentOutput:
        normalized = _normalize_order_id(order_id)
        with self._lock:
            if normalized not in self._orders:
                raise BusinessToolError(f"未找到订单 {normalized}")
            payments = list(self._payments.get(normalized, []))

        captured = [record for record in payments if record.status == "captured"]
        groups: Dict[tuple[Decimal, str], List[PaymentRecord]] = {}
        for record in captured:
            groups.setdefault((record.amount, record.currency), []).append(record)
        duplicate_ids = [
            record.payment_id
            for records in groups.values()
            if len(records) > 1
            for record in records
        ]
        return QueryPaymentOutput(
            order_id=normalized,
            payments=payments,
            duplicate_payment_detected=bool(duplicate_ids),
            duplicate_payment_ids=duplicate_ids,
        )

    def create_refund_request(
        self,
        value: CreateRefundRequestInput,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> CreateRefundRequestOutput:
        order_id = value.order_id
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                raise BusinessToolError(f"未找到订单 {order_id}")
            if order.status in {"pending", "cancelled"}:
                raise BusinessToolError(f"订单 {order_id} 当前状态不支持创建退款申请")

            captured = [
                record
                for record in self._payments.get(order_id, [])
                if record.status == "captured"
            ]
            if not captured:
                raise BusinessToolError(f"订单 {order_id} 没有可退款的成功支付记录")

            payment = next(
                (record for record in captured if record.payment_id == value.payment_id),
                None,
            )
            if value.payment_id and payment is None:
                raise BusinessToolError(
                    f"支付记录 {value.payment_id} 不属于订单 {order_id} 或不可退款"
                )
            if payment is None:
                payment = captured[-1]

            amount = value.amount or payment.amount
            if amount > payment.amount:
                raise BusinessToolError(
                    f"退款金额 {amount} 超过支付记录可退金额 {payment.amount}"
                )

            idempotency_key = value.idempotency_key or self._derived_idempotency_key(
                value,
                payment.payment_id,
                amount,
                context,
            )
            existing = self._refunds_by_key.get(idempotency_key)
            if existing is not None:
                return existing.model_copy(update={"idempotent_replay": True})

            created_at = datetime.now(timezone.utc).isoformat()
            result = CreateRefundRequestOutput(
                request_id=f"RR-{uuid.uuid4().hex[:12].upper()}",
                order_id=order_id,
                payment_id=payment.payment_id,
                amount=amount,
                currency=payment.currency,
                reason=value.reason,
                status="pending_review",
                idempotency_key=idempotency_key,
                created_at=created_at,
                message="退款申请已创建，当前仅进入人工审核队列，尚未执行真实退款。",
            )
            self._refunds_by_key[idempotency_key] = result
            return result

    @staticmethod
    def _derived_idempotency_key(
        value: CreateRefundRequestInput,
        payment_id: str,
        amount: Decimal,
        context: Optional[Dict[str, Any]],
    ) -> str:
        request_context = context or {}
        stable_scope = request_context.get("conv_id") or request_context.get("request_id") or "direct"
        raw = "|".join(
            [
                str(stable_scope),
                value.order_id,
                payment_id,
                str(amount),
                value.reason,
            ]
        )
        return "refund:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class BusinessToolService:
    def __init__(self, repository: Optional[MockBusinessRepository] = None) -> None:
        self.repository = repository or MockBusinessRepository()

    async def get_order_status(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        del context
        value = GetOrderStatusInput.model_validate(params)
        return self.repository.get_order_status(value.order_id).model_dump(mode="json")

    async def query_payment(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        del context
        value = QueryPaymentInput.model_validate(params)
        return self.repository.query_payment(value.order_id).model_dump(mode="json")

    async def create_refund_request(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        value = CreateRefundRequestInput.model_validate(params)
        return self.repository.create_refund_request(
            value,
            context=context,
        ).model_dump(mode="json")


def register_business_tools(
    manager: MCPToolManager,
    repository: Optional[MockBusinessRepository] = None,
) -> BusinessToolService:
    """注册默认业务工具并返回服务，便于测试和演示检查内存状态。"""
    service = BusinessToolService(repository)
    manager.register(
        Tool(
            name="get_order_status",
            description=(
                "查询真实订单状态、商品、金额和时间。用户提到具体订单号并询问订单、"
                "售后或支付问题时应先调用；不得编造订单状态。"
            ),
            handler=service.get_order_status,
            schema=GetOrderStatusInput.model_json_schema(),
            input_model=GetOrderStatusInput,
            output_model=GetOrderStatusOutput,
            allowed_agents={"general", "technical", "billing"},
            risk_level="read",
            cache_ttl=30.0,
            timeout_s=3.0,
        )
    )
    manager.register(
        Tool(
            name="query_payment",
            description=(
                "查询订单的支付记录并判断是否存在相同金额的重复成功扣款。"
                "仅 BillingAgent 可以调用。"
            ),
            handler=service.query_payment,
            schema=QueryPaymentInput.model_json_schema(),
            input_model=QueryPaymentInput,
            output_model=QueryPaymentOutput,
            allowed_agents={"billing"},
            risk_level="read",
            cache_ttl=15.0,
            timeout_s=3.0,
        )
    )
    manager.register(
        Tool(
            name="create_refund_request",
            description=(
                "创建待人工审核的退款申请，不会执行真实退款。仅当用户明确要求退款，"
                "且已查询订单和支付记录后调用；重复扣款应指定需要退回的 payment_id。"
            ),
            handler=service.create_refund_request,
            schema=CreateRefundRequestInput.model_json_schema(),
            input_model=CreateRefundRequestInput,
            output_model=CreateRefundRequestOutput,
            allowed_agents={"billing"},
            risk_level="medium_write",
            cache_ttl=0.0,
            timeout_s=3.0,
        )
    )
    return service
