"""Structured, deterministic synthesis for multi-agent responses.

The synthesizer treats tool output as evidence and model prose as advice.  It
never promotes model prose to a verified fact, and it surfaces conflicting
high-trust evidence instead of silently choosing one value.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


SOURCE_PRIORITY = {
    "tool": 300,
    "knowledge_base": 200,
    "agent": 100,
}
BLOCKING_FACT_PREFIXES = ("order:", "payment:", "refund:")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


@dataclass
class FactRecord:
    key: str
    label: str
    value: Any
    source: str
    source_kind: str = "tool"
    agent_type: str = ""
    confidence: float = 1.0
    evidence_sources: List[str] = field(default_factory=list)

    @property
    def priority(self) -> int:
        return SOURCE_PRIORITY.get(self.source_kind, 0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    agent_type: str
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConflictRecord:
    key: str
    label: str
    candidates: List[Dict[str, Any]]
    resolved: bool
    blocking: bool
    resolution: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentWorkResult:
    agent_type: str
    success: bool
    summary: str = ""
    confirmed_facts: List[FactRecord] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    tool_executions: List[Dict[str, Any]] = field(default_factory=list)
    recommended_actions: List[Recommendation] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    requires_approval: bool = False
    requires_human: bool = False
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_agent_response(cls, response: Any) -> "AgentWorkResult":
        agent_type = getattr(getattr(response, "agent_type", None), "value", None)
        agent_type = agent_type or str(getattr(response, "agent_type", "unknown"))
        content = _normalize_text(getattr(response, "content", ""))
        success = bool(getattr(response, "success", False))
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        result = cls(
            agent_type=agent_type,
            success=success,
            summary=content,
            tool_executions=tool_calls,
            requires_human=bool(getattr(response, "escalate", False)),
        )

        if success and content:
            result.recommended_actions.append(
                Recommendation(agent_type=agent_type, content=content)
            )
        if not success:
            result.errors.append(content or f"{agent_type} Agent 处理失败")

        for call in tool_calls:
            tool_name = str(call.get("tool_name", "unknown"))
            if not call.get("success", False):
                error = _normalize_text(str(call.get("error") or "未知错误"))
                result.unresolved_questions.append(
                    f"工具 {tool_name} 调用失败：{error}"
                )
                continue

            call_result = call.get("result")
            if not isinstance(call_result, dict):
                continue
            result.confirmed_facts.extend(
                _facts_from_tool_result(tool_name, call_result, agent_type)
            )
            if tool_name == "create_refund_request":
                result.requires_approval = call_result.get("status") == "pending_review"

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "success": self.success,
            "summary": self.summary,
            "confirmed_facts": [fact.to_dict() for fact in self.confirmed_facts],
            "assumptions": list(self.assumptions),
            "tool_executions": list(self.tool_executions),
            "recommended_actions": [
                action.to_dict() for action in self.recommended_actions
            ],
            "unresolved_questions": list(self.unresolved_questions),
            "requires_approval": self.requires_approval,
            "requires_human": self.requires_human,
            "errors": list(self.errors),
        }


def _fact(
    *,
    key: str,
    label: str,
    value: Any,
    tool_name: str,
    agent_type: str,
) -> FactRecord:
    source = f"tool:{tool_name}"
    return FactRecord(
        key=key,
        label=label,
        value=value,
        source=source,
        source_kind="tool",
        agent_type=agent_type,
        evidence_sources=[source],
    )


def _facts_from_tool_result(
    tool_name: str,
    value: Dict[str, Any],
    agent_type: str,
) -> List[FactRecord]:
    order_id = str(value.get("order_id") or "").strip()
    facts: List[FactRecord] = []

    if tool_name == "get_order_status" and order_id:
        field_labels = {
            "product": "商品",
            "status": "订单状态",
            "amount": "订单金额",
            "currency": "币种",
        }
        for field_name, label in field_labels.items():
            if field_name in value:
                facts.append(
                    _fact(
                        key=f"order:{order_id}:{field_name}",
                        label=f"订单 {order_id} {label}",
                        value=value[field_name],
                        tool_name=tool_name,
                        agent_type=agent_type,
                    )
                )

    elif tool_name == "query_payment" and order_id:
        if "duplicate_payment_detected" in value:
            facts.append(
                _fact(
                    key=f"payment:{order_id}:duplicate_detected",
                    label=f"订单 {order_id} 是否存在重复扣款",
                    value=value["duplicate_payment_detected"],
                    tool_name=tool_name,
                    agent_type=agent_type,
                )
            )
        duplicate_ids = value.get("duplicate_payment_ids")
        if duplicate_ids is not None:
            facts.append(
                _fact(
                    key=f"payment:{order_id}:duplicate_ids",
                    label=f"订单 {order_id} 重复支付记录",
                    value=duplicate_ids,
                    tool_name=tool_name,
                    agent_type=agent_type,
                )
            )
        for payment in value.get("payments", []):
            if not isinstance(payment, dict) or not payment.get("payment_id"):
                continue
            payment_id = str(payment["payment_id"])
            for field_name, label in (
                ("status", "状态"),
                ("amount", "金额"),
                ("currency", "币种"),
                ("channel", "渠道"),
            ):
                if field_name in payment:
                    facts.append(
                        _fact(
                            key=f"payment:{payment_id}:{field_name}",
                            label=f"支付 {payment_id} {label}",
                            value=payment[field_name],
                            tool_name=tool_name,
                            agent_type=agent_type,
                        )
                    )

    elif tool_name == "create_refund_request":
        request_id = str(value.get("request_id") or "unknown")
        for field_name, label in (
            ("status", "状态"),
            ("order_id", "订单"),
            ("payment_id", "支付记录"),
            ("amount", "申请金额"),
            ("currency", "币种"),
        ):
            if field_name in value:
                facts.append(
                    _fact(
                        key=f"refund:{request_id}:{field_name}",
                        label=f"退款申请 {request_id} {label}",
                        value=value[field_name],
                        tool_name=tool_name,
                        agent_type=agent_type,
                    )
                )

    return facts


@dataclass
class SynthesisOutcome:
    response: str
    confirmed_facts: List[FactRecord]
    assumptions: List[str]
    conflicts: List[ConflictRecord]
    tool_calls: List[Dict[str, Any]]
    partial_failure: bool
    requires_approval: bool
    requires_human: bool
    work_results: List[AgentWorkResult]

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "confirmed_facts": [
                fact.to_dict() for fact in self.confirmed_facts
            ],
            "assumptions": list(self.assumptions),
            "conflicts": [
                conflict.to_dict() for conflict in self.conflicts
            ],
            "partial_failure": self.partial_failure,
            "requires_approval": self.requires_approval,
            "requires_human": self.requires_human,
            "agents": [
                {
                    "agent_type": item.agent_type,
                    "success": item.success,
                    "fact_count": len(item.confirmed_facts),
                    "tool_call_count": len(item.tool_executions),
                    "errors": list(item.errors),
                }
                for item in self.work_results
            ],
        }


class SynthesisAgent:
    """Merge AgentWorkResult objects and render one evidence-bounded answer."""

    def synthesize(self, responses: Iterable[Any]) -> SynthesisOutcome:
        work_results = [
            getattr(response, "work_result", None)
            or AgentWorkResult.from_agent_response(response)
            for response in responses
        ]
        facts, conflicts = self._merge_facts(
            fact
            for work_result in work_results
            for fact in work_result.confirmed_facts
        )
        recommendations = self._dedupe_recommendations(
            action
            for work_result in work_results
            for action in work_result.recommended_actions
        )
        assumptions = self._dedupe_text(
            assumption
            for work_result in work_results
            for assumption in work_result.assumptions
        )
        unresolved = self._dedupe_text(
            question
            for work_result in work_results
            for question in work_result.unresolved_questions
        )
        tool_calls = self._dedupe_tool_calls(
            call
            for work_result in work_results
            for call in work_result.tool_executions
        )
        partial_failure = any(not item.success for item in work_results)
        requires_approval = any(item.requires_approval for item in work_results)
        blocking_conflict = any(
            conflict.blocking and not conflict.resolved for conflict in conflicts
        )
        requires_human = (
            blocking_conflict
            or any(item.requires_human for item in work_results)
            or (partial_failure and not facts)
        )
        response = self._render(
            work_results=work_results,
            facts=facts,
            recommendations=recommendations,
            conflicts=conflicts,
            unresolved=unresolved,
            partial_failure=partial_failure,
            requires_approval=requires_approval,
            requires_human=requires_human,
        )
        return SynthesisOutcome(
            response=response,
            confirmed_facts=facts,
            assumptions=assumptions,
            conflicts=conflicts,
            tool_calls=tool_calls,
            partial_failure=partial_failure,
            requires_approval=requires_approval,
            requires_human=requires_human,
            work_results=work_results,
        )

    @staticmethod
    def _merge_facts(
        facts: Iterable[FactRecord],
    ) -> tuple[List[FactRecord], List[ConflictRecord]]:
        grouped: Dict[str, Dict[str, List[FactRecord]]] = {}
        for fact in facts:
            grouped.setdefault(fact.key, {}).setdefault(
                _canonical(fact.value), []
            ).append(fact)

        merged: List[FactRecord] = []
        conflicts: List[ConflictRecord] = []
        for key, value_groups in grouped.items():
            for group in value_groups.values():
                primary = group[0]
                primary.evidence_sources = list(
                    dict.fromkeys(
                        source
                        for item in group
                        for source in (item.evidence_sources or [item.source])
                    )
                )

            if len(value_groups) == 1:
                merged.append(next(iter(value_groups.values()))[0])
                continue

            candidates = [group[0] for group in value_groups.values()]
            highest_priority = max(candidate.priority for candidate in candidates)
            highest = [
                candidate
                for candidate in candidates
                if candidate.priority == highest_priority
            ]
            label = candidates[0].label
            if len(highest) == 1:
                winner = highest[0]
                merged.append(winner)
                conflicts.append(
                    ConflictRecord(
                        key=key,
                        label=label,
                        candidates=[
                            {
                                "value": candidate.value,
                                "source": candidate.source,
                                "source_kind": candidate.source_kind,
                            }
                            for candidate in candidates
                        ],
                        resolved=True,
                        blocking=False,
                        resolution=(
                            f"采用高优先级来源 {winner.source} 的结果"
                        ),
                    )
                )
                continue

            blocking = key.startswith(BLOCKING_FACT_PREFIXES)
            conflicts.append(
                ConflictRecord(
                    key=key,
                    label=label,
                    candidates=[
                        {
                            "value": candidate.value,
                            "source": candidate.source,
                            "source_kind": candidate.source_kind,
                        }
                        for candidate in candidates
                    ],
                    resolved=False,
                    blocking=blocking,
                    resolution="同级可信来源结果不一致，需要人工确认",
                )
            )

        return merged, conflicts

    @staticmethod
    def _dedupe_recommendations(
        actions: Iterable[Recommendation],
    ) -> List[Recommendation]:
        result: List[Recommendation] = []
        seen: set[str] = set()
        for action in actions:
            normalized = _normalize_text(action.content)
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            result.append(
                Recommendation(
                    agent_type=action.agent_type,
                    content=normalized,
                )
            )
        return result

    @staticmethod
    def _dedupe_text(values: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = _normalize_text(value)
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result

    @staticmethod
    def _dedupe_tool_calls(
        calls: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for call in calls:
            identity = str(
                call.get("tool_use_id")
                or call.get("execution_id")
                or _canonical(
                    {
                        "tool_name": call.get("tool_name"),
                        "input": call.get("input"),
                        "result": call.get("result"),
                    }
                )
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(call)
        return result

    @staticmethod
    def _render(
        *,
        work_results: List[AgentWorkResult],
        facts: List[FactRecord],
        recommendations: List[Recommendation],
        conflicts: List[ConflictRecord],
        unresolved: List[str],
        partial_failure: bool,
        requires_approval: bool,
        requires_human: bool,
    ) -> str:
        if not any(item.success for item in work_results):
            return "抱歉，本次参与协作的 Agent 均未能完成处理，建议稍后重试或转人工客服。"

        role_names = {
            "general": "服务协调 Agent",
            "technical": "技术可靠性 Agent",
            "billing": "收入与合规 Agent",
            "escalation": "人工升级通道",
        }
        lines = ["我已综合各专业 Agent 的处理结果。"]

        if facts:
            lines.extend(["", "已核实信息："])
            for fact in facts:
                lines.append(f"- {fact.label}：{_display_value(fact.value)}")

        unresolved_conflicts = [
            conflict for conflict in conflicts if not conflict.resolved
        ]
        if unresolved_conflicts:
            lines.extend(["", "存在待确认的冲突："])
            for conflict in unresolved_conflicts:
                values = "、".join(
                    _display_value(candidate["value"])
                    for candidate in conflict.candidates
                )
                lines.append(f"- {conflict.label}出现不一致结果：{values}")

        if recommendations:
            lines.extend(["", "处理建议："])
            for action in recommendations:
                role = role_names.get(action.agent_type, action.agent_type)
                lines.append(f"- {role}：{action.content}")

        if unresolved:
            lines.extend(["", "尚未完成："])
            lines.extend(f"- {question}" for question in unresolved)

        if partial_failure:
            lines.extend(
                ["", "部分 Agent 未完成处理，以上结论仅基于当前可用结果。"]
            )
        if requires_approval:
            lines.extend(
                ["", "已创建的退款申请仅进入待审核状态，尚未发生实际退款。"]
            )
        if requires_human:
            lines.extend(
                ["", "当前存在需要人工确认的信息，高风险操作不应继续自动执行。"]
            )

        return "\n".join(lines)
