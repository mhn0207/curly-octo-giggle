"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAgent

并行协作：
  - 复杂问题（如"技术问题 + 账单问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from agents.result_synthesizer import AgentWorkResult, SynthesisAgent
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL   = "general"    # 服务协调
    TECHNICAL = "technical"  # 技术可靠性
    BILLING   = "billing"    # 收入与合规
    ESCALATION = "escalation" # 人工升级通道（占位）


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    work_result: Optional[AgentWorkResult] = None


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: Optional[Dict[str, Any]] = None


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str
    allowed_tools: tuple[str, ...] = ()

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        tool_manager: Optional[Any] = None,
        max_tool_steps: int = 4,
        request_timeout_s: float = 60.0,
        tool_calling_enabled: bool = True,
    ):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager
        self._tool_manager = tool_manager
        self._max_tool_steps = max(1, min(int(max_tool_steps), 8))
        self._request_timeout_s = max(1.0, float(request_timeout_s))
        self._tool_calling_enabled = tool_calling_enabled
        self.stats = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        tool_calls: List[Dict[str, Any]] = []
        try:
            content = await asyncio.wait_for(
                self._call_llm(req, tool_calls),
                timeout=self._request_timeout_s,
            )
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            response = AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tool_calls=tool_calls,
            )
            response.work_result = AgentWorkResult.from_agent_response(response)
            return response
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            response = AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tool_calls=tool_calls,
            )
            response.work_result = AgentWorkResult.from_agent_response(response)
            return response

    async def _call_llm(
        self,
        req: Request,
        tool_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        tools = self._available_tool_definitions()
        if not tools:
            return await self._call_llm_legacy(req)

        trace = tool_trace if tool_trace is not None else []
        messages = self._initial_messages(req)
        system_prompt = self._build_system_prompt(req)

        for _ in range(self._max_tool_steps):
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )
            blocks = list(getattr(response, "content", []) or [])
            tool_uses = [
                block
                for block in blocks
                if self._block_attr(block, "type", "") == "tool_use"
            ]
            if not tool_uses:
                text = self._response_text(blocks)
                return text or "No response content was generated."

            messages.append(
                {
                    "role": "assistant",
                    "content": [self._block_to_dict(block) for block in blocks],
                }
            )
            tool_results = []
            for block in tool_uses:
                tool_name = str(self._block_attr(block, "name", ""))
                tool_use_id = str(self._block_attr(block, "id", ""))
                raw_input = self._block_attr(block, "input", {})
                tool_input = dict(raw_input) if isinstance(raw_input, dict) else {}
                result = await self._tool_manager.call(
                    tool_name,
                    tool_input,
                    context={
                        "agent_type": self.agent_type.value,
                        "request_id": req.request_id,
                        "user_id": req.user_id,
                        "conv_id": req.conv_id,
                    },
                )
                trace.append(
                    {
                        "tool_name": tool_name,
                        "tool_use_id": tool_use_id,
                        "input": tool_input,
                        "success": result.success,
                        "cached": result.cached,
                        "latency_ms": round(result.latency_ms, 1),
                        "error": result.error,
                        "result": result.data,
                    }
                )
                payload = {
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                    "cached": result.cached,
                }
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(payload, ensure_ascii=False, default=str),
                        "is_error": not result.success,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        messages.append(
            {
                "role": "user",
                "content": (
                    "The controlled tool-call limit has been reached. "
                    "Use only the tool results already returned, do not claim any unobserved action, "
                    "and provide a concise final answer."
                ),
            }
        )
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return self._response_text(list(getattr(response, "content", []) or []))

    def _available_tool_definitions(self) -> List[Dict[str, Any]]:
        if not self._tool_calling_enabled or self._tool_manager is None:
            return []
        return self._tool_manager.definitions_for(
            self.agent_type.value,
            list(self.allowed_tools),
        )

    def _initial_messages(self, req: Request) -> List[Dict[str, Any]]:
        def clean(value: str) -> str:
            return value.encode("utf-8", errors="ignore").decode("utf-8")

        messages: List[Dict[str, Any]] = []
        background = []
        if req.context:
            background.append(f"[context]\n{clean(req.context)}")
        if req.entities and any(req.entities.values()):
            entity_text = json.dumps(req.entities, ensure_ascii=False)
            background.append(f"[extracted_entities]\n{clean(entity_text)}")
        if background:
            messages.append({"role": "user", "content": "\n\n".join(background)})
            messages.append({"role": "assistant", "content": "Context received."})
        messages.append({"role": "user", "content": clean(req.message)})
        return messages

    @staticmethod
    def _block_attr(block: Any, name: str, default: Any = None) -> Any:
        if isinstance(block, dict):
            return block.get(name, default)
        return getattr(block, name, default)

    @classmethod
    def _block_to_dict(cls, block: Any) -> Dict[str, Any]:
        if isinstance(block, dict):
            return dict(block)
        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        result: Dict[str, Any] = {}
        for name in ("type", "id", "name", "input", "text"):
            value = cls._block_attr(block, name, None)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def _response_text(cls, blocks: List[Any]) -> str:
        return "\n".join(
            str(cls._block_attr(block, "text", "")).strip()
            for block in blocks
            if cls._block_attr(block, "text", "")
        ).strip()

    async def _call_llm_legacy(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        background = []
        if req.context:
            background.append(f"[背景信息]\n{_clean(req.context)}")
        if req.entities and any(req.entities.values()):
            entity_text = json.dumps(req.entities, ensure_ascii=False)
            background.append(f"[已提取实体]\n{_clean(entity_text)}")
        if background:
            messages.append({"role": "user", "content": "\n\n".join(background)})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self._build_system_prompt(req),
            messages=messages,
        )
        return resp.content[0].text

    def _build_system_prompt(self, req: Request) -> str:
        base_prompt = self._build_system_prompt_legacy(req)
        if not self._available_tool_definitions():
            return base_prompt
        return base_prompt + (
            "\n\n[受控业务工具]\n"
            "当请求涉及具体订单、支付记录或退款状态时，优先使用本角色获准调用的工具核验，"
            "不得编造订单状态、支付流水、业务标识或操作结果。工具报错或无结果只表示暂时无法核验，"
            "不能当作业务事实。创建退款申请前必须先核验订单和支付记录；只有用户明确提出退款诉求时"
            "才可调用 create_refund_request。该工具只创建 pending_review 状态的待审核申请，不会直接"
            "划转资金。最终回复必须区分已核实事实、处理建议和待审核操作，并明确说明下一责任方。"
        )

    def _build_system_prompt_legacy(self, req: Request) -> str:
        """把动态加载的 Skills 拼入 system prompt，让业务规则随请求生效。"""
        if self._skill_manager is None:
            return self.system_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "人工客服", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    allowed_tools = ("get_order_status",)
    system_prompt = """你是「知应 AI 企业服务协同 Agent 平台」中的服务协调 Agent。

你的职责是统一承接复杂业务请求，处理通用咨询、订单与物流、会员权益、基础账户问题、信息澄清和跨领域协调。你的目标是识别真实诉求、组织已有信息，并推动问题进入可执行的下一步。

[执行原则]
- 优先解决当前核心诉求；复合问题要拆分，并清楚标记需要技术可靠性 Agent、收入与合规 Agent 或人工通道接手的部分。
- 可使用的上下文包括会话记忆、历史摘要、用户画像、已提取实体、企业知识和动态 Skills。只使用与当前请求相关的内容，不把历史推测当作本轮事实。
- 事实可信度按“受控工具结果 > 企业知识库 > 用户明确提供的信息 > 一般经验”排序。依据不足时直接说明尚不能确认，不得编造订单、政策、权益、处理进度或后台操作结果。
- 只收集推进处理所必需的信息；不得索要密码、验证码、支付密码、完整银行卡号等敏感信息。
- 不替其他专业角色作出技术根因、退款结论或合规承诺。高风险、低置信度、强烈投诉或需要后台权限的事项，应总结已知信息并建议进入人工升级通道。

[回复要求]
先给结论或当前判断，再说明依据、下一步和需要补充的信息。语言自然、克制、专业；简单问题简短回答，复杂问题使用清晰的分项或步骤。明确区分“已完成”“待核验”“待审核”和“建议操作”。"""


class TechnicalAgent(BaseAgent):
    agent_type    = AgentType.TECHNICAL
    allowed_tools = ("get_order_status",)
    system_prompt = """你是「知应 AI 企业服务协同 Agent 平台」中的技术可靠性 Agent。

你的职责是处理登录失败、错误码、页面或应用崩溃、接口异常、配置与部署问题、性能退化和数据同步异常，帮助用户以低风险、可验证的方式缩小故障范围。

[执行原则]
- 先确认故障现象、发生时间、影响范围和最近变更，再判断可能原因；没有日志、错误码或工具证据时，不得断言根因。
- 优先给出低风险、低成本且可回退的排查步骤，并说明每一步的验证信号。不要把清缓存、重启或重装当作没有条件的通用答案。
- 结合会话记忆、已提取错误码、企业知识和动态 Skills 工作。知识与当前现象冲突时，以已核实的运行事实为准，并明确指出不确定性。
- 不编造服务状态、后台日志、修复结果或工单进度；不要求用户公开密码、验证码、完整 Token、API Key 或私钥。
- 对支付、退款、发票等非技术结论，只描述技术侧观察，并标记需收入与合规 Agent 协作。生产大面积故障、数据丢失、权限异常、安全事件或必须后台操作的情况，应进入人工或二线技术升级通道。

[回复要求]
优先采用“当前判断 / 排查步骤 / 验证方式 / 升级条件”的结构。引用用户已提供的错误码或现象，步骤具体但不过量，并明确哪些内容是可能原因、哪些是已核实事实。"""


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    allowed_tools = ("get_order_status", "query_payment", "create_refund_request")
    system_prompt = """你是「知应 AI 企业服务协同 Agent 平台」中的收入与合规 Agent。

你的职责是处理支付异常、重复扣款、退款、发票、订阅、账单核验及相关合规边界。你的回答必须保守、准确、可追溯，尤其不能把申请、审核和实际到账混为一谈。

[执行原则]
- 涉及具体资金、订单或支付状态时必须先通过获准工具核验；不能仅凭用户描述或模型推测宣布扣款原因、退款成功、到账时间或发票处理结果。
- 明确区分订单金额、实付金额、退款金额、到账金额和手续费；明确区分“已提交”“pending_review”“审核通过”“已退款”和“已到账”。
- 结合会话记忆、结构化实体、企业政策和动态 Skills 工作。政策未覆盖、证据冲突或工具不可用时，说明当前无法确认，并给出人工财务核验所需的最少信息。
- 只收集订单号、交易时间、金额、渠道等必要信息；不得索要支付密码、验证码、完整银行卡号或无关身份材料。
- 不作税务、法律或监管保证，不承诺无条件退款、补偿或固定到账时间。技术故障线索交由技术可靠性 Agent；大额、异常扣款、发票作废重开、合同费用调整和争议事项进入人工或财务审核通道。

[回复要求]
优先采用“已核实信息 / 当前结论 / 下一步处理 / 审核与时效边界”的结构。先回应资金或票据焦点，再解释规则；所有不确定结果都使用可核验、不过度承诺的表达。"""


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.TECHNICAL:  AgentType.TECHNICAL,
        IntentCategory.BILLING:    AgentType.BILLING,
        IntentCategory.ACCOUNT:    AgentType.BILLING,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        # 其余意图 → GENERAL（默认）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        tool_manager: Optional[Any] = None,
        max_tool_steps: int = 4,
        request_timeout_s: float = 60.0,
        tool_calling_enabled: bool = True,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._tool_manager = tool_manager
        self._synthesis_agent = SynthesisAgent()
        agent_options = {
            "skill_manager": skill_manager,
            "tool_manager": tool_manager,
            "max_tool_steps": max_tool_steps,
            "request_timeout_s": request_timeout_s,
            "tool_calling_enabled": tool_calling_enabled,
        }

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [GeneralAgent(client, model, **agent_options)],
            AgentType.TECHNICAL: [TechnicalAgent(client, model, **agent_options)],
            AgentType.BILLING: [BillingAgent(client, model, **agent_options)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_tool_manager(self, tool_manager: Optional[Any]) -> None:
        """Replace the shared tool runtime for all agent instances."""
        self._tool_manager = tool_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._tool_manager = tool_manager


    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.urgency = intent_result.urgency
            req.entities = intent_result.entities

        # 复杂问题自动并行协作，例如同一句同时涉及登录故障和扣款/退款。
        collaboration = self._collaboration_targets(req)
        if len(collaboration) > 1:
            return await self.run_parallel(req, collaboration)

        # 2. 路由：选择 Agent 类型
        agent_type = self._route(req.intent, req.urgency)

        # 3. 执行（含降级）
        response = await self._execute(req, agent_type)

        # 4. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.ESCALATION:
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处创建工单、通知人工客服

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            tool_calls=response.tool_calls,
        )

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及技术和账单）。
        """
        t0 = time.monotonic()
        tasks = [self._execute(req, at) for at in agent_types]
        raw_responses = await asyncio.gather(*tasks, return_exceptions=True)
        responses: List[AgentResponse] = []
        for agent_type, response in zip(agent_types, raw_responses):
            if isinstance(response, AgentResponse):
                responses.append(response)
                continue
            logger.error(
                "并行 Agent 执行异常: agent=%s error_type=%s",
                agent_type.value,
                type(response).__name__,
            )
            failed = AgentResponse(
                agent_type=agent_type,
                content="该专业 Agent 暂时无法完成处理。",
                success=False,
            )
            failed.work_result = AgentWorkResult.from_agent_response(failed)
            responses.append(failed)

        synthesis_agent = getattr(self, "_synthesis_agent", None)
        if synthesis_agent is None:
            synthesis_agent = SynthesisAgent()
            self._synthesis_agent = synthesis_agent
        outcome = synthesis_agent.synthesize(responses)
        escalated = (
            outcome.requires_human
            or any(response.escalate for response in responses)
        )

        return OrchestratorResult(
            request_id=req.request_id,
            response=outcome.response,
            agent_type=agent_types[0],
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            tool_calls=outcome.tool_calls,
            synthesis=outcome.to_public_dict(),
        )

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"登录报错且被重复扣款"需要技术和账单 Agent 同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        technical_kws = ["崩溃", "报错", "error", "crash", "无法登录", "登录失败", "500", "401"]
        billing_kws = ["退款", "扣款", "发票", "账单", "支付", "订阅", "refund", "invoice"]

        if req.intent == IntentCategory.TECHNICAL or any(kw in msg for kw in technical_kws):
            targets.append(AgentType.TECHNICAL)
        if req.intent in (IntentCategory.BILLING, IntentCategory.ACCOUNT) or any(kw in msg for kw in billing_kws):
            targets.append(AgentType.BILLING)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
