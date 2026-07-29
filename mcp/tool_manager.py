"""
亮点：MCP 工具调用框架

核心问题：工具调用出错（检索不全、召回不好）怎么优化？

本模块的答案：
  1. 查询改写（Query Rewriting）—— 用 LLM 把用户原始问题扩写成多个角度的子查询，
     再合并去重，解决"召回不全"问题。
  2. 结果重排（Reranking）—— 对召回结果用 LLM 打分，按相关性重新排序，
     解决"召回不好/排序差"问题。
  3. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩。
  4. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用。
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。
"""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from core.llm_factory import create_structured_invoker
from core.structured_invoker import parse_json_array
from core.structured_schemas import QueryRewriteOutput, RerankOutput

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # 是否经过重排


@dataclass
class ToolExecution:
    """Sanitized audit record for one tool execution."""
    execution_id: str
    tool_name: str
    agent_type: str
    request_id: str
    params: Dict[str, Any]
    success: bool
    status: str
    latency_ms: float
    cached: bool = False
    error: Optional[str] = None
    result_preview: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0
    recent_latencies_ms: List[float] = field(default_factory=list)

    def record_latency(self, latency_ms: float) -> None:
        self.total_latency_ms += latency_ms
        self.recent_latencies_ms.append(latency_ms)
        if len(self.recent_latencies_ms) > 200:
            del self.recent_latencies_ms[:50]

    def percentile_latency_ms(self, percentile: float) -> float:
        if not self.recent_latencies_ms:
            return 0.0
        values = sorted(self.recent_latencies_ms)
        index = min(len(values) - 1, max(0, int((len(values) - 1) * percentile)))
        return values[index]

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN：放行一次探测

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    supports_rerank: bool = False            # 是否支持结果重排
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any
    input_model: Optional[Type[BaseModel]] = None
    output_model: Optional[Type[BaseModel]] = None
    allowed_agents: Set[str] = field(default_factory=set)
    risk_level: str = "read"

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    MCP 工具调用框架。

    核心优化链路（针对检索类工具）：
      用户查询 → 查询改写（多角度子查询）→ 并行召回 → 结果重排 → 返回 Top-K
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        rewrite_invoker: Optional[Any] = None,
        rerank_invoker: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model = model
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at)
        self._executions: List[ToolExecution] = []
        self._rewrite_invoker = rewrite_invoker or create_structured_invoker(
            api_key=api_key,
            model=model,
            base_url=base_url,
            component="query_rewrite",
            temperature=0.3,
            max_tokens=256,
        )
        self._rerank_invoker = rerank_invoker or create_structured_invoker(
            api_key=api_key,
            model=model,
            base_url=base_url,
            component="rag_rerank",
            temperature=0.0,
            max_tokens=256,
        )

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,
    ) -> ToolResult:
        """Controlled tool call with permissions, schemas, timeout, circuit breaker, and audit."""
        legacy_tool = self._tools.get(name)
        if (
            legacy_tool is not None
            and legacy_tool.input_model is None
            and legacy_tool.output_model is None
            and not legacy_tool.allowed_agents
        ):
            return await self._call_legacy(
                name,
                params,
                context,
                use_cache=use_cache,
                rerank_top_k=rerank_top_k,
            )
        raw_params = dict(params or {})
        tool = self._tools.get(name)
        if tool is None:
            return self._finalize_result(
                ToolResult(success=False, data=None, tool_name=name, error=f"Tool not found: {name}"),
                raw_params,
                context,
            )

        try:
            self._validate_permission(tool, context)
            normalized_params = self._validate_params(tool, raw_params)
        except Exception as ex:
            error = self._safe_error(ex)
            logger.warning("Tool call rejected: tool=%s type=%s", name, type(ex).__name__)
            return self._finalize_result(
                ToolResult(success=False, data=None, tool_name=name, error=error),
                raw_params,
                context,
            )

        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, normalized_params)
            if cached is not None:
                tool.stats.total += 1
                tool.stats.success += 1
                return self._finalize_result(
                    ToolResult(success=True, data=cached, tool_name=name, cached=True),
                    normalized_params,
                    context,
                )

        if not tool.breaker.allow():
            error = f"Tool circuit is open: {name}; retry later"
            result = await self._fallback_result(tool, normalized_params, context, error)
            return self._finalize_result(result, normalized_params, context)

        started = time.monotonic()
        tool.stats.total += 1
        try:
            data = await asyncio.wait_for(
                tool.handler(normalized_params, context),
                timeout=tool.timeout_s,
            )
            data = self._validate_output(tool, data)
            latency = (time.monotonic() - started) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.record_latency(latency)
            tool.breaker.record_success()

            if tool.cache_ttl > 0:
                self._set_cache(name, normalized_params, data, tool.cache_ttl)

            reranked = False
            if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
                query = normalized_params.get("query", "")
                data = await self._rerank(query, data, rerank_top_k)
                reranked = True

            return self._finalize_result(
                ToolResult(
                    success=True,
                    data=data,
                    tool_name=name,
                    latency_ms=latency,
                    reranked=reranked,
                ),
                normalized_params,
                context,
            )
        except asyncio.TimeoutError:
            latency = (time.monotonic() - started) * 1000
            tool.stats.record_latency(latency)
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error("Tool timed out: %s (%.1fs)", name, tool.timeout_s)
            result = await self._fallback_result(tool, normalized_params, context, "execution timed out")
            result.latency_ms = latency
            return self._finalize_result(result, normalized_params, context)
        except Exception as ex:
            latency = (time.monotonic() - started) * 1000
            tool.stats.record_latency(latency)
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            error = self._safe_error(ex)
            logger.error("Tool failed: %s type=%s", name, type(ex).__name__)
            result = await self._fallback_result(tool, normalized_params, context, error)
            result.latency_ms = latency
            return self._finalize_result(result, normalized_params, context)

    async def _call_legacy(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,          # >0 时对结果重排，取 Top-K
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 缓存写入 → 可选重排
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params)
            if cached is not None:
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(success=True, data=cached, tool_name=name, cached=True)

        # 熔断检查
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            # 参数校验（根据 JSON Schema 的 required 和 properties.type）
            self._validate_params(tool, params)

            data = await asyncio.wait_for(tool.handler(params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.record_latency(latency)
            tool.breaker.record_success()

            # 写缓存
            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl)

            # 重排（针对返回列表的检索工具）
            reranked = False
            if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
                query = params.get("query", "")
                data, reranked = await self._rerank(query, data, rerank_top_k), True

            return ToolResult(success=True, data=data, tool_name=name,
                              latency_ms=latency, reranked=reranked)

        except asyncio.TimeoutError:
            tool.stats.record_latency((time.monotonic() - t0) * 1000)
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.record_latency((time.monotonic() - t0) * 1000)
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    # ── 查询改写（解决召回不全）────────────────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """将原始查询改写为多个角度，并执行长度、空值和数量校验。"""
        query = self._clean_text(query).strip()[:500]
        if not query:
            return []
        n = min(max(int(n), 1), 8)
        task_prompt = self._clean_text(f"""将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面；不要回答问题。
原始查询: "{query}"""
        )
        legacy_prompt = self._clean_text(
            task_prompt
            + '\n返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]'
        )

        async def legacy_call() -> QueryRewriteOutput:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": legacy_prompt}],
            )
            raw = resp.content[0].text
            values = parse_json_array(raw)

            if not isinstance(values, list):
                raise ValueError("查询改写结果不是数组")
            return QueryRewriteOutput(queries=values[:8])

        try:
            if self._rewrite_invoker is not None:
                output = await self._rewrite_invoker.ainvoke(
                    QueryRewriteOutput,
                    [{"role": "user", "content": task_prompt}],
                    legacy_fallback=legacy_call,
                )
            else:
                output = await legacy_call()

            normalized = [query]
            for value in output.queries:
                candidate = self._clean_text(value).strip()[:500]
                if candidate and candidate not in normalized:
                    normalized.append(candidate)
                if len(normalized) >= n + 1:
                    break
            return normalized
        except Exception as ex:
            logger.warning(f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        完整的检索优化链路：查询改写 → 并行召回 → 去重 → 重排 → Top-K

        这是解决"检索不全、召回不好"的完整方案。
        """
        # 1. 查询改写：生成多角度子查询
        sub_queries = await self.rewrite_query(query, n=3)
        logger.info(f"查询改写: {query!r} → {sub_queries}")

        # 2. 并行召回：所有子查询同时检索
        recall_k = max(top_k, 5)
        tasks = [
            self.call(tool_name, {"query": q, "top_k": recall_k}, context, use_cache=True)
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 合并去重（按内容哈希去重）
        seen, merged = set(), []
        for r in results:
            if isinstance(r, ToolResult) and r.success and isinstance(r.data, list):
                for item in r.data:
                    key = self._dedupe_key(item)
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)

        if not merged:
            return ToolResult(success=False, data=[], tool_name=tool_name, error="所有子查询均无结果")

        # 4. 重排：用 LLM 对合并结果按相关性打分，取 Top-K
        reranked = await self._rerank(query, merged[:20], top_k)
        public_results = [self._public_item(item) for item in reranked]
        return ToolResult(success=True, data=public_results, tool_name=tool_name, reranked=True)

    # ── 结果重排（解决召回不好）──────────────────────────────────────────────

    async def _rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        """用结构化索引列表重排候选，并过滤重复、越界和缺失索引。"""
        top_k = max(0, int(top_k))
        if top_k == 0:
            return []
        items = items[:20]
        if len(items) <= top_k:
            return items

        items_text = "\n".join(
            f"{i}. {json.dumps(item, ensure_ascii=False, default=str)[:300]}"
            for i, item in enumerate(items)
        )
        task_prompt = self._clean_text(f"""根据用户查询，对以下检索结果按相关性从高到低排序。
用户查询: "{query}"
检索结果:
{items_text}

请返回按相关性降序排列的候选索引。""")
        legacy_prompt = self._clean_text(
            task_prompt + "\n只返回 JSON 索引数组，例如: [2, 0, 1]。"
        )

        async def legacy_call() -> RerankOutput:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": legacy_prompt}],
            )
            raw = resp.content[0].text
            values = parse_json_array(raw)

            if not isinstance(values, list):
                raise ValueError("重排结果不是数组")
            return RerankOutput(ordered_indexes=values)

        try:
            if self._rerank_invoker is not None:
                output = await self._rerank_invoker.ainvoke(
                    RerankOutput,
                    [{"role": "user", "content": task_prompt}],
                    legacy_fallback=legacy_call,
                )
            else:
                output = await legacy_call()

            order: List[int] = []
            seen_indexes = set()
            for index in output.ordered_indexes:
                if 0 <= index < len(items) and index not in seen_indexes:
                    seen_indexes.add(index)
                    order.append(index)
            order.extend(index for index in range(len(items)) if index not in seen_indexes)
            return [items[index] for index in order[:top_k]]
        except Exception as ex:
            logger.warning(f"重排失败，返回原始顺序: {ex}")
            return items[:top_k]

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe_key(item: Any) -> str:
        if isinstance(item, dict):
            chunk_id = item.get("_chunk_id") or item.get("chunk_id")
            if chunk_id:
                return f"chunk:{chunk_id}"
            document_id = item.get("_document_id") or item.get("document_id")
            chunk_index = item.get("chunk", item.get("chunk_index"))
            if document_id and chunk_index is not None:
                return f"document:{document_id}:{chunk_index}"
            content = " ".join(str(item.get("content", "")).split())
        else:
            content = " ".join(str(item).split())
        return "content:" + hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _public_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        return {key: value for key, value in item.items() if not key.startswith("_")}

    def definitions_for(
        self,
        agent_type: str,
        allowed_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return Anthropic tool definitions filtered by agent permissions."""
        name_filter = set(allowed_names) if allowed_names is not None else None
        definitions: List[Dict[str, Any]] = []
        for name, tool in self._tools.items():
            if name_filter is not None and name not in name_filter:
                continue
            if tool.allowed_agents and agent_type not in tool.allowed_agents:
                continue
            definitions.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.schema,
                }
            )
        return definitions

    def tool_catalog(self) -> List[Dict[str, Any]]:
        """Return a public tool catalog without executable handlers."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "risk_level": tool.risk_level,
                "allowed_agents": sorted(tool.allowed_agents) if tool.allowed_agents else ["all"],
                "timeout_s": tool.timeout_s,
                "cache_ttl": tool.cache_ttl,
            }
            for tool in self._tools.values()
        ]

    def get_recent_executions(
        self,
        limit: int = 50,
        *,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        values = self._executions
        if request_id:
            values = [item for item in values if item.request_id == request_id]
        bounded_limit = max(1, min(int(limit), 200))
        return [asdict(item) for item in values[-bounded_limit:]]

    @staticmethod
    def _validate_permission(tool: Tool, context: Optional[Dict[str, Any]]) -> None:
        if not tool.allowed_agents:
            return
        agent_type = str((context or {}).get("agent_type") or "")
        if not agent_type:
            raise PermissionError(f"Tool {tool.name} requires an agent identity")
        if agent_type not in tool.allowed_agents:
            raise PermissionError(f"Agent {agent_type} cannot call tool {tool.name}")

    @staticmethod
    def _validate_output(tool: Tool, data: Any) -> Any:
        if tool.output_model is None:
            return data
        value = (
            data
            if isinstance(data, tool.output_model)
            else tool.output_model.model_validate(data)
        )
        return value.model_dump(mode="json")

    def _finalize_result(
        self,
        result: ToolResult,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> ToolResult:
        call_context = context or {}
        if result.cached:
            status = "cached"
        elif result.success and result.error:
            status = "degraded"
        elif result.success:
            status = "success"
        else:
            status = "failed"
        execution = ToolExecution(
            execution_id=f"tool-{uuid.uuid4().hex[:12]}",
            tool_name=result.tool_name,
            agent_type=str(call_context.get("agent_type") or "system"),
            request_id=str(call_context.get("request_id") or ""),
            params=self._sanitize_for_audit(params),
            success=result.success,
            status=status,
            latency_ms=round(result.latency_ms, 1),
            cached=result.cached,
            error=result.error,
            result_preview=self._sanitize_for_audit(result.data),
        )
        self._executions.append(execution)
        if len(self._executions) > 1000:
            del self._executions[:250]
        return result

    @classmethod
    def _sanitize_for_audit(cls, value: Any, key: str = "") -> Any:
        sensitive_markers = ("password", "secret", "token", "api_key", "card", "verification_code")
        if any(marker in key.lower() for marker in sensitive_markers):
            return "***"
        if isinstance(value, dict):
            return {
                str(item_key): cls._sanitize_for_audit(item_value, str(item_key))
                for item_key, item_value in list(value.items())[:30]
            }
        if isinstance(value, (list, tuple)):
            return [cls._sanitize_for_audit(item, key) for item in list(value)[:30]]
        if isinstance(value, str):
            return value if len(value) <= 500 else value[:500] + "..."
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)[:500]

    @staticmethod
    def _safe_error(ex: BaseException) -> str:
        if isinstance(ex, ValidationError):
            details = []
            for error in ex.errors(include_input=False, include_url=False)[:5]:
                location = ".".join(str(part) for part in error.get("loc", ()))
                prefix = f"{location}: " if location else ""
                details.append(prefix + str(error.get("msg", "validation failed")))
            return "Parameter validation failed: " + "; ".join(details)
        message = str(ex).strip()
        return message[:500] if message else type(ex).__name__

    def invalidate_cache(self, tool_name: Optional[str] = None) -> None:
        """知识变更或 collection 切换后使旧检索结果失效。"""
        if tool_name is None:
            self._cache.clear()
            return
        prefix = f"{tool_name}:"
        for key in [key for key in self._cache if key.startswith(prefix)]:
            del self._cache[key]

    def structured_output_stats(self) -> Dict[str, Any]:
        return {
            "query_rewrite": (
                self._rewrite_invoker.stats_snapshot() if self._rewrite_invoker else {"mode": "legacy"}
            ),
            "rerank": (
                self._rerank_invoker.stats_snapshot() if self._rerank_invoker else {"mode": "legacy"}
            ),
        }

    def _cache_key(self, name: str, params: Dict) -> str:
        return f"{name}:{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict) -> Optional[Any]:
        key = self._cache_key(name, params)
        if key in self._cache:
            data, expire_at = self._cache[key]
            if time.monotonic() < expire_at:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, name: str, params: Dict, data: Any, ttl: float) -> None:
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params)] = (data, time.monotonic() + ttl)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> Dict[str, Any]:
        """Use strict Pydantic schemas when available; preserve legacy JSON-schema tools."""
        if tool.input_model is not None:
            value = (
                params
                if isinstance(params, tool.input_model)
                else tool.input_model.model_validate(params)
            )
            return value.model_dump(mode="json", exclude_none=True)
        self._validate_params_legacy(tool, params)
        return dict(params)

    def _validate_params_legacy(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "p50_latency_ms": round(t.stats.percentile_latency_ms(0.50), 1),
                "p95_latency_ms": round(t.stats.percentile_latency_ms(0.95), 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
