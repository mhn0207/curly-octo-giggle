"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_factory import create_structured_invoker
from core.structured_invoker import parse_json_object
from core.structured_schemas import IntentLLMOutput

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 要求升级/转人工
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单/退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float

@dataclass
class VoteResult:
    """多路识别的融合结果。confidence 对应最终胜出意图的加权分数。"""
    intent: IntentCategory
    confidence: float
    scores: Dict[IntentCategory, float]



# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"],
    IntentCategory.COMPLAINT:  ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST:    ["帮我取消订单", "我需要修改地址", "请协助退款"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "转人工客服", "找你们经理"],
    IntentCategory.TECHNICAL:  ["应用一直崩溃", "无法登录", "出现500错误"],
    IntentCategory.BILLING:    ["为什么扣了两次款？", "申请退款", "发票问题"],
    IntentCategory.ACCOUNT:    ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
        structured_invoker: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        self._structured_invoker = structured_invoker or create_structured_invoker(
            api_key=api_key,
            model=model,
            base_url=base_url,
            component="intent_recognizer",
            temperature=0.1,
            max_tokens=512,
        )
        # 第三方兼容 API（如 DeepSeek）通常不支持 Embedding，禁用该策略。
        # 官方 Anthropic SDK 当前没有 embeddings 资源，因此下面会使用稳定的
        # 本地字符 n-gram 向量作为轻量兜底，保证三路融合链路真实可跑。
        self._embedding_enabled = not bool(base_url)

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        vote = self._vote(llm, emb, pat)
        local_entities = self._extract_entities_locally(message)
        entities = self._merge_entities(local_entities, llm.get("entities"))
        urgency = self._urgency(message, vote.intent)

        result = IntentResult(
            intent=vote.intent,
            confidence=vote.confidence,
            urgency=urgency,
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            self._cache.clear()  # 模板变化后旧识别结果失效
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]
        )
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        task_prompt = f"""你是「知应 AI 企业服务协同 Agent 平台」的请求理解模块。请根据用户当前消息、示例和最近对话，选择最能代表本轮主要诉求的一个意图。

分类边界：
- query：查询订单、物流、政策、权益或其他业务信息
- request：要求取消、修改、提交申请等业务操作
- technical：登录、错误码、崩溃、接口、配置或系统异常
- billing：支付、扣款、退款、发票、订阅或账务争议
- account：账户资料、账号状态或账户管理
- complaint：表达不满但没有明确要求人工升级
- escalation：明确要求人工、投诉升级或负责人介入
- greeting / feedback / other：问候、正向反馈或无法归类

如果消息同时包含多个领域，选择用户最需要优先解决的主诉求，并在理由中指出复合问题，供下游多 Agent 协作。不要因为出现负面措辞就直接判为 escalation，也不要把“咨询退款规则”和“明确申请退款”混为一类。最近对话只用于消解指代和补全上下文，不得覆盖当前消息中的明确诉求。

示例:
{examples}
{ctx}
用户消息: "{message}"

请给出意图、0-1 置信度、一句话理由，并提取订单号、产品、日期、金额和错误码实体。只提取消息或最近对话中明确出现的值，不要推测缺失实体。
可选意图: {", ".join(c.value for c in IntentCategory)}"""
        task_prompt = self._clean_text(task_prompt)
        legacy_prompt = self._clean_text(task_prompt + """

返回格式（仅 JSON，不要其他文字）:
{
  "intent": "<意图值>",
  "confidence": <0-1>,
  "reasoning": "<一句话说明>",
  "entities": {
    "order_id": [], "product": [], "date": [], "amount": [], "error_code": []
  }
}""")

        async def legacy_call() -> IntentLLMOutput:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.1,
                messages=[{"role": "user", "content": legacy_prompt}],
            )
            raw = resp.content[0].text
            data = parse_json_object(raw)

            if not isinstance(data, dict):
                raise ValueError("LLM 返回值不是 JSON 对象")
            try:
                intent = IntentCategory(data.get("intent", "other")).value
            except (TypeError, ValueError):
                intent = IntentCategory.OTHER.value
            try:
                confidence = float(data.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            return IntentLLMOutput(
                intent=intent,
                confidence=min(max(confidence, 0.0), 1.0),
                reasoning=self._clean_text(data.get("reasoning", "")),
                entities=self._normalize_entities(data.get("entities")),
            )

        try:
            if self._structured_invoker is not None:
                output = await self._structured_invoker.ainvoke(
                    IntentLLMOutput,
                    [{"role": "user", "content": task_prompt}],
                    legacy_fallback=legacy_call,
                )
            else:
                output = await legacy_call()
            return {
                "intent": IntentCategory(output.intent),
                "confidence": output.confidence,
                "reasoning": self._clean_text(output.reasoning),
                "entities": output.entities.model_dump(),
            }
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {
                "intent": IntentCategory.OTHER,
                "confidence": 0.0,
                "reasoning": "LLM 失败",
                "entities": self._empty_entities(),
                "failed": True,
            }

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        patterns = {
            IntentCategory.ESCALATION: ["投诉", "经理", "转人工", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    ["退款", "扣款", "发票", "refund"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
        }
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> VoteResult:
        """加权投票，并返回最终意图对应的融合置信度。"""
        if llm.get("failed"):
            for fallback in (emb, pat):
                intent = fallback.get("intent", IntentCategory.OTHER)
                try:
                    confidence = float(fallback.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                confidence = min(max(confidence, 0.0), 1.0)
                if intent != IntentCategory.OTHER and confidence > 0:
                    return VoteResult(intent=intent, confidence=confidence, scores={intent: confidence})
            return VoteResult(
                intent=IntentCategory.OTHER,
                confidence=0.0,
                scores={IntentCategory.OTHER: 0.0},
            )

        weights = (
            [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
            if self._embedding_enabled
            else [(llm, 0.85), (pat, 0.15)]
        )
        scores: Dict[IntentCategory, float] = {}
        for strategy, weight in weights:
            intent = strategy.get("intent", IntentCategory.OTHER)
            if not isinstance(intent, IntentCategory):
                try:
                    intent = IntentCategory(intent)
                except (TypeError, ValueError):
                    intent = IntentCategory.OTHER
            try:
                confidence = float(strategy.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = min(max(confidence, 0.0), 1.0)
            scores[intent] = scores.get(intent, 0.0) + weight * confidence

        best_intent, best_score = max(scores.items(), key=lambda item: item[1])
        final_intent = best_intent if best_score >= self.threshold else IntentCategory.OTHER
        return VoteResult(intent=final_intent, confidence=best_score, scores=scores)

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_entities() -> Dict[str, List[str]]:
        return {"order_id": [], "product": [], "date": [], "amount": [], "error_code": []}

    @classmethod
    def _normalize_entities(cls, value: Any) -> Dict[str, List[str]]:
        """把 LLM 返回的实体统一为字段固定、值为去重字符串列表的结构。"""
        normalized = cls._empty_entities()
        if not isinstance(value, dict):
            return normalized

        for key in normalized:
            raw_items = value.get(key, [])
            if raw_items is None:
                continue
            if not isinstance(raw_items, (list, tuple, set)):
                raw_items = [raw_items]
            for item in raw_items:
                text = cls._clean_text(item).strip()
                if text and text not in normalized[key]:
                    normalized[key].append(text)
        return normalized

    @classmethod
    def _merge_entities(cls, *sources: Any) -> Dict[str, List[str]]:
        merged = cls._empty_entities()
        for source in sources:
            normalized = cls._normalize_entities(source)
            for key, values in normalized.items():
                for value in values:
                    if value not in merged[key]:
                        merged[key].append(value)
        return merged

    def _extract_entities_locally(self, message: str) -> Dict[str, List[str]]:
        """用确定性规则补充常见实体，不再为实体提取单独调用一次 LLM。"""
        text = self._clean_text(message)
        order_ids = re.findall(
            r"(?:订单(?:号)?|order\s*(?:id)?)\s*[:：#-]?\s*([A-Za-z0-9-]{4,32})",
            text,
            flags=re.IGNORECASE,
        )
        amounts = re.findall(
            r"(?:[¥￥]\s*)?\d+(?:\.\d{1,2})?\s*(?:元|人民币|CNY|RMB)",
            text,
            flags=re.IGNORECASE,
        )
        dates = re.findall(
            r"(?:\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?|\d{1,2}月\d{1,2}日)",
            text,
        )
        error_codes = re.findall(r"(?<!\d)(?:4\d{2}|5\d{2})(?!\d)", text)
        return self._normalize_entities({
            "order_id": order_ids,
            "date": dates,
            "amount": amounts,
            "error_code": error_codes,
        })

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """基于当前消息、最近上下文和模型生成不可逆缓存键。"""
        recent_history = []
        for item in (history or [])[-3:]:
            if not isinstance(item, dict):
                continue
            recent_history.append({
                "role": self._clean_text(item.get("role", ""))[:32],
                "content": self._clean_text(item.get("content", ""))[:500],
            })
        payload = {
            "message": self._clean_text(message)[:500],
            "history": recent_history,
            "model": self.model,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    def structured_output_stats(self) -> Dict[str, Any]:
        invoker = getattr(self, "_structured_invoker", None)
        if invoker is None:
            return {"intent_recognizer": {"mode": "legacy"}}
        return {"intent_recognizer": invoker.stats_snapshot()}

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
