"""
亮点：多轮对话记忆管理

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（ChromaDB）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（ChromaDB）—— 从对话中提炼的长期偏好和实体

关键设计：
  - 上下文构建时三级记忆融合，按重要性 + 时效性排序
  - 工作记忆超过阈值时自动压缩（LLM 摘要），防止 context 爆炸
  - 所有 Embedding 通过 Anthropic API 生成，无本地模型
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
from anthropic import AsyncAnthropic
from redis import asyncio as redis_async

from core.llm_factory import create_structured_invoker
from core.structured_invoker import parse_json_object
from core.structured_schemas import UserProfileOutput

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 用户画像：偏好、常用实体
    summary:          str             # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-3-5-sonnet-20241022",
        structured_invoker: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model = model
        self._profile_invoker = structured_invoker or create_structured_invoker(
            api_key=api_key,
            model=model,
            base_url=base_url,
            component="user_profile",
            temperature=0.0,
            max_tokens=512,
        )

        self._redis = redis_async.from_url(redis_url, decode_responses=True)
        self._profile_tasks: set[asyncio.Task[None]] = set()
        self._profile_versions: Dict[str, int] = {}
        self._profile_locks: Dict[str, asyncio.Lock] = {}
        self._closed = False

        # ChromaDB：优先连接独立服务（docker compose 模式），连不上则降级为本地嵌入式
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # 测试连接
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 情景记忆：存储历史对话片段
        self._episodic = chroma.get_or_create_collection("episodic")
        # 用户画像：存储提炼出的偏好和实体
        self._profile  = chroma.get_or_create_collection("user_profile")

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        # 追加到 Redis 列表（左推，最新在前）
        await self._redis.lpush(key, json.dumps({
            "role":      msg.role.value,
            "content":   msg.content,
            "ts":        msg.timestamp.isoformat(),
            "metadata":  msg.metadata,
        }))
        await self._redis.expire(key, 86400)  # 24h TTL

        # Compress working memory after reaching the threshold.
        if await self._redis.llen(key) >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)

    def schedule_profile_update(self, user_id: str, conv_id: str) -> asyncio.Task[None]:
        """Schedule a managed profile update; stale versions are skipped."""
        if self._closed:
            raise RuntimeError("MemoryManager is closed")

        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        version = self._profile_versions.get(user_id, 0) + 1
        self._profile_versions[user_id] = version
        task_suffix = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8]
        task = asyncio.create_task(
            self._run_profile_update(user_id, conv_id, version),
            name=f"profile-update:{task_suffix}:{version}",
        )
        self._profile_tasks.add(task)
        task.add_done_callback(self._profile_task_done)
        return task

    def _profile_task_done(self, task: asyncio.Task[None]) -> None:
        """Remove a completed task and consume failures so none go unobserved."""
        self._profile_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning("Managed profile task failed: %s", error)

    async def _run_profile_update(
        self,
        user_id: str,
        conv_id: str,
        version: int,
    ) -> None:
        """Serialize updates per user and discard stale versions after locking."""
        lock = self._profile_locks.setdefault(user_id, asyncio.Lock())
        try:
            async with lock:
                if self._profile_versions.get(user_id) != version:
                    return
                await self.update_profile(
                    user_id,
                    conv_id,
                    expected_version=version,
                )
        finally:
            if self._profile_versions.get(user_id) == version:
                self._profile_versions.pop(user_id, None)
                if self._profile_locks.get(user_id) is lock:
                    self._profile_locks.pop(user_id, None)

    async def drain_profile_updates(self, timeout_s: float = 30.0) -> None:
        """Wait for managed profile tasks and cancel them on timeout."""
        tasks = [task for task in self._profile_tasks if not task.done()]
        if not tasks:
            return

        _, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_s))
        if pending:
            logger.warning("Profile task drain timed out; cancelling %s task(s)", len(pending))
            for task in pending:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self, timeout_s: float = 30.0) -> None:
        """Drain profile tasks and close the Redis connection pool."""
        if self._closed:
            return
        self._closed = True
        await self.drain_profile_updates(timeout_s=timeout_s)
        await self._redis.aclose()

    async def update_profile(
        self,
        user_id: str,
        conv_id: str,
        *,
        expected_version: Optional[int] = None,
    ) -> None:
        """Build a structured user profile and persist it in ChromaDB."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return
        if (
            expected_version is not None
            and self._profile_versions.get(user_id) != expected_version
        ):
            return

        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
        task_prompt = self._safe_text(f"""从以下对话中提炼用户偏好和关键实体。
对话:
{text}

请提取 preferences、products 和 issue_types；没有信息时返回空列表。""")
        legacy_prompt = self._safe_text(
            task_prompt
            + '\n只返回 JSON，例如: {"preferences": ["..."], "entities": {"products": [], "issue_types": []}}'
        )

        def string_list(value: Any) -> List[str]:
            if value is None:
                return []
            values = value if isinstance(value, list) else [value]
            return [self._safe_text(item).strip() for item in values if self._safe_text(item).strip()]

        async def legacy_call() -> UserProfileOutput:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": legacy_prompt}],
            )
            raw = resp.content[0].text
            data = parse_json_object(raw)

            if not isinstance(data, dict):
                raise ValueError("用户画像结果不是对象")
            entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
            return UserProfileOutput(
                preferences=string_list(data.get("preferences")),
                entities={
                    "products": string_list(entities.get("products", entities.get("产品"))),
                    "issue_types": string_list(entities.get("issue_types", entities.get("问题类型"))),
                },
            )

        try:
            if self._profile_invoker is not None:
                output = await self._profile_invoker.ainvoke(
                    UserProfileOutput,
                    [{"role": "user", "content": task_prompt}],
                    legacy_fallback=legacy_call,
                )
            else:
                output = await legacy_call()

            if (
                expected_version is not None
                and self._profile_versions.get(user_id) != expected_version
            ):
                logger.debug("Skipping stale profile write: %s/%s", user_id, conv_id)
                return

            doc_id = f"{user_id}_profile_{conv_id}"
            doc_text = self._safe_text(output.model_dump_json())
            await asyncio.to_thread(
                self._replace_profile,
                doc_id,
                doc_text,
                user_id,
                conv_id,
            )
            logger.info(f"User profile updated: {user_id}")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    def structured_output_stats(self) -> Dict[str, Any]:
        if self._profile_invoker is None:
            return {"user_profile": {"mode": "legacy"}}
        return {"user_profile": self._profile_invoker.stats_snapshot()}

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索）
        history = await self._search_episodic(user_id, query or (recent[-1].content if recent else ""))

        # 3. 用户画像
        profile = await self._get_profile(user_id)

        # 4. 会话摘要（如果已压缩过）
        summary = await self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存 Redis（覆盖旧摘要）
          3. 旧消息存入情景记忆（ChromaDB）供跨会话检索
          4. 工作记忆只保留最近 5 条
        """
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # 保留最近 5 条
        keep        = messages[-5:]

        # LLM 摘要
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{text}")
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(resp.content[0].text).strip()
        except Exception:
            summary = f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        # 存摘要到 Redis
        skey = self._summary_key(user_id, conv_id)
        old_summary = await self._redis.get(skey) or ""
        new_summary = self._safe_text(f"{old_summary}\n{summary}").strip()
        await self._redis.setex(skey, 86400, new_summary)

        # 旧消息存入情景记忆
        await self._store_episodic(user_id, conv_id, text, summary)

        # 重置工作记忆为最近 5 条
        key = self._wm_key(user_id, conv_id)
        await self._redis.delete(key)
        for m in reversed(keep):
            await self._redis.lpush(key, json.dumps({
                "role": m.role.value, "content": m.content,
                "ts": m.timestamp.isoformat(), "metadata": m.metadata,
            }))
        await self._redis.expire(key, 86400)
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = await self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, query: str) -> List[str]:
        """语义检索情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            # 直接传 query_texts，ChromaDB 内置模型自动生成向量做匹配
            results = await asyncio.to_thread(
                self._episodic.query,
                query_texts=[query_text],
                n_results=self.HISTORY_TOP_K,
                where={"user_id": self._safe_text(user_id)},
            )
            docs = results["documents"][0] if results["documents"] else []
            return [self._safe_text(doc) for doc in docs if isinstance(doc, str) and doc.strip()]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        """将压缩后的对话片段存入情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            # 直接传 documents，ChromaDB 内置模型自动生成 embedding
            await asyncio.to_thread(
                self._episodic.add,
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "full_text": self._safe_text(text[:500])}],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    def _replace_profile(
        self,
        doc_id: str,
        doc_text: str,
        user_id: str,
        conv_id: str,
    ) -> None:
        """Replace one conversation profile from a worker thread."""
        try:
            self._profile.delete(ids=[doc_id])
        except Exception:
            pass
        self._profile.add(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{
                "user_id": user_id,
                "conv_id": conv_id,
                "ts": datetime.now().isoformat(),
            }],
        )

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """Return the newest profile by timestamp, independent of Chroma order."""
        try:
            results = await asyncio.to_thread(
                self._profile.get,
                where={"user_id": self._safe_text(user_id)},
                include=["documents", "metadatas"],
            )
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []
            if not documents:
                return {}

            latest_index = max(
                range(len(documents)),
                key=lambda index: self._profile_sort_key(
                    metadatas[index] if index < len(metadatas) else {},
                    index,
                ),
            )
            return json.loads(documents[latest_index])
        except Exception as ex:
            logger.warning("Failed to read user profile: %s", ex)
            return {}

    @staticmethod
    def _profile_sort_key(metadata: Any, index: int) -> tuple[float, int]:
        """Parse ISO timestamps and fall back deterministically on invalid values."""
        raw_timestamp = str((metadata or {}).get("ts", ""))
        try:
            timestamp = datetime.fromisoformat(
                raw_timestamp.replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            timestamp = float("-inf")
        return timestamp, index

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
