"""统一的 LangChain 结构化输出调用、重试、降级和统计。"""
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from packaging.version import Version
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
LegacyFallback = Callable[[], Awaitable[T]]


def _extract_json_value(raw: str, expected_type: type, opening: str) -> Any:
    """Decode the first complete JSON value of the requested container type."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw or ""):
        if character != opening:
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, expected_type):
            return value
    raise ValueError(f"LLM 返回值不包含 JSON {expected_type.__name__}")


def parse_json_object(raw: str) -> Dict[str, Any]:
    return _extract_json_value(raw, dict, "{")


def parse_json_array(raw: str) -> list[Any]:
    return _extract_json_value(raw, list, "[")

def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off", "disabled"}


class StructuredOutputMode(str, Enum):
    NATIVE = "native"
    TOOL = "tool"
    LEGACY_JSON = "legacy_json"

    @classmethod
    def from_value(cls, value: str) -> "StructuredOutputMode":
        try:
            return cls((value or "").strip().lower())
        except ValueError:
            logger.warning("未知结构化输出模式 %r，回退到 tool", value)
            return cls.TOOL


@dataclass
class StructuredCallStats:
    total: int = 0
    success: int = 0
    model_failures: int = 0
    validation_failures: int = 0
    retries: int = 0
    fallbacks: int = 0
    fallback_failures: int = 0
    shadow_calls: int = 0
    shadow_mismatches: int = 0
    shadow_structured_failures: int = 0
    shadow_legacy_failures: int = 0
    latency_ms_total: float = 0.0
    latency_ms_max: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StructuredOutputCapabilityError(RuntimeError):
    """当前集成或供应商不支持所请求的结构化输出能力。"""


class StructuredInvoker:
    """为各业务模块提供统一、可观测的结构化模型调用。"""

    def __init__(
        self,
        model: Any,
        *,
        mode: StructuredOutputMode = StructuredOutputMode.TOOL,
        fallback_enabled: bool = True,
        validation_retries: int = 1,
        component: str = "unknown",
        shadow_enabled: bool = False,
    ):
        self._model = model
        self._mode = mode
        self._fallback_enabled = fallback_enabled
        self._validation_retries = max(0, min(validation_retries, 2))
        self._component = component
        self._shadow_enabled = shadow_enabled
        self._bound_models: Dict[type[BaseModel], Any] = {}
        self.stats = StructuredCallStats()

    def stats_snapshot(self) -> Dict[str, Any]:
        total = self.stats.total
        return {
            **self.stats.to_dict(),
            "mode": self._mode.value,
            "fallback_enabled": self._fallback_enabled,
            "shadow_enabled": self._shadow_enabled,
            "avg_latency_ms": round(self.stats.latency_ms_total / total, 1) if total else 0.0,
            "validation_failure_rate": self.stats.validation_failures / total if total else 0.0,
            "fallback_rate": self.stats.fallbacks / total if total else 0.0,
        }

    def _record_latency(self, started: float) -> None:
        latency_ms = (time.monotonic() - started) * 1000
        self.stats.latency_ms_total += latency_ms
        self.stats.latency_ms_max = max(self.stats.latency_ms_max, latency_ms)

    async def ainvoke(
        self,
        schema: type[T],
        messages: Any,
        *,
        legacy_fallback: Optional[LegacyFallback[T]] = None,
    ) -> T:
        """调用结构化模型；失败时按配置执行模块自己的 legacy 路径。"""
        self.stats.total += 1
        started = time.monotonic()
        last_error: Optional[BaseException] = None

        if (
            self._shadow_enabled
            and self._mode != StructuredOutputMode.LEGACY_JSON
            and legacy_fallback is not None
        ):
            return await self._invoke_shadow(
                schema,
                messages,
                legacy_fallback,
                started,
            )

        if self._mode != StructuredOutputMode.LEGACY_JSON:
            attempts = self._validation_retries + 1
            for attempt in range(attempts):
                try:
                    runnable = self._structured_model(schema)
                    value = await runnable.ainvoke(messages)
                    output = value if isinstance(value, schema) else schema.model_validate(value)
                    self.stats.success += 1
                    logger.info(
                        "结构化调用成功: component=%s schema=%s mode=%s latency_ms=%.1f",
                        self._component,
                        schema.__name__,
                        self._mode.value,
                        (time.monotonic() - started) * 1000,
                    )
                    self._record_latency(started)
                    return output
                except Exception as ex:
                    last_error = ex
                    validation_error = self._is_validation_error(ex)
                    if validation_error:
                        self.stats.validation_failures += 1
                    else:
                        self.stats.model_failures += 1
                    logger.warning(
                        "结构化调用失败: component=%s schema=%s mode=%s type=%s",
                        self._component,
                        schema.__name__,
                        self._mode.value,
                        type(ex).__name__,
                    )
                    if not validation_error or attempt + 1 >= attempts:
                        break
                    self.stats.retries += 1

        legacy_primary = self._mode == StructuredOutputMode.LEGACY_JSON
        if legacy_fallback is not None and (legacy_primary or self._fallback_enabled):
            if not legacy_primary:
                self.stats.fallbacks += 1
            try:
                output = await legacy_fallback()
                validated = output if isinstance(output, schema) else schema.model_validate(output)
                if legacy_primary:
                    self.stats.success += 1
                self._record_latency(started)
                return validated
            except Exception as ex:
                if legacy_primary:
                    self.stats.model_failures += 1
                else:
                    self.stats.fallback_failures += 1
                last_error = ex
                logger.warning(
                    "legacy JSON 调用失败: component=%s schema=%s type=%s",
                    self._component,
                    schema.__name__,
                    type(ex).__name__,
                )

        if last_error is not None:
            self._record_latency(started)
            raise last_error
        self._record_latency(started)
        raise StructuredOutputCapabilityError(
            f"{self._component} 未配置可用的结构化输出或 legacy fallback"
        )

    async def _invoke_shadow(
        self,
        schema: type[T],
        messages: Any,
        legacy_fallback: LegacyFallback[T],
        started: float,
    ) -> T:
        self.stats.shadow_calls += 1
        structured_invoker = StructuredInvoker(
            self._model,
            mode=self._mode,
            fallback_enabled=False,
            validation_retries=self._validation_retries,
            component=f"{self._component}:shadow",
            shadow_enabled=False,
        )
        structured_invoker._bound_models = self._bound_models

        async def validated_legacy() -> T:
            value = await legacy_fallback()
            return value if isinstance(value, schema) else schema.model_validate(value)

        structured_result, legacy_result = await asyncio.gather(
            structured_invoker.ainvoke(schema, messages),
            validated_legacy(),
            return_exceptions=True,
        )
        self.stats.model_failures += structured_invoker.stats.model_failures
        self.stats.validation_failures += structured_invoker.stats.validation_failures
        self.stats.retries += structured_invoker.stats.retries

        structured_failed = isinstance(structured_result, BaseException)
        legacy_failed = isinstance(legacy_result, BaseException)
        if structured_failed:
            self.stats.shadow_structured_failures += 1
        if legacy_failed:
            self.stats.shadow_legacy_failures += 1

        if not structured_failed and not legacy_failed:
            if structured_result.model_dump(mode="json") != legacy_result.model_dump(mode="json"):
                self.stats.shadow_mismatches += 1
            self.stats.success += 1
            self._record_latency(started)
            return legacy_result
        if not legacy_failed:
            self.stats.success += 1
            self._record_latency(started)
            return legacy_result
        if not structured_failed:
            self.stats.success += 1
            self.stats.fallbacks += 1
            self._record_latency(started)
            return structured_result

        self.stats.fallback_failures += 1
        self._record_latency(started)
        raise legacy_result

    def _structured_model(self, schema: type[T]) -> Any:
        cached = self._bound_models.get(schema)
        if cached is not None:
            return cached
        if self._model is None:
            raise StructuredOutputCapabilityError("结构化模型未初始化")

        if self._mode == StructuredOutputMode.NATIVE:
            if not self._supports_native_json_schema():
                raise StructuredOutputCapabilityError(
                    "当前 langchain-anthropic 不支持原生 JSON Schema"
                )
            runnable = self._model.with_structured_output(schema, method="json_schema")
        else:
            # 0.2.x 使用强制 tool calling；该版本会忽略 method 参数，因此这里不传。
            runnable = self._model.with_structured_output(schema)

        self._bound_models[schema] = runnable
        return runnable

    @staticmethod
    def _supports_native_json_schema() -> bool:
        try:
            return Version(version("langchain-anthropic")) >= Version("1.1.0")
        except PackageNotFoundError:
            return False

    @staticmethod
    def _is_validation_error(ex: BaseException) -> bool:
        return isinstance(ex, ValidationError) or type(ex).__name__ in {
            "OutputParserException",
            "ValidationError",
        }
