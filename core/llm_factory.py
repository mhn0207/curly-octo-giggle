"""知应 AI 的 LangChain Chat Model 工厂。"""
import logging
import os
from typing import Any, Optional

from core.structured_invoker import StructuredInvoker, StructuredOutputMode, env_bool

logger = logging.getLogger(__name__)


def create_chat_model(
    *,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> Any:
    """创建与当前锁定版本兼容的 ChatAnthropic。"""
    from langchain_anthropic import ChatAnthropic

    options: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if base_url:
        options["base_url"] = base_url
    return ChatAnthropic(**options)


def create_structured_invoker(
    *,
    api_key: str,
    model: str,
    base_url: Optional[str],
    component: str,
    temperature: float,
    max_tokens: int,
) -> Optional[StructuredInvoker]:
    """根据环境开关创建调用器；关闭时返回 None，业务模块继续走旧路径。"""
    if not env_bool("ZHIYING_LANGCHAIN_STRUCTURED_OUTPUT", False):
        return None

    mode = StructuredOutputMode.from_value(
        os.getenv("ZHIYING_STRUCTURED_OUTPUT_MODE", "tool")
    )
    fallback_enabled = env_bool("ZHIYING_STRUCTURED_OUTPUT_FALLBACK", True)
    try:
        validation_retries = int(
            os.getenv("ZHIYING_STRUCTURED_OUTPUT_VALIDATION_RETRIES", "1")
        )
    except ValueError:
        logger.warning("非法的结构化输出重试次数，使用默认值 1")
        validation_retries = 1
    try:
        chat_model = None
        if mode != StructuredOutputMode.LEGACY_JSON:
            chat_model = create_chat_model(
                api_key=api_key,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(os.getenv("ZHIYING_LLM_TIMEOUT", "30")),
                max_retries=int(os.getenv("ZHIYING_LLM_MAX_RETRIES", "2")),
            )
        return StructuredInvoker(
            chat_model,
            mode=mode,
            fallback_enabled=fallback_enabled,
            validation_retries=validation_retries,
            component=component,
            shadow_enabled=env_bool("ZHIYING_STRUCTURED_OUTPUT_SHADOW", False),
        )
    except Exception as ex:
        logger.warning(
            "结构化调用器初始化失败，%s 继续使用 legacy JSON: %s",
            component,
            type(ex).__name__,
        )
        return None
