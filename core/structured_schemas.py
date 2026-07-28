"""知应 AI 内部 LLM 结构化输出 Schema。

这些模型只约束 LLM 与业务层之间的边界，不改变 FastAPI 的外部请求/响应模型。
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictOutputModel(BaseModel):
    """拒绝模型臆造的字段，避免无声丢弃非法输出。"""

    model_config = ConfigDict(extra="forbid")


IntentName = Literal[
    "query",
    "complaint",
    "request",
    "greeting",
    "escalation",
    "technical",
    "billing",
    "account",
    "feedback",
    "other",
]


class IntentEntities(StrictOutputModel):
    order_id: list[str] = Field(default_factory=list)
    product: list[str] = Field(default_factory=list)
    date: list[str] = Field(default_factory=list)
    amount: list[str] = Field(default_factory=list)
    error_code: list[str] = Field(default_factory=list)


class IntentLLMOutput(StrictOutputModel):
    intent: IntentName
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    entities: IntentEntities = Field(default_factory=IntentEntities)


class QueryRewriteOutput(StrictOutputModel):
    queries: list[str] = Field(min_length=1, max_length=8)


class RerankOutput(StrictOutputModel):
    ordered_indexes: list[int] = Field(default_factory=list)


class UserEntities(StrictOutputModel):
    products: list[str] = Field(default_factory=list)
    issue_types: list[str] = Field(default_factory=list)


class UserProfileOutput(StrictOutputModel):
    preferences: list[str] = Field(default_factory=list)
    entities: UserEntities = Field(default_factory=UserEntities)


class JudgeOutput(StrictOutputModel):
    relevance: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    helpfulness: float = Field(ge=0.0, le=1.0)
