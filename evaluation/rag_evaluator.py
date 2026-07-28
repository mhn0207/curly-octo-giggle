"""Repeatable retrieval metrics and old/new RAG comparison."""

import asyncio
import inspect
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Dict, List, Sequence


@dataclass(frozen=True)
class RAGTestCase:
    query: str
    expected_titles: tuple[str, ...]


DEFAULT_RAG_CASES = [
    RAGTestCase("退款审核通过后多久到账", ("退款政策",)),
    RAGTestCase("订单发货七天还没收到怎么办", ("订单查询",)),
    RAGTestCase("如何开启两步验证保护账号", ("账户安全",)),
    RAGTestCase("登录提示 401 错误怎么处理", ("技术故障排查",)),
    RAGTestCase("金卡会员有什么折扣", ("会员与积分",)),
    RAGTestCase("加急配送需要几天和多少运费", ("配送说明",)),
]

SearchCallable = Callable[[str, int], Any]


class RAGEvaluator:
    """Compute Recall@K, MRR, top-1 accuracy, and latency percentiles."""

    def __init__(self, recall_ks: Sequence[int] = (1, 3, 5)) -> None:
        values = sorted({int(value) for value in recall_ks if int(value) > 0})
        if not values:
            raise ValueError("recall_ks must contain a positive integer")
        self._recall_ks = tuple(values)

    async def evaluate(
        self,
        search: SearchCallable,
        cases: Sequence[RAGTestCase] = DEFAULT_RAG_CASES,
    ) -> Dict[str, Any]:
        max_k = max(self._recall_ks)
        recalls = {value: 0 for value in self._recall_ks}
        reciprocal_ranks: List[float] = []
        latencies_ms: List[float] = []
        details: List[Dict[str, Any]] = []

        for case in cases:
            started = time.monotonic()
            results = await self._invoke_search(search, case.query, max_k)
            latencies_ms.append((time.monotonic() - started) * 1000)
            titles = [
                str(item.get("title", "")).strip()
                for item in results
                if isinstance(item, dict)
            ]
            expected = {title.strip() for title in case.expected_titles}
            rank = next(
                (index for index, title in enumerate(titles, start=1) if title in expected),
                None,
            )
            for value in self._recall_ks:
                if any(title in expected for title in titles[:value]):
                    recalls[value] += 1
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            details.append(
                {
                    "query": case.query,
                    "expected_titles": list(case.expected_titles),
                    "returned_titles": titles,
                    "first_relevant_rank": rank,
                }
            )

        total = len(cases)
        return {
            "cases": total,
            "recall": {
                f"recall_at_{value}": recalls[value] / total if total else 0.0
                for value in self._recall_ks
            },
            "mrr": sum(reciprocal_ranks) / total if total else 0.0,
            "top1_accuracy": recalls.get(1, 0) / total if total else 0.0,
            "latency_ms": {
                "p50": self._percentile(latencies_ms, 0.50),
                "p95": self._percentile(latencies_ms, 0.95),
            },
            "details": details,
        }

    async def compare(
        self,
        legacy_search: SearchCallable,
        langchain_search: SearchCallable,
        cases: Sequence[RAGTestCase] = DEFAULT_RAG_CASES,
    ) -> Dict[str, Any]:
        legacy = await self.evaluate(legacy_search, cases)
        langchain = await self.evaluate(langchain_search, cases)
        deltas = {
            key: langchain["recall"][key] - legacy["recall"][key]
            for key in legacy["recall"]
        }
        return {
            "legacy": legacy,
            "langchain": langchain,
            "delta": {
                **deltas,
                "mrr": langchain["mrr"] - legacy["mrr"],
                "top1_accuracy": langchain["top1_accuracy"] - legacy["top1_accuracy"],
                "p95_latency_ms": (
                    langchain["latency_ms"]["p95"] - legacy["latency_ms"]["p95"]
                ),
            },
            "recall_gate_passed": all(value >= 0.0 for value in deltas.values()),
            "cases": [asdict(case) for case in cases],
        }

    @staticmethod
    async def _invoke_search(search: SearchCallable, query: str, top_k: int) -> List[Dict[str, Any]]:
        if inspect.iscoroutinefunction(search):
            value = await search(query, top_k)
        else:
            value = await asyncio.to_thread(search, query, top_k)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, list):
            raise TypeError("RAG search callable must return a list")
        return value

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * percentile)))
        return round(ordered[index], 1)
