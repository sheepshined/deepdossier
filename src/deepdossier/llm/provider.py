"""LLM Provider 抽象层。

- OpenAICompatProvider：OpenAI 兼容协议（DeepSeek / GLM / OpenRouter / Ollama / vLLM ...），
  结构化输出 = prompt 注入 schema + JSON 提取 + pydantic 校验 + 失败修复重试（兼容所有国产模型）；
  内置单 run 调用预算（max_calls），防止成本失控。
- StubProvider：确定性离线桩，测试与无 key 演示用。
- EmbeddingClient / StubEmbeddingClient：记忆检索用向量（OpenAI 兼容 /embeddings）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("deepdossier.llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMBudgetError(LLMError):
    """预算熔断（次数/token/金额任一触发），调用方按"部分成果"路径落终态（R3/R27）。"""


def estimate_cost_cny(usage: dict[str, int], price_in: float, price_out: float) -> float:
    """按价格表（¥/百万 token）估算成本（R27）。"""
    return usage.get("prompt_tokens", 0) / 1e6 * price_in + usage.get("completion_tokens", 0) / 1e6 * price_out


# ---------------------------------------------------------------- JSON 工具

def extract_json(content: str) -> str:
    """从模型输出中提取 JSON 串（容忍 markdown 代码块与前后杂文）。"""
    fence = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if fence:
        content = fence.group(1)
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"输出中未找到 JSON 对象: {content[:200]!r}")
    return content[start : end + 1]


def parse_structured(content: str, schema: type[T]) -> T:
    return schema.model_validate(json.loads(extract_json(content)))


def schema_hint(schema: type[BaseModel]) -> str:
    fields = schema.model_json_schema().get("properties", {})
    lines = [f"- {name}: {spec.get('type', 'any')}（{spec.get('description', '')}）" for name, spec in fields.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------- Provider

class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        fast: bool = False,
    ) -> str | T:
        """返回校验后的 pydantic 对象（schema 非空）或纯文本。"""

    @property
    @abstractmethod
    def usage(self) -> dict[str, int]:
        """{calls, prompt_tokens, completion_tokens}"""

    async def aclose(self) -> None:  # noqa: B027 — 默认无底层资源需要释放
        return None


class OpenAICompatProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        fast_model: str = "",
        temperature: float = 0.3,
        max_retries: int = 2,
        max_calls: int | None = None,
        timeout_seconds: float = 120.0,
        max_tokens_budget: int = 0,
        max_cost_cny: float = 0.0,
        price_input_cny_per_m: float = 2.0,
        price_output_cny_per_m: float = 8.0,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.fast_model = fast_model or model
        self.temperature = temperature
        self.max_retries = max_retries
        self._max_calls = max_calls
        self._max_tokens = max_tokens_budget
        self._max_cost = max_cost_cny
        self._price_in = price_input_cny_per_m
        self._price_out = price_output_cny_per_m
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_seconds)

    async def aclose(self) -> None:
        """关闭底层 HTTP 客户端（应用级共享 provider 关停时调用）。"""
        await self._client.close()

    @property
    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    def _check_budget(self) -> None:
        if self._max_calls is not None and self._usage["calls"] >= self._max_calls:
            raise LLMBudgetError(f"LLM 调用预算耗尽（max_calls={self._max_calls}），终止本 run")
        if self._max_tokens and self._usage["prompt_tokens"] + self._usage["completion_tokens"] >= self._max_tokens:
            raise LLMBudgetError(
                f"LLM token 预算耗尽（{self._usage['prompt_tokens'] + self._usage['completion_tokens']}"
                f"/{self._max_tokens}），终止本 run"
            )
        if self._max_cost:
            cost = estimate_cost_cny(self._usage, self._price_in, self._price_out)
            if cost >= self._max_cost:
                raise LLMBudgetError(f"LLM 金额预算耗尽（约 ¥{cost:.2f}/{self._max_cost}），终止本 run")

    async def complete(self, *, system, user, schema=None, temperature=None, max_tokens=None, fast=False):
        self._check_budget()
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if schema is not None:
            user = (
                f"{user}\n\n[输出要求] 只输出一个 JSON 对象，不要任何解释或 markdown 代码块。"
                f"字段要求：\n{schema_hint(schema)}"
            )
        messages.append({"role": "user", "content": user})

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.fast_model if fast else self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                u = resp.usage
                if u:
                    self._usage["calls"] += 1
                    self._usage["prompt_tokens"] += u.prompt_tokens or 0
                    self._usage["completion_tokens"] += u.completion_tokens or 0
                if schema is None:
                    return content
                try:
                    return parse_structured(content, schema)
                except (ValueError, json.JSONDecodeError, ValidationError) as e:
                    last_err = e
                    logger.warning("结构化输出校验失败（%s），进入修复重试", type(e).__name__)
                    # 修复环：把错误信息回灌，要求模型重新输出
                    messages.append({"role": "assistant", "content": content[:2000]})
                    messages.append({
                        "role": "user",
                        "content": f"上面的输出不符合要求（{type(e).__name__}: {str(e)[:300]}）。"
                                   "请重新输出，只输出一个合法 JSON 对象。",
                    })
            except LLMError:
                raise
            except Exception as e:  # noqa: BLE001 — 网络/限流等，退避后重试
                last_err = e
                logger.warning("LLM 调用异常（第 %d 次）: %s", attempt + 1, e)
            await asyncio.sleep(1.5**attempt)
        # R1：修复重试耗尽 → 换快模型再走一轮完整修复环（仅结构化输出场景）
        if (
            schema is not None
            and not fast
            and self.fast_model != self.model
            and isinstance(last_err, (ValueError, json.JSONDecodeError, ValidationError))
        ):
            logger.warning("结构化输出在主模型上修复失败，切换快模型 %s 最后重试", self.fast_model)
            try:
                return await self.complete(
                    system=system, user=user, schema=schema,
                    temperature=temperature, max_tokens=max_tokens, fast=True,
                )
            except LLMError as e2:
                raise LLMError(f"结构化输出在主/快模型上均失败: {e2}") from e2
        raise LLMError(f"LLM 调用失败（重试 {self.max_retries} 次）: {last_err}") from last_err


class StubProvider(BaseLLMProvider):
    """确定性离线桩：按 schema 类型返回固定结构，内容从 user prompt 中提取关键信息。"""

    def __init__(self) -> None:
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @property
    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    async def complete(self, *, system, user, schema=None, temperature=None, max_tokens=None, fast=False):
        self._usage["calls"] += 1
        self._usage["completion_tokens"] += len(user) // 4

        if schema is None:
            return f"[stub 摘要] {user[:120]}"

        name = schema.__name__
        if name == "QuestionRoute":
            from deepdossier.agents.schemas import QuestionRoute

            return QuestionRoute(route="research", reason="stub 默认走完整调研")
        if name == "DraftReview":
            from deepdossier.agents.schemas import DraftReview

            return DraftReview(needs_revision=False, issues=[])  # stub 草稿免审
        if name == "ResearchPlan":
            from deepdossier.agents.schemas import ResearchPlan

            return ResearchPlan(
                title=f"《{user[:30]}》调研报告",
                subquestions=[
                    {"id": "q1", "question": f"{user[:40]} 的核心概念与现状是什么？", "perspective": "技术视角"},
                    {"id": "q2", "question": f"{user[:40]} 的主要争议与反面观点有哪些？", "perspective": "批判视角"},
                ],
            )
        if name == "EvidenceBundle":
            from deepdossier.agents.schemas import EvidenceBundle

            query = user[:60]
            return EvidenceBundle(
                evidence=[
                    {
                        "question_id": "q1",
                        "claim": f"关于「{query}」的 stub 事实要点",
                        "detail": "这是离线演示的确定性证据内容，用于验证编排与引用链路。",
                        "source_url": "https://example.com/stub",
                        "source_title": "Stub Source",
                        "confidence": 0.5,
                    }
                ]
            )
        if name == "Critique":
            from deepdossier.agents.schemas import Critique

            if "当前轮次: 2" in user or "当前轮次: 3" in user:
                return Critique(sufficient=True, gaps=[], followup_queries=[], reasoning="证据已覆盖计划。")
            return Critique(
                sufficient=False,
                gaps=["缺少量化的对比数据"],
                followup_queries=[f"{user[:20]} 最新数据对比"],
                reasoning="第一轮证据不足，需补搜。",
            )
        if name == "SourceCuration":
            import re as _re

            from deepdossier.agents.schemas import SourceCuration

            m = _re.search(r"共 (\d+) 条", user)
            n = int(m.group(1)) if m else 0
            return SourceCuration(keep=list(range(n)))  # stub 全保留，行为与过滤关闭时一致
        if name == "Report":
            from deepdossier.agents.schemas import Report

            question = user.split("\n")[0][:60]
            return Report(
                title=f"《{question}》调研报告",
                markdown=(
                    f"# 《{question}》调研报告\n\n"
                    "## 摘要\n本报告由 DeepDossier 离线演示模式（Stub Provider）生成。\n\n"
                    "## 主要发现\n1. stub 事实要点 [1]\n\n"
                    "## 参考来源\n1. [Stub Source](https://example.com/stub)\n"
                ),
                sources=[{"n": 1, "title": "Stub Source", "url": "https://example.com/stub"}],
            )
        raise LLMError(f"StubProvider 未实现 schema: {name}")


# ---------------------------------------------------------------- Embedding

class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    async def aclose(self) -> None:  # noqa: B027 — 默认无资源需要释放
        return None


class EmbeddingClient(BaseEmbedder):
    """OpenAI 兼容 /embeddings 接口。"""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=60)

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(model=self.model, input=text[:4000])
        return list(resp.data[0].embedding)

    async def aclose(self) -> None:
        await self._client.close()


class StubEmbeddingClient(BaseEmbedder):
    """确定性桩向量（sha256 前 16 字节归一化），离线测试用。"""

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]


def build_provider(settings) -> BaseLLMProvider:
    if settings.llm_provider == "stub":
        return StubProvider()
    return OpenAICompatProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fast_model=settings.fast_model,
        temperature=settings.temperature,
        max_retries=settings.llm_max_retries,
        max_calls=settings.max_llm_calls,
        timeout_seconds=settings.llm_timeout_seconds,
        max_tokens_budget=settings.max_run_tokens,
        max_cost_cny=settings.max_run_cost_cny,
        price_input_cny_per_m=settings.price_input_cny_per_m,
        price_output_cny_per_m=settings.price_output_cny_per_m,
    )


def build_embedder(settings) -> BaseEmbedder | None:
    """未配置 embedding_model 时返回 None，调用方退化为 n-gram 检索。"""
    if not settings.embedding_model:
        return None
    if settings.llm_provider == "stub":
        return StubEmbeddingClient()
    return EmbeddingClient(settings.llm_base_url, settings.llm_api_key, settings.embedding_model)
