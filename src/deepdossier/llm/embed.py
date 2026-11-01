"""问题文本 → 向量（带 SQLite 缓存）：embedding 可用时为记忆检索提供语义匹配。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepdossier.store.runs import RunStore

logger = logging.getLogger("deepdossier.embed")


async def embed_question(
    embedder: Any,
    store: RunStore,
    text: str,
) -> list[float] | None:
    """返回 text 的向量（优先命中缓存）；embedder 为 None 或失败时返回 None（退化 n-gram）。"""
    if embedder is None:
        return None
    cached = store.get_query_vec(text)
    if cached:
        return cached
    try:
        vec = await embedder.embed(text)
    except Exception as e:  # noqa: BLE001 — 向量失败不阻断研究，退化为字面匹配
        logger.warning("embedding 调用失败，记忆检索退化为 n-gram: %s", e)
        return None
    store.set_query_vec(text, vec)
    return vec
