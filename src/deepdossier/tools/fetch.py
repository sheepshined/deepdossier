"""网页抓取与正文抽取（httpx + BeautifulSoup）。"""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 DeepDossier/0.1"
)

_STRIP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe")


def extract_text(html: str, max_chars: int = 6000) -> tuple[str, str]:
    """从 HTML 提取 (标题, 正文文本)。"""
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "")[:200]
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return title, text[:max_chars]


async def fetch_readable(
    client: httpx.AsyncClient,
    url: str,
    max_chars: int = 6000,
    timeout: float = 15.0,
) -> tuple[str | None, str]:
    """抓取 URL 并抽取正文，返回 (正文, 错误信息)；成功时错误信息为空串。"""
    try:
        resp = await client.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if ctype and "html" not in ctype and "text" not in ctype and "xml" not in ctype:
            return None, f"跳过非文本类型 {ctype}"
        title, text = extract_text(resp.text, max_chars=max_chars)
        if len(text) < 80:
            return None, "正文过短，可能为反爬页面"
        return (f"{title}\n\n{text}" if title else text), ""
    except Exception as e:  # noqa: BLE001 — 单页失败不阻断整体研究
        return None, f"{type(e).__name__}: {str(e)[:150]}"
