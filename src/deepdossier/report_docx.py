"""报告 DOCX 导出：markdown → Word 文档（python-docx，零模板）。

块级解析与 web 渲染器同一套约定（标题/段落/列表/表格/引用块/代码块/分隔线）；
行内支持粗体/斜体/行内代码/链接（真实 hyperlink）/上标引用 [n]。
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?:[^)\s]+)\)")
_INLINE_RE = re.compile(
    r"(\*\*.+?\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\(https?:[^)\s]+\)|\[\d+\])"
)


def _add_hyperlink(paragraph, text: str, url: str) -> None:
    """python-docx 无内建超链接 API，直接构造 oxml。"""
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = paragraph._p.makeelement(qn("w:hyperlink"), {qn("r:id"): r_id})
    new_run = paragraph._p.makeelement(qn("w:r"), {})
    rPr = paragraph._p.makeelement(qn("w:rPr"), {})
    color = paragraph._p.makeelement(qn("w:color"), {qn("w:val"): "0B62D6"})
    u = paragraph._p.makeelement(qn("w:u"), {qn("w:val"): "single"})
    rPr.append(color)
    rPr.append(u)
    new_run.append(rPr)
    t = paragraph._p.makeelement(qn("w:t"), {})
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def _add_internal_link(paragraph, anchor: str, text: str) -> None:
    """内部锚点超链接（上标引用 [n] → 跳到文末来源书签）。"""
    hyperlink = paragraph._p.makeelement(qn("w:hyperlink"), {qn("w:anchor"): anchor})
    new_run = paragraph._p.makeelement(qn("w:r"), {})
    rPr = paragraph._p.makeelement(qn("w:rPr"), {})
    vert = paragraph._p.makeelement(qn("w:vertAlign"), {qn("w:val"): "superscript"})
    color = paragraph._p.makeelement(qn("w:color"), {qn("w:val"): "C26A00"})
    rPr.append(vert)
    rPr.append(color)
    new_run.append(rPr)
    t = paragraph._p.makeelement(qn("w:t"), {})
    t.text = text
    t.set(qn("xml:space"), "preserve")
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def _add_bookmark(paragraph, name: str, bid: int) -> None:
    """在段落上打书签（参考来源条目 → src_{n}）。"""
    start = paragraph._p.makeelement(qn("w:bookmarkStart"), {qn("w:id"): str(bid), qn("w:name"): name})
    end = paragraph._p.makeelement(qn("w:bookmarkEnd"), {qn("w:id"): str(bid)})
    paragraph._p.insert(0, start)
    paragraph._p.append(end)


def _add_inline(paragraph, text: str) -> None:
    for tok in _INLINE_RE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = paragraph.add_run(tok[2:-2])
            r.bold = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2 and not tok.startswith("**"):
            r = paragraph.add_run(tok[1:-1])
            r.italic = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1])
            r.font.name = "Consolas"
        elif _LINK_RE.fullmatch(tok):
            m = _LINK_RE.fullmatch(tok)
            assert m is not None
            _add_hyperlink(paragraph, m.group(1), m.group(2))
        elif re.fullmatch(r"\[\d+\]", tok):
            # 上标引用 + 内部锚点跳转（F15：docx 引用为上标可点击超链接）
            _add_internal_link(paragraph, f"src_{tok[1:-1]}", tok)
        else:
            paragraph.add_run(tok)


def _split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def render_docx(run: dict[str, Any], version: str = "") -> bytes:
    """把一条 run 的 report_md 渲染为 DOCX 字节流。"""
    md = str(run.get("report_md", ""))
    doc = Document()
    props = doc.core_properties
    props.title = md.split("\n", 1)[0].lstrip("# ").strip() or "深度调研报告"
    props.author = f"DeepDossier {version}".strip()
    props.created = datetime.now().replace(microsecond=0)

    # 元信息头
    meta = doc.add_paragraph()
    r = meta.add_run(
        f"调研问题：{run.get('question', '')}    状态：{run.get('status', '')}"
    )
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x67, 0x70, 0x7D)
    usage = run.get("usage") or {}
    if usage:
        r2 = meta.add_run(
            f"    LLM {usage.get('calls', 0)} 次调用 · "
            f"输入 {usage.get('prompt_tokens', 0)} / 输出 {usage.get('completion_tokens', 0)} tokens"
        )
        r2.font.size = Pt(9)
        r2.font.color.rgb = RGBColor(0x67, 0x70, 0x7D)

    lines = md.replace("\r", "").split("\n")
    i, in_code, code_buf = 0, False, []
    in_sources = False  # 进入"参考来源"区后，有序列表条目打 src_{n} 书签供引用跳转
    bookmark_id = 100
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            if in_code:
                p = doc.add_paragraph()
                rc = p.add_run("\n".join(code_buf))
                rc.font.name = "Consolas"
                rc.font.size = Pt(9)
                code_buf = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        h = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if h:
            level = min(len(h[1]), 3)
            if "参考来源" in h[2]:
                in_sources = True
            p = doc.add_heading("", level=level)
            _add_inline(p, h[2])
            i += 1
            continue
        if re.match(r"^(---|\*\*\*)\s*$", ln):
            p = doc.add_paragraph("─" * 46)
            for r3 in p.runs:
                r3.font.color.rgb = RGBColor(0xD8, 0xDD, 0xE3)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue
        if re.match(r"^\s*>\s?", ln):
            quote = []
            while i < len(lines) and re.match(r"^\s*>\s?", lines[i]):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(18)
            _add_inline(p, " ".join(quote))
            for r4 in p.runs:
                r4.font.color.rgb = RGBColor(0x4A, 0x55, 0x63)
            continue
        if re.match(r"^\s*\|.*\|", ln) and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            head = _split_row(ln)
            i += 2
            rows = []
            while i < len(lines) and re.match(r"^\s*\|.*\|", lines[i]):
                rows.append(_split_row(lines[i]))
                i += 1
            table = doc.add_table(rows=1 + len(rows), cols=len(head))
            table.style = "Table Grid"
            for j, c in enumerate(head):
                cell_p = table.rows[0].cells[j].paragraphs[0]
                run_bold = cell_p.add_run(c)
                run_bold.bold = True
            for ri, row in enumerate(rows, start=1):
                for j, c in enumerate(row[: len(head)]):
                    _add_inline(table.rows[ri].cells[j].paragraphs[0], c)
            continue
        if re.match(r"^\s*[-*+]\s+", ln):
            while i < len(lines) and re.match(r"^\s*[-*+]\s+", lines[i]):
                p = doc.add_paragraph(style="List Bullet")
                _add_inline(p, re.sub(r"^\s*[-*+]\s+", "", lines[i]))
                i += 1
            continue
        if re.match(r"^\s*\d+\.\s+", ln):
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                num_m = re.match(r"^\s*(\d+)\.\s+", lines[i])
                p = doc.add_paragraph(style="List Number")
                if in_sources and num_m:
                    _add_bookmark(p, f"src_{num_m.group(1)}", bookmark_id)
                    bookmark_id += 1
                _add_inline(p, re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            continue
        p = doc.add_paragraph()
        _add_inline(p, ln.strip())
        i += 1

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
