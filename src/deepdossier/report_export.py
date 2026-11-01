"""报告独立 HTML 导出：零外部依赖的单文件，可直接浏览器打开或 Ctrl+P 存 PDF。

设计：markdown 原文以 JSON 内嵌，渲染由文件内 JS 完成（与 web/index.html 的
mdRender 保持同一实现，改动需两处同步）；正文引用 [n] 锚点跳转到文末来源表。
"""

from __future__ import annotations

import json
import time
from typing import Any

_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { margin: 0; background: #f5f6f8; color: #1c2430;
         font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .page { max-width: 860px; margin: 0 auto; background: #fff; padding: 48px 56px 64px; }
  .meta { color: #67707d; font-size: 13px; line-height: 1.8; border-bottom: 1px solid #e3e6ea;
          padding-bottom: 14px; margin-bottom: 28px; }
  .meta b { color: #1c2430; }
  h1 { font-size: 24px; margin: 0 0 20px; line-height: 1.4; }
  #report { line-height: 1.85; font-size: 15px; }
  #report h2 { font-size: 18px; border-bottom: 1px solid #e3e6ea; padding-bottom: 6px; margin-top: 30px; }
  #report h3 { font-size: 16px; margin-top: 24px; }
  #report a { color: #0b62d6; text-decoration: none; }
  #report sup a { color: #c26a00; font-size: 11px; padding: 0 1px; }
  #report blockquote { margin: 12px 0; padding: 8px 16px; border-left: 3px solid #d8dde3;
          background: #f8f9fb; color: #4a5563; }
  #report code { background: #f1f3f6; border-radius: 4px; padding: 1px 5px; font-size: 13px; }
  #report pre { background: #f1f3f6; border-radius: 6px; padding: 12px 14px; overflow-x: auto; }
  #report pre code { background: none; padding: 0; }
  #report table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; }
  #report th, #report td { border: 1px solid #d8dde3; padding: 6px 10px; text-align: left; }
  #report th { background: #f4f6f9; }
  #report hr { border: none; border-top: 1px solid #e3e6ea; margin: 24px 0; }
  ol.src { padding-left: 24px; line-height: 2; font-size: 14px; }
  ol.src li { margin-bottom: 4px; }
  ol.src .t { color: #67707d; }
  .foot { margin-top: 40px; color: #98a1ac; font-size: 12px; text-align: center; }
  @media print {
    body { background: #fff; }
    .page { max-width: none; padding: 0; }
    a { color: inherit; }
    #report sup a { color: #555; }
  }
</style>
</head>
<body>
<div class="page">
  <div class="meta">
    <b>调研问题：</b>__QUESTION__<br>
    <b>生成时间：</b>__TIME__　<b>耗时：</b>__DURATION__　<b>状态：</b>__STATUS__<br>
    <b>LLM 用量：</b>__USAGE__　<b>参考来源：</b>__NSRC__ 条
  </div>
  <h1>__TITLE__</h1>
  <div id="report"></div>
  <div class="foot">由 DeepDossier 深度调研智能体生成 · Planner → Researcher → Critic → Writer · __VERSION__</div>
</div>
<script>
(function () {
  var md = __MD_JSON__;
  document.getElementById("report").innerHTML = mdRender(md);

  function escapeHtml(s) { var d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
  function inline(s) {
    s = escapeHtml(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\\*\\*([^*]+)\\*\\*/g, "<b>$1</b>");
    s = s.replace(/\\*([^*]+)\\*/g, "<i>$1</i>");
    s = s.replace(/\\[([^\\]]+)\\]\\((https?:[^)\\s]+)\\)/g, '<a href="$2" target="_blank">$1</a>');
    s = s.replace(/(^|[\\s(])((?:https?:\\/\\/)[^\\s<)]*[A-Za-z0-9_/&%=?#~-])/g, '$1<a href="$2" target="_blank">$2</a>');
    s = s.replace(/\\[(\\d+)\\]/g, '<sup><a href="#src-$1">[$1]</a></sup>');
    return s;
  }
  function mdRender(md) {
    var lines = md.replace(/\\r/g, "").split("\\n"), out = [], i = 0, inCode = false, code = [];
    while (i < lines.length) {
      var ln = lines[i];
      if (/^```/.test(ln)) {
        if (inCode) { out.push("<pre><code>" + escapeHtml(code.join("\\n")) + "</code></pre>"); code = []; }
        inCode = !inCode; i++; continue;
      }
      if (inCode) { code.push(ln); i++; continue; }
      if (/^\\s*$/.test(ln)) { i++; continue; }
      var h = ln.match(/^(#{1,4})\\s+(.*)$/);
      if (h) { var n = h[1].length; out.push("<h" + n + ">" + inline(h[2]) + "</h" + n + ">"); i++; continue; }
      if (/^(---|\\*\\*\\*)\\s*$/.test(ln)) { out.push("<hr>"); i++; continue; }
      if (/^\\s*>\\s?/.test(ln)) {
        var q = [];
        while (i < lines.length && /^\\s*>\\s?/.test(lines[i])) { q.push(lines[i].replace(/^\\s*>\\s?/, "")); i++; }
        out.push("<blockquote>" + q.map(inline).join("<br>") + "</blockquote>"); continue;
      }
      if (/^\\s*\\|.*\\|/.test(ln) && i + 1 < lines.length && /^\\s*\\|?[\\s:|-]+\\|?\\s*$/.test(lines[i + 1])) {
        var rows = [], head = splitRow(ln);
        i += 2;
        while (i < lines.length && /^\\s*\\|.*\\|/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
        var t = "<table><tr>" + head.map(function (c) { return "<th>" + inline(c) + "</th>"; }).join("") + "</tr>";
        rows.forEach(function (r) { t += "<tr>" + r.map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("") + "</tr>"; });
        out.push(t + "</table>"); continue;
      }
      if (/^\\s*[-*+]\\s+/.test(ln)) {
        var ul = [];
        while (i < lines.length && /^\\s*[-*+]\\s+/.test(lines[i])) { ul.push("<li>" + inline(lines[i].replace(/^\\s*[-*+]\\s+/, "")) + "</li>"); i++; }
        out.push("<ul>" + ul.join("") + "</ul>"); continue;
      }
      if (/^\\s*\\d+\\.\\s+/.test(ln)) {
        var ol = [];
        while (i < lines.length && /^\\s*\\d+\\.\\s+/.test(lines[i])) { ol.push("<li>" + inline(lines[i].replace(/^\\s*\\d+\\.\\s+/, "")) + "</li>"); i++; }
        out.push("<ol>" + ol.join("") + "</ol>"); continue;
      }
      var para = [inline(ln)]; i++;
      while (i < lines.length && !/^\\s*$/.test(lines[i]) && !/^(#{1,4}\\s|\\s*[-*+]\\s|\\s*\\d+\\.\\s|\\s*>|\\s*\\||```)/.test(lines[i])) {
        para.push(inline(lines[i])); i++;
      }
      out.push("<p>" + para.join("<br>") + "</p>");
    }
    return out.join("");
  }
  function splitRow(ln) {
    return ln.replace(/^\\s*\\|/, "").replace(/\\|\\s*$/, "").split("|").map(function (c) { return c.trim(); });
  }
})();
</script>
</body>
</html>
"""


def _fmt_duration(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m} 分 {s} 秒" if m else f"{s} 秒"


def render_export_html(run: dict[str, Any], version: str = "") -> str:
    """把一条 run 记录渲染为可独立分发的 HTML 报告。"""
    usage = run.get("usage") or {}
    usage_desc = (
        f"{usage.get('calls', 0)} 次调用 · 输入 {usage.get('prompt_tokens', 0)} tokens · "
        f"输出 {usage.get('completion_tokens', 0)} tokens"
        if usage
        else "—"
    )
    title = run.get("report_md", "").split("\n", 1)[0].lstrip("# ").strip() or "深度调研报告"

    html = _TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__QUESTION__", str(run.get("question", "")))
    created = run.get("created_at")
    html = html.replace(
        "__TIME__",
        time.strftime("%Y-%m-%d %H:%M", time.localtime(created)) if created else "—",
    )
    duration = (run.get("finished_at") or 0) - (run.get("created_at") or 0) if run.get("finished_at") else None
    html = html.replace("__DURATION__", _fmt_duration(duration))
    html = html.replace("__STATUS__", {"completed": "完成", "failed": "失败"}.get(run.get("status", ""), run.get("status", "")))
    html = html.replace("__USAGE__", usage_desc)
    sources = run.get("sources") or []
    html = html.replace("__NSRC__", str(len(sources)))
    html = html.replace("__VERSION__", version)
    # JSON 内嵌防注入：拆开 "</"，避免 markdown 中出现 </script> 时提前终止脚本标签
    md_json = json.dumps(str(run.get("report_md", "")), ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("__MD_JSON__", md_json)
    return html
