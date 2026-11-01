"""Agent 数据契约（pydantic，所有结构化输出的唯一事实来源）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: str = Field(description="短标识，如 q1")
    question: str = Field(description="子问题文本")
    perspective: str = Field(default="", description="提问视角，如 技术/商业/批判/用户")


class ResearchPlan(BaseModel):
    title: str = Field(description="调研主题标题")
    subquestions: list[SubQuestion] = Field(description="3-5 个不同视角的子问题")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class EvidenceCard(BaseModel):
    question_id: str = Field(description="对应的子问题 id")
    claim: str = Field(description="一句话事实要点")
    detail: str = Field(description="支撑细节，2-4 句")
    source_url: str
    source_title: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="证据可信度 0-1")
    snippet_only: bool = Field(
        default=False,
        description="R9：正文抓取全失败时仅基于搜索摘要提炼，可信度打折的显式标注",
    )


class EvidenceBundle(BaseModel):
    evidence: list[EvidenceCard]


class Critique(BaseModel):
    sufficient: bool = Field(description="证据是否足以支撑报告")
    gaps: list[str] = Field(default_factory=list, description="证据缺口描述")
    conflicts: list[str] = Field(default_factory=list, description="来源间的矛盾点")
    followup_queries: list[str] = Field(default_factory=list, description="定向补搜查询词，最多 3 条")
    reasoning: str = ""


class QuestionRoute(BaseModel):
    """开头问题路由输出：research=启动完整调研，direct=直接回答（无需检索）。"""

    route: str = Field(description="'research' 或 'direct'")
    reason: str = Field(default="", description="路由理由，一句话")


class SourceCuration(BaseModel):
    """来源过滤（source curation）输出：保留的搜索结果编号。"""

    keep: list[int] = Field(description="保留的结果编号列表，如 [0, 2]")


class DraftReview(BaseModel):
    """草稿事实审查输出：是否需要修订 + 具体问题清单。"""

    needs_revision: bool = Field(description="草稿是否需要修订")
    issues: list[str] = Field(
        default_factory=list,
        description="具体问题清单，每条指明位置与问题（无引用支撑的断言/引用错位/与证据冲突的表述）",
    )


class SourceRef(BaseModel):
    n: int = Field(description="引用编号，与 markdown 上标 [n] 一致")
    title: str = ""
    url: str


class Report(BaseModel):
    title: str
    markdown: str = Field(description="完整 Markdown 报告，引用以上标 [n] 形式指向参考来源")
    sources: list[SourceRef] = Field(
        default_factory=list,
        description="结构化参考来源列表，编号与 markdown 中的 [n] 引用一致",
    )
