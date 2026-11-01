"""全局配置（pydantic-settings，环境变量前缀 DOSSIER_，支持 .env）。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/deepdossier/config.py → 项目根


def _resolve_env_file() -> str:
    """.env 发现：优先当前工作目录；回退项目根。

    服务可能从任意目录启动（如 `deepdossier serve` 不在项目根执行），
    仅用相对路径会导致 .env 静默丢失、真实 LLM 调用 401（密钥为空）。
    """
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        return str(cwd_env)
    root_env = _PROJECT_ROOT / ".env"
    if root_env.is_file():
        return str(root_env)
    return str(cwd_env)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOSSIER_",
        env_file=_resolve_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    llm_provider: str = "openai_compatible"  # openai_compatible | stub
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_fast_model: str = ""
    temperature: float = 0.3
    llm_max_retries: int = 2
    max_llm_calls: int = 150  # 单 run LLM 调用硬预算，超出立即失败
    llm_timeout_seconds: int = 120  # 单次 LLM 调用客户端超时（R28）
    slow_call_seconds: float = 60.0  # 单次调用超过该阈值发 slow_node 事件（R28）
    max_run_tokens: int = 600_000  # 单 run token 预算（输入+输出），0=关闭（R27）
    max_run_cost_cny: float = 10.0  # 单 run 金额预算（按价格表估算），0=关闭（R27）
    price_input_cny_per_m: float = 2.0  # 价格表：输入 ¥/百万 token（R27）
    price_output_cny_per_m: float = 8.0  # 价格表：输出 ¥/百万 token（R27）
    embedding_model: str = ""  # 空 → 记忆检索退化为字符 n-gram 匹配

    # ---- 搜索 ----
    retriever: str = "duckduckgo"  # duckduckgo | so360 | bing | tavily | serper | stub
    retriever_fallbacks: str = ""  # 降级链（逗号分隔），主检索器异常/零结果时依次尝试
    mcp_servers: str = ""  # MCP 数据源（JSON 数组）：[{"name":"x","transport":"stdio","command":"...","args":[...]}, {"transport":"http","url":"..."}]
    search_top_k: int = 5
    tavily_api_key: str = ""
    serper_api_key: str = ""

    # ---- 研究行为 ----
    question_routing: bool = True  # 开头问题路由：闲聊/简单询问直接回答，不进检索流水线
    max_subquestions: int = 5
    max_research_rounds: int = 2
    max_sources_per_question: int = 3
    source_curation: bool = True  # 检索结果先经轻量模型相关性过滤（GPT-Researcher 式 source curation）
    draft_review: bool = True  # 草稿级事实审查：Writer 草稿经 Verifier 审查、按需单轮修订
    report_language: str = "zh"
    fetch_pages: bool = True  # false 时仅用搜索摘要（离线演示/省流量）

    # ---- 服务 ----
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""  # 非空时所有 /api/* 需要 X-API-Key 请求头
    cors_origins: str = ""  # 逗号分隔来源列表；空 = 不启用 CORS
    rate_limit_per_minute: int = 10  # 创建任务限流（按 key），0 = 关闭
    max_concurrent_runs: int = 4  # 后台研究并发上限
    run_timeout_seconds: int = 600  # 单 run 硬超时
    inline_execution: bool = False  # POST 请求内联执行（测试/调试）
    data_dir: Path = Path("./data")

    # ---- 存储保留策略 ----
    checkpointing: bool = True  # LangGraph 断点续跑（AsyncSqliteSaver，data_dir/checkpoints.sqlite3）
    retention_days: int = 30
    max_lessons: int = 500

    @property
    def fast_model(self) -> str:
        return self.llm_fast_model or self.llm_model

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "deepdossier.sqlite3"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)


def load_settings(**overrides: object) -> Settings:
    """加载配置；overrides 用于 CLI/demo 场景强制覆盖（如 stub provider）。"""
    settings = Settings(**overrides)  # type: ignore[arg-type]
    settings.ensure_dirs()
    return settings
