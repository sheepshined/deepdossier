# DeepDossier · 深度调研智能体

> 当前版本 v0.4.0 · 139 项离线测试 · 企业级韧性验收（docs/04）

多 Agent 协同的深度研究系统：输入一个复杂问题，输出一份**逐结论挂引用、经批判循环与事实审查双重检验、矛盾显式呈现**的 Markdown 调研报告。

```
Router(问题路由) → Planner(多视角拆解) → Researcher×N(并行检索+来源过滤) → Critic(批判/补搜循环)
        → Writer(带引用报告) → Verifier(事实审查/修订)
        ↑____________________策略记忆(跨 run 复盘)____________________↓
```

架构与竞品选型依据见 [docs/01-竞品调研与选型.md](docs/01-竞品调研与选型.md) / [docs/02-实现方案.md](docs/02-实现方案.md)；
最终功能目标与验收标准见 [docs/03-最终功能目标与验收标准.md](docs/03-最终功能目标与验收标准.md)；
**系统学习入口见 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)**（概念地图 → 分阶段路线 → 精读清单 → 动手实验）。

## 特性

**研究质量**
- **多 Agent 协同编排**（LangGraph）：`Send` 并行扇出 + 条件循环 + 证据去重归约，编排即图
- **问题智能路由**：闲聊/简单询问由快模型直接回答（秒级、零检索消耗），复杂问题才进完整流水线；路由失败 fail-open
- **对抗性批判循环**：Critic 找证据缺口与来源矛盾 → 定向补搜，报告质量显著高于单轮 RAG
- **草稿级事实审查（Verifier）**：Writer 草稿对照证据审查幻觉引用/无支撑断言/编号错位，按需单轮修订（实测曾单次抓出 8 处引用错位）
- **来源质量过滤**：搜索结果在抓取正文前经轻量模型相关性筛选（GPT-Researcher 式 curation），实测砍掉 25-50% 噪音
- **矛盾显式呈现**：Critic 发现的来源冲突传入 Writer，落进「争议与不确定性」一节，不和稀泥
- **多视角问题分解**：子问题带 perspective 标注，避免同质化（借鉴 STORM）
- **策略记忆**：每次调研沉淀搜索策略经验，注入未来同类问题；中文模糊匹配（n-gram）或语义检索（embedding 余弦，可配）双模式

**数据源**
- **可插拔检索器 ×6**：DuckDuckGo / 360 搜索（国内网络推荐，免 key）/ Bing / Tavily / Serper / Stub（离线），环境变量一键切换
- **检索降级链**：主引擎异常或零结果自动降级（降级过程在事件流可见）
- **MCP 数据源**：标准 `{"mcpServers":{...}}` 配置（stdio/http），领域工具支持多步调用链与 LLM 参数提取；免 key 直连的 360 检索与任意 MCP 数据源均可配置
- **混合并行检索**：带关键词的领域 MCP 按查询词选择性并行补充，web/MCP 结果合并去重
- **检索透明化**：每条结果带引擎指纹（via），时间线展示各引擎贡献与过滤前后对比

**交付形态**
- **报告多格式导出**：Markdown / 独立 HTML（单文件、打印即 PDF）/ Word .docx（上标引用可点击跳转到文末来源书签）
- **报告对比评审**：任意两份 run 报告 LLM 对比（结论一致性/分歧点/来源覆盖差异），结果持久化缓存不重复计费
- **Web 时间线**：零构建单文件前端——实时 Agent 事件（SSE）、历史列表（区分直答/调研）、来源面板、引用悬停预览、失败续跑入口，断线自动重连零丢失

**工程底座**
- **Provider 无关**：DeepSeek / GLM / OpenRouter / Ollama / vLLM，一个 `base_url` 切换；单 run LLM 调用硬预算
- **断点续跑**：LangGraph SqliteSaver 每步落 checkpoint，崩溃/超时/重启后从断点恢复；run 参数持久化、事件 seq 续接不覆盖历史
- **免 key 开箱即用**：默认配置零 API key 跑通全链路（`deepdossier demo`）
- **任务治理**：后台并发上限 + 单 run 硬超时 + 优雅关停 + 重启清理僵尸 run；API 令牌桶限流
- **全程可观测**：17 种结构化事件（含逐次 llm_call 调用）三通道落地——SQLite（恢复+回放）、JSONL trace（自托管）、Console（CLI 实时）
- **存储治理**：SQLite（WAL）runs/events/lessons/embeddings/comparisons，保留策略（按天/按量）自动清理
- **安全**：所有 `/api/*` 端点（含 SSE）可选 X-API-Key 鉴权（常量时间比较），CORS 可配
- **质量保障**：**139 项全离线测试**（不出网、不花 token，含首事件延迟/断线重连等体验级验收测试），CI 含 Docker 镜像内 Web UI 回归验证

## 快速开始

```bash
# 1) 安装
cd deepdossier
pip install -e .

# 2) 离线演示（无任何 API key）
deepdossier demo

# 3) 真实研究：配置 .env（见 .env.example），至少需要 LLM key
cp .env.example .env   # 填 DOSSIER_LLM_API_KEY 等

deepdossier research "多智能体系统在企业的落地路径与风险？" --depth 2 -o report.md

# 4) Web 服务（时间线可视化 / 报告对比 / 导出）
deepdossier serve      # 打开 http://127.0.0.1:8000
```

Docker：

```bash
cp .env.example .env && vi .env
docker compose up --build   # http://localhost:8000
```

## Python API

```python
import asyncio
from deepdossier.config import load_settings
from deepdossier.store.runs import RunStore
from deepdossier.runner import run_research

settings = load_settings()
store = RunStore(settings.db_path)
run_id = store.new_run_id()
store.create_run(run_id, "RAG 与微调如何选型？")

result = asyncio.run(run_research("RAG 与微调如何选型？", settings, store, run_id))
print(result["report_md"])
```

更多用法见 [examples/run_demo.py](examples/run_demo.py)。

## 环境变量（前缀 `DOSSIER_`）

| 变量 | 说明 | 默认 |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI 兼容 LLM | deepseek |
| `LLM_FAST_MODEL` | 轻任务快模型（路由/过滤/提取等，留空同主模型） | - |
| `MAX_LLM_CALLS` | 单 run 调用次数预算 | 150 |
| `MAX_RUN_TOKENS` / `MAX_RUN_COST_CNY` | token / 金额熔断（配价格表 `PRICE_*_CNY_PER_M`），任一触发即停 | 600000 / 10 |
| `LLM_TIMEOUT_SECONDS` / `SLOW_CALL_SECONDS` | 单次调用超时 / 慢调用事件阈值 | 120 / 60 |
| `EMBEDDING_MODEL` | 记忆语义检索用；留空退化为 n-gram | - |
| `QUESTION_ROUTING` | 开头问题路由：闲聊/简单询问直接回答，不进检索流水线 | true |
| `RETRIEVER` | `duckduckgo`（免key）/ `so360`（国内网络推荐，免key）/ `bing` / `tavily` / `serper` / `stub` | duckduckgo |
| `RETRIEVER_FALLBACKS` | 降级链（逗号分隔）：主检索器异常/零结果时依次尝试 | 空 |
| `MCP_SERVERS` | MCP 数据源（标准 `{"mcpServers":{...}}` 或数组，stdio/http），领域工具支持 call_chain + $llm 提参 | 空 |
| `MAX_SUBQUESTIONS` / `MAX_RESEARCH_ROUNDS` | 拆解数 / 补搜轮数上限 | 5 / 2 |
| `SOURCE_CURATION` | 来源质量过滤：检索结果先经轻量模型相关性筛选 | true |
| `DRAFT_REVIEW` | 草稿级事实审查：报告草稿经 Verifier 审查、按需修订 | true |
| `REPORT_LANGUAGE` | 报告语言 `zh` / `en` | zh |
| `CHECKPOINTING` | LangGraph 断点续跑（失败 run 可从 checkpoint 恢复） | true |
| `API_KEY` | 所有 `/api/*`（含 SSE）需要 `X-API-Key` | 空 |
| `CORS_ORIGINS` | 允许跨域来源，逗号分隔；空=不启用 | 空 |
| `RATE_LIMIT_PER_MINUTE` | 创建任务限流（按 key/IP），0=关 | 10 |
| `MAX_CONCURRENT_RUNS` / `RUN_TIMEOUT_SECONDS` | 后台并发 / 单 run 超时 | 4 / 600 |
| `RETENTION_DAYS` / `MAX_LESSONS` | 存储保留策略（启动时执行） | 30 / 500 |
| `DATA_DIR` | SQLite 与 trace 存放目录 | ./data |

## 项目结构

```
src/deepdossier/
├── agents/          # router（问题路由）/ planner / researcher（含来源过滤）/ critic / writer / verifier（事实审查）+ prompts + 契约
├── graph/           # LangGraph 状态（证据去重归约）、依赖容器、编排组装（可挂 checkpointer）
├── llm/             # Provider（预算/修复重试/Stub/事件代理）+ embedding + 向量缓存
├── tools/           # Retriever 插件（so360/bing/ddg/tavily/serper/stub）+ 降级链 + 混合并行 + MCP 数据源 + 网页抓取
├── store/           # SQLite：runs/events/lessons/embeddings/comparisons + 后台事件写线程
├── api/             # FastAPI(lifespan) + SSE 通知中心 + TaskManager + 限流 + 对比评审
├── web/             # 单文件前端：实时时间线/历史/来源面板/对比/导出/续跑（随包分发）
├── report_export.py # 独立 HTML 报告导出（打印即 PDF）
├── report_docx.py   # Word (.docx) 报告导出（引用书签跳转）
├── runner.py        # 端到端运行器（CLI/API 共用；断点续跑；llm_call 事件代理）
├── trace.py         # 事件体系与 JSONL 追踪
└── cli.py           # research / serve / demo
tests/              # 139 项全离线 pytest（Stub 双件套 + MockTransport 断网验证，含体验级验收测试）
```

## 路线图

- [x] LangGraph SqliteSaver 图级 checkpoint（节点级断点续跑）
- [x] 报告导出 HTML / Word（PDF 经 HTML 打印）
- [x] MCP 数据源接入（对齐 GPT-Researcher 的 hybrid research）
- [x] 草稿级事实审查（Verifier）与报告对比评审
- [ ] 证据向量索引与 cross-encoder 重排
- [ ] 多租户与任务队列（Celery/Redis）

## License

MIT
