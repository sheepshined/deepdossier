# DeepDossier 学习手册

> 本手册回答一个问题：**怎么学这个项目**。面向要"吃透" v0.4.0（4137 行源码 / 139 项离线测试）的读者：给出分阶段路线、每阶段的通关检验与可回滚的动手实验。
> 总预算 **4~5 天**。一条原则：**先跑通、再主线、后支线；每阶段动手做一次实验，不做"只读不练"的学习。**

## 阅读路径

| 顺序 | 材料 | 用途 |
|---|---|---|
| 1 | 本手册 §0-§1 | 建立概念地图与学习顺序 |
| 2 | 本手册 §2-§4 | 主线：分阶段路线 → 精读清单 → 动手实验 |
| 3 | README.md | 产品能力与运行方式（数字已校准） |
| 4 | docs/05 降级对照表 | 学韧性设计时的速查 |
| 5 | CHANGELOG.md + docs/01~04 | 演进脉络与背景（§7 有分级导读） |

## 0. 定位与技术栈

**定位**：应用型多 Agent 深度调研系统——输入复杂问题，经「路由 → 拆解 → 并行检索 → 批判循环 → 带引用写作 → 事实审查」产出可溯源报告，全程可观测、可断点续跑、免 key 可离线演示。**不做框架**（docs/03 §3 明确边界）。

| 层 | 选型 | 落点 |
|---|---|---|
| 编排 | LangGraph：StateGraph + Send 扇出 + AsyncSqliteSaver 断点 | `src/deepdossier/graph/builder.py:63` |
| Agent | router / planner / researcher / critic / writer / verifier 六节点 | `src/deepdossier/agents/` |
| LLM | OpenAI 兼容（DeepSeek/GLM/Ollama…）＋ 三重预算熔断 ＋ JSON 修复重试 | `src/deepdossier/llm/provider.py:85` |
| 检索 | 6 种 web 检索器 ＋ 降级链 ＋ hybrid MCP ＋ 网页抓取 | `src/deepdossier/tools/{search,mcp_source,fetch}.py` |
| 存储 | SQLite（WAL）：runs / events / lessons / embeddings / comparisons | `src/deepdossier/store/runs.py:25` |
| 服务 | FastAPI（lifespan）＋ SSE 双通道 ＋ TaskManager 治理 | `src/deepdossier/api/app.py:61` |
| 前端 | 零构建单文件（原生 JS，EventSource 重连，989 行） | `src/deepdossier/web/index.html` |
| 导出 | 独立 HTML（打印即 PDF）／ DOCX（python-docx 书签跳转） | `report_export.py:141` / `report_docx.py:103` |
| 测试 | pytest 139 项全离线（Stub 双件套 + MockTransport 断网断言） | `tests/` |

**目录职责**（`src/deepdossier/`）：`agents` 各节点实现＋提示词＋数据契约；`graph` 状态/依赖容器/图组装；`llm` Provider 与 embedding；`tools` 检索与抓取；`store` SQLite 与事件写线程；`api` 服务层；`web` 前端；根级 `runner.py`（run 生命周期）、`trace.py`（事件体系）、`config.py`、`cli.py`、`report_export.py`、`report_docx.py`。

---

## 1. 前置知识地图（概念 → 项目落点）

### 后端

| 概念 | 一句话解释 | 本项目落点 |
|---|---|---|
| StateGraph | 节点＋边组成的状态机，compile 后可 invoke | `graph/builder.py:63` |
| 条件边 | 由函数根据状态决定下一跳（返回节点名或 Send 列表） | `builder.py:74`、`builder.py:77` |
| Send API（map-reduce） | 动态扇出到同一节点，各分支结果归约回主状态 | `builder.py:31-60` |
| Reducer 归约器 | 并发写入同一状态时的合并函数 | `graph/state.py:22` |
| TypedDict 状态 | 图的共享状态；**键必须声明否则被静默丢弃** | `state.py:39`（陷阱注释 `:42`） |
| Checkpointer | 每超级步持久化状态 → 断点续跑 | `runner.py:47`、`runner.py:240-243` |
| SSE | 服务端事件流（text/event-stream ＋ after_seq 续传） | `api/app.py:311` |
| 结构化输出 | prompt 注入 schema → JSON 提取 → pydantic 校验 | `llm/provider.py:42-60`、`:139-143` |
| 修复重试环 | 校验失败把错误信息回灌让模型重答 | `provider.py:163-174` |
| fail-open / fail-closed | 失败时"放行"还是"拒绝"的安全默认选择 | 放行：`router.py:34`；拦截：`verifier.py:121` |
| 令牌桶限流 | 按速率补充令牌的限流算法 | `api/ratelimit.py:17` |
| WAL | SQLite 写前日志，读写并发不互斥 | `store/runs.py:104` |
| 生产者-消费者 | emit 只入队，后台线程批量落库 | `store/writer.py:27-89` |
| 事件契约 | 17 种事件类型构成的前端/回放契约 | `trace.py:23`＋全仓 emit 点 |
| 引用覆盖率 | 长正文行中带 `[n]` 的比例（>40 字非列表行） | `agents/writer.py:89` |
| 部分成果 | 失败/超时时从 checkpoint 抢救证据落库 | `runner.py:166` |
| MCP | 模型上下文协议：stdio/http 接外部工具为数据源 | `tools/mcp_source.py:85` |
| Embedding 余弦 | 语义相似度检索（可选，缺省退化为 bigram） | `store/runs.py:86`、`:276` |
| 幂等迁移 | ALTER 已存在列时吞掉 OperationalError | `store/runs.py:69-73`、`:107-111` |
| 结构契约校验 | 章节/编号越界/断号的程序化检查 | `agents/verifier.py:28` |

### 前端

| 概念 | 一句话解释 | 本项目落点 |
|---|---|---|
| EventSource | 浏览器原生 SSE 客户端，支持重连 | `web/index.html:557` |
| 断线续传 | 重连时带 `after_seq` 补齐丢失事件 | `index.html:556-591` |
| watchdog | 10 秒无事件强制重连的守护定时器 | `index.html:546-554` |
| seq 去重 | seenSeqs 集合防重放渲染重复 | `index.html:569-573` |
| 事件委托 | data-action 属性统一分发，无内联 onclick | `index.html:263-334` |
| mdRender | 手写 markdown→HTML（与导出器同约定、双实现） | `index.html:877`、`report_export.py:80-123` |

---

## 2. 分阶段学习路线

### 阶段 0（0.5 天）：跑起来并点通全流程

**做什么**：`pip install -e .` → `deepdossier demo`（零 key 离线）→ `deepdossier serve` 打开 http://127.0.0.1:8000 → 用页面示例问题跑一次 → 观察时间线、导出 HTML、看历史列表。
**主线问题**：一次研究从提问到出报告，中间经过哪些节点、各自做什么？
**通关检验**：
1. demo 输出里 router→planner→researcher×2→critic→补搜→writer→verifier 各做了什么？（线索：`trace.py:85` console_sink）
2. 时间线上 `llm_call` 与 `tool_call` 事件的 payload 各含什么字段？（`:108-116` 与 `agents/researcher.py:93-103`）
3. 报告头部若出现"未联网核实"说明发生了什么？（`agents/writer.py:134-137`）

### 阶段 1（0.5 天）：状态与编排骨架

**读什么**：`graph/state.py`、`graph/builder.py`、`graph/deps.py` 全文（共 189 行）。
**主线问题**：状态如何被并发安全地写入与合并？图有几条路径、几种边？
**通关检验**：
1. 为什么 `evidence` 用归约器而 `round` 不用？（`state.py:22-36` vs `:44-47`）
2. 什么输入会让 router 走到 `direct_answer`？失败时走哪条？（`builder.py:84`、`router.py:33-36`）
3. 新增状态键最容易踩的坑是什么？（`state.py:42` 注释；docs/02 §10 记录了这次教训）

### 阶段 2（1 天）：检索链与研究员

**读什么**：`tools/search.py`（重点 `:226-349`）、`tools/mcp_source.py`（`:85-186`）、`tools/fetch.py`、`agents/researcher.py` 全文。
**主线问题**：一次子问题的"搜索→过滤→抓取→提炼"如何做到既健壮又省钱？
**通关检验**：
1. FallbackRetriever 在"全链干净零结果"与"全链异常"时行为有何不同？（`search.py:238-253`）
2. HybridRetriever 什么情况下完全不调用 MCP？（`search.py:272-276`）
3. 搜索零结果时为什么可以不调 LLM？省了什么？（`researcher.py:114-121`）
4. `snippet_only` 何时置位、谁消费它？（`researcher.py:168-173` → `writer.py:29-45`）

### 阶段 3（1 天）：批判循环与成稿工程

**读什么**：`agents/critic.py`、`agents/writer.py`、`agents/prompts.py`（重点 WRITER/VERIFIER/REVISE 三段）。
**主线问题**：补搜循环如何终止不失控？引用的编号一致性如何工程化保证？
**通关检验**：
1. 补搜循环的终止条件有哪三个？（`critic.py:48-57`）为什么决策收口在节点内而不是路由函数？
2. `merge_citations` 如何保证正文编号与来源列表严格一致？（`writer.py:48-69`）
3. `citation_coverage` 的统计口径——哪些行算"疑似事实段落"？（`writer.py:89-106`）
4. 为什么修订后必须再次 merge_citations？（`verifier.py:154`）

### 阶段 4（0.5 天）：质量门禁与韧性

**读什么**：`agents/verifier.py`、`runner.py:47-300`、`store/writer.py`。
**主线问题**：T1"永不空手而归"/T2"结论必须可信"在代码里落在哪几行？
**通关检验**：
1. Verifier 四道防线分别拦什么？哪几道不依赖 LLM？（`verifier.py:1-8`、`:28`、`:58`、`:90`、`:149-163`）
2. `draft_review=false` 时哪些检查仍然生效？（`verifier.py:93-96`）
3. run 失败时用户为什么还能拿到东西？抢救数据从哪来？（`runner.py:166-206`、`:290-297`）
4. 事件写失败为什么既不丢事件也不拖垮 run？（`store/writer.py:46-62`）

### 阶段 5（0.5 天）：服务层与前端

**读什么**：`api/app.py`、`api/tasks.py`、`api/hub.py`、`web/index.html`（重点 `:531-660`、`:857-925`）。
**主线问题**：SSE 如何做到"实时又可靠"？后台任务如何被治理？
**通关检验**：
1. SSE"即时唤醒＋回放"双通道各自解决什么问题？（`app.py:311-338`、`hub.py:15-47`）
2. 为什么 resume 只允许 failed 的 run？（`app.py:180-188`）
3. 前端如何保证断线零丢失？（`index.html:556-591`、`:569-573`）
4. 排队状态从哪来、在哪展示？（`app.py:164-168`、`tasks.py:43-49`）

### 阶段 6（0.5 天）：测试与工程视角

**做什么**：`python -m pytest -q`（139 项）→ 精读三个代表性测试 → 跑 `scripts/check_release.py`。
**精读**：`tests/test_resilience.py:463`（混沌 100 run ≥99% 终态）、`tests/test_checkpoint.py:41`（断点续跑往返）、`tests/test_experience.py:32`（首事件 <3s）。
**通关检验**：
1. 混沌测试的注入点为什么避开 planner/writer？（`test_resilience.py:456-460`）
2. 断点续跑测试为什么对 planner 注错而不是 researcher？（`test_checkpoint.py:14-19`）
3. 测试如何保证完全不出网？（`tests/conftest.py:64-70` 的 offline_http 夹具）

---

## 3. 精读代码清单

| # | 位置 | 学什么 / 为什么必读 |
|---|---|---|
| 1 | `graph/state.py:14-49` | 证据归约器：claim+URL 归一化去重 ＋ 400 条内存护栏（R30）；`route` 未声明即被丢弃的 LangGraph 陷阱 |
| 2 | `graph/builder.py:63-85` | 七节点图全貌：Send 扇出 vs 字符串路由两种条件边；direct 分支与 research 主线的分流 |
| 3 | `graph/deps.py:24-55` | 依赖容器与资源释放顺序（先关检索器/事件线程再关 http）；per-run 双缓存的位置 |
| 4 | `agents/planner.py:17-44` | 多视角拆解＋策略记忆注入；embedding 向量与 lessons 检索的衔接 |
| 5 | `agents/researcher.py:38-72` | curate_sources：fail-open ＋ 全判无关留底 1 条，防"审核环节饿死研究" |
| 6 | `agents/researcher.py:75-182` | 单查询执行器全链：缓存→过滤→抓取→提炼；零结果短路与 snippet_only 标注 |
| 7 | `agents/critic.py:13-71` | 决策收口模式：can_continue 三条件；`[降级]` reasoning 的消费方（writer.py:41） |
| 8 | `agents/prompts.py:89-143` | 写作/审查/修订三段提示词：结构要求、"宁可漏报不可误报"的审查立场 |
| 9 | `agents/writer.py:48-106` | merge_citations 引用去重重编号＋citation_coverage 覆盖率口径（质量度量的地基） |
| 10 | `agents/verifier.py:28-173` | 四道防线＋回退护栏：程序化契约、幻觉检测、覆盖率门禁、缩水/下降则放弃修订 |
| 11 | `llm/provider.py:85-196` | 三重预算熔断（次数/token/金额）、修复重试环、快模型兜底（R1） |
| 12 | `runner.py:209-300` | run 生命周期：resume 守卫、事件先于终态的顺序保证、部分成果抢救、CancelledError 分支 |
| 13 | `runner.py:47-67` | checkpointer 生命周期：aiosqlite 与事件循环绑定，为什么每次 run 独立开关 |
| 14 | `store/runs.py:25-73`、`:328-373` | 五张表 schema＋幂等迁移；对比结果缓存；保留策略（prune 不动 running） |
| 15 | `store/writer.py:27-96` | 事件写线程：批量落库、失败内存缓冲重试（R16）、close 排空语义 |
| 16 | `tools/search.py:226-349` | FallbackRetriever/HybridRetriever 的组合语义；via 引擎指纹如何生成 |
| 17 | `api/app.py:135-204`、`:311-338` | 创建/续跑路径与 SSE 双通道实现；鉴权与限流的挂载点 |
| 18 | `web/index.html:531-660`、`:857-925` | SSE 重连/watchdog/seq 去重；resume 入口与 mdRender 约定 |

其余文件按 §2 阶段路线顺带读，不必逐行。

---

## 4. 动手实验清单

> **守则**：实验只改工作区（`.env` 或源码常量），不提交；每做完一个回到对应阶段解释"为什么出现这个现象"。
> **本仓库不是 git 仓库，回滚方式 = 把下表的"原值"改回去**（或删掉临时加的环境变量重新启动服务）。

| 实验 | 改动（文件:行号 / 环境变量） | 预期现象与理解点 | 原值（回滚用） |
|---|---|---|---|
| L1 关闭问题路由 | `DOSSIER_QUESTION_ROUTING=false` 后重启 serve，问"你是谁" | 不再秒回直答，而是走完整流水线；`route` 事件 reason 变为"路由已关闭"（router.py:21-23）。理解：路由是 START 后第一跳 | `true` |
| L2 关闭来源过滤 | `DOSSIER_SOURCE_CURATION=false` | 时间线上 `curate` 事件消失，抓取的来源条数变多。理解：curation 在抓取前拦截噪音（researcher.py:43） | `true` |
| L3 关闭草稿审查 | `DOSSIER_DRAFT_REVIEW=false` | 无 `draft_review` 事件；但覆盖率不足的"引用不充分"标注**仍然出现**——程序化部分不依赖 LLM（verifier.py:93-96） | `true` |
| L4 破坏性：覆盖率门禁拉满 | `agents/verifier.py:20` 的 `COVERAGE_GATE = 0.80` 改为 `1.01` | 任何报告都触发强制修订＋顶部"引用不充分"；跑 `python -m pytest tests/test_verifier.py -q` 观察哪些测试变红。理解：门禁是硬拦截不是展示 | `0.80` |
| L5 证据上限砍到 2 | `graph/state.py:19` 的 `_MAX_EVIDENCE_CARDS = 400` 改为 `2` | `deepdossier demo` 报告证据骤减；`test_resilience.py::test_r30_evidence_cap` 变红。理解：内存护栏的取舍（保先到者） | `400` |
| L6 预算熔断体验 | `DOSSIER_MAX_LLM_CALLS=3` 后跑 demo（CLI 会报错退出）；更好的方式：跑 `python -m pytest tests/test_resilience.py::test_r3_budget_exhaustion_keeps_partial_results -q` | run 失败但落"部分成果"报告（已收集证据＋计划）。理解：T1 永不空手而归（runner.py:166） | `150` |
| L7 慢调用事件 | `DOSSIER_SLOW_CALL_SECONDS=0.001` 后跑 demo | 每个 `llm_call` 后紧跟 `slow_node` 事件（schema/耗时）。理解：观测层不改流程只上报（runner.py:117-121） | `60` |
| L8 降级链语义（读代码验证） | 无改动：`python -m pytest tests/test_search_fetch.py -k fallback -q`，再读 `search.py:238-253` | 体会"干净零结果返回 []"与"全链异常抛异常"两种终态的区别，以及调用方如何各自处理 | — |
| L9 断点守卫 | `DOSSIER_CHECKPOINTING=false` 后重启 serve，对任意 failed run 调 `POST /api/researches/{id}/resume` | 返回 409 "checkpointing is disabled"（app.py:186-187）。理解：resume 的守卫是显式拒绝而非静默丢状态 | `true` |

---

## 5. 常见疑问预答（FAQ）

**Q1 为什么用 LangGraph 而不是纯 asyncio 编排？**
Send/条件边/checkpointer 恰好直接映射"并行＋循环＋恢复"三个需求；代价是重依赖，所以编排集中在 `builder.py:63` 单文件里，爆炸半径可控。

**Q2 为什么 `evidence` 用归约器而 `round` 不用？**
并行扇出会并发写 evidence（必须合并），round 只在 critic 单点写（后写覆盖即可）。把并发语义显式化在类型上，是 `state.py:22-47` 的设计。

**Q3 为什么 Critic 决策收口节点内、路由保持纯函数？**
v0.2 的真实 bug：先递增 round 再路由，导致 `round >= max` 恒真、补搜永不触发。修法＝决策写进 `pending_followups`，路由只负责扇出（`critic.py:48-57` / `builder.py:52-60`）。

**Q4 为什么 `route` 键必须声明在 TypedDict 里？**
LangGraph 会**静默丢弃**未声明的状态键，direct 路由会因此失效。`state.py:42` 的注释就是 v0.3 踩坑后留下的警示。

**Q5 为什么检索/抓取缓存是 per-run dict，不做全局缓存？**
补搜轮增量执行＋并行研究员去重是本 run 内的需求；跨 run 复用会串数据、失效策略复杂化。`deps.py:38-40`。

**Q6 为什么不用 JSON mode / response_format，而是 prompt 注入 schema？**
要兼容 DeepSeek/GLM/Ollama 等全部 OpenAI 兼容端。代价是偶发非法 JSON，用修复环＋快模型兜底补偿（`provider.py:139-143`、`:163-195`）。

**Q7 为什么 Critic 用快模型、Verifier 用主模型？**
Critic 每轮必调、判断粗粒度（成本敏感）；Verifier 一次定稿、判断细粒度（质量敏感）。`critic.py:33` vs `verifier.py:105-106`。

**Q8 为什么覆盖率门禁是 80%，且修订后下降要回退？**
覆盖率是启发式指标，模型可能"为达标而砍内容"；回退护栏保证只接受真实改善（`verifier.py:20`、`:149-163`）。

**Q9 为什么 checkpointer 每次 run 独立开关连接？**
aiosqlite 连接与事件循环绑定，跨 loop 复用会死锁（`runner.py:43-47` 注释）；本地 sqlite 开关开销可忽略。

**Q10 为什么"部分成果"从 checkpoint 抢救而不是内存状态？**
失败可能发生在任意 await 点，内存状态不可靠；checkpoint 每超级步已落盘，是唯一可信来源（`runner.py:166-206`）。

**Q11 为什么事件写失败要有内存重试缓冲？**
观测层故障既不拖垮主流程、也不静默丢事件（R16）；缓冲有 2000 条上限防内存膨胀（`store/writer.py:46-62`）。

**Q12 为什么前端 mdRender 与 report_export.py 的渲染器是两份实现？**
一份跑浏览器、一份生成静态文件，无法共享代码；代价是改动需两处同步（`report_export.py:4` 注释明示）。

**Q13 为什么 direct 回答不写策略记忆？**
记忆是"搜索策略"经验，直答没有检索过程，写进去是噪音（`runner.py:249` 的 `if report_md and sources`）。

---

## 6. 观测与调试手段速查

| 想看什么 | 手段 |
|---|---|
| 一次研究的实时事件流 | Web 时间线（serve 后 http://127.0.0.1:8000）；或 `curl -N "…/api/researches/<id>/events?after_seq=0"` |
| 事件历史（离线复盘） | `sqlite3 data/deepdossier.sqlite3 "SELECT seq,type,node,payload_json FROM events WHERE run_id='<id>' ORDER BY seq"` |
| 逐次 LLM 调用明细 | 时间线"LLM 明细"开关；或 events 表 `llm_call` 的 schema/tokens/耗时字段 |
| 全链路 trace | `data/traces/<run_id>.jsonl`（每行一个事件） |
| 断点状态 | `data/checkpoints.sqlite3`；resume 后核对 `max_event_seq` 是否续接 |
| 服务日志 | `deepdossier serve -v`；logger 命名空间 `deepdossier.{runner,llm,events,api,mcp,tasks}` |
| run 用量/来源/参数 | `GET /api/researches/<id>`（usage / sources / params） |
| 测试 | `python -m pytest -q`；单文件如 `python -m pytest tests/test_resilience.py -q` |
| 发布自检 | `python scripts/check_release.py`；事实基准 `benchmarks/fact_benchmark.json` |

**排查口诀**：
- "报告没生成/run 失败" → 先看 run 的 status/error，再看 events 尾部 `error` 事件；失败通常也有"部分成果"（`runner.py:166`）。
- "搜不到东西" → 看 `tool_call` 的 `engines` 字段与 `fallback` 事件，确认实际贡献引擎（`researcher.py:104-111`）。
- "时间线卡住" → POST 响应里 `queued`/`active`/`waiting`；刷新页面靠 after_seq 续传（`app.py:311`）。
- "真实调研 401" → 检查 `.env` 位置（`config.py:10` 先 CWD 后项目根）与 `DOSSIER_LLM_API_KEY`。
- "引用编号乱/缺章节" → 看 `quality` 与 `draft_review` 事件，确认 verifier 是否执行了修订（`verifier.py:112-171`）。

---

## 7. 其他文档导读

| 文档 | 定位 | 建议 |
|---|---|---|
| README.md | 可读（产品向；数字已于 2026-09-26 校准：139 项测试 / 17 种事件） | 入口第一读 |
| docs/01-竞品调研与选型.md | 可读（选型历史，星数为 2026-09 快照） | 想理解"为什么做这个产品"时读 |
| docs/02-实现方案.md | 部分过时：§1-4 仍反映现状；§5 里程碑表已过时（M5 标 roadmap 但实际全完成）；§10 是 v0.3 增量记录 | 架构速览 |
| docs/03-最终功能目标与验收标准.md | 历史快照（2026-09-13 实测 101 项；本轮已在 §5.1 后补 139 说明） | 理解 F1-F25 需求来源 |
| docs/04-Codex级实现要求.md | 历史要求文档（R1-R40 已全部落地，见 CHANGELOG 0.4.0） | 学韧性设计时的需求索引 |
| docs/05-降级对照表.md | **长期有效**：21 种降级在事件/报告/界面三处的呈现 | 调试降级行为时必查 |
| CHANGELOG.md | 推荐先读 | 5 分钟看懂 v0.2→v0.3→v0.4 演进 |

> 当前唯一反映项目现状的文档：**README.md ＋ 本手册 ＋ 代码本身**。

---

## 8. 核实与修正记录（2026-09-26 重学核实）

| 位置 | 原文 | 改为 | 依据 |
|---|---|---|---|
| README.md:3 / :48 / :137 | "101+ 项 / 101 项离线测试" | "139 项" | `pytest -q` 实测 `139 passed`（139 = docs/04 基线 103 ＋ v0.4 韧性套件 36） |
| README.md:45 | "13 种结构化事件" | "17 种" | 全仓 emit 类型字符串去重：node_start/node_end/llm_call/tool_call/evidence_added/critique/final/error/done/route/curate/fallback/quality/degraded/hallucination/draft_review/slow_node |
| README.md:31 | "免 key 直连的 360 + 高德官方 MCP 开箱可配" | "免 key 直连的 360 检索与任意 MCP 数据源均可配置" | 高德 MCP 实际存在于用户本地 `.env`（amap-weather/amap-poi 已真实验证），但仓库分发的 `.env.example` 无任何示例，对全新克隆者不可复现 → 改为通用能力描述 |
| docs/03 §2.2 | "当前基线 90 项" | "当前基线 139 项" | 同上实测 |
| docs/03 §5.1 | "101 passed"（09-13 快照） | 保留原文，表后追加 139 更新说明 | 历史快照不改写，注明演进 |

**未能验证、保留待核的断言**（仓库内无出处）：
- README.md:22 "实测曾单次抓出 8 处引用错位"——无法判定，保留待核。
- README.md:23 curation "实测砍掉 25-50% 噪音"——仓库内无文档出处；开发期 A/B 记录（来源 18→9 全唯一、prompt 开销约 +6%）在项目工作笔记中有案，可视为有依据。
