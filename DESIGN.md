# memory-hub 设计描述文档（本地 + 云端）

> 本文档是 memory-hub 的完整设计描述，整合了**本地优先的 MCP 长期记忆中枢**与可选的**云端进化层**扩展。面向 AI 阅读，目标是在不直接读源码的情况下，让另一个 AI 完整理解系统的设计理念、模块边界、数据流、安全模型与扩展点。后续可基于本文档进行创意发散、文章撰写或方案改造。
> 
> 文档范围：
> - **本地层（L0）**：memory-hub v0.1.0 已实现的核心系统，包括 MemoryStore、Provider 系统、Evolution 循环。
> - **云端层（L1/L2）**：基于本地层扩展的群体级记忆进化与分发设计（v0.1 草案），默认关闭，opt-in 启用。

---

## 1. 项目定位

**memory-hub** 是从 Hermes Agent 抽取的**共享长期记忆服务**。它让多个 AI agent（Kimi Code、Claude Code、Hermes 等）通过 **MCP（Model Context Protocol）** 读写同一份本地长期记忆，并内置一套“进化循环”让记忆自动自省、提炼策略、治理冗余。

在本地层之上，memory-hub 还设计了可选的**云端进化层**：在用户知情、许可的前提下，将本地记忆中“可分享的部分”上行到云端，经群体级进化（聚合、蒸馏、淘汰）形成团队共识，再以下发（push）或按需拉取（pull）的方式回流到各成员的本地记忆系统。

一句话概括：

> 一个文件优先、本地优先、MCP 协议封装、可插拔外部记忆后端、带有人工审批式进化循环的长期记忆中枢；可选扩展到群体级云端共识与分发。

---

## 2. 核心设计目标

| 目标 | 说明 |
|------|------|
| **多 Agent 共享** | 同一份 `~/.memory-hub/` 数据目录，多个 agent 通过 MCP stdio/HTTP 接入。 |
| **文件优先** | 主记忆是 markdown 文件（`MEMORY.md` / `USER.md`），人类可直接阅读、编辑、版本控制。 |
| **本地优先 / 零云依赖** | 默认无需网络；可选 `holographic` provider 用 SQLite + HRR 向量检索。 |
| **前缀缓存友好** | 系统提示中的记忆块在会话开始时冻结为快照，会话中写盘但不改系统提示。 |
| **安全第一** | 注入/渗出扫描、漂移检测、写入门、原子写、审批制进化，默认全部手动。 |
| **可进化** | 从会话转录中自动提取长期事实、发现重复模式、合并近重复、归档过期条目。 |
| **可插拔后端** | ProviderManager 支持一个外部 provider；内置 holographic 作为示例与生产选项。 |
| **可选云端共识** | 在严格隐私、安全与审批控制下，将公共经验聚合成团队共识并回流本地。 |

---

## 3. 系统架构总览

### 3.1 本地层架构（L0）

```
┌─────────────────────────────────────────────────────────────┐
│                       MCP Clients                            │
│     Kimi Code (stdio)   Claude Code (stdio)   Hermes (HTTP)  │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │      memory_hub.server.mcp_server      │
        │   FastMCP: stdio / streamable-http     │
        └───────────┬───────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────────┐   ┌───────────────┐   ┌──────────────────┐
│ MemoryStore │   │   Evolution   │   │ ProviderManager  │
│  (M1 文件)  │   │     Loop      │   │ (≤1 ext provider)│
│             │   │review/strategy│   │   holographic    │
│             │   │   /curator    │   │                  │
└──────┬──────┘   └───────┬───────┘   └─────────┬────────┘
     │                │                    │
     ▼                ▼                    ▼
MEMORY.md      review_queue.json      hub.db (SQLite)
USER.md        strategies/*.md        FTS5 + HRR
archive/*.md   transcripts/*.jsonl    fact_store tools
```

三个子系统：

1. **M1 — MemoryStore（文件记忆内核）**：`core/store.py`，两个 markdown 文件，§ 分隔，字符预算。
2. **M2 — Provider 系统**：`providers/`，可插拔外部记忆后端。
3. **M3 — Evolution 循环**：`evolution/`，转录驱动的记忆自省与治理。

### 3.2 本地 + 云端完整架构（L0 + L1 + L2）

```
┌─────────────────────────────────────────────────────────────────┐
│  L0 本地层（每个用户机器）                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐   │
│  │MemoryStore│  │Evolution │  │Provider  │  │ 云端扩展: Uplink│   │
│  │ M1 文件   │  │循环 M3    │  │Manager   │  │ Gate + 共识命名 │   │
│  │           │  │           │  │ M2       │  │ 空间 TEAM.md    │   │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘   │
│        个人记忆 MEMORY.md/USER.md —— 默认永不上行                  │
└──────────────┬───────────────────────────────▲──────────────────┘
               │ 上行: 白名单结构化条目            │ 下行: 签名 delta
               │ (渗出扫描+审批后)                │ (扫描+审批后)
┌──────────────▼───────────────────────────────┴──────────────────┐
│  L1 云端聚合层（可选服务）                                         │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐   │
│  │隔离候选区  │→ │ k-匿名聚合  │→ │ 共识库     │→ │ 自然选择    │   │
│  │Quarantine │  │ +蒸馏管线   │  │(分层分频道)│  │ TTL/反馈淘汰│   │
│  └───────────┘  └────────────┘  └───────────┘  └────────────┘   │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐                    │
│  │信誉/身份层 │  │ Canary 监控│  │ 失败蒸馏   │                    │
│  │(Sybil 防御)│  │(渗透检测)   │  │ 独立管道   │                    │
│  └───────────┘  └────────────┘  └───────────┘                    │
└──────────────────────────────┬───────────────────────────────────┘
                               │ 频道化发布（签名 + 版本化）
┌──────────────────────────────▼───────────────────────────────────┐
│  L2 分发层                                                        │
│  频道订阅（topic channels）/ delta 增量包 / 按需 pull API           │
└───────────────────────────────────────────────────────────────────┘
```

云端层是本地 Evolution 循环的群体扩展：

- **上行**：本地记忆 →（白名单过滤 + 渗出扫描 + 用户审批）→ 云端隔离候选区 →（k-匿名聚合 + 蒸馏）→ 团队共识库。
- **下行**：团队共识库 →（频道订阅 + 增量 delta + 签名版本）→ 本地独立命名空间 `TEAM.md` →（严格扫描 + 用户审批）→ 进入 agent 上下文。

---

## 4. 数据目录约定

默认根目录：`~/.memory-hub/`（可被 `MEMORY_HUB_HOME` 环境变量覆盖）。

```
~/.memory-hub/
├── config.yaml                 # 配置：预算、provider、evolution LLM、cloud
├── memories/
│   ├── MEMORY.md               # agent 笔记（环境事实、项目约定、坑点）
│   ├── USER.md                 # 用户画像（偏好、风格、习惯）
│   ├── TEAM.md                 # 云端下行共识（独立命名空间，可选扩展）
│   ├── MEMORY.md.lock          # 文件锁（独立 .lock 文件）
│   ├── USER.md.lock
│   ├── TEAM.md.lock
│   └── *.bak.<ts>              # 漂移检测快照
├── transcripts/
│   └── YYYYMMDD.jsonl          # 会话转录：turn / session_end 记录
├── strategies/
│   └── <slug>.md               # 提取的策略文档（YAML frontmatter）
├── archive/
│   └── MEMORY.md / USER.md     # curator 归档（可恢复，永不物理删除）
├── review_queue.json           # 待审提案队列（写入门 + 进化提案共用）
├── hub.db                      # holographic provider 的 SQLite 事实库
├── hub.db-wal / hub.db-shm     # SQLite WAL 文件
├── .review_state.json          # 自省水位（防重复评审）
├── .strategy_state.json        # 策略提取水位
├── .consensus_state.json       # 云端共识同步水位（可选扩展）
└── context_ledger/             # 上下文账本：记录进过上下文的 consensus_id
    └── YYYYMMDD.jsonl
```

### 4.1 主记忆文件格式

`MEMORY.md`、`USER.md` 与 `TEAM.md` 以段落符号 `§`（section sign, U+00A7）分隔条目：

```markdown
User prefers concise replies in Chinese.
§
Project uses Python 3.11 + pytest.
§
Always write UTF-8 explicitly on Windows.
```

- 条目可多行。
- 字符预算（默认）：`MEMORY.md` 2200 字符，`USER.md` 1375 字符，`TEAM.md` 1500 字符（云端共识）。
- 用字符而非 token，保证模型无关性。
- 加载时去重（保留首次出现）。

---

## 5. 核心模块详解

### 5.1 `paths.py` / `config.py`

- `get_memory_hub_home()`：解析 `MEMORY_HUB_HOME` 环境变量，否则 `~/.memory-hub`。每次调用动态解析，方便测试切换。
- `cfg_get(dotted_key, default)`：读取 `config.yaml` 的点路径值，如 `memory.memory_char_limit`。加载失败时返回空字典，永不抛错。
- `DEFAULT_CONFIG_TEMPLATE`：`init` 命令写入的默认配置模板。

### 5.2 `core/store.py` — MemoryStore

这是项目最核心的模块，语义移植自 Hermes 的 `tools/memory_tool.py`。

#### 双态模型

```
_system_prompt_snapshot   # 会话开始时冻结，用于注入系统提示
memory_entries / user_entries / team_entries   # 实时态，工具调用时读写
```

- `load_from_disk()`：启动时读取文件，构建快照与实时态。
- 会话中 `add/replace/remove/batch` 只改实时态并落盘，**不改快照**。
- 这保证系统提示前缀缓存稳定，避免每写一次记忆就使上下文前缀失效。
- `TEAM.md` 与 `MEMORY.md`/`USER.md` 遵循相同双态模型，但物理分离、独立预算、独立快照。

#### 主要 API

| 方法 | 行为 |
|------|------|
| `add(target, content)` | 追加条目，预算校验，去重，注入扫描。 |
| `replace(target, old_text, new_content)` | 按唯一子串替换条目。 |
| `remove(target, old_text)` | 按唯一子串删除条目。 |
| `apply_batch(target, operations)` | 原子批量 add/replace/remove，按最终状态校验预算。 |
| `format_for_system_prompt(target)` | 返回冻结快照。 |

#### 写操作的安全流程

以 `replace/remove/batch` 为例：

1. 注入扫描新内容（`strict` scope）。
2. 若 `require_approval=True`，将操作 stage 到 `review_queue.json`。
3. 获取 `.lock` 文件锁。
4. 重新读取磁盘文件（`skip_drift=False`）。
5. 检测外部漂移：
   - 圆trip 不匹配（解析后再序列化 != 原字节）。
   - 任一单条长度超过该 store 的字符上限。
6. 若漂移，备份为 `.bak.<ts>` 并拒绝写入。
7. 应用操作，按最终状态检查字符预算。
8. 原子写盘（临时文件 + `os.replace`，处理 symlink、跨设备回退）。

`add` 省略漂移检测，因为追加不会覆盖已有内容；但仍会检测“文件存在但读失败”的情况，防止把已有记忆覆盖成单条。

#### 合并失败防循环

`_consolidation_failure()`：当预算不足或匹配失败时，前 3 次返回带 `current_entries` 的错误，提示模型在本轮内合并/删除后重试；超过 3 次返回 `done=True` 的终止结果，避免模型陷入记忆调用循环而阻塞用户回复。

### 5.3 `core/threat_patterns.py`

统一注入/渗出/后门模式库，按 scope 分级：

- `all`：经典 prompt injection、exfiltration（curl/wget/cat  secrets）。
- `context`：role hijack、C2/promptware、已知红队框架名。
- `strict`：persistence/SSH backdoor、修改 agent 配置、硬编码 secret、向 URL 发送上下文。

扫描器还会检测不可见 Unicode（zero-width、方向隔离符等），并在 NFKC 归一化后跑正则，防御同形字攻击。

MemoryStore 在写入与加载快照时都使用 `scope="strict"`；云端上行/下行同样复用 `scope="strict"`。

### 5.4 `providers/base.py` / `manager.py`

#### MemoryProvider ABC（M2 契约）

必须实现 5 个生命周期方法：

- `initialize(session_id, **kwargs)`
- `prefetch(query) -> str`
- `sync_turn(user, assistant)`
- `on_session_end(messages)`
- `shutdown()`

可选钩子：

- `on_memory_write(action, target, content, metadata)`：镜像内置 MemoryStore 的写操作。
- `get_tool_schemas()` / `handle_tool_call()`：暴露 provider 专属工具。

#### ProviderManager

- 最多允许 **一个外部 provider**（通过 `memory.provider` 配置），避免 schema 膨胀与后端冲突。
- 所有 `sync_turn` / `on_session_end` 扔到**单线程后台工作队列**，保证：
  - 慢 provider 不阻塞 MCP 请求路径。
  - 写入按顺序落地（turn N 在 turn N+1 之前）。
- 所有 provider 错误都被捕获并记录，永不抛给调用方。
- `notify_memory_write()`：内置 MemoryStore 提交后，把 add/replace/remove 镜像给 provider。

### 5.5 `providers/holographic/` — 结构化事实库

零云依赖的本地深度记忆后端，移植自 Hermes 的 `plugins/memory/holographic/`。

#### `store.py`

SQLite 事实库，核心表：

```sql
facts           -- fact_id, content(UNIQUE), category, tags, trust_score,
                --    retrieval_count, helpful_count, created_at, updated_at, hrr_vector
entities        -- entity_id, name, entity_type, aliases
fact_entities   -- 事实-实体关联
facts_fts       -- FTS5 虚拟表（content + tags）
memory_banks    -- 按 category 聚合的 HRR 向量
```

特性：

- 进程级共享连接 + RLock，避免多 writer 竞争导致的 “database is locked”。
- WAL 模式，NFS/SMB/FUSE 上自动回退到 DELETE journal。
- 实体提取：大写多词短语、引号内术语、aka 别名。
- `add_fact` 按 content 去重，返回已存在 fact_id。
- `record_feedback`：helpful +0.05 trust，unhelpful -0.10（不对称惩罚错误记忆）。

#### `retrieval.py` — FactRetriever

多策略检索，最终分数 = relevance × trust_score × 可选时间衰减。

 relevance 组合：

- `fts_weight`（默认 0.4）：FTS5 全文检索，查询先做 stopword 去除 + OR 连接，避免 FTS5 默认 AND 导致的自然语言查询召回低。
- `jaccard_weight`（默认 0.3）：查询词与事实内容的 token Jaccard。
- `hrr_weight`（默认 0.3）：HRR 向量相似度。

专用查询动作：

- `search`：关键词搜索。
- `probe`：实体召回（用 HRR unbind 从记忆库提取与该实体关联的内容）。
- `related`：发现与实体结构关联的事实。
- `reason`：多实体组合查询（AND 语义，找同时关联多个实体的事实）。
- `contradict`：自动化记忆卫生，找出共享实体但内容向量相似度低的事实对。

#### `holographic.py` — HRR 向量代数

使用**相位向量**实现 Holographic Reduced Representations：

- `encode_atom(word, dim)`：SHA-256 确定性生成相位向量。
- `bind(a, b)`：循环卷积 = 相位相加。
- `unbind(memory, key)`：循环相关 = 相位相减。
- `bundle(*vectors)`：叠加 = 复指数平均取角度。
- `encode_fact(content, entities)`：将内容绑定到 `__hrr_role_content__`，每个实体绑定到 `__hrr_role_entity__`，再 bundle 成事实向量。

支持无 numpy 回退：无 numpy 时 HRR 权重自动转给 FTS + Jaccard。

#### `compaction.py`

防止自动事实提取时把上下文压缩交接摘要误当成用户事实。检测标记包括：

- `_compressed_summary` metadata key。
- `[CONTEXT COMPACTION — REFERENCE ONLY]` 等前缀。
- merge-into-tail 消息中的 `[PRIOR CONTEXT]` / `[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]` 分隔符。

### 5.6 `evolution/` — 本地进化循环

本地进化循环的目标不是“让 AI 自动改记忆”，而是**让 AI 帮用户起草记忆修改提案，用户审完再落盘**。它把 LLM 当作一个提建议的实习生，而不是有写权限的管理员。

#### 设计哲学

| 原则 | 实现 |
|------|------|
| **默认手动，可选自动** | `auto_review=false`、`auto_apply=false`。 |
| **审批制** | 所有写入/合并/归档提案先进 `review_queue.json`。 |
| **永不自动删除** | curator 只 stage 归档提案；归档时先把原文本写到 `archive/<TARGET>.md`，再删除 live 条目。 |
| **可审计** | transcript、提案、策略文档都是本地可读文件。 |
| **故障隔离** | LLM 调用失败、JSON 解析失败、执行失败都不推进水位，记忆不被改动。 |

#### 四大组件

| 组件 | 文件 | 职责 |
|------|------|------|
| LLM Client | `llm_client.py` | OpenAI 兼容 client，进化的“大脑”。 |
| Transcripts | `transcripts.py` | 转录读取 + 水印（防重复处理）。 |
| 三阶段 | `review.py` / `strategy.py` / `curator.py` | 自省 / 策略 / 治理。 |
| Approval | `approval.py` | 统一审批队列与回放。 |

#### `llm_client.py`

轻量 OpenAI 兼容 chat client，基于 `httpx`，无 SDK 依赖：

- 从 `config.yaml` 的 `evolution.llm` 构造。
- 支持 `${ENV_VAR}` 环境变量占位符展开。
- 支持 `responder` 注入，用于离线测试。
- 未配置时抛出 `LLMNotConfiguredError`，进化功能优雅降级，不影响其他功能。

#### `transcripts.py`

- 转录按天分文件：`transcripts/YYYYMMDD.jsonl`。
- 每行 JSON：turn 记录或 session_end 记录。
- `.review_state.json` / `.strategy_state.json` 记录每个文件已处理到的**行偏移**。
- `read_new_turns(state, since, max_turn_chars)`：只读未处理过的 turn，并截断单条消息以控制 prompt 大小。
- 水位只在成功处理后才推进，失败时下次重试，**不会漏审也不会重复审**。

#### `review.py` — 记忆自省

**触发方式**：

- 手动：`memory_review()` MCP 工具、CLI `memory-hub review`。
- 自动：`memory_session_end()` 时，若 `evolution.auto_review=true`，在独立 daemon 线程启动 `_run_review_safely()`。

**流程**：

1. 读取未评审的 transcript turns（最多 80 条）。
2. 构造 prompt：现有 `memory` 条目 + 现有 `user` 条目 + 新 turns。
3. 要求 LLM 返回严格 JSON：
   ```json
   {"operations": [
     {"target": "memory|user", "action": "add|replace|remove",
      "content": "...", "old_text": "...", "reason": "..."}
   ]}
   ```
4. `validate_operations()` 过滤格式错误、target/action 非法、内容缺失的操作。
5. 按 target 分组：
   - `auto_apply=false`（默认）：`stage_write()` 进 `review_queue.json`。
   - `auto_apply=true`：直接 `store.apply_batch(target, ops)`。
6. 成功后 `save_watermark()` 推进 `.review_state.json`。

**判断标准**（prompt 中告知 LLM）：

- **target=user**：用户画像、偏好、沟通风格、工作习惯、对 assistant 的期望。
- **target=memory**：环境事实、项目约定、工具用法、坑点。
- **不保存**：任务进度、一次性请求、临时状态、原始日志。
- 不重复现有记忆条目。

#### `strategy.py` — 策略提取

**核心规则：重复才值得固化。** 模式必须出现 **≥2 次** 才会生成策略文档，防止一次性指令被错误固化。

**Prompt 关注点**：

- 反复纠正的风格、格式、长度、语气。
- 反复表达的偏好（"I prefer...", "always...", "never..."）。
- 反复出现的工作流程模式。

**输出**：`strategies/<slug>.md`，带 YAML frontmatter：

```yaml
---
name: chinese-replies
description: Reply in Chinese, keep it concise.
created: 2026-07-26T16:00:00
source_sessions: ["20260726"]
times_observed: 3
---
markdown body...
```

**Upsert 机制**：

- 已存在的策略通过相同 `name` 更新，不重复创建。
- `times_observed` 累加。
- `source_sessions` 合并。
- 正文由 LLM 重新 refine。

**流程**：`memory-hub strategy` → `run_strategy_extraction()` → 读取未分析 turns → LLM 返回 `{"strategies": [...]}` → 过滤 `times_observed < 2` → 创建/更新文件 → 推进 `.strategy_state.json`。

#### `curator.py` — 记忆治理

对 M1 的 live entries 做轻量治理，**全部 stage 到 review queue，不自动执行**。

**近重复合并**：

- 使用 `difflib.SequenceMatcher`，ratio ≥ 0.8 视为近重复。
- 调用 LLM 生成合并后的单一条目。
- Stage 为 batch 提案：
  ```json
  {"action": "batch", "target": "...", "operations": [
    {"action": "replace", "old_text": "A", "content": "merged"},
    {"action": "remove", "old_text": "B"}
  ]}
  ```

**过期/矛盾归档**：

- 把当前 target 的所有条目编号列给 LLM。
- LLM 返回 `{"stale": ["<exact entry text>", ...], "reason": "..."}`。
- 只归档确实仍存在于 live store 中的条目。
- Stage 为 `archive` 提案：
  ```json
  {"action": "archive", "target": "...", "old_texts": [...]}
  ```

**归档回放**（`archive_entries()`）：

1. 将要删除的完整原文本追加到 `archive/<TARGET>.md`。
2. 再调用 `store.remove()` 从 live store 删除。
3. 保证误删可恢复。

#### `approval.py`

统一审批队列 `review_queue.json`，与写入门（`memory.write_approval: review`）共用，云端扩展后还支持 `uplink_candidate`、`consensus_import`、`consensus_rollback` 等 payload 类型：

- 每条记录：
  ```json
  {"id": "mem-<ms>", "ts": 1234567890, "summary": "...", "payload": {...}}
  ```
- `list_pending()`：列出所有待审提案。
- `approve(pending_id, store, approve=True)`：
  - 批准：`_replay()` 执行 payload。
  - 拒绝：从队列移除。
  - 执行失败：保留在队列中，可重试或拒绝。
- `archive` payload 有独立回放路径，最终调用 `archive_entries()`。

---

#### 本地 Evolution 配置接口

```yaml
evolution:
  llm:
    base_url: https://api.moonshot.cn/v1
    api_key: ${MOONSHOT_API_KEY}
    model: moonshot-v1-8k
    timeout: 60
  auto_apply: false      # true: 自省提案直接写；false: 进 review queue
  auto_review: false     # true: session_end 后后台自省
```

---

#### 本地 Evolution 主要取舍与限制

| 取舍 | 原因 |
|------|------|
| **LLM 调用成本高** | 自省、策略、治理都依赖外部 LLM，默认手动以控制成本。 |
| **策略需 ≥2 次** | 防一次性指令固化，但可能漏掉重要但只出现一次的长效偏好。 |
| **curator 用 difflib** | 轻量、离线、无需 embedding，但不如语义相似度精细。 |
| **审批增加延迟** | 自动化的好处被人工把关平衡，适合对个人记忆敏感的场景。 |
| **转录只按天文件** | 简单、可审计，但长期可能产生大文件，未来可按大小/条数 rollover。 |

---

### 5.7 Evolution 的潜在扩展：Skill、Trace 与 Session Experience

> 本节内容为**设想性设计**，尚未在代码中实现，用于展示 evolution 模块可扩展处理更多“重要成分”的方向。

#### 背景与动机

当前 evolution 主要处理三类输入：

- `transcripts/*.jsonl` → `review.py` → memory entries
- recurring patterns → `strategy.py` → strategy documents
- live entries → `curator.py` → merge/archive proposals

但 agent 实际运行中还存在三类高价值“记忆成分”未被直接纳入：

1. **Skill（技能文档）**：可复用的工作流、能力说明，通常存放在 `~/.config/agents/skills/<name>/SKILL.md`。
2. **Skill Trace（技能执行轨迹）**：某次调用 skill 时的输入、中间步骤、用户反馈、修正、最终结果。
3. **Session Experience（会话上下文中的经验）**：一次完整会话中产生的、尚未沉淀为长期记忆的中间洞察，如某类 bug 的排查路径、工具组合用法、临时但被反复提及的偏好。

如果 evolution 能处理这三类成分，memory-hub 就从“共享记事本”升级为“会学习的工具箱”，把一次性会话经验逐步转化为可复用的技能与长期记忆。

#### 设计原则

- **不改动 M1/M2/M3 核心语义**：只把 skill/trace/experience 作为新的输入源。
- **审批制仍然适用**：所有生成/修改提案仍进 `review_queue.json`。
- **Skill 保持人类可读**：仍用 Markdown + YAML frontmatter，可版本控制。
- **Trace 与 experience 只读蒸馏**：它们本身不直接写入长期存储，只作为原料被提炼。
- **可配置关闭**：用户可单独关闭 skill/trace/experience 的进化处理。

#### Skill 的 Evolution 处理

**Skill 作为记忆对象**

Skill 文档可视为一种“高结构化的长期记忆”。Evolution 可以：

- **发现 skill 缺口**：当 transcript 中反复出现某类任务需求（如“帮我画架构图”），而现有 skill 未覆盖时，stage 一个“新建 skill 提案”。
- **更新现有 skill**：当用户多次以相似方式纠正某个 skill 的使用方式时，生成 `SKILL.md` 的 diff 提案。
- **合并/拆分 skill**：当两个 skill 职责重叠，或一个 skill 过于臃肿时，生成合并或拆分提案。

**Skill 元数据扩展（设想）**

skill frontmatter 可增加进化相关字段：

```yaml
---
name: architecture-diagram
created: 2026-01-15
times_used: 12
last_used: 2026-09-03
evolution_state: stable    # stable | candidate | deprecated
related_memories: [memory-id-1, memory-id-2]
related_strategies: [chinese-replies]
---
```

Evolution 根据 `times_used`、用户反馈、trace 成功率等指标判断 skill 是否需要更新或归档。

#### Skill Trace 的处理

**Trace 的来源与格式（设想）**

每次 agent 调用 skill 时，可记录：

```json
{
  "type": "skill_trace",
  "ts": 1234567890,
  "skill": "architecture-diagram",
  "session_id": "sess-abc",
  "input_summary": "用户要求画 memory-hub 架构图",
  "steps": [
    {"tool": "search_web", "result": "ok"},
    {"tool": "generate_svg", "result": "ok"}
  ],
  "user_feedback": "颜色太深，改浅色",
  "correction": "使用更浅的配色方案",
  "outcome": "success"
}
```

Trace 可存入 `~/.memory-hub/traces/YYYYMMDD/<skill>/<trace-id>.jsonl`，或扩展 `hub.db` 中的 trace 表。

**Trace 的 evolution 用途**

- **成功案例蒸馏**：把高频成功 trace 中的有效步骤固化为 skill 正文补充。
- **失败/修正模式提取**：当同一 skill 多次出现相似修正时，stage 更新提案。
- **skill 效果评估**：统计成功率、平均修正次数，作为 skill 是否 stable/deprecated 的依据。
- **生成示例库**：从优秀 trace 中提取典型用例，写入 skill 的 EXAMPLES 段落。

#### Session Experience 的处理

**什么是 Session Experience**

一次完整会话中，除了最终沉淀到 `MEMORY.md` / `USER.md` 的事实外，还存在大量“中间经验”：

- 某个工具的特殊用法组合。
- 解决某类 bug 的排查路径。
- 用户临时提出但可能未来会重复的工作偏好。
- 跨工具、跨文件的协作模式。

这些经验如果直接写入主记忆会污染它；如果完全丢弃又浪费学习机会。

**Session Experience 的暂存形式（设想）**

作为一种“短期工作记忆”存于：

```
~/.memory-hub/experiences/YYYYMMDD.md
```

格式类似主记忆，但带 TTL 或 session 标记：

```markdown
---
session_id: sess-abc
ttl_days: 7
source_skill: architecture-diagram
---

发现用户喜欢 SVG 输出后手动调整配色。
§
本次排查使用 grep + test_store.py 快速定位问题。
```

**Session Experience 的 evolution 流程**

1. **采集**：`memory_session_end()` 把 `_session_messages` 和本次会话的“经验摘要”交给 evolution。
2. **蒸馏**：LLM 从 experience 中识别：
   - 可沉淀为长期记忆的事实 → `review.py` 处理。
   - 重复出现的工作模式 → `strategy.py` 处理。
   - 可固化为 skill 的能力 → skill evolution 处理。
   - 一次性或临时信息 → 丢弃或过期。
3. **暂存**：所有提案进入 `review_queue.json`。
4. **审批**：用户批准后分别写入 memory / strategy / skill。
5. **过期清理**：TTL 到期且未被升级的 experience 自动归档或删除。

#### 与现有 Evolution 的集成（设想架构）

```
transcript turns ──► review.py ──► memory proposals
        │
        ├─► skill traces ─────────► skill evolution ──► skill update/create proposals
        │
        ├─► session experiences ─► experience distiller
        │                              │
        │              ├─► memory proposals
        │              ├─► strategy proposals
        │              └─► skill proposals
        │
        └─► recurring patterns ──► strategy.py ──► strategy documents
```

可新增一个可选阶段 `experience.py`：

- 输入：session experience + skill trace + recent transcripts。
- 输出：分类后的 proposals（memory / strategy / skill / archive）。
- 所有 proposals 统一进 `review_queue.json`，由 `approval.py` 审批。

#### 审批粒度的扩展

当前 `review_queue.json` 的 payload 以 memory operations 为主。扩展后可支持更多类型：

```json
{
  "id": "mem-123",
  "type": "skill_update",
  "summary": "update architecture-diagram skill: add light-theme example",
  "payload": {
    "skill_path": ".../skills/architecture-diagram/SKILL.md",
    "diff": "...",
    "reason": "3 traces show user prefers light theme"
  }
}
```

`approval.py` 的 `_replay()` 可扩展对 `skill_update` / `skill_create` / `experience_archive` 类型的回放逻辑（设想中）。

#### 取舍与挑战

| 挑战 | 设想中的应对 |
|------|-------------|
| Skill 更新可能破坏既有用法 | 所有 skill 更新进审批；旧版本保留到 `archive/skills/`。 |
| Trace 量过大 | 只保留有反馈或异常的 trace；成功 trace 采样。 |
| Experience 噪声高 | TTL 机制 + 重复出现才升级；一次性 experience 自动过期。 |
| 多源冲突 | skill / strategy / memory 可能给出矛盾指示；增加冲突检测阶段。 |
| 隐私敏感 | trace 和 experience 默认本地存储；用户可配置是否允许从中学习。 |

---

## 6. MCP 工具契约

所有工具都是 `server/mcp_server.py` 中的普通函数，用 `mcp.tool()` 注册，便于单测。

| 工具 | 作用 |
|------|------|
| `memory_read(target="both")` | 读 live entries。 |
| `memory_add(target, content)` | 追加条目。 |
| `memory_replace(target, old_text, content)` | 替换条目。 |
| `memory_remove(target, old_text)` | 删除条目。 |
| `memory_batch(target, operations)` | 原子批量操作。 |
| `memory_prefetch(query="")` | 返回 `<memory-context>` 块：快照 + provider 召回。 |
| `memory_sync_turn(user, assistant)` | 写入 transcript，收集 session messages，后台同步 provider。 |
| `memory_session_end()` | 写 session_end 标记，触发 provider `on_session_end`；`auto_review=true` 时后台自省。 |
| `memory_provider_tool(name, args)` | 透传调用 provider 工具（如 `fact_store`）。 |
| `memory_review()` | 手动触发记忆自省。 |
| `memory_pending()` | 列出待审提案。 |
| `memory_approve(pending_id, approve=True)` | 批准/拒绝提案。 |
| `memory_strategies_list()` | 列出策略文档。 |
| `memory_strategies_read(name)` | 读取单个策略文档。 |
| `memory_status()` | 条目数、字符用量/预算、路径、provider 状态。 |
| `memory_uplink_pending()` * | 列出待审批的上行候选（云端扩展）。 |
| `memory_consensus_status()` * | 查看已订阅频道与本地共识版本（云端扩展）。 |
| `memory_consensus_rollback(version)` * | 回退 TEAM.md 到指定版本（云端扩展）。 |

\* 云端扩展工具，仅在 `cloud.enabled=true` 时暴露。

### 写镜像

`memory_add/replace/remove/batch` 在 `result.get("success") == True` 且非 staged 时，会调用 `ProviderManager.notify_memory_write()`，把写操作同步给外部 provider。holographic provider 会把这些写入作为 `general` / `user_pref` 事实存入 SQLite。

---

## 7. 数据流详细说明

### 7.1 会话启动

1. Agent 调用 `memory_prefetch`。
2. MemoryStore 加载 `MEMORY.md` / `USER.md` / `TEAM.md`，生成冻结快照。
3. 快照注入系统提示。
4. ProviderManager 若配置了外部 provider，则初始化并加载。
5. 若启用云端，检查订阅频道版本并提示有新 delta（不自动导入）。

### 7.2 会话中写入

1. Agent 调用 `memory_add/replace/remove/batch`。
2. MemoryStore 校验、落盘、返回结果。
3. 写操作镜像给 provider（如果配置）。
4. 若启用云端上行，符合白名单的本地条目进入 `uplink_candidate` 审批队列。
5. **系统提示中的快照不变。**

### 7.3 转录上报

1. 每完成一轮对话，接入方调用 `memory_sync_turn(user, assistant)`。
2. 写入 `transcripts/YYYYMMDD.jsonl`。
3. 加入内存中的 `_session_messages`。
4. provider 的 `sync_turn` 进入后台队列（holographic 当前为 no-op）。

### 7.4 会话结束

1. 调用 `memory_session_end()`。
2. 写 `session_end` 标记。
3. provider 的 `on_session_end` 进入后台队列（holographic 可配置 `auto_extract` 提取事实）。
4. 若 `evolution.auto_review=true`，后台线程启动 `_run_review_safely()`。
5. `_session_messages` 清空。
6. 若启用云端反馈回传，将 consensus 的纯计数反馈（helpful/unhelpful/counter）随下次上行捎带。

### 7.5 记忆自省

1. 手动或自动触发 `memory_review()` / CLI `memory-hub review`。
2. `review.py` 读取未评审 turns，构造 prompt。
3. LLM 返回 JSON 操作。
4. 默认 stage 到 `review_queue.json`。
5. 用户通过 `memory_approve` 或 CLI `memory-hub approve` 批准。
6. `approval.py` 的 `_replay()` 调用 `apply_pending()` 或直接 `archive_entries()`。

### 7.6 策略提取

1. CLI `memory-hub strategy` 调用 `run_strategy_extraction()`。
2. 读取未分析 turns，构造 prompt。
3. LLM 返回 `{"strategies": [...]}`。
4. 过滤掉 `times_observed < 2` 的提案。
5. 创建或更新 `strategies/<slug>.md`。

---

## 8. 安全模型

### 8.1 本地层安全

| 层级 | 机制 |
|------|------|
| **写入扫描** | 所有 add/replace/batch 新内容过 `strict` threat patterns；命中则拒绝写入。 |
| **快照消毒** | 加载时把命中威胁的条目替换为 `[BLOCKED: ...]` 占位符进入系统提示；原始文本保留在文件中供用户删除。 |
| **不可见字符** | 检测 zero-width、方向隔离符等，报告具体 Unicode codepoint。 |
| **漂移检测** | 防止外部写入（patch 工具、shell 追加、并发 session）被静默覆盖；检测到则备份并拒绝。 |
| **原子写** | 临时文件 + fsync + rename，避免 truncate-before-lock 数据丢失。 |
| **文件锁** | 单独 `.lock` 文件，Unix `fcntl.flock` / Windows `msvcrt.locking`。 |
| **写入门** | `memory.write_approval: review` 时，所有写入先 stage 到队列，审批后才生效。 |
| **进化默认手动** | `auto_apply=false`、`auto_review=false`、curator 永不自动执行、归档可恢复。 |
| **Provider 故障隔离** | provider 错误被捕获记录，不阻塞主路径。 |

### 8.2 云端层安全（扩展）

云端层引入三类新风险，设计在后续第 13 章详细展开：

| 风险 | 本地防御 |
|------|----------|
| **隐私泄漏（坑一）** | 上行白名单 schema、k-匿名隔离区、用户审批、双层身份、频道不开“人”主题。 |
| **共识毒化（坑二）** | 下行 strict 扫描、五层记忆注入检测、独立 `TEAM.md` 命名空间、签名版本化、上下文账本、一键回滚。 |
| **context collapse（坑三）** | 共识 gen ≤ 2 硬约束、condition 交集机械校验、TTL/反证淘汰、supersede 链。 |

---

## 9. 配置模型

### 9.1 完整 `config.yaml` 示例

```yaml
memory:
  memory_char_limit: 2200
  user_char_limit: 1375
  team_char_limit: 1500
  write_approval: auto        # auto | review
  provider: ""                # 空 = 只用内置文件记忆；可选 holographic

evolution:
  llm:
    base_url: https://api.moonshot.cn/v1
    api_key: ${MOONSHOT_API_KEY}
    model: moonshot-v1-8k
    timeout: 60
  auto_apply: false           # true: 自省提案直接写；false: 进 review queue
  auto_review: false          # true: 每次 session_end 后台自省

plugins:
  hermes-memory-store:
    db_path: $MEMORY_HUB_HOME/hub.db
    auto_extract: false
    default_trust: 0.5
    min_trust_threshold: 0.3
    temporal_decay_half_life: 0
    hrr_dim: 1024
    hrr_weight: 0.3

cloud:
  enabled: false                    # 总开关，默认关；开启后仍需逐项 opt-in
  hub_url: "https://hub.example.com"
  uplink:
    enabled: false                  # 上行开关
    auto_submit: false              # true: 过扫描后直接上行；false（默认）: 审批后上行
    allowed_fact_types: ["env_fact", "tool_pitfall", "api_behavior", "workflow_guardrail"]
    aggregate_risk_check: true      # 聚合风险检查
  consensus:
    enabled: false                  # 下行开关
    subscriptions: ["coding/python-packaging"]   # 订阅频道
    auto_import_delta: false        # true: 常规 delta 免审（仍可回滚）；false: 全部审批
    context_injection: true         # 是否注入系统提示（否则仅可查询）
    keep_versions: 10               # 本地保留的共识版本数
    feedback_uplink: true           # 反馈计数信号回传（纯计数，无内容）
  identity:
    reputation_key: "auto"          # 本地生成/加载长期信誉密钥
    rotate_contributor_id: "per_session"
```

### 9.2 配置哲学

- 默认全关 + 逐项 opt-in，与现有 `auto_apply: false`、`auto_review: false` 的“默认手动”哲学一致。
- `auto_submit` / `auto_import_delta` 两个自动化开关分开，让用户可以独立选择“上行严格、下行宽松”或反之。

---

## 10. 扩展性设计

### 10.1 新增 Provider

在 `memory_hub/providers/<name>/` 下创建模块，暴露 `create_provider(config=None) -> MemoryProvider`，并在 `config.yaml` 设置 `memory.provider: "<name>"`。

### 10.2 新增 Evolution 阶段

可复用 `transcripts.py` 的水印机制，新增一个 `state_file` + `run_*()` 函数，通过 CLI/MCP 工具触发。云端扩展即采用此模式：新增 `evolution/uplink.py` 与 `.consensus_state.json`。

### 10.3 自定义预算/分隔符

当前分隔符 `§` 与默认预算是 Hermes 语义的一部分，非必要不动。预算可通过 `config.yaml` 覆盖。

### 10.4 客户端接入

- stdio：任何支持 MCP stdio 的客户端（Kimi Code、Claude Code）配置 command + args 即可。
- streamable-http：启动 `--http` 暴露 `http://host:port/mcp`。

---

## 11. 与 Hermes Agent 的关系

memory-hub 是 Hermes 记忆子集的**保真抽取 + 简化封装**：

| memory-hub | Hermes 来源 | 变化 |
|------------|-------------|------|
| `core/store.py` | `tools/memory_tool.py` | 全量语义保留；atomic_replace 内联；写入门简化。 |
| `core/threat_patterns.py` | `tools/threat_patterns.py` | 原样复制。 |
| `providers/base.py` | `agent/memory_provider.py` | 简化为 M2 契约。 |
| `providers/manager.py` | `agent/memory_manager.py` | 保留 ≤1 provider、串行 worker。 |
| `providers/holographic/` | `plugins/memory/holographic/` | 全量移植；WAL fallback 简化。 |
| `evolution/review.py` | `agent/background_review.py` | fork-agent 改为 JSON 提案 + 审批。 |
| `evolution/strategy.py` | `agent/learn_prompt.py` 思路 | 策略文档化。 |
| `evolution/curator.py` | `agent/curator.py` 思路 | 轻量 difflib 实现。 |
| `evolution/uplink.py` * | 新增 | 云端上行门与共识同步。 |

\* 云端扩展为新增模块。

---

## 12. 已知限制与取舍

### 12.1 本地层

| 取舍 | 说明 |
|------|------|
| 单外部 provider | 防止 schema 冲突；需要多后端时需自行 fork 或封装聚合 provider。 |
| 字符预算非 token | 简单、模型无关，但可能与真实上下文成本不完全对齐。 |
| HRR 容量上限 | `dim=1024` 时约存 sqrt(dim)≈32 条高保真，超过后 SNR 下降会告警。 |
| 策略提取需 ≥2 次 | 避免一次性指令被固化；但也可能漏掉真正重要但只出现一次的长效偏好。 |
| curator 用 difflib | 轻量、离线、无需 embedding，但不如语义相似度精细。 |
| 无分布式同步 | 同一机器多进程共享文件 + SQLite；不支持跨机器同步。 |
| 进化 LLM 成本 | 自省/策略/治理都需调用外部 LLM；默认手动，用户可控。 |

### 12.2 云端层（扩展）

| 取舍 | 说明 |
|------|------|
| 云端依赖 | 开启后需要网络与 hub 服务，违背“零云依赖”默认目标。 |
| 隐私风险 | 白名单 + k-匿名把风险压到很低，但推断型隐私无理论完备解。 |
| 长尾坑点可能死在隔离区 | k-匿名门槛使稀有但真实的问题无法形成共识。 |
| 工程复杂度高 | 身份凭证、签名版本、Sybil 检测、Canary 等需要额外基础设施。 |
| 冷启动困境 | 早期用户少，k 凑不齐，共识稀少，下行价值低。 |

---

## 13. 云端进化层设计

> 本节整合自《memory-hub 云端进化层设计文档》，作为本地层的可选群体级扩展。默认全部关闭，用户逐项 opt-in 后才启用。

### 13.1 定位与边界

#### 13.1.1 本层是什么

本地进化循环（review / strategy / curator 三阶段）是**单机自省**：从自己的会话转录中提炼自己的记忆。云端进化层是它的群体扩展：

- **上行**：本地记忆 →（白名单过滤 + 渗出扫描 + 用户审批）→ 云端隔离候选区 →（k-匿名聚合 + 蒸馏）→ 团队共识库。
- **下行**：团队共识库 →（频道订阅 + 增量 delta + 签名版本）→ 本地独立命名空间 →（严格扫描 + 用户审批）→ 进入 agent 上下文。

#### 13.1.2 本层不是什么（防止范围漂移）

| 不是 | 说明 |
|------|------|
| 不是 skill 市场 | 本层处理的是**记忆**（事实、坑点、偏好、guardrail），不是可复用工作流。skill 是稳定程序，一人写好万人照抄；记忆带时效、带情境、带个体噪声，其价值靠群体统计显影。skill 共享是另一个项目。 |
| 不是原始轨迹同步 | 会话转录、skill trace、session experience **永不上行**。上行的只有经过白名单 schema 结构化后的条目。 |
| 不是全局统一知识库 | 共识分层（全局 / 情境 / 个人），允许带情境标签的矛盾共识共存，不做强行仲裁。 |
| 不改变本地三权 | 本地 MemoryStore 的写入门、漂移检测、审批制全部保留；云端只是多了一个“提案来源”，用户对下行条目保留否决权。 |

#### 13.1.3 继承的设计哲学

原设计的核心立场原样继承并放大：**不信任任何单点判断；所有关键决策要么可审计、要么可回滚、要么有结构硬约束**。云端化之后，每一条都更必须是“机制”（代码强制）而不是“约定”（文档承诺）。

---

### 13.2 设计依据：论文映射表

本设计的每个核心机制都有明确的学术出处。四篇核心论文提供进化机制，若干攻防论文提供威胁模型与对策。

#### 13.2.1 四篇核心论文

| 论文 | 出处 | 核心机制 | 本设计借用点 |
|------|------|----------|--------------|
| **G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems** | arXiv:2506.07398，NeurIPS 2025（NUS / 同济 / UCLA / A*STAR / NTU） | 受组织记忆理论启发，将多智能体协作历史组织为三层图：insight graph（可泛化抽象洞察）、query graph（任务层）、interaction graph（细粒度轨迹）；任务到来时双向遍历取“高层洞察 + 细粒度轨迹”，任务后整层吸收新轨迹持续进化 | ① 分层晋升漏斗：共识层只存 insight 级抽象，原始证据只留哈希指针；② 消融实验表明去掉 insight 层或 interaction 层都会掉点——“抽象共识”与“原始细节”必须双层并存；③ **反面证据**：单机记忆方案直接搬进多 agent 场景某些任务上反而降性能——说明云端层不能是单机循环的简单放大，必须重新设计聚合语义 |
| **ACE: Agentic Context Engineering** | arXiv:2510.04618，ICLR 2026（Stanford / UC Berkeley / SambaNova） | 上下文作为“进化中的 playbook”；Generator 产轨迹、Reflector 提炼、Curator 做**增量 delta 合并**（非 LLM 的确定性逻辑，可并行）；明确命名两个失败模式：brevity bias（摘要丢细节）与 context collapse（反复改写侵蚀知识） | ① 共识下发采用 delta 增量条目 + 本地确定性 merge，禁整包覆盖；② 提炼代数上限（generation ≤ 2）硬编码防群体尺度坍缩；③ 共识条目双写“陈述 + 证据指针”，摘要永远可回溯 |
| **ReasoningBank** | arXiv 预印本，2025-09（Google DeepMind） | 记忆条目同时从**成功与失败**轨迹蒸馏：成功经验提供已验证策略，失败经验提供反事实信号与坑点，用于打磨 guardrail；观察到条目随时间的“成熟演化”（操作型 → 自检型 → 组合型） | ① 云端单独设“失败蒸馏管道”，坑点/guardrail 型共识下发权重高于成功策略；② 共识条目带成熟度阶段标记，作为质量信号参与排序 |
| **A-MEM: Agentic Memory for LLM Agents** | arXiv:2502.12110，NeurIPS 2025（Rutgers） | Zettelkasten 卡片盒：每条记忆是带关键词/标签/上下文描述的原子笔记；写入时自动建链；**新记忆触发对既有历史记忆的更新**，整个网络持续精炼 | ① 云端共识库条目间建立链接网络，新证据到达时触发既有共识的 refine（受 generation 上限约束）；② 条目的“结构化笔记”形态直接对应本设计的白名单 schema |

#### 13.2.2 攻防与支撑文献

| 论文/工作 | 出处 | 对本设计的作用 |
|-----------|------|----------------|
| **Collaborative Memory** | arXiv:2505.18279（2025-05） | 多用户记忆共享 + 动态访问控制的直接参照：每片段标注 provenance（来源/时间），检索分用户私有层与跨用户层；写入策略含 transformation 模式（落盘前改写/匿名化/脱敏）——本设计上行前处理的原型 |
| **Memory Sharing for LLM-based Agents** | arXiv:2404.09982（2024-04） | 早期奠基：开放式任务共享记忆池 + “记忆对未来任务的潜在效用”评估——本设计“反馈信号上行”的前身 |
| **MemCollab** | arXiv:2603.23234（2026-03） | 多 agent 在**相同任务**上对比轨迹、蒸馏任务级模式、滤除个体噪声；检索先按任务类别过滤再语义搜索——本设计“情境共识”与频道化分发的依据 |
| **Murmur** | arXiv:2511.17671（2025-11，Princeton 等） | **威胁模型核心文献**：实证跨用户共享记忆可被“chatter”毒化并击穿协作 agent 群组——证明本设计的云端共识天然打开攻击面 |
| **MINJA / AgentPoison** | AgentPoison 为 NeurIPS 2024 | 下行毒化的攻击基线：MINJA 仅需查询即可注入记忆（成功率 >95%）；AgentPoison 系统性验证投毒记忆库的红队方法——本设计“下行一律视为不可信输入”的依据 |
| **MemAudit** | arXiv:2605.23723（2026） | 事后审计范式：因果归因 + 结构异常检测定位投毒记忆——本设计“上下文账本 + 可回滚”的依据 |
| **Adaptive Attacks Break Defenses** | arXiv:2503.00061 | 针对静态攻击调优的防御会被自适应攻击绕过——本设计“假设失陷，检测+止血+回滚优先于单点防住”的心态依据 |
| **Typed Memory / provenance-role collapse** | 2026 年工作 | 形式化“反复压缩后来源/角色/情境元信息最先被磨平”；解法是 typed memory representation（来源/角色/情境为独立类型字段）——本设计共识 schema 强制独立字段的依据 |
| **联邦学习 Byzantine 鲁棒性（LIE 攻击等）** | 联邦学习文献 | 中位数/Krum 等纯统计过滤可被精心构造的攻击绕过（恶意更新与诚实更新统计上不可区分）——本设计“独立用户计数必须叠加身份与历史维度”的依据 |

---

### 13.3 总体架构

#### 13.3.1 三层结构

完整架构见第 3 章。云端层内部包含：

- **L1 云端聚合层**：隔离候选区、聚合/蒸馏管线、共识库、自然选择、信誉/身份层、Canary 监控、失败蒸馏管道。
- **L2 分发层**：频道订阅、delta 增量包、按需 pull API。

#### 13.3.2 记忆条目的完整生命周期（状态机）

```
本地会话
   │  review.py / strategy.py（现有）
   ▼
个人记忆条目（MEMORY.md / strategies/）        ←—— 永不上行
   │  Uplink Gate：白名单 schema 校验 + 渗出扫描 + 用户审批
   ▼
上行条目（白名单结构化事实）
   │  云端接收 → 语义聚类
   ▼
隔离候选区 QUARANTINE（独立来源 < k，永不下发、永不聚合出区）
   │  聚类内独立信誉身份 ≥ k 且通过 Sybil 聚类检测
   ▼
共识候选 candidate（生成 consensus 草案，generation=1，限定语校验）
   │  限定语保留校验 + Canary 管线无污染
   ▼
共识激活 active（进入频道，可下发/pull）
   │  持续反馈：fresh 观测续命 TTL / 反证计数累积
   ▼
 ├─ TTL 到期且无新鲜观测 → 降级 degraded → 归档 archived
 ├─ 反证超过阈值 → 降级 degraded → 仲裁（分叉为两条情境共识或归档）
 └─ generation 已达 2 且需更新 → 禁止再提炼，到期消亡
```

**设计理由（为什么用状态机而不是一次性流程）**：记忆的群体可信性不是一次性判定，而是随证据流持续变化。k-匿名、反馈淘汰、反证降级都是“随时间发生的事件”，只有状态机能统一表达。同时每个状态转移都有明确的机械判据（计数、阈值、TTL），不依赖任何单点 LLM 判断——这是“机制而非约定”的落地方式。

**优点**：全生命周期可审计；每个状态可独立测试；归档而非删除，误杀可恢复（继承 curator 哲学）。

---

### 13.4 核心数据模型

#### 13.4.1 上行条目 schema（白名单，本地生成）

**设计：只允许结构化类型，拒绝自由文本。**

```yaml
uplink_entry:
  schema_version: 1
  fact_type: env_fact | tool_pitfall | api_behavior | workflow_guardrail
  subject: "pip"                      # 受控词表：工具/库/API 名，枚举或受控注册
  predicate: "has_bug_in_version"     # 受控关系词表
  object: "25.0 在 Windows 上解析本地路径失败"   # 唯一允许的自由文本，限长 200 字符
  condition:                          # 强制字段，只允许枚举值，禁止自由描述
    tool: ["pip"]
    version_range: ["25.0.x"]
    task_category: ["python-packaging"]
    os: ["windows"]
  evidence_type: first_hand_observation | doc_reference | inferred
  observed_at: "2026-08-30"
  contributor_id: "anon-session-9f3a..."   # 轮换标识，每次会话更换
  reputation_attestation: "sig(...)"        # 信誉密钥的匿名凭证
  content_hash: "sha256(canonical_form)"   # 规范化后哈希，用于证据指针
```

**设计理由**：
1. **判定逻辑从“识别敏感”翻转为“白名单放行”**。检测式脱敏（黑名单）在原理上赢不了马赛克效应——单条无害、组合可识别。攻击面是“组合”而非“单条”，黑名单永远枚举不完。白名单思路反过来：定义什么**可以**分享，其余默认拒绝。个人生活、身份、关系类记忆天然塞不进 `{subject, predicate, condition}` 这个结构，这一刀切掉约九成隐私风险面。
2. **condition 字段只允许枚举值**，把“我们公司的 X 项目”这类结构性身份信息在 schema 层面物理排除——不是“扫出来再删”，而是“根本写不进去”。
3. `content_hash` 对规范化文本（NFKC、去空白、词表归一）取哈希，使云端能判重、能对齐同一事实的多次独立报告，而不需要保存原始自由文本。

**优点**：隐私防线是结构性的而非检测性的；条目可机器校验；规范化哈希使跨用户聚类成本极低。

**代价/残余风险**：推断型隐私理论上无完备解——白名单把风险压到很低但不是零；受控词表需要治理成本（见 13.9）。

#### 13.4.2 共识条目 schema（云端生成，下行分发）

```yaml
consensus_entry:
  consensus_id: "C-7f3a2b"
  channel: "coding/python-packaging"
  statement: "pip 25.0.x 在 Windows 上解析含中文的本地路径会失败，安装前应升级至 ≥25.1"
  condition:                          # 强制字段，限定语保护校验的对象
    tool: ["pip"]
    version_range: ["25.0.x"]
    os: ["windows"]
  generation: 1                       # 提炼代数，硬上限 2
  maturity: operational | self_checking | compositional   # ReasoningBank 成熟阶段
  evidence:                           # 证据指针三要素——缺失则物理上无法写入共识层
    independent_sources: 5            # 经 Sybil 聚类合并后的独立信誉身份数
    total_observations: 12            # 总观测次数（与上者分开记账）
    first_seen: "2026-06-11"
    last_seen: "2026-08-29"
    evidence_hashes: ["sha256:...", ...]   # 仅哈希，原始条目不可读
  confidence: 0.82                    # 由来源数/反证数/时间衰减计算，公式公开
  counter_evidence: 1                 # 反证计数（用户标记"对我失效"）
  status: candidate | active | degraded | archived
  ttl_days: 90                        # 需新鲜观测续命，否则自动降级
  generation_lineage: ["Q-cluster-331"]   # 来源聚类指针，供审计
  links: ["C-8e1c4d"]                 # A-MEM 式条目间链接
  signature: "hub-sign(consensus_id + version + payload)"
  release_version: 2026.09.03-14      # 所属发布版本，支持回滚
```

**设计理由**：
1. **证据指针是存在前提**。没有 `{evidence_hashes, independent_sources, 时间窗}` 的条目物理上无法写入共识层。这是 context collapse 对策的结构化落地：共识永远可回溯证据规模，“权威光环的解药是透明，不是谨慎措辞”。
2. **独立来源数与总观测数分开记账**。共识≠正确：群体统计压得住个体噪声，压不住相关性偏差（同源用户群会把同一个错误“独立验证”出高置信度）。`independent_sources=5, total_observations=12` 比 `12 次观测` 诚实得多；confidence 公式对两者分别加权。
3. **generation 上限 = 2，硬编码**。gen 2 条目禁止作为提炼原料，要么回源证据重归纳（实际不可行），要么到期消亡。把“禁止对摘要做摘要”从约定变成结构约束。
4. **condition 强制 + 限定语校验**（见 13.5.2）：对应 provenance-role collapse 研究——反复压缩后最先丢的就是“在什么条件下成立”，所以条件做成 typed 字段而非揉在自由文本里，并加机械校验。
5. **maturity 标记**：借 ReasoningBank 的观察（条目从操作型成熟到组合型），让消费方能区分“一条刚固化的具体操作”与“一条历经多次验证的组合策略”。

**优点**：透明、可审计、可计算置信度；TTL 与反证使共识库有“死亡”，不会只膨胀；链接网络支持 A-MEM 式 refine。

**代价**：schema 较重，云端实现成本高；`evidence_hashes` 只存哈希意味着无法真正“回源重读”（见 13.6.4 的显式取舍）。

#### 13.4.3 身份模型：信誉与内容的分离

**设计：双层身份。**

| 层 | 载体 | 生命周期 | 云端可见 |
|----|------|----------|----------|
| 信誉身份 | 长期 pseudonymous 信誉密钥（本地生成，永不暴露私钥） | 长期 | 只挂信誉分（历史验证为真的比例、账户龄、速率），**不挂任何上行内容** |
| 内容身份 | 轮换 contributor_id | 每次会话更换 | 只挂本次上行的条目，**跨会话不可链接** |

上行时携带的是“信誉密钥的匿名凭证”（证明“此上行来自信誉 0.9 的贡献者”，但不证明“是哪一个贡献者”）。

**设计理由**：隐私要求匿名（持久标识会拼出个人画像），Sybil 防御要求信誉（需要纵向历史）——两者直接冲突，单层身份无解。分层后：云端知道“这是一条高信誉上行”，但无法把不同时间段的条目拼成个人轨迹。

**优点**：同时满足两个对抗性需求；新账户权重压到接近零，Sybil 成本从“注册 N 个号”抬到“养 N 个有历史的号”。

**代价/残余风险**：信誉分本身也是指纹——高分贡献者群体很小，可被间接识别。缓解：信誉分按区间粗粒度披露（如 0.8-0.9 一档），不暴露精确值。这是目前攻防两边都认账的最佳折中，但不是完备解，记在第 13.9 章。

#### 13.4.4 频道（channel）模型

**设计**：共识库按主题分频道，层级式命名：

```
coding/python-packaging
coding/toolchain-windows
data/finance-data-sources
agent/mcp-ecosystem
```

每个频道独立的 k 阈值、TTL 默认值、订阅者集合。**涉及“人”的主题（人物评价、组织信息）不开频道**——结构性排除，而不是靠审核。

**设计理由**：① 接 MemCollab 的 category-aware 检索——先按任务类别过滤再语义匹配，防跨领域错误迁移；② k 阈值 per-频道可配，代码工具类 k=3 即可，敏感领域可以 k=5 或干脆不开放，把隐私/时效的 trade-off 颗粒化；③ 订阅制让用户只暴露于自己关心的主题，减小下行攻击面。

---

### 13.5 上行管道（本地 → 云端）

#### 13.5.1 七步流水线

```
[1] 候选发现      review.py/strategy.py 产出的本地条目中，Uplink Gate 扫描
                  符合白名单 fact_type 的条目（纯本地，无网络）
       │
[2] 结构化校验    条目必须能完整填入 13.4.1 schema；填不进 → 终止，不上行
       │
[3] 渗出扫描      过现有 threat_patterns scope="strict"（复用 core/threat_patterns.py）
       │
[4] 聚合风险检查  与“本会话已批准上行队列”做组合检查：若新条目与已批准条目
                  组合后可识别个人身份 → 拒绝并提示用户（启发式规则 + LLM 二审）
       │
[5] 用户审批      进 review_queue.json，payload 类型 "uplink_candidate"；
                  用户看到的就是要传的原文（结构化 YAML 明文），一字不差
       │
[6] 规范化 + 签名  canonical form → content_hash；附轮换 contributor_id +
                  信誉匿名凭证；TLS 上行
       │
[7] 云端入隔离区  语义聚类到 candidate cluster；来源计数 +1
```

#### 13.5.2 关键步骤的理由与优点

**步骤 [2] 白名单结构化是主防线**。理由见 13.4.1：黑名单检测在组合攻击面上原理性失败，白名单放行把判定从“AI 猜这条敏不敏感”变成“结构塞不塞得进”——前者是概率判断，后者是机械判断。

**步骤 [4] 聚合风险检查是对马赛克效应的正面回应**。单条过白名单不代表组合无害。实现为两级：
- 机械级：本会话已批准条目与本条目的 condition/枚举值做交集分析，若交集收敛到过小的群体（如“同城 + 同行业 + 同设备”三枚举同时命中），拒绝；
- LLM 级：把已批准队列与新条目一起给本地 LLM 问一句“这些组合能推断出此人身份吗”，仅作建议，最终判定权在用户。

*诚实声明*：这一步是启发式，不可能完备。它的定位是“抬高意外泄漏的概率门槛”，不是“保证不泄漏”。

**步骤 [5] 上行审批继承现有写入门哲学**。原设计里“LLM 是提建议的实习生，不是有写权限的管理员”；上行同理——云端是提建议的实习生，**上行审批是用户最重要的一道隐私防线**。明文展示原文，禁止“摘要式确认”（只给用户看“将上传 3 条记忆”）。

**步骤 [6] 轮换标识 + 匿名凭证**：见 13.4.3。上行内容不含任何可跨会话链接的标识。

#### 13.5.3 本地侧改动清单（与现有模块的对接）

| 现有模块 | 改动 |
|----------|------|
| `core/threat_patterns.py` | 复用，不改动；uplink 复用 `scope="strict"` |
| `evolution/approval.py` | 新增 payload 类型 `uplink_candidate`、`consensus_import`、`consensus_rollback`，复用 `_replay()` 机制 |
| `evolution/` | 新增 `uplink.py`（Gate：校验+扫描+聚合检查+审批 stage） |
| `config.yaml` | 新增 `cloud:` 配置节（见第 9 章） |
| MCP 工具 | 新增 `memory_uplink_pending()`、`memory_consensus_status()`、`memory_consensus_rollback(version)` |

**设计理由**：全部复用现有审批与扫描基础设施，云端层对本地内核（MemoryStore 双态模型、原子写、漂移检测）**零侵入**——新功能以“新 payload 类型 + 新 evolution 阶段”挂接，正是原设计 §10.2 预留的扩展点。

---

### 13.6 云端进化管线

#### 13.6.1 隔离候选区（Quarantine）

**设计**：所有上行条目先进入隔离候选区，按语义聚类成 cluster。隔离区条目有三个“永不”：**永不下发、永不直接进共识、来源数 < k 的 cluster 永不聚合出区**。

**设计理由**：这是 k-匿名晋升门槛的落点。独有记忆（k=1,2）永远锁在隔离区——独有记忆恰恰是个人隐私浓度最高的部分（你的特殊环境、你的特殊坑）；能出区的必然是 ≥k 个独立用户共有的“公共经验”，而公共经验几乎不含个人身份信息。**马赛克效应在这里被结构性对冲**：不是检测出隐私再删，而是独有内容物理上不出区。

**优点**：隐私保护不依赖任何 AI 判断的正确性；k 值 per-频道可调。

**代价**：长尾坑点（稀有但真实的问题）可能永远凑不齐 k 个报告，死在隔离区。接受此代价——宁可漏报不可错传；缓解手段是 k 值按频道风险和反馈数据动态调（冷启动期可临时 k=2，见 13.9）。

#### 13.6.2 聚合与蒸馏（Quarantine → Consensus）

**触发条件**（全部满足才启动）：
1. cluster 内独立信誉身份数 ≥ k（Sybil 聚类检测合并后，见 13.6.4）；
2. cluster 内条目 evidence_type 不全是 `inferred`（至少有 1 条 first_hand）；
3. Canary 管线当前无告警（见 13.6.5）。

**蒸馏规则**：
- LLM 从 cluster 生成共识草案，`generation = max(原料代数) + 1`，**结果 > 2 则拒绝生成**（硬约束）；
- **限定语保留校验（机械检查，不依赖 LLM 自觉）**：产物的 condition 字段必须是所有原料 condition 的**交集**而非并集；若产物丢失了任一原料的条件约束，判提炼失败。这一条直接对应 context collapse——共识坍缩的第一征兆就是限定语丢失，而条件字段的集合运算是可以机械验证的；
- 产物通过后才写入共识库，状态 `candidate`，观察期 N 天后转 `active`。

**优点**：聚合有明确的数学门槛（k、交集、代数上限），LLM 只在门槛内做文字工作；限定语校验是机械判据，可单测。

#### 13.6.3 失败蒸馏独立管道

**设计**：`fact_type = tool_pitfall | workflow_guardrail` 的上行走**独立聚类池与独立 k 阈值**（可低于普通 env_fact，如 k=2），蒸馏产物打 `guardrail` 标记，下发时在频道内排序权重上调。

**设计理由**（ReasoningBank）：成功路径大同小异，**坑各有各的坑，汇总价值最高**；失败轨迹提供的反事实信号专门用于打磨 guardrail。且坑点类记忆的“误报代价”（多一次无谓检查）远低于“漏报代价”（踩坑返工），不对称性支持更低的 k 门槛与更高的下发权重。

#### 13.6.4 Sybil 防御与信誉加权

**设计**：
1. **信誉加权计数**：cluster 的“独立来源数”不是人头数，是 `Σ reputation_weight(identity)`，新账户权重 ≈ 0；
2. **Sybil 聚类检测**：对 cluster 内上行条目做语义指纹 + 时间模式分析（行文 embedding 相似度、上报间隔分布、condition 枚举选择模式），高度相似的“独立用户”合并为一个计票单位；
3. **速率限制**：单信誉身份的 topic 级上行速率有先验上限，超限进入人工/延迟审核。

**设计理由**：联邦学习 Byzantine 文献的教训——中位数/Krum 类纯统计过滤可被精心构造的攻击（如 LIE）绕过，恶意更新可以做到与诚实更新统计不可区分。所以独立计数必须叠加**身份历史维度**（信誉）与**行为指纹维度**（聚类检测），单维度都不够。

#### 13.6.5 Canary 监控

**设计**：云端持有一批已知标准答案的“锚点事实”（如“SQLite WAL 模式的作用”），维持其在共识库中的正常表示；持续监控这些条目是否被扭曲、稀释或关联到异常 cluster。扭曲即告警：冻结相关频道的新共识晋升，进入人工审计。

**设计理由**：自适应攻击研究表明静态防御必然被绕过，所以要从“防住”切换到“检测 + 止血 + 回滚”。Canary 是渗透的烟雾报警器：攻击者不知道哪些是 canary，污染共识库时大概率会碰到锚点。

#### 13.6.6 自然选择（反馈回路 + TTL）

**设计**：
- 本地对下行共识条目的 `retrieval_count / helpful_count / unhelpful_count` 以及显式的“**对我失效**”反证标记，作为**纯计数信号**（不含任何记忆内容）随下次上行捎带回传；
- 共识条目 TTL 到期且窗口内无新鲜观测（新上行 cluster 命中）→ `degraded`；再 30 天无观测 → `archived`；
- `counter_evidence / (independent_sources + counter_evidence)` 超过阈值（默认 30%）→ `degraded` 并触发仲裁：若反证集中携带不同 condition，则**分叉**为两条各带情境标签的共识共存；否则归档。

**设计理由**：① 进化要有死亡，否则共识库只会膨胀成噪音；② context collapse 的本质是条目活得太久且无人复核，TTL 强制复核节奏；③ 分叉而非统一仲裁——同一主题下允许两条矛盾共识各带情境标签共存，由 pull 方的上下文决定取哪条，这比强行合并成一条更诚实（现实世界的“坑”经常是真的分情境的）。

**优点**：反馈信号纯计数，零隐私内容；共识库规模自限；反证有结构性出路。

#### 13.6.7 A-MEM 式链接与 refine

**设计**：共识条目间维护链接网络（共享 condition 主体、共享 evidence cluster 的条目互链）；新 cluster 晋升时，触发对链上既有共识的 refine 检查——但 refine 产物代数 +1 且受 gen ≤ 2 约束，超限则另立新条目并标记 supersede 关系，而非原地改写。

**设计理由**（A-MEM）：记忆网络的价值在新旧相互作用中增长；但在群体尺度上，原地改写是 collapse 的温床，所以用“代数约束下的新条目 + supersede 链”替代 A-MEM 的原地更新——保留进化能力，掐断坍缩路径。

---

### 13.7 下发 / Pull 管道（云端 → 本地）

#### 13.7.1 分发机制：频道订阅 + 签名 delta

**设计**：
- 用户按频道订阅；每个频道的发布是**单调递增版本号 + delta 包**（新增/修改/降级/归档的条目级增量），签名覆盖 `channel + version + delta payload`；
- 支持两种消费模式：**push**（订阅频道有新版本时通知，用户决定何时拉）与 **pull**（按需查询，可带 condition 过滤，如“只要 windows + python-packaging”）；
- 本地保存共识版本历史（默认保留最近 10 版），支持 `memory_consensus_rollback(version)` 整体回退。

**设计理由**（ACE）：ACE 的核心工程教训是**增量 delta 合并 + 确定性 merge 逻辑**，而不是全文重写——brevity bias 与 context collapse 都源于“每次重写一遍”。下发同理：整包覆盖会让本地 merge 变成 LLM 判断题（会丢细节），delta 条目级合并是确定性操作（按 consensus_id upsert），无歧义、可并行、可回放。签名 + 版本化 + 回滚则是“假设失陷”心态的直接落地：不要求每次发布都干净，要求**任何一次污染都能被检测后整体回退**。

#### 13.7.2 本地接收：五道工序

```
[1] 签名校验      hub 公钥验签；版本单调性检查（防重放旧版本）
[2] 威胁扫描      delta 内每条过 threat_patterns strict 扫描
[3] 记忆注入检测  五层检测：
                  指令覆盖 / 人格操纵 / 跨会话持久化 / 编码混淆(base64|hex|rot13) / 渗出诱导
[4] 独立命名空间  写入 TEAM.md（与 MEMORY.md/USER.md 物理分离），
                  永不覆盖个人条目；冲突时并列呈现并提示用户
[5] 审批          首次导入或高变更量版本进 review_queue；
                  常规 delta 可配 auto-import 但始终可回滚
```

**设计理由**：下行毒化（MINJA >95% 成功率、AgentPoison、Murmur）说明共识条目本身就是注入载体——它是“将被读进 LLM 上下文的文本”，必须按不可信输入处理。独立命名空间 + 数据形态呈现（注入上下文时以引用包裹、标注来源与置信度）是纵深防御：即使一条恶意共识穿过检测进了 TEAM.md，它在上下文里也是“被引用的、带置信度的外部数据”，而非与 MEMORY.md 平权的“自己的记忆”。

#### 13.7.3 上下文注入规范

共识条目注入系统提示时的固定格式：

```
<team-consensus source="C-7f3a2b" confidence="0.82" independent_sources="5">
pip 25.0.x 在 Windows 上解析含中文的本地路径会失败……（条件：windows, pip 25.0.x）
</team-consensus>
```

**设计理由**：① 权威光环的解药是透明——置信度与来源数随行呈现，让 agent（和用户）能据此加权；② 结构化包裹使其在上下文里可被程序识别，支持事后审计（13.7.4）；③ 与本地快照的冻结语义兼容：共识块在会话开始时的快照中，会话中 delta 到达只落盘不改快照（继承双态模型）。

#### 13.7.4 事后审计与回滚

**设计**：本地维护**上下文账本**（context ledger）：每次会话记录“哪些 consensus_id 进入了上下文”。当用户发现 agent 行为异常（或云端 canary 告警广播）时：
1. 按 consensus_id 回溯“最近哪些会话受过它影响”；
2. 一键 `consensus_rollback(version)` 回退 TEAM.md 到指定版本；
3. 回滚事件与涉事 id 上行（纯元数据），供云端做 MemAudit 式归因。

**设计理由**（MemAudit + 自适应攻击文献）：事前拦截必然有漏网，防御的重心必须是“漏网之后能多快归因、多快止血”。上下文账本把“是哪几条记忆进过上下文”从不可知变成一行查询——这是事后归因的物理前提，成本只是每会话记几个 id。

**代价（显式记账）**：云端 raw 证据只存哈希与聚合统计（计数、时间窗），原始条目不可读——这意味着 context collapse 对策里的“回源重归纳”实际上做不到，gen 2 到期条目只能消亡重来。**接受此代价：消亡本来就是进化的一部分**，而可读的原始轨迹一旦上云就是不可逆的隐私敞口。

---

### 13.8 关键设计决策汇总表

#### 13.8.1 进化机制决策（对应四篇核心论文）

| # | 决策 | 理由 | 依据 | 优点 | 代价/残余风险 |
|---|------|------|------|------|----------------|
| D1 | 分层晋升漏斗：个人记忆 → 隔离区 → 共识层，共识只存 insight 级抽象 | 单机记忆方案直接搬进多 agent 场景已被实证会降性能；群体层必须有自己的聚合语义 | G-Memory | 个体噪声在漏斗中被统计压掉；抽象共识可跨环境迁移 | 细粒度细节不上行，云端共识“够用但不完整”——本地原始经验仍是第一来源 |
| D2 | 共识下发用 delta 增量 + 确定性 merge，禁整包覆盖 | 全文重写是 brevity bias 与 context collapse 的温床 | ACE | merge 无歧义、可并行、可回放、可回滚 | 版本管理工程成本 |
| D3 | 提炼代数上限 gen ≤ 2，硬编码 | 群体尺度的坍缩带“团队共识”权威光环，比单机坍缩危害大且更难被质疑 | ACE（collapse）+ provenance-role collapse | 坍缩从“靠自觉防”变成“结构上不可能发生超过两代” | gen 2 到期条目消亡重来，共识库有持续重建成本 |
| D4 | 证据指针三要素（哈希/独立来源数/时间窗）为共识存在前提 | 权威光环的解药是透明；无证据链的共识不可审计也不可信 | ACE + 组织记忆理论 | 置信度可计算、可展示、可追责 | 证据只存哈希 → 无法回源重读（显式接受） |
| D5 | 失败蒸馏独立管道，guardrail 型共识下发加权 | 坑的汇总价值高于成功路径；失败信号专门打磨 guardrail；误报/漏报代价不对称 | ReasoningBank | 长尾坑点群体免疫，新成员冷启动价值最大 | 坑点误报会造成“无谓检查”开销 |
| D6 | 共识条目带 maturity 阶段标记 | 条目从操作型到组合型有成熟过程，消费方应能区分 | ReasoningBank | 排序与加权更精细 | 阶段判定目前依赖 LLM，粒度粗 |
| D7 | 共识库条目互链，新证据触发 refine，但用“新条目 + supersede”替代原地改写 | 保留记忆网络的进化性，掐断原地改写的坍缩路径 | A-MEM + ACE | 网络价值随证据增长；改写历史可审计 | 链接维护与 supersede 链的工程复杂度 |

#### 13.8.2 三坑对策决策

| # | 决策 | 对应坑 | 理由 | 依据 | 代价/残余风险 |
|---|------|--------|------|------|----------------|
| S1 | 上行白名单 schema + condition 枚举化，拒绝自由文本 | 坑一（隐私） | 黑名单检测在组合攻击面上原理性失败；结构性排除优于检测性删除 | Collaborative Memory（transformation 写入）的强化版 | 推断型隐私无完备解；受控词表需治理 |
| S2 | k-匿名晋升门槛 + 隔离区三“永不” | 坑一（隐私） | 独有记忆不出区，能下发的必是多人公共经验——马赛克效应结构性对冲 | k-匿名思想 + Collaborative Memory 分层检索 | 长尾坑点可能永远凑不齐 k，死在隔离区（显式接受） |
| S3 | 上行审批明文展示原文，继承 review_queue | 坑一（隐私） | 用户否决权是最重要的隐私防线 | memory-hub 原设计哲学 | 审批负担；自动化开关与此有张力 |
| S4 | 双层身份：长期信誉密钥（只挂分）+ 轮换内容标识 | 坑一 × 坑二交叉 | 信誉需持久身份，隐私需匿名，单层身份无解 | 联邦学习身份分层实践 | 信誉分本身是指纹，粗粒度区间披露缓解但不根除 |
| S5 | 下行一律视为不可信输入：strict 扫描 + 五层记忆注入检测 + 独立命名空间 | 坑二（毒化） | 共识条目是将被读进上下文的文本，本身就是注入载体 | MINJA（>95% 注入成功率）、AgentPoison、Murmur、OWASP AMG 分类 | 检测有漏网率——所以必须配 S6 |
| S6 | 签名 + 版本化 + 本地回滚 + 上下文账本 | 坑二（毒化） | 自适应攻击必然绕过静态防御；重心放在检测后归因与止血 | Adaptive Attacks、MemAudit | 账本与版本存储成本（很低：每会话几个 id） |
| S7 | 信誉加权计数 + Sybil 聚类检测 + 速率限制 | 坑二（毒化） | 纯统计过滤可被 LIE 类攻击绕过，必须加身份历史与行为指纹维度 | 联邦学习 Byzantine 文献 | 冷启动期信誉体系稀薄，防御弱 |
| S8 | Canary 锚点条目 + 扭曲告警冻结 | 坑二（毒化） | 渗透的烟雾报警器；攻击者不知哪些是 canary | 蜜罐思想 | canary 选题需随领域更新，运营成本 |
| S9 | 限定语保护：condition 交集机械校验，丢失即判提炼失败 | 坑三（坍缩） | 坍缩的第一征兆是限定语丢失；条件字段的集合运算可机械验证 | provenance-role collapse / typed memory | 交集过严时产物过于保守 |
| S10 | TTL + 反证计数 + 分叉仲裁（矛盾共识带情境标签共存） | 坑三（坍缩） | 坍缩的本质是条目活太久无人复核；强行统一仲裁不诚实 | 自然选择 + MemCollab 情境化 | 分叉过多时消费方选择负担 |

#### 13.8.3 三坑对策 → 模块落点速查

| 坑 | 本地模块 | 云端模块 | 分发模块 |
|----|----------|----------|----------|
| 坑一 隐私 | Uplink Gate（白名单校验/渗出扫描/聚合风险检查/审批） | 隔离区 + k 门槛；双层身份 | 频道不开“人”主题；信誉分区间披露 |
| 坑二 毒化 | 五层注入检测；独立命名空间；上下文账本；回滚 | 信誉加权 + Sybil 检测 + 速率限制；Canary | 签名 + 单调版本；告警广播 |
| 坑三 坍缩 | 注入时置信度/来源数随行呈现 | gen ≤ 2 硬约束；限定语交集校验；TTL/反证淘汰；supersede 链 | delta 增量；证据指针随行 |

---

### 13.9 未解决问题（诚实清单）

以下问题本设计**没有**解决，只做缓解。列出以防后续误判为“已覆盖”：

1. **推断型隐私无理论完备解**。白名单 + k 门槛把风险压到很低，但“多条公共经验 + 公开信息”的跨源推断永远存在。涉及个人身份的主题依赖“不开频道”这一管理手段，而非技术手段。
2. **信誉指纹化**。高信誉贡献者是小群体，信誉区间披露只能提高识别成本，不能消除。长期看可能需要更强的匿名凭证方案（如群签名），工程复杂度高，v0.1 不纳入。
3. **冷启动困境**。系统早期用户少：k 门槛凑不齐 → 共识稀少 → 下行价值低 → 新用户不愿开遥测。缓解：冷启动期允许运营方以“种子共识”（人工审核的公共知识，如官方文档勘误）灌库，并临时下调 k 至 2；但这引入运营方信任假设，必须在文档中向用户显性声明。
4. **受控词表治理**。subject/predicate/condition 的枚举词表需要持续维护（新工具、新版本段），词表更新本身需要版本化与社区流程，否则白名单会腐烂成“什么都填不进去”或“什么都算枚举值”。
5. **语义聚类的对抗操纵**。攻击者可构造与合法 cluster 语义接近的恶意条目“搭车”晋升。聚类算法本身成为攻击面，目前只有信誉加权 + canary 的间接防御。
6. **回源不可达**。证据只存哈希，gen 2 条目到期只能消亡重建；若某领域上行源持续流失，对应共识会“灭绝”——这是隐私优先选择的必然代价。
7. **maturity 阶段判定**依赖 LLM，目前无可机械验证的判据。

---

### 13.10 分阶段实施路线

每个里程碑独立可用、独立可验，且随时可以停在任何一个里程碑而不损害现有功能。

| 里程碑 | 内容 | 验收标准 | 依赖 |
|--------|------|----------|------|
| **M1 本地上行门** | 白名单 schema 定义与校验器；`evolution/uplink.py`；审批 payload `uplink_candidate`；config `cloud.uplink` | 无云端也能跑：上行条目被生成、校验、审批、打包到本地 outbox | 现有 approval.py |
| **M2 云端聚合最小核** | 隔离区存储；规范化哈希聚类；k 门槛晋升；共识库 schema；只读 API | 模拟 N 个客户端上行，验证 k 门槛与聚类正确性 | M1 |
| **M3 下行链路** | 频道订阅；delta 生成与签名；本地 TEAM.md 命名空间；五层扫描；`consensus_import` 审批 | 端到端：上行 → 共识 → 下发 → 注入上下文，全程审批可见 | M2 |
| **M4 反馈与自然选择** | 反馈计数回传；TTL/降级/归档状态机；反证分叉仲裁；上下文账本 + 回滚 | 构造反证场景验证降级与分叉；回滚后上下文账本能列出受影响会话 | M3 |
| **M5 防御强化** | 双层身份与匿名凭证；信誉加权；Sybil 聚类检测；Canary 管线 | 红队演练：模拟 Sybil 上行毒化与下行注入，验证检测/止血/回滚闭环 | M4 |

**顺序理由**：M1–M3 是价值闭环（先让系统转起来），M4 是健康闭环（让系统能老能死），M5 是生存闭环（让系统挨打不死）。把最重的身份密码学放 M5 是有意的——先用简单身份把数据流跑通，Sybil 防御在真实流量上调，比在空系统上设计有效得多。

---

## 14. 后续可展开的创意方向（供 AI 参考）

基于以上完整设计（本地 + 云端），可进一步思考并发表文章的切入点：

1. **“文件优先的长期记忆”**：为什么用 markdown + § 分隔，而不是向量数据库？讨论可审计性、人类可读性、版本控制、与 LLM 上下文的张力。
2. **“冻结快照 vs 实时态”**：前缀缓存工程与记忆新鲜度之间的权衡。
3. **“MCP 作为 agent 间共享记忆的总线”**：stdio / HTTP 双传输、多客户端共享同一 home 的架构意义。
4. **“本地优先的 RAG 替代”**：holographic provider 的 HRR 向量代数、FTS5 + 信任分数 + 时间衰减的混合检索设计。
5. **“记忆的免疫系统”**：threat patterns、漂移检测、写入门、审批制进化如何构成多层防御。
6. **“从对话到策略”**：strategy extraction 如何把重复的用户纠正固化为可复用策略文档。
7. **“记忆治理的伦理与控制权”**：为什么 curator 不自动执行、归档可恢复、进化默认手动——用户最终保有对记忆的否决权。
8. **“AI 的记忆如何像人类记忆一样工作”**：短期上下文、长期文件记忆、结构化事实库、策略文档、归档，分别对应人类记忆的哪些层面。
9. **“Memory-hub 作为个人知识基础设施”**：它不只是 agent 的记忆，也是用户个人事实、偏好、项目约定的长期仓库。
10. **“可插拔 provider 的开放设计”**：如何为 memory-hub 写一个自己的 provider（示例：Notion、Obsidian、Zotero、本地向量库）。
11. **“从对话到 Skill”**：evolution 如何从 transcript 和 skill trace 中发现能力缺口，自动生成或更新 SKILL.md。
12. **“Trace 驱动的技能进化”**：用 skill 执行轨迹（成功/失败/修正）持续 refine skill 指令，类似人类的“反思性练习”。
13. **“Session Experience 作为短期记忆”**：在 `MEMORY.md` 长期记忆与一次性上下文之间增加一层可过期的 experience 层。
14. **“记忆的层次结构”**：短期 experience → 长期 memory / strategy → 可复用 skill，三层之间的晋升与退化机制。
15. **“群体记忆的统计学”**：k-匿名、信誉加权、Sybil 检测如何把“多人共有经验”从噪声中显影为可信共识。
16. **“云端共识的权威光环”**：为什么证据指针、限定语校验、反证机制是抵抗“团队共识”盲目信任的解药。
17. **“记忆的进化伦理”**：当 AI 开始从群体中学习和遗忘，用户如何保留对“自己 agent 知道什么”的最终控制权。

---

## 15. 关键代码索引

| 文件 | 职责 |
|------|------|
| `memory_hub/paths.py` | home 目录解析 |
| `memory_hub/config.py` | 配置加载、默认模板 |
| `memory_hub/core/store.py` | MemoryStore 内核 |
| `memory_hub/core/threat_patterns.py` | 注入/渗出扫描库 |
| `memory_hub/providers/base.py` | MemoryProvider ABC |
| `memory_hub/providers/manager.py` | ProviderManager |
| `memory_hub/providers/holographic/__init__.py` | HolographicMemoryProvider |
| `memory_hub/providers/holographic/store.py` | SQLite 事实库 |
| `memory_hub/providers/holographic/retrieval.py` | 多策略检索 |
| `memory_hub/providers/holographic/holographic.py` | HRR 向量代数 |
| `memory_hub/evolution/llm_client.py` | OpenAI 兼容 client |
| `memory_hub/evolution/transcripts.py` | 转录与水印 |
| `memory_hub/evolution/review.py` | 记忆自省 |
| `memory_hub/evolution/strategy.py` | 策略提取 |
| `memory_hub/evolution/curator.py` | 记忆治理 |
| `memory_hub/evolution/approval.py` | 审批队列 |
| `memory_hub/evolution/uplink.py` * | 云端上行门、共识同步、上下文账本（新增） |
| `memory_hub/server/mcp_server.py` | FastMCP 服务与工具 |
| `memory_hub/cli.py` | 命令行入口 |

\* 云端扩展新增模块。

---

## 16. 总结

memory-hub 的完整设计可以压缩为三层理解：

1. **本地层是根基**：文件优先、本地优先、审批制进化，所有核心安全机制（写入扫描、漂移检测、原子写、写入门、双态快照）都在这里。云端不是替代本地，而是本地之上可选的群体扩展。

2. **云端层是群体放大的进化循环**：它把单机自省扩展为“上行 → 隔离 → 聚合 → 共识 → 下发 → 反馈”的完整群体进化。进化机制来自论文（G-Memory、ACE、ReasoningBank、A-MEM），防御机制来自攻击文献（Murmur、MINJA、AgentPoison、MemAudit、自适应攻击、Byzantine 鲁棒性）。

3. **三个核心风险的对策全是结构，不是检测**：
   - **隐私（坑一）**：白名单 schema + k-匿名隔离区 + 双层身份 + 用户审批；
   - **毒化（坑二）**：strict 扫描 + 五层注入检测 + 签名版本化 + 上下文账本 + 一键回滚 + Canary 监控 + Sybil 防御；
   - **context collapse（坑三）**：gen ≤ 2 硬约束 + condition 交集校验 + TTL/反证淘汰 + supersede 链。

贯穿始终的是同一个立场：**用户否决权不可让渡**。上行要审批，下行要审批，自动化全是 opt-in，一切可回滚。云端进化只是把 memory-hub “LLM 是实习生”的哲学放大到群体尺度——放大之后，每条原则都更需要是机制，而不是约定。

---

*文档版本：基于 memory-hub v0.1.0 源码与云端进化层 v0.1 草案整合。*
