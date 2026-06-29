# Cloud Skill Evolution Platform — 云端协同进化平台

> 基于 **CoEvoSkills** + **SkillMAS** + **CoMAS** 三篇论文构建的云端 Skill 协同进化服务。  
> 本地 Agent 持续上报执行证据，云端执行有界进化、社区评审与组织重组，再将候选 Skill 下发回本地进行 Shadow A/B 测试与自动晋升。

---

## 一、系统架构总览

```text
┌─────────────────────────────────────────────────────────────────┐
│                         云端 (Cloud)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Trace Ingest │→ │ Utility Learn│→ │ Retained Evidence    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│           ↓                                              ↓      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Skill Evolution (核心进化循环)                │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐  │  │
│  │  │Bounded      │→   │ CoEvoSkills  │→   │ Validation  │  │  │
│  │  │Diagnosis    │    │ Generator↔Verifier↔Oracle   │    │  │  │
│  │  └─────────────┘    └──────────────┘    └─────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│           ↓                              ↓                      │
│  ┌──────────────┐              ┌──────────────────┐            │
│  │ CoMAS Review │              │ MAS Restructuring│            │
│  │ (社区评审层)  │              │ (组织重组)        │            │
│  └──────────────┘              └──────────────────┘            │
│           ↓                                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Candidate Deployer (候选下发)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ SSE / Poll
┌─────────────────────────────────────────────────────────────────┐
│                        本地 (Local Agent)                        │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────────────┐   │
│  │Trace     │→ │ Cloud Sync  │→ │ Skill Bank               │   │
│  │Collector │  │ (上报/拉取)  │  │ (active/shadow/archive)  │   │
│  └──────────┘  └─────────────┘  └──────────────────────────┘   │
│                                          ↓                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Shadow A/B Test & Promotion                 │  │
│  │  主链路使用 active skill，后台并行执行 shadow skill 对比   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、核心进化流程

整个协同进化是一个**持续闭环**，每一轮称为一个 **Adaptation Round**。

### Step 1 — Trace 采集与上报（本地）

本地 Agent 在执行任务时，由 `TraceCollector` 采集详细的执行证据：

- **used_skills**：实际执行支持的 skill（严格区分于仅被检索的 skill）
- **participating_executors**：参与执行的执行器
- **success / error_type / outcome_reward**：执行结果与错误分类
- **context / execution**：任务上下文与执行细节

Traces 经 gzip 压缩后批量上报云端（失败时写入本地 SQLite 队列，稍后重试）。

### Step 2 — Trace Ingestion（云端接收）

云端 `TraceIngestionService` 接收 traces 后执行三个动作：

1. **持久化**：写入 `traces` 表，原始证据永久保存
2. **Utility Learning**：调用算法引擎更新 Skill 与 Executor 的 Q-value
3. **构造 Retained Evidence Set**：按优先级筛选值得保留的失败证据
   - 重复失败（同一 skill + task_type 连续失败 ≥ 2 次）
   - 接近成功（outcome_reward 在 0.3~0.8）
   - 检索/执行不匹配（retrieved 与 used_skills 不一致）

### Step 3 — Bounded Diagnosis（有界诊断）

对 retained failure traces，`BoundedSkillEvolution` 引擎执行有界诊断：

| 错误类型 (error_type) | BoundedTag | 含义 |
|---|---|---|
| hallucination_prop | `add-guard` | 添加边界守卫 |
| wrong_order | `reorder-step` | 调整动作顺序 |
| retrieval_mismatch | `tighten-retrieval` | 收紧检索范围 |
| skill_overload | `split-skill` | 拆分 skill |
| routing_ambiguity | `handoff-to-structure` | 移交结构层重组 |

- **有界**：只能产出 **6 个枚举值**，禁止随意扩展，防止 Skill 库无限膨胀
- **可识别性**：只有当错误可被 uniquely identify 时，才进入下一步进化

### Step 4 — CoEvoSkills 进化循环（核心）

对需要进化的 skill，启动 **Generator ↔ Surrogate Verifier ↔ Oracle** 的协同循环：

```
初始化: S = 当前skill, V = 空测试集, n = 0, r = 0, R_best = 0

while n < N_max 且 r < M_max:
    1. 执行当前 skill S，获取输出 x
    
    2. 检查上下文是否溢出（feedback history 超过阈值则退出）
    
    3. Surrogate Verifier 评估:
       - 信息隔离：Verifier 只看 Task Instruction I 和 Output x
       - 生成测试断言 V，检查正确性
       - 若未全通过 → 输出诊断 F，Generator 基于 (S, C, F) 精炼 skill，r += 1，继续循环
    
    4. 若 Surrogate 全通过 → 进入 Ground-Truth Oracle:
       - Oracle 在 fresh env 中执行，只返回 pass/fail bit（不泄露测试内容，防过拟合）
       - 若 reward == 1.0 → 完美通过，提前退出
       - 若 reward > R_best → 更新最佳快照 S_best
       - Verifier Escalation：升级测试套件 V，n += 1

循环结束，将 S_best 存入 Validation Pool (P_r)
```

### Step 5 — CoMAS Review（社区评审）

进入 Validation Pool 的 candidate skill 会接受社区评审：

1. 每个 reviewer 提交 **solution proposal**（改进建议）
2. 其他 reviewer 提交 **evaluation**（批判性评审）
3. 独立 scorer 对 (solution, evaluation) 打分（1~3 分）
4. 计算零和奖励：
   - `r(solution) = (score - 1) / 2`
   - `r(evaluation) = (3 - score) / 2`

评审结果作为 Utility Learning 的辅助奖励信号，不直接修改 skill。

### Step 6 — MAS Restructuring（组织重组）

当 Bounded Diagnosis 的 tag 为 `handoff-to-structure`，或满足以下条件时，触发 MAS 组织重组：

- 存在低效用 Executor（Q-value < 阈值）
- 某 task_type 集中失败 ≥ 3 次且已有 skill update
- Skill 重叠度 > 0.8（冗余严重）

**重组决策**：`{keep, add, merge/remove, modify}`，每轮最多执行一次，防止拓扑震荡。

### Step 7 — Candidate Deploy（候选下发）

Validation Pool 中验证通过的 candidate skill 被封装为 payload，通过 SSE 或轮询接口下发给本地 Agent：

- 包含：skill 内容、parent_version、diff、confidence、测试用例
- `target_agent=None` 时全量广播

### Step 8 — Shadow A/B Test & Promotion（本地）

本地 Agent 接收 candidate 后注入 **shadow** 状态：

1. **Shadow 执行**：主链路继续使用 active skill（不阻塞用户），后台并行执行 shadow skill
2. **A/B 对比**：收集 success rate、latency、output 等指标
3. **晋升条件**（同时满足）：
   - 绝对提升 > 15%
   - 置信度 > 0.9
   - 连续执行 ≥ 10 次无异常
4. **晋升**：shadow → active，原 active → archive
5. **回滚**：若 active 连续失败 ≥ 3 次，紧急回滚到最近 archive 版本

---

## 三、核心算法原理

### 3.1 Utility Learning（SkillMAS Eq.3-4）

基于 Q-learning 的效用更新，用于评估 Skill 和 Executor 在特定 task_type 上的表现：

```
Q⁺(x, z) ← Q(x, z) + α · (R(ξ) - Q(x, z))

其中:
  α = 1 / (1 + N(x, z))     // 自适应步长，N 为历史更新次数
  R(ξ) = 1.0 (success) / 0.0 (failure)
```

- **Skill Utility**：按 `used_skills` 分配信用（credit assignment）
- **Executor Utility**：按 `participating_executors` 分配信用
- 步长 α 随更新次数递减，确保老 skill 评分稳定、新 skill 快速收敛

### 3.2 Bounded Evolution（SkillMAS Section 2.3）

将开放式 skill 修改约束为 **6 种原子操作**（BoundedTag）：

```
Action Set = {add-guard, reorder-step, tighten-retrieval, split-skill, handoff-to-structure, empty}
```

优势：
- 防止 LLM-based Generator 无限制膨胀 prompt
- 保证 Skill 库可收敛（convergence guarantee）
- 诊断 → 标签 → 定向修改，形成可解释、可追溯的进化链路

### 3.3 Surrogate Verifier（CoEvoSkills Algorithm 1）

**信息隔离设计**是核心：

- Verifier 的 Prompt **绝不包含 Generator 的内部推理状态**
- Verifier 只看到：Task Instruction + Output Files + Previous Tests
- 目的：防止 confirmation bias（Verifier 继承 Generator 的错误）

**Verifier Escalation**：当 surrogate pass 但 oracle fail 时，Verifier 独立升级测试套件，使下一轮验证更严格。

### 3.4 Oracle 信号设计

- Oracle 是 ground-truth 评估器（通常是人类验证或真实环境执行）
- **只返回 pass/fail bit**，不返回具体测试内容
- 目的：防止 Generator 过拟合到 held-out test set

---

## 四、核心数据模型

| 表/概念 | 对应论文符号 | 说明 |
|---|---|---|
| `traces` | T_r | 原始执行证据 |
| `skills` | L_r | Skill 库（active / candidate / archive）|
| `skill_utility` | Q^s_r | Skill 效用表 (skill_id × task_type → q_value) |
| `executor_utility` | Q^a_r | Executor 效用表 (executor_id × task_type → q_value) |
| `validation_pool` | P_r | 验证池，存放进化后的 candidate skill |
| `executors` | A_r | MAS 组织架构 |
| `skill_reviews` | — | CoMAS 社区评审记录 |
| `candidate_queue` | — | 待下发候选队列 |

**SQLite 约定**：所有数组字段（`used_skills`, `bounded_tags`, `owned_skills`, `scripts`）均使用 **JSON 字符串** 存储。

---

## 五、关键约束（设计红线）

| 约束 | 位置 | 违反后果 |
|---|---|---|
| `used_skills` 必须只含实际执行支持的 skill | TraceCollector | Utility Learning 信用分配错误，skill 排名失真 |
| Surrogate Verifier Prompt 不得包含 Generator 内部状态 | `_build_isolated_prompt` | Confirmation bias，Verifier 继承 Generator 错误 |
| Oracle 只返回 pass/fail bit，不泄露测试内容 | `_ground_truth_oracle` | Generator 过拟合到 held-out tests |
| α = 1/(1+N)，禁止固定步长 | `UtilityLearningEngine.batch_update` | 老 skill 评分剧烈波动，系统不稳定 |
| Bounded Tag 只能是 6 个枚举值 | `BoundedSkillEvolution.diagnose_trace` | Skill 库无限膨胀，无法收敛 |
| MAS 重组每轮最多一次 | `MASRestructuringService.evaluate_restructure` | 组织拓扑震荡，路由混乱 |
| Shadow 执行不阻塞主链路 | `ShadowRunner.run_ab_test` | 影响用户正常请求响应时间 |
| 本地队列持久化（失败重试）| `CloudSyncClient._enqueue_for_retry` | Trace 丢失，进化信号不完整 |

---

## 六、技术栈

- **Runtime**: Python 3.12 + FastAPI
- **Database**: SQLite (aiosqlite)，文件型轻量方案
- **LLM**: Kimi API / DeepSeek（驱动 Generator 与 Verifier）
- **本地存储**: JSON + SQLite（Skill Bank、Trace 队列、A/B 历史）

---

## 七、参考文献映射

| 模块 | 论文 | 章节/算法 |
|---|---|---|
| Utility Learning | SkillMAS | Section 2.2, Eq.3-4 |
| Bounded Evolution | SkillMAS | Section 2.3 |
| Retained Evidence | SkillMAS | Section 2.1 |
| MAS Restructuring | SkillMAS | Section 2.4, Alg.1 (Appendix A.1) |
| Surrogate Verifier | CoEvoSkills | Algorithm 1, Section 3.3 |
| CoMAS Review | CoMAS | Section 3.1-3.2, Fig.2 |

---

> **一句话总结**：本地 Agent 持续上报 `Trace`，云端基于 `Utility Learning` 评估、`Bounded Evolution` 诊断、`CoEvoSkills` 循环进化、`CoMAS` 社区评审，最终将候选 Skill 下发回本地进行 `Shadow A/B 测试` 与自动晋升——形成一个完整的**云-端协同进化闭环**。
