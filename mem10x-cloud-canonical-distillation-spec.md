# mem10x Cloud Canonical Distillation Spec
## Async Collaborative Canonical Distillation (ACCD) v1.0

> **Status**: Stable Baseline  
> **Date**: 2026-06-29  
> **Scope**: Cloud Pod 内部处理流程，定义从 PR 捞取 sense/experience 到生成 canonical sense 的完整算法  
> **Based on**: Meta-Team (arXiv:2605.29790) + Mem²Evolve (arXiv:2604.10923)，经批判性修正适配 mem10x 异步分布式场景

---

## 1. 设计定位

### 1.1 要解决的问题
Local agent 的 sense/experience 经过本地验证后，通过 Git PR 异步上传到云端。Cloud Pod 需要：
- **消除 agent-specific bias**（不同 agent 的模型、项目、习惯带来的噪声）
- **提取跨 agent 的 reasoning invariants**（稳定不变的判断规则）
- **检测并仲裁冲突**（同一场景下不同 agent 的矛盾策略）
- **生成可回推的 canonical sense**（权威版本，供所有 local agent 拉取）

### 1.2 非目标
- **不处理实时协作**：mem10x 是异步生态，Cloud Pod 不做 Meta-Team 式的同步 post-task 通信
- **不替代本地验证**：Cloud 只做"交叉验证"，不做"首次验证"
- **不强制覆盖本地**：canonical sense 以建议形式回推，local agent 保留拒绝权
- **不处理 DKD 规则生成**：DKD 进化由独立模块处理，本 spec 只输出 `dkd-candidates/`

---

## 2. 输入/输出契约

### 2.1 输入（Ingestion Payload）

Cloud Pod 从指定 Git PR 或目录捞取，输入为 JSONL 格式：

```json
{
  "batchId": "2026-06-29T00:00:00Z",
  "sources": [
    {
      "agentId": "agent-xyz",
      "team": "team-alpha",
      "senses": [
        {
          "id": "sense-xyz-1",
          "semanticType": "sense",
          "skillFamily": "frontend-react",
          "trigger": "用户要求生成 React 组件且未指定 props 类型",
          "obligation": "优先使用 TypeScript interface 定义 props",
          "antipattern": ["直接使用 any 类型"],
          "boundary": {"applicable": ["新项目"], "notApplicable": ["legacy 项目"]},
          "localMetrics": {
            "recallCount": 5,
            "validationScore": 0.85,
            "utilityScore": 0.82,
            "conflictScore": 0.15
          },
          "lineage": ["task-abc", "task-def"],
          "submittedAt": "2026-06-28T10:00:00Z"
        }
      ],
      "experiences": [
        {
          "id": "exp-xyz-1",
          "semanticType": "experience",
          "skillFamily": "frontend-react",
          "title": "Common Pitfalls When Parsing Inconsistent CSV Data Formats",
          "whatWentWrong": "...",
          "rootCause": "...",
          "correctiveActions": "...",
          "warningSigns": "...",
          "relatedSenseId": "sense-xyz-1",
          "localMetrics": {
            "recallCount": 3,
            "validationScore": 0.90
          }
        }
      ]
    }
  ]
}
```

### 2.2 输出（Canonical Package）

Cloud Pod 处理完成后，推送到 Git 的输出目录结构：

```
cloud-output/
├── canonical/
│   └── {skillFamily}/
│       └── {canonical-id}.canonical.json      # 权威 sense
├── variations/
│   └── {skillFamily}/
│       └── {canonical-id}.variations.json     # 差异化边界变体
├── conflicts/
│   └── {batch-date}/
│       ├── unresolved.json                    # 需人工仲裁的冲突
│       └── resolved.json                      # 自动仲裁记录
└── dkd-candidates/
    └── {batch-date}/
        └── {canonical-id}.dkd-candidate.json   # 供 DKD 进化模块消费
```

---

## 3. 核心算法：七阶段流水线

### 阶段 1：Quality Gate（质量过滤）

**目标**：清洗输入，过滤掉低质量内容，避免污染 canonical sense。

**过滤规则**：

| 规则 | 阈值 | 动作 | 理论依据与修正 |
|------|------|------|-------------|
| `validationScore < 0.7` | 0.7 | 丢弃 | Mem²Evolve 的 Self-Correction Loop 本地门槛通常为 0.5；**修正**：多 agent 交叉场景下，门槛提高到 0.7，防止弱证据污染全局 |
| `recallCount < 3` | 3 | 丢弃 | Meta-Team 的 evidence 来自同一任务的多次执行；**修正**：独立 agent 的 sense 需要至少 3 次独立验证，排除偶然成功 |
| Schema 校验失败 | — | 丢弃 | 基础校验，确保 downstream 处理安全 |
| `semanticType` 不在白名单 | — | 丢弃 | 仅处理 `sense` 和 `experience` |

**输出**：`filtered_senses[]`, `filtered_experiences[]`

---

### 阶段 2：Trigger Clustering（按场景聚类）

**目标**：将描述同一类场景的 sense 分到一组，为后续对比蒸馏做准备。

**算法**：

```
1. 提取每个 sense 的 trigger 文本
2. 用 embedding 模型编码（推荐 text-embedding-3-large 或同等质量模型）
3. 聚类：
   - 首选 HDBSCAN，min_cluster_size=2, metric='cosine'
   - 备选 K-Means（当数据量 < 20 时），距离阈值 0.25
4. 对每个聚类 C_k，生成聚类中心语义描述（由 LLM 生成最通用的 trigger 描述）
5. 噪声点（未聚类的 sense）单独标记，进入 `variations/` 的 orphan 目录
```

**示例**：

```
Cluster A: "React 组件 props 类型定义"
  - sense-1 (agent-xyz): "生成 React 组件且未指定 props 类型"
  - sense-2 (agent-abc): "用户要求创建 React FC 组件"
  - sense-3 (agent-def): "React 函数组件缺少 TypeScript 类型"

Cluster B: "API 错误处理"
  - sense-4: ...
```

**批判性修正**：
- Meta-Team 的聚类是天然任务分解（Planner/Developer/Reviewer 的角色天然对应），无需额外计算。
- mem10x 的 agent 是独立工作者，没有共同任务结构，**必须显式语义聚类**。
- 不能用简单文本匹配（不同 agent 措辞差异大），必须用向量嵌入。

---

### 阶段 3：Intra-Cluster Contrastive Distillation（聚类内对比蒸馏）——核心阶段

**目标**：在每个聚类内部，对比多个 agent 的 sense，提取**稳定不变式**（invariants）和**冲突**。

**输入**：聚类 $C = \{s_1, s_2, ..., s_n\}$，每个 $s_i$ 包含 `{trigger, obligation, antipattern, boundary, validationScore, recallCount}`

#### Step 3.1：Obligation Invariant Extraction

**算法**：

```python
def extract_obligation_invariants(C):
    obligations = [s.obligation for s in C]
    invariants = []
    variants = []
    
    for obligation in obligations:
        compatible_count = 0
        for o in obligations:
            if llm_judge_compatibility(obligation, o):  # LLM 判断逻辑兼容性
                compatible_count += 1
        
        agreement_rate = compatible_count / len(obligations)
        
        if agreement_rate >= 0.7:  # 70% 共识阈值
            invariants.append({
                "obligation": obligation,
                "agreementRate": agreement_rate,
                "supporters": [s.agentId for s in C if llm_judge_compatibility(obligation, s.obligation)]
            })
        elif agreement_rate >= 0.4:
            variants.append({
                "obligation": obligation,
                "agreementRate": agreement_rate,
                "note": "soft recommendation"
            })
    
    # 去重：如果多个 obligation 逻辑等价，合并为一条
    invariants = deduplicate_by_semantic_equivalence(invariants)
    
    return invariants, variants
```

**LLM 兼容性判断 Prompt 模板**：

```
You are a logic compatibility judge. Compare two rules and determine if they are logically compatible.

Rule A: {obligation_a}
Rule B: {obligation_b}

Answer in JSON:
{
  "compatible": true/false,       // 是否逻辑兼容（可同时成立）
  "entails": "A->B" | "B->A" | "none", // 是否有蕴含关系
  "overlap": 0.0-1.0,             // 语义重叠度
  "reasoning": "..."
}
```

**示例输出**：

| Obligation | 同意 Agent 数 | 共识率 | 分类 |
|-----------|-------------|--------|------|
| "使用 interface 定义 props" | 3/3 | 100% | **Invariant (Hard)** |
| "优先使用 interface 而非 type" | 2/3 | 67% | **Invariant (Hard)** |
| "为可选字段添加 ? 标记" | 2/3 | 67% | **Invariant (Hard)** |
| "使用 React.FC 泛型" | 1/3 | 33% | **Variant (Soft)** |

#### Step 3.2：Antipattern Invariant Extraction

与 Step 3.1 同理，提取被多数 agent 标记的 antipattern。

**特殊处理**：
- 如果某 antipattern 只出现在 failure experience 中，但无对应 sense 的 obligation，**标记为 DKD Candidate**（说明这是"禁止行为"，但"正确行为"尚未形成共识）。

#### Step 3.3：Boundary Merge & Conflict Detection

**算法**：

```python
def process_boundaries(C):
    boundaries = [s.boundary for s in C]
    
    # 3.3.1 合并互补边界
    merged = merge_complementary(boundaries)
    # 例如："仅适用于新项目" + "适用于 legacy 项目（已有约定除外）" → 合并为完整边界描述
    
    # 3.3.2 检测冲突边界
    conflicts = []
    for i, b1 in enumerate(boundaries):
        for j, b2 in enumerate(boundaries[i+1:], i+1):
            if llm_judge_boundary_conflict(b1, b2):
                conflicts.append({
                    "type": "boundary-conflict",
                    "senseA": C[i].id,
                    "senseB": C[j].id,
                    "boundaryA": b1,
                    "boundaryB": b2,
                    "severity": "elevate" if abs(C[i].validationScore - C[j].validationScore) > 0.2 else "arbitrate"
                })
    
    return merged, conflicts
```

**冲突类型判定**：

| 冲突类型 | 判定标准 | 示例 |
|---------|---------|------|
| **边界互补** | 同一 trigger，boundary 不同但不重叠 | "仅适用于新项目" vs "适用于 legacy 项目" |
| **策略优劣** | 同一 trigger + 重叠 boundary，obligation 不同，validationScore 差异 > 0.2 | "必须用 interface" vs "可以用 type"，分数 0.9 vs 0.6 |
| **策略平局** | 同上，但分数差异 < 0.2 | 分数 0.85 vs 0.82，无法自动判定 |
| **本质矛盾** | obligation 逻辑互斥 | "必须用 A" vs "禁止用 A" |

**批判性修正**：
- MemCollab 的对比蒸馏基于**同一任务的执行轨迹**（有 ground truth 对齐，可以逐 token 对比），提取的是 agent-agnostic 的 reasoning constraints。
- mem10x 的对比基于**不同 agent 的 sense**（没有共同任务，只有语义相似），所以**不能用逐 token 对比**，必须用**结构化 slot 的语义兼容性判断**（LLM-as-a-Judge）。
- Meta-Team 的协作是同步的、有共同目标的（agents 一起完成任务），证据交换是实时的。mem10x 的协作是异步的、独立目标的，所以"证据交换"被替换为**结构化 sense 的交叉验证**（用 LLM 判断兼容性，而非对话协商）。

---

### 阶段 4：Canonical Sense Generation（规范 Sense 生成）

**目标**：将每个聚类的蒸馏结果，打包成一个 canonical sense。

**生成规则**：

| 字段 | 生成方式 | 说明 |
|------|---------|------|
| `id` | `{skillFamily}-{slug}-{version}` | 如 `frontend-react-props-typing-v3` |
| `trigger` | LLM 生成聚类中心语义 | 最通用的场景描述，覆盖聚类内所有 trigger 的交集 |
| `obligation.hard` | Step 3.1 的 invariants（共识率 ≥ 70%） | 必须遵守的规则 |
| `obligation.soft` | Step 3.1 的 variants（共识率 40%-70%） | 建议性规则 |
| `antipattern` | Step 3.2 的 invariants | 禁止行为 |
| `boundary` | Step 3.3 的 merged boundaries | 适用/不适用边界，标注来源 agent |
| `confidence` | 加权平均 | `Σ(recallCount_i × validationScore_i) / Σ(recallCount_i)` |
| `lineage` | 溯源数组 | 所有贡献 sense 的 ID |
| `version` | 批次日期 | 如 `2026-06-29` |

**示例输出**：

```json
{
  "id": "frontend-react-props-typing-v3",
  "skillFamily": "frontend-react",
  "semanticType": "sense",
  "trigger": "为 React 函数组件定义 props 类型",
  "obligation": {
    "hard": [
      "使用 TypeScript interface 定义 props 结构",
      "为可选字段添加 ? 标记"
    ],
    "soft": [
      "优先使用 interface 而非 type alias"
    ]
  },
  "antipattern": [
    "使用 any 作为 props 类型",
    "完全省略 props 类型定义"
  ],
  "boundary": {
    "applicable": ["新项目", "TypeScript 项目"],
    "notApplicable": ["纯 JavaScript 项目", "legacy 项目（已有约定除外）"],
    "sources": {
      "新项目": ["agent-xyz", "agent-abc"],
      "legacy 项目": ["agent-def"]
    }
  },
  "confidence": 0.87,
  "lineage": ["sense-xyz-1", "sense-abc-2", "sense-def-3"],
  "version": "2026-06-29"
}
```

---

### 阶段 5：Synthetic Validation（合成验证）

**目标**：对生成的 canonical sense 进行"思维实验"，验证其在未见过场景下的有效性。

**参考来源**：Mem²Evolve 的 LLM-as-a-Judge（Trajectory Evaluation）。

**批判性修正**：
- Mem²Evolve 的 Judge 评估的是**实际执行轨迹**（有明确的 task 和 answer，可以运行验证）。
- canonical sense 是**判断规则**，没有实际执行。所以需要**合成测试场景**（synthetic scenarios）来验证。

**算法**：

```python
def synthetic_validate(canonical_sense, num_scenarios=10):
    """
    返回: (pass_rate, details[])
    """
    # 1. 生成测试场景
    scenarios = llm_generate(
        prompt=f"""
基于以下 trigger 和 obligation，生成 {num_scenarios} 个不同的测试场景。
要求场景覆盖典型情况、边界情况和潜在陷阱。

Trigger: {canonical_sense.trigger}
Obligation: {json.dumps(canonical_sense.obligation)}
Antipattern: {json.dumps(canonical_sense.antipattern)}
Boundary: {json.dumps(canonical_sense.boundary)}

输出 JSON 数组，每个元素包含：scenario, expected_behavior
""",
        temperature=0.8
    )
    
    results = []
    for scenario in scenarios:
        # 2. 让 LLM 应用 canonical sense 的 obligation
        applied = llm_apply_sense(
            scenario=scenario["scenario"],
            sense=canonical_sense
        )
        
        # 3. 检查是否违反 antipattern
        violation = check_antipattern_violation(applied, canonical_sense.antipattern)
        
        # 4. 检查决策合理性
        reasonableness = llm_judge(
            prompt=f"""
场景: {scenario["scenario"]}
Agent 决策: {applied}
预期行为: {scenario["expected_behavior"]}

这个决策是否合理？是否遵循了规则？
输出: {{"reasonable": true/false, "reasoning": "..."}}
"""
        )
        
        results.append({
            "scenario": scenario["scenario"],
            "violation": violation,
            "reasonableness": reasonableness["reasonable"],
            "reasoning": reasonableness["reasoning"]
        })
    
    pass_rate = sum(1 for r in results if not r["violation"] and r["reasonableness"]) / len(results)
    
    return pass_rate, results
```

**判定标准**：

| 结果 | 动作 |
|------|------|
| `pass_rate >= 0.8` | 进入 Stage 6 |
| `pass_rate < 0.8` | 标记为 `rejected`，写入 `conflicts/{batch}/rejected-by-synthetic.json`，不进入 canonical |

---

### 阶段 6：Conflict Arbitration（冲突仲裁）

**目标**：处理阶段 3 检测到的冲突，决定最终 canonical sense 的形态。

**仲裁策略表**：

| 冲突类型 | 判定标准 | 仲裁策略 | 输出 |
|---------|---------|---------|------|
| **边界互补** | 同一 trigger，boundary 不同且不重叠 | **Split**：生成多个 canonical sense，各带不同 boundary | `canonical/` 下多个文件 + `variations/` 记录关联 |
| **策略优劣** | 同一 trigger + 重叠 boundary，obligation 不同，validationScore 差异 > 0.2 | **Elevate**：选择高分版本作为 canonical，低分版本进入 `variations/` | 单 canonical + 变体记录 |
| **策略平局** | 同上，但分数差异 < 0.2 | **Unresolved**：标记为人工仲裁，进入 `conflicts/unresolved.json` | 无 canonical 生成，等待人工 |
| **本质矛盾** | obligation 逻辑互斥（如"必须用 A" vs "禁止用 A"） | **Unresolved**：必须人工介入 | 同上 |

**批判性修正**：
- Meta-Team 的 agents 可以实时讨论、协商、达成共识。mem10x 的 agents 是异步独立的，没有实时协商能力。
- 所以引入 **validationScore 作为客观权重**，减少主观争论。当客观权重无法区分时（平局），必须**人工兜底**。

---

### 阶段 7：Packaging & Push（打包与推送）

**目标**：将处理结果结构化输出，推送到 Git，供 local agent 拉取。

**输出目录与文件格式**：

```
cloud-output/
├── canonical/
│   └── {skillFamily}/
│       └── {canonical-id}.canonical.json
├── variations/
│   └── {skillFamily}/
│       └── {canonical-id}.variations.json
├── conflicts/
│   └── {batch-date}/
│       ├── unresolved.json
│       └── resolved.json
└── dkd-candidates/
    └── {batch-date}/
        └── {canonical-id}.dkd-candidate.json
```

#### 7.1 canonical/{skillFamily}/{id}.canonical.json

见阶段 4 的示例输出。

#### 7.2 variations/{skillFamily}/{id}.variations.json

记录被 Split 或 Elevate 策略淘汰的差异化版本：

```json
{
  "canonicalId": "frontend-react-props-typing-v3",
  "variations": [
    {
      "type": "boundary-variant",
      "boundary": {"applicable": ["legacy 项目"]},
      "obligation": "使用 type alias 而非 interface",
      "sourceAgent": "agent-def",
      "reason": "legacy 项目已有 type alias 约定"
    },
    {
      "type": "soft-recommendation",
      "obligation": "使用 React.FC 泛型",
      "sourceAgent": "agent-ghi",
      "agreementRate": 0.33
    }
  ]
}
```

#### 7.3 conflicts/{batch-date}/unresolved.json

```json
{
  "batchId": "2026-06-29",
  "conflicts": [
    {
      "id": "conflict-001",
      "type": "essential-contradiction",
      "skillFamily": "frontend-react",
      "trigger": "React 组件状态管理",
      "obligationA": "必须使用 useState",
      "obligationB": "禁止使用 useState，改用 useReducer",
      "agentA": "agent-xyz",
      "agentB": "agent-abc",
      "validationScoreA": 0.85,
      "validationScoreB": 0.83,
      "status": "pending-human-arbitration",
      "deadline": "2026-07-06"
    }
  ]
}
```

#### 7.4 dkd-candidates/{batch-date}/{id}.dkd-candidate.json

从 experience 和 antipattern 中提取的 DKD 规则候选：

```json
{
  "sourceCanonicalId": "frontend-react-props-typing-v3",
  "derivedFrom": ["exp-xyz-1", "exp-abc-2"],
  "level": "L2",
  "ruleType": "structural",
  "condition": "memoryWrite.semanticType == 'sense' && content.includes('React') && !content.includes('interface')",
  "requiredAction": "添加 TypeScript interface 定义",
  "forbiddenAction": "使用 any 类型作为 props",
  "exclusionScope": "legacy-project",
  "confidence": 0.87,
  "status": "pending-dkd-evaluation"
}
```

**推送机制**：
- 推送到 `mem10x-cloud-hub` repo（或原 repo 的 `cloud/` 分支）
- 触发 webhook 或 Git tag（如 `canonical-2026-06-29`）通知各 local agent
- Local agent 通过 `git pull` 或 API 拉取更新

---

## 4. 关键参数表

| 参数 | 值 | 可调性 | 理由 |
|------|-----|--------|------|
| `validationScore_gate` | 0.7 | 高 | 高于 Mem²Evolve 本地标准（0.5），多 agent 交叉需更严格 |
| `recallCount_gate` | 3 | 中 | 至少 3 次独立验证，排除偶然成功 |
| `obligation_consensus_threshold` | 0.7 | 中 | 70% agent 同意才算 invariant；低于此值为 soft recommendation |
| `variant_threshold` | 0.4 | 中 | 40%-70% 共识为 variant；低于 40% 忽略 |
| `clustering_distance_threshold` | 0.25 | 高 | HDBSCAN 的 cosine 距离阈值；需根据 embedding 模型调整 |
| `synthetic_scenario_count` | 10 | 高 | 覆盖多样性，成本可控 |
| `synthetic_pass_rate_threshold` | 0.8 | 中 | 允许 20% edge case 失败；低于此值拒绝 |
| `elevation_score_gap` | 0.2 | 中 | validationScore 差异 > 0.2 才可自动 Elevate；否则人工仲裁 |
| `batch_schedule` | nightly | 高 | 每日凌晨运行；高频场景可改为 event-driven |

---

## 5. 错误处理与兜底机制

| 场景 | 处理策略 |
|------|---------|
| 某 skillFamily 只有一个 agent 提交 | 跳过对比蒸馏，直接标记为 `single-source`，confidence 打折（×0.7），进入 canonical 但标注"待交叉验证" |
| LLM-as-a-Judge 调用失败 | 重试 3 次（指数退避）；仍失败则标记该聚类为 `degraded`，使用规则化 fallback（如简单多数投票） |
| 所有 sense 都无法聚类 | 全部作为 orphan 进入 `variations/`，不生成 canonical |
| Synthetic validation 全部失败 | 拒绝生成 canonical，记录到 `conflicts/rejected-by-synthetic.json` |
| Unresolved 冲突积压超过 7 天 | 自动告警（webhook/email），通知管理员人工仲裁 |
| Cloud Pod 运行中断 | 基于 batchId 的幂等设计：重新运行同一 batch 时，跳过已处理的 sense（通过 checksum 去重） |

---

## 6. 与源论文的映射与修正

| 本 Spec 组件 | 源论文概念 | 直接照搬的问题 | 修正方式 |
|-------------|-----------|-------------|---------|
| Stage 2 Trigger Clustering | Meta-Team 的分布式任务结构 | Meta-Team 的聚类是天然角色分解，无需计算 | 显式语义聚类（HDBSCAN + embedding） |
| Stage 3 Contrastive Distillation | MemCollab 的 Contrastive Trajectory Distillation | MemCollab 对比的是同一任务的执行轨迹（有 ground truth 对齐） | 对比结构化 sense slot（LLM 判断兼容性） |
| Stage 3 证据交换 | Meta-Team 的 post-task communication | Meta-Team 是同步实时通信 | 异步结构化交叉验证（LLM-as-a-Judge） |
| Stage 5 Synthetic Validation | Mem²Evolve 的 LLM-as-a-Judge | Mem²Evolve 评估实际执行轨迹 | 合成测试场景进行思维实验 |
| Stage 6 Conflict Arbitration | Meta-Team 的 collective discussion | Meta-Team 的 agents 可实时协商 | validationScore 加权 + 人工兜底 |
| 整体架构 | Mem²Evolve 的双 Memory 闭环 | Mem²Evolve 是单 agent 的 forward-backward | 扩展到多 agent 分布式，增加 L2/L3 层 |

---

## 7. 实现建议

### 7.1 技术栈
- **Embedding**: `text-embedding-3-large` 或同等级模型（支持 3072 维）
- **Clustering**: `hdbscan`（Python）或 `sklearn.cluster.HDBSCAN`
- **LLM 调用**: 统一封装 LLM client，支持重试、退避、fallback
- **Git 操作**: `gitpython` 或 shell `git` 命令
- **调度**: Kubernetes CronJob（nightly）或 Argo Workflows（event-driven）

### 7.2 监控指标
- `batch_processing_time`：每批次处理耗时
- `senses_ingested` / `senses_rejected`：摄入与拒绝比例
- `clusters_formed` / `clusters_orphaned`：聚类效果
- `canonical_generated` / `canonical_rejected`：生成与拒绝比例
- `conflicts_unresolved`：未解决冲突数量（告警阈值：>5）
- `synthetic_pass_rate_avg`：合成验证平均通过率

### 7.3 扩展点
- **增量处理**：只处理新增/修改的 sense，避免全量重算
- **多语言支持**：embedding 和 LLM 判断需支持中文 sense
- **A/B 测试**：新版本的 canonical sense 可先推送给部分 agent，观察 validationScore 变化

---

## 8. 附录

### 附录 A：LLM Prompt 模板

#### A.1 聚类中心描述生成

```
你是一位领域专家。以下是一组描述相似场景的 trigger 文本，请生成一个最通用、最准确的中心描述，覆盖所有场景的核心意图。

Triggers:
{cluster_triggers}

要求：
- 描述应简洁（20-50 字）
- 覆盖所有 trigger 的交集，不遗漏关键条件
- 使用专业术语

输出纯文本，不要解释。
```

#### A.2 Synthetic Scenario 生成

见阶段 5 的 prompt 模板。

#### A.3 Boundary Conflict 检测

```
判断以下两个边界条件是否冲突（在同一 trigger 下不能同时成立）：

边界 A: {boundary_a}
边界 B: {boundary_b}

冲突类型：
- "complementary"：互补（不重叠，可共存）
- "overlapping-compatible"：重叠但兼容
- "overlapping-conflicting"：重叠且矛盾
- "subsumed"：一个包含另一个

输出 JSON：{"conflictType": "...", "reasoning": "..."}
```

### 附录 B：版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06-29 | 初始稳定版本 |
