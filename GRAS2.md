# GRAS 预测预警架构：Decision-Making & Presentation 设计

> 说明：本文件补充在你已有的 Data / Analysis 设计之上，聚焦：
> - Decision-Making：如何基于预测结果做“预警 + 决策 + 闭环”  
> - Presentation：如何给不同角色（应用负责人、运维、Cyber、安全合规、管理层）可视化呈现与协同

---

## 一、Decision-Making 层（智能预警与闭环决策）

### 1. 设计目标

围绕 GRAS 指标（G=应用质量、R=运维可靠性、A=可用性/可访问性、S=安全/cyber），决策层的目标是把 **“预测结果” 转成 “可执行行动”**：

1. 将预测的数值 → 转化为统一的风险等级（Risk Level）。
2. 给出明确的 **告警策略** 与 **处置建议 / 自动行动**。
3. 实现 **人机协同**：先“建议”，再根据成熟度逐步“自动执行”。
4. 打通 **闭环**：事后反馈预测准确率、处置效果，用于迭代模型和规则。

---

### 2. 核心概念与数据模型

#### 2.1 对象：应用 / 维度 / 时间

- 应用实体：`Application`  
  - 字段示例：`appId`, `appName`, `ownerTeam`, `businessCriticality`
- 指标维度：`Metric`  
  - GRAS 四大类，每类可再细分，如：
    - `G`: error rate, latency p95, user complaint index
    - `R`: MTTR, change failure rate
    - `A`: uptime %, SLA breach count
    - `S`: vulnerability backlog, failed login rate, abnormal traffic
- 时间维度：`timestamp` / `timeBucket`（如 5min / 1h / 1day）

#### 2.2 决策输入：预测结果与上下文

- 预测结果（从 Analysis 层输出）：
  - `metricId`, `appId`
  - `timestamp`（预测的时间点）
  - `forecastValue`（预测值）
  - `lower`, `upper`（置信区间）
  - `modelVersion`
  - `predictionHorizon`（预测提前量，如 2h / 24h）
- 上下文：
  - 当前/历史实测值（来自数据湖或在线库）
  - 当前事件状态（是否已有 P0/P1 事件）
  - 变更信息（最近是否有发布 / 配置变更）
  - 安全事件信息（如最近扫描出的高危漏洞）

#### 2.3 决策输出：风险与行动

- 风险评分（Risk Score 0–100）
- 风险等级（Risk Level：例如 `INFO / WARN / MINOR / MAJOR / CRITICAL`）
- 告警实体 `Alert`：
  - `alertId`, `appId`, `metricId`
  - `riskLevel`, `riskScore`
  - `predictedBreachTime`（预计触达阈值时间）
  - `recommendation`（建议行动）
  - `status`：`PENDING / ACKED / MITIGATED / FALSE_POSITIVE`
- 决策行动 `Action`：
  - 类型：`NOTIFY`, `CREATE_TICKET`, `RUN_RUNBOOK`, `AUTO_SCALING`, `FEATURE_FLAG_ROLLBACK` 等
  - 是否自动执行：`AUTO / REQUIRE_APPROVAL`

---

### 3. 决策逻辑（规则 + 模型）

#### 3.1 阈值与 SLO/SLA 映射

每个 `appId + metricId` 配置自己的决策配置：

- **静态阈值**：如 `error_rate > 2%` 视为异常。
- **动态阈值 / 基线**：基于预测模型的“正常区间”自动推导。
- **预测触发条件**：
  - 条件示例：
    - 在未来 2 小时内，预测 `A(availability)` 连续低于 99.9% 超过 30 分钟 → 触发 `MAJOR` 预警。
    - 未来 24 小时，`S(vulnerability backlog)` 无法在变更窗口前清零 → 触发 `WARN`。
- **业务权重**：
  - 不同应用 `businessCriticality` 不同，对应不同的风险映射表。

可以抽象为一个规则表（可存 DynamoDB / Firestore / RDS）：

| appId | metricId | conditionExpression                    | riskLevel | actions                     |
|------|----------|-----------------------------------------|----------|-----------------------------|
| A1   | G.error  | forecastValue > 2% for 3 buckets       | MAJOR    | NOTIFY+CREATE_TICKET        |
| A1   | S.vuln   | forecastValue > 10 high vuln on release | CRITICAL | NOTIFY+RUN_RUNBOOK+ESCALATE |

#### 3.2 复合决策（跨维度/跨系统）

- 同一应用内的 **多维度组合**：
  - 例：错误率升高 (G) + 变更密集 (R) → 提升一个等级。
  - 例：可用性降低 (A) + 检测到异常访问 (S) → 直接标记为 `CRITICAL`。
- 跨系统关联：
  - 例如一个关键支付服务的风险，需同时考虑下游渠道、网关、防火墙的 GRAS 状态，按拓扑聚合。

可以使用 **决策树/规则引擎** 抽象，如 Drools / 自定义规则 JSO# GRAS 预测预警架构：Decision-Making & Presentation 设计

> 说明：本文件补充在你已有的 Data / Analysis 设计之上，聚焦：
> - Decision-Making：如何基于预测结果做“预警 + 决策 + 闭环”  
> - Presentation：如何给不同角色（应用负责人、运维、Cyber、安全合规、管理层）可视化呈现与协同

---

## 一、Decision-Making 层（智能预警与闭环决策）

<!-- ... existing code ... -->

## 二、Presentation 层（可视化与协同工作台）

### 1. 设计目标

- 为不同角色提供统一视角：
  - 管理层：业务线 / 应用的 GRAS 总体健康与风险走势。
  - 应用团队：自己负责应用的当前状态 + 未来 24–72 小时风险。
  - 运维 / SRE：可快速 **从告警跳转到根因线索**（指标 → 日志 / Trace）。
  - 安全 / Cyber 团队：安全向的 S 指标与全局风险视图。
- 实现：
  - 可视化的 **GRAS 评分面板**（当前 + 预测）。
  - 统一的 **预警和事件工作台**。
  - 支持 **下钻分析**和 **操作入口**（执行 runbook、更新阈值、标记误报）。

---

### 2. 视图设计（按用户角色与颗粒度）

#### 2.1 管理层视图（Executive Dashboard）

- 维度：业务域 / 产品线 / 关键应用 Top N。
- 关键可视化元素：
  - GRAS 综合评分雷达图：对某业务线聚合 G/R/A/S 四向量。
  - 按日/周的 GRAS 趋势线（实际 vs 预测）。
  - P0/P1 事件数量与平均恢复时间（MTTR）趋势。
  - “未来 7 天预测风险” 热力图：  
    行 = 应用，列 = 日期，颜色 = 风险等级。
- 目的：
  - 一眼看到“哪个业务在未来一周最值得关注”。

#### 2.2 应用/团队视图（Owner Dashboard）

- 面向具体 `appId`：
  - GRAS 四个维度的当前值、历史曲线、预测曲线。
  - 每个维度中关键指标的 Actual vs Forecast（折线图 + 阈值线）。
  - 未来 24 小时将触发的预警列表（按时间线排序）。
- 交互：
  - 点击某一时间点的预测点，可看到：
    - 预测的区间上下限
    - 当前系统上下文（部署版本、变更记录）
    - 评估为该风险的 TOP 特征（模型解释，如 SHAP）
  - “一键跳转”：
    - 到日志查询（CloudWatch Logs / Cloud Logging）
    - 到分布式追踪（X-Ray / Cloud Trace）
    - 到变更管理系统（CI/CD 平台、变更单）

#### 2.3 运维 / SRE 视图（Operations / NOC Dashboard）

- 以“告警与事件时间轴 + 地图式拓扑”为核心：
  - 时间轴：按时间显示预测预警、实际故障、恢复动作。
  - 拓扑图：应用依赖关系，节点颜色代表当前 GRAS 状态。
- 功能：
  - 告警工作台：
    - 按 Severity / 应用 / 区域 / Tag 过滤。
    - 显示告警来源（预测 vs 实时监控），可区分“提前预警”。
    - 状态管理：`ACK`, `In Progress`, `Resolved`, `False Positive`。
  - 一键访问 Runbook：
    - 每类告警关联标准操作手册。
    - 如已支持自动化，可提供“模拟执行 / 实际执行”按钮。

#### 2.4 安全 / Cyber 视图（Security Dashboard）

- 针对 S（Security）指标：
  - 漏洞积压、补丁 SLA 达标率、异常登录、攻击事件等。
  - 未来某个审计点前是否能完成所有高危项整改的预测。
- 与其他维度关联：
  - 显示 “安全风险高但业务仍勉强可用” 的区域，作为高优先级整改目标。
  - 支持查看单个应用的综合 GRAS 安全面板。

#### 2.5 视图清单与展现形式（按页面/模块拆解）

> 本小节把上面的角色视图，具体拆成几类“页面/模块”，明确每一类展示的内容和图表形式，便于按页面实现。

1. **全局 GRAS 概览页（Landing / Overview）**

   - 目标：类似 Datadog Service 总览或业务 Cockpit，作为所有角色的入口。
   - 展示内容：
     - 应用列表 / 矩阵：
       - 每个应用一张“小卡片”：`appName`、当前 GRAS 综合评分、当前风险等级（颜色）、最近 24h 事件数。
     - 全局汇总指标：
       - 全公司 / 某业务线的 GRAS 平均得分。
       - 最近 7 天 P0/P1 事件数量、平均 MTTR。
     - 未来风险热力图：
       - 行：应用或业务线。
       - 列：未来 7 天的日期（或 24 小时的时间段）。
       - 格子颜色：预测风险等级。
   - 展现形式：
     - **卡片 + 表格**：可排序、可过滤（按业务线/团队/标签）。
     - **热力图（Heatmap）**：用于一眼看出“未来哪天、哪些应用风险高”。
     - 顶部 **数字大盘卡片**：展示关键 KPI（GRAS 总分、事件数、SLO 达标率）。

2. **单应用 GRAS 详情页（Application Detail View）**

   - 目标：模仿 Grafana / Datadog 单服务详情页，用于应用团队和 SRE。
   - 展示内容：
     - 应用基础信息：`appName`、Owner 团队、重要级别、关键依赖。
     - GRAS 四维总览：
       - G/R/A/S 四个子评分的雷达图 or 条形图。
     - 指标时序：
       - 对每个维度至少 1~2 个核心指标：
         - G：错误率、P95 延迟。
         - R：变更失败率、MTTR。
         - A：可用性%、SLA breach 次数。
         - S：高危漏洞数、异常登录数。
       - 每条指标展示：
         - 历史实际值（折线）
         - 预测值（另一条折线）
         - 预测区间（用阴影区域表示上下置信区间）
         - 阈值/SLO 线（水平参考线）
     - 即将触发的预警：
       - 时间轴或列表形式：显示“预计何时、因为什么条件触发哪个风险等级的告警”。
   - 展现形式：
     - **动态折线图（Time-Series Line Chart）**：
       - X 轴：时间（支持缩放和拖动）。
       - Y 轴：指标数值。
       - 多条曲线叠加：实际 / 预测 / 阈值。
     - **雷达图 / 条形图**：
       - 展示当前某一时刻的 GRAS 四向量。
     - **事件时间轴（Timeline）**：
       - 用点 + 标签显示预测预警和真实事件、变更窗口等。

3. **预警与事件工作台（Alerts & Incidents Workspace）**

   - 目标：模仿 PagerDuty / Opsgenie 的 incident 列表 + 详情页，统一管理所有 GRAS 预警与实际故障。
   - 展示内容：
     - 预警/事件列表：
       - 列：`alertId/incidentId`、应用、维度(G/R/A/S)、风险等级、状态、预测提前量（预测时间距预计 breach 的时间）、当前责任人。
       - 支持过滤：按 Severity、维度、应用、时间范围、状态。
     - 单个预警/事件详情：
       - 完整时间线：预测 → 告警发送 → ACK → 处置动作 → 结束时间。
       - 对应的关键指标时序截面（小型折线图），突出异常区间。
       - 推荐行动 / 关联 Runbook。
   - 展现形式：
     - **数据表格（带筛选和排序）**：主视图。
     - **右侧详情抽屉 / 弹窗**：展开时展示时间线 + 小图。
     - **时间线组件（Timeline）**：用图标 + 颜色区分预测预警 vs 实际告警 vs 恢复。

4. **拓扑与运行视图（Service Map + Operations View）**

   - 目标：给 SRE / NOC 一个“系统地图”，感知哪些节点在当前或未来有风险。
   - 展示内容：
     - 服务拓扑图：
       - 节点：应用 / 服务。
       - 连线：依赖关系（调用链）。
       - 节点颜色：当前 GRAS 状态（绿/黄/红），可叠加“未来 X 小时风险”的边框/光晕。
     - 悬浮详情：
       - 鼠标悬停显示当前 G/R/A/S 四维评分、最近事件、最近变更。
   - 展现形式：
     - **网络图（force-directed graph 或 service map）**。
     - 节点颜色 + 大小编码：颜色=风险，大小=调用量或业务重要度。
     - 支持点击节点跳转至“单应用 GRAS 详情页”。

5. **安全 / Cyber 视图（Security Risk View）**

   - 目标：给安全团队和应用 Owner 看 S 维度及其对业务的潜在影响。
   - 展示内容：
     - 按应用/业务线的安全评分列表：
       - 当前 S-score、未修复高危漏洞数、未关闭安全事件数。
     - 未来审计/大促前的整改预测：
       - 折线图：高危漏洞 backlog 随时间的实际+预测变化。
       - 标记审计/大促日期，显示在该日期前是否预期能清零。
     - 安全事件时间轴：攻击事件、异常访问峰值等。
   - 展现形式：
     - **条形图/排序表**：按 S-score 或高危漏洞数排序。
     - **折线图（Actual vs Forecast）**：跟踪整改进度。
     - **时间轴**：重要安全事件。

6. **规则与配置视图（Rule & Config Management，内部使用）**

   - 目标：给平台团队管理阈值、规则、模型版本与展示配置。
   - 展示内容：
     - 每个 `appId + metricId` 的规则列表（与决策层规则表对应）。
     - 展示参数：静态阈值、动态基线策略、SLO 目标、规则版本、启用状态。
   - 展现形式：
     - **配置表格 + 编辑表单**。
     - 可选地提供 **变更审计时间线**：记录何时谁调整了哪个阈值，以便回溯误报/漏报原因。

---

### 3. 实现方式（AWS / GCP 参考）

<!-- ... existing code ... -->，并由一个轻量服务执行。

---

### 4. 决策引擎架构（云原生实现参考）

#### 4.1 AWS 实现参考

**数据流：**

1. Analysis 层（Fargate/Batch/EKS 上的 Python 任务）将预测结果写入：
   - 冷数据：S3 `predictions/metric=G/...`
   - 热数据：DynamoDB `gras_predictions` 表
2. 同时向 **EventBridge** 推送事件：
   - `detail-type: "GRASPredictionUpdated"`
   - `detail` 中包含 `appId`, `metricId`, `forecastValue`, `timeRange`, `modelVersion` 等。
3. EventBridge 规则将事件路由到 **Lambda / Step Functions**：
   - 读取规则表（DynamoDB / Parameter Store / AppConfig）。
   - 计算 Risk Score、Risk Level。
   - 生成 Alert 实体，写入 `gras_alerts`（DynamoDB）。
4. 根据决策结果触发后续行动：
   - 告警通知：SNS → Email / SMS / ChatOps（Slack, Teams）。
   - 工单：调用 ServiceNow / Jira API。
   - 自动化：触发 SSM Automation / Incident Manager / Auto Scaling 相关 API。
   - 回写：决策结果写入 DynamoDB / S3，供后续分析和可视化。

**组件建议：**

- **事件总线**：Amazon EventBridge  
- **决策执行**：AWS Lambda（轻量规则）、AWS Step Functions（多步决策和人工审批）  
- **规则存储**：DynamoDB（规则表 + 规则版本）、AWS AppConfig（灰度发布决策规则）  
- **告警与协同**：SNS + Chatbot + Systems Manager Incident Manager  

#### 4.2 GCP 实现参考

1. Analysis 层将预测写入：
   - 冷数据：GCS / BigQuery
   - 热数据：Firestore（当前/近未来风险快照）
2. 向 **Pub/Sub Topic** 发送 `gras-prediction-updated` 消息。
3. Pub/Sub 触发 **Cloud Functions / Cloud Run**：
   - 加载规则（存 Firestore / Config Connector 管理的 YAML）。
   - 计算风险等级，写入 `gras_alerts` 集合 / BigQuery 表。
4. 后续行动：
   - Cloud Functions 调用：
     - Cloud Monitoring API 创建告警事件 / 注入自定义事件。
     - 第三方 API（PagerDuty, OpsGenie, Slack）。
   - 使用 **Workflows** 编排复杂流程（例如：先通知值班人员 → 等待 ACK → 超时自动升级）。

---

### 5. 闭环与学习反馈

决策层应将“结果”反馈回 Data/Analysis：

- 保存告警的 **实际结果**：
  - 是否真的发生故障？（`truePositive` / `falsePositive`）
  - 实际影响时长、业务损失。
- 记录 **处置效果**：
  - 哪个动作最有效（提早扩容 vs 限流 vs 回滚）。
- 这些数据作为新特征/标签，回流至数据湖（S3 / GCS），供下一版模型训练与阈值优化使用。

---

## 二、Presentation 层（可视化与协同工作台）

### 1. 设计目标

- 为不同角色提供统一视角：
  - 管理层：业务线 / 应用的 GRAS 总体健康与风险走势。
  - 应用团队：自己负责应用的当前状态 + 未来 24–72 小时风险。
  - 运维 / SRE：可快速 **从告警跳转到根因线索**（指标 → 日志 / Trace）。
  - 安全 / Cyber 团队：安全向的 S 指标与全局风险视图。
- 实现：
  - 可视化的 **GRAS 评分面板**（当前 + 预测）。
  - 统一的 **预警和事件工作台**。
  - 支持 **下钻分析**和 **操作入口**（执行 runbook、更新阈值、标记误报）。

---

### 2. 视图设计（按用户角色与颗粒度）

#### 2.1 管理层视图（Executive Dashboard）

- 维度：业务域 / 产品线 / 关键应用 Top N。
- 关键可视化元素：
  - GRAS 综合评分雷达图：对某业务线聚合 G/R/A/S 四向量。
  - 按日/周的 GRAS 趋势线（实际 vs 预测）。
  - P0/P1 事件数量与平均恢复时间（MTTR）趋势。
  - “未来 7 天预测风险” 热力图：  
    行 = 应用，列 = 日期，颜色 = 风险等级。
- 目的：
  - 一眼看到“哪个业务在未来一周最值得关注”。

#### 2.2 应用/团队视图（Owner Dashboard）

- 面向具体 `appId`：
  - GRAS 四个维度的当前值、历史曲线、预测曲线。
  - 每个维度中关键指标的 Actual vs Forecast（折线图 + 阈值线）。
  - 未来 24 小时将触发的预警列表（按时间线排序）。
- 交互：
  - 点击某一时间点的预测点，可看到：
    - 预测的区间上下限
    - 当前系统上下文（部署版本、变更记录）
    - 评估为该风险的 TOP 特征（模型解释，如 SHAP）
  - “一键跳转”：
    - 到日志查询（CloudWatch Logs / Cloud Logging）
    - 到分布式追踪（X-Ray / Cloud Trace）
    - 到变更管理系统（CI/CD 平台、变更单）

#### 2.3 运维 / SRE 视图（Operations / NOC Dashboard）

- 以“告警与事件时间轴 + 地图式拓扑”为核心：
  - 时间轴：按时间显示预测预警、实际故障、恢复动作。
  - 拓扑图：应用依赖关系，节点颜色代表当前 GRAS 状态。
- 功能：
  - 告警工作台：
    - 按 Severity / 应用 / 区域 / Tag 过滤。
    - 显示告警来源（预测 vs 实时监控），可区分“提前预警”。
    - 状态管理：`ACK`, `In Progress`, `Resolved`, `False Positive`。
  - 一键访问 Runbook：
    - 每类告警关联标准操作手册。
    - 如已支持自动化，可提供“模拟执行 / 实际执行”按钮。

#### 2.4 安全 / Cyber 视图（Security Dashboard）

- 针对 S（Security）指标：
  - 漏洞积压、补丁 SLA 达标率、异常登录、攻击事件等。
  - 未来某个审计点前是否能完成所有高危项整改的预测。
- 与其他维度关联：
  - 显示 “安全风险高但业务仍勉强可用” 的区域，作为高优先级整改目标。
  - 支持查看单个应用的综合 GRAS 安全面板。

---

### 3. 实现方式（AWS / GCP 参考）

#### 3.1 AWS 展示实现方案

数据来源：

- 预测与告警：DynamoDB `gras_predictions`, `gras_alerts`
- 历史曲线：S3 (`bronze/silver + predictions/`) + Athena 查询
- 事件与 Runbook：Systems Manager / Incident Manager / 工单系统 API

可选实现：

1. **Amazon QuickSight**
   - BI 类仪表盘（趋势、聚合指标、管理视图）。
   - 使用 Athena / Redshift / SPICE 作为数据源。
   - 配置 Row-Level Security，实现按 BU / Team 授权。
2. **Managed Grafana + CloudWatch**
   - 面向 SRE / 运维，展示时间序列、阈值线、告警状态。
   - 使用 CloudWatch Metrics + Logs Insights + 在 Grafana 中混合 DynamoDB/Athena 数据源。
3. **自研 Web Portal（推荐作为统一入口）**
   - 前端：React / Vue / Angular
   - 后端：API Gateway + Lambda / Fargate（读取 DynamoDB / Athena）
   - 集成：
     - SSO：Cognito 或公司统一 IdP
     - ChatOps：嵌入 Slack / Teams 接入，展示告警与操作按钮

#### 3.2 GCP 展示实现方案

数据来源：

- BigQuery（历史 + 汇总 + 预测）
- Firestore / Bigtable（热数据、告警）
- Cloud Monitoring（实时监控指标）

可选实现：

1. **Looker Studio / Looker**
   - 连接 BigQuery，构建管理层与分析型仪表盘。
   - 支持数据行级权限控制。
2. **Cloud Monitoring 自带 Dashboard**
   - 用于实时运维视角，看时间序列和告警。
   - 通过自定义指标将预测和风险等级注入 Monitoring。
3. **自研 Portal（Cloud Run + 前端）**
   - 前端同上，自托管于 Cloud Run / GCE。
   - 后端 API 读写 BigQuery / Firestore / Pub/Sub。
   - 支持将告警和行动按钮嵌入到日常协作工具（如 ChatOps 机器人）。

---

### 4. 交互与可用性（UX）要点

1. **默认视图简洁，复杂度逐级下钻**：
   - 首页只展示：当前健康状况 + 未来 24–72h 风险矩阵。
   - 深度诊断才暴露所有时序、特征贡献等细节。
2. **解释性（Explainability）展示**：
   - 对每个高风险预测，展示：
     - 哪些特征贡献最大（例如：最近 3 次发布失败、CPU 利用率异常、扫描出的高危漏洞）。
     - 简化的自然语言解释：“在当前流量与失败率下，模型预计在 X 小时内 SLA 将被打破。”
3. **行动优先（Action-First）**：
   - 告警详情页顶部放置建议行动按钮：
     - “执行扩容 Runbook”
     - “创建 P1 工单”
     - “标记为误报并提供原因” → 用于模型持续学习。
4. **历史对比与验证**：
   - 在 Presentation 中支持选中历史事件，查看：
     - 当时的预测 vs 实际。
     - 当时执行的决策和效果。
   - 帮助业务和技术团队建立对预测体系的信任。

---

### 5. 权限与多租户

- 按 **应用 / 团队 / 业务线** 做授权隔离：
  - 应用负责人只看自己负责的应用。
  - SRE / NOC / 安全团队可查看全局视图。
  - 管理层可按业务线、区域、重要程度聚合。
- 权限控制落在：
  - AWS：Cognito + IAM + QuickSight Row-Level Security / API 层过滤。
  - GCP：IAM + Looker Row-Level Security / API 层过滤。

---

### 6. 与现有 GRAS 文档的集成建议

- 在原 GRAS.md 中的四层框架概览里，把本文件简要链接为：
  - “Decision-Making 层：见 GRAS-decision-presentation.md 第 1–5 章”
  - “Presentation 层：见 GRAS-decision-presentation.md 第 2–5 章”
- Data / Analysis 层输出的：
  - 预测结果表结构
  - Feature Store 设计
  - 模型管理（SageMaker / Vertex / 自建 MLflow）
  与本文件中决策层的数据模型（Alert / Risk Score / Action）做字段对齐，确保：
  - Analysis 层 → Decision 层 → Presentation 层 全链路上字段名称、时间粒度和应用标识一致。