# SkillForge — 前端设计文档（Web Dashboard 版）

> **版本**: v1.0  
> **范围**: 纯前端展示层（Mock 数据驱动），覆盖三层角色的核心视图  
> **目标**: 为 SkillForge 的 Web Dashboard 前端提供可直接进入编码阶段的详细设计蓝图  
> **角色覆盖**: 一般使用者（SRE 工程师）/ Team Leader / 公司高管（VP/CTO/Director）

---

## 目录

1. [项目概述与设计前提](#一项目概述与设计前提)
2. [技术架构与选型](#二技术架构与选型)
3. [项目目录结构](#三项目目录结构)
4. [Mock 数据层设计](#四mock-数据层设计)
5. [路由与页面架构](#五路由与页面架构)
6. [全局布局与导航系统](#六全局布局与导航系统)
7. [第一层：一般使用者页面设计](#七第一层一般使用者页面设计)
8. [第二层：Team Leader 页面设计](#八第二层team-leader-页面设计)
9. [第三层：高管层页面设计](#九第三层高管层页面设计)
10. [通用组件设计](#十通用组件设计)
11. [状态管理与数据流](#十一状态管理与数据流)
12. [交互设计规范](#十二交互设计规范)

---

## 一、项目概述与设计前提

### 1.1 产品形态说明

SkillForge 在实际交付中存在三种产品形态：

1. **VS Code 插件**：嵌入式诊断面板、操作录制、上下文助手（第一层核心入口）
2. **Teams / Slack Bot**：推送通知、频道助手、问答捕获（第一层 + 第二层的信息通道）
3. **Web Dashboard**：能力可视化、管理报表、治理控制台（三层角色均会使用，但视角不同）

本文档仅针对 **Web Dashboard 前端** 进行完整设计。VS Code 插件和 Bot 的 UI 以 Web Dashboard 中的对应模块为视觉与交互基准。

### 1.2 设计前提约束

- **无后端**：所有数据来自 Mock 层，接口定义先行，后续可无缝替换为真实 API。
- **三层角色共用一套代码库**：通过角色切换 + 路由级权限控制（前端假鉴权）实现不同视图。
- **响应式优先桌面端**：核心用户在工作时间使用桌面浏览器，平板适配，手机仅保证信息可读。
- **深色/浅色双主题**：SRE 工程师常有夜间 On-call 场景，默认深色模式，一键切换。

### 1.3 核心设计原则

1. **信息密度分层**：高管层极度压缩（一屏核心结论），Team Leader 中等密度（数据 + 建议），使用者层信息最全（可执行细节）。
2. **数据叙事优先**：每个图表/数字都必须附带"这意味着什么"的叙事文本，禁止裸数据展示。
3. **渐进披露**：默认展示最关键信息，详情通过折叠、抽屉、下钻获取。
4. **一致性语言**：全站使用统一的状态色标（🟢🟡🔴⚪）和成熟度等级语言（L1-L5）。

---

## 二、技术架构与选型

### 2.1 技术栈

| 层级 | 技术选型 | 理由 |
|------|---------|------|
| 框架 | **React 19** + **TypeScript 5.x** | 类型安全，生态成熟，Concurrent Features 提升交互体验 |
| 构建 | **Vite 6.x** | 极速 HMR，现代打包，配置简洁 |
| 样式 | **Tailwind CSS 4.x** | 原子化样式，设计系统友好，暗色模式原生支持 |
| UI 组件 | **shadcn/ui** + 自研业务组件 | 高质量可定制基础组件，无样式入侵 |
| 图表 | **Recharts** + **Tremor** | React 声明式图表，Tremor 提供现成 Dashboard 卡片 |
| 状态管理 | **Zustand** + **Immer** | 轻量、无样板代码、支持派生状态 |
| 路由 | **React Router v7** | 声明式路由，支持 loader（Mock 数据预取） |
| Mock 服务 | **MSW (Mock Service Worker)** | 拦截真实 HTTP 请求，后续切真实 API 零成本 |
| 图标 | **Lucide React** | 风格统一，轻量化 SVG |
| 日期处理 | **date-fns** | 模块化，体积小 |
| 数据表格 | **TanStack Table v8** | 高性能表格，排序/筛选/分页能力完备 |

### 2.2 架构模式

采用 **Feature-Based Folder Structure（按功能分模块）** + **Shared Layer（共享层）**：

```
src/
├── features/           # 按业务功能组织（第一层 / 第二层 / 第三层 / 公共）
│   ├── layer1/         # 一般使用者功能模块
│   ├── layer2/         # Team Leader 功能模块
│   ├── layer3/         # 高管层功能模块
│   └── shared/         # 跨模块共享（SkillCard、UserAvatar 等）
├── components/         # shadcn/ui + 全局通用组件（Layout、Nav 等）
├── hooks/              # 全局自定义 Hooks
├── stores/             # Zustand Store（按角色/领域拆分）
├── lib/                # 工具函数、常量、类型定义
├── mocks/              # MSW Handlers + Mock 数据工厂
├── types/              # 全局 TypeScript 类型
└── routes/             # 路由配置与路由守卫
```

---

## 三、项目目录结构

```
SkillForge-Web/
├── public/
│   └── mock-data/              # 静态 JSON 备用
├── src/
│   ├── main.tsx                # 应用入口
│   ├── App.tsx                 # 根组件（Provider 汇聚）
│   ├── index.css               # Tailwind 入口 + CSS 变量
│   │
│   ├── components/ui/          # shadcn/ui 基础组件
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── badge.tsx
│   │   ├── dialog.tsx
│   │   ├── drawer.tsx
│   │   ├── tabs.tsx
│   │   ├── table.tsx
│   │   ├── tooltip.tsx
│   │   ├── progress.tsx
│   │   ├── avatar.tsx
│   │   ├── skeleton.tsx
│   │   ├── dropdown-menu.tsx
│   │   └── ...
│   │
│   ├── components/layout/      # 全局布局组件
│   │   ├── AppShell.tsx        # 外壳：Sidebar + Header + Content
│   │   ├── Sidebar.tsx         # 侧边导航（按角色动态）
│   │   ├── Header.tsx          # 顶栏：面包屑、通知、角色切换、主题切换
│   │   ├── RoleSwitcher.tsx    # 角色切换器（Mock 鉴权核心）
│   │   └── BreadcrumbNav.tsx   # 面包屑导航
│   │
│   ├── components/charts/      # 图表封装
│   │   ├── RadarChart.tsx      # 技能雷达图
│   │   ├── TrendLine.tsx       # 趋势线图
│   │   ├── SankeyDiagram.tsx   # 桑基图（指标-能力映射）
│   │   ├── ForceGraph.tsx      # 力导向图（知识流动）
│   │   ├── HeatmapCalendar.tsx # 热力日历
│   │   └── GaugeChart.tsx      # 仪表盘/进度环
│   │
│   ├── components/shared/      # 跨模块业务组件
│   │   ├── SkillCard.tsx       # Skill 信息卡片
│   │   ├── SkillStatusBadge.tsx # Skill 健康度色标
│   │   ├── UserAvatar.tsx      # 用户头像 + 在线状态
│   │   ├── IncidentBadge.tsx   # Incident 等级徽章
│   │   ├── MaturityBadge.tsx   # 成熟度等级徽章 L1-L5
│   │   ├── MetricCard.tsx      # 指标概览卡片（Tremor 风格）
│   │   ├── TimeRangePicker.tsx # 时间范围选择器
│   │   ├── EmptyState.tsx      # 空状态占位
│   │   └── LoadingOverlay.tsx  # 加载遮罩
│   │
│   ├── features/layer1/        # 第一层：一般使用者
│   │   ├── pages/
│   │   │   ├── DiagnosePage.tsx      # 智能诊断中心
│   │   │   ├── MySkillsPage.tsx      # 我的 Skill 工坊
│   │   │   ├── SnippetVaultPage.tsx  # 命令片段库
│   │   │   ├── LearningMapPage.tsx   # 学习地图
│   │   │   ├── ArenaPage.tsx         # 实战演练场
│   │   │   └── ProfilePage.tsx       # 个人档案 / 成就
│   │   ├── components/
│   │   │   ├── DiagnoseInput.tsx     # 诊断输入框
│   │   │   ├── SkillChainList.tsx    # Skill 链路推荐列表
│   │   │   ├── SkillStepper.tsx      # Skill 步骤引导器
│   │   │   ├── SnippetSearch.tsx     # 片段搜索框
│   │   │   ├── SnippetItem.tsx       # 片段条目
│   │   │   ├── SkillTree.tsx         # 技能树可视化
│   │   │   ├── ArenaScenarioCard.tsx # 演练场景卡片
│   │   │   └── AchievementTimeline.tsx # 成就时间线
│   │   └── hooks/
│   │       └── useDiagnose.ts
│   │
│   ├── features/layer2/        # 第二层：Team Leader
│   │   ├── pages/
│   │   │   ├── TeamOverviewPage.tsx   # 团队概览（每日脉搏大屏）
│   │   │   ├── SkillRadarPage.tsx     # 团队技能雷达
│   │   │   ├── MTTRAnalysisPage.tsx   # MTTR 趋势与归因
│   │   │   ├── MembersPage.tsx        # 人员管理与成长追踪
│   │   │   ├── SchedulingPage.tsx     # 排班与技能覆盖
│   │   │   └── ReportsPage.tsx        # 周报 / 月报 / 汇报
│   │   ├── components/
│   │   │   ├── TeamPulseCard.tsx      # 团队脉搏卡片
│   │   │   ├── IncidentLiveCard.tsx   # Incident 实时态势卡片
│   │   │   ├── SkillRadarChart.tsx    # 雷达图封装（团队版）
│   │   │   ├── MemberRow.tsx          # 成员列表行
│   │   │   ├── OnboardingTracker.tsx  # 新人成长追踪面板
│   │   │   ├── ScheduleGrid.tsx       # 排班网格
│   │   │   └── ReportPreview.tsx      # 报告预览
│   │   └── hooks/
│   │       └── useTeamData.ts
│   │
│   ├── features/layer3/        # 第三层：高管
│   │   ├── pages/
│   │   │   ├── ExecutiveDashboardPage.tsx  # 组织级一页纸仪表板
│   │   │   ├── AIGovernancePage.tsx        # AI 治理控制台
│   │   │   ├── StrategyAlignPage.tsx       # 战略对齐（指标-能力映射）
│   │   │   ├── OrgPlannerPage.tsx          # 组织能力规划器
│   │   │   ├── MaturityAssessmentPage.tsx  # 成熟度评估
│   │   │   └── BoardReportPage.tsx         # 集团汇报材料
│   │   ├── components/
│   │   │   ├── OnePageStatus.tsx           # 一页纸状态摘要
│   │   │   ├── ExecutiveIncidentBrief.tsx  # 重大事故简报
│   │   │   ├── ROICard.tsx                 # ROI 概览卡片
│   │   │   ├── AIPolicyEditor.tsx          # AI 策略编辑器
│   │   │   ├── ConflictTable.tsx           # Skill 冲突检测表
│   │   │   ├── SankeyAlignChart.tsx        # 指标-能力桑基图
│   │   │   ├── MaturityLadder.tsx          # 成熟度阶梯图
│   │   │   ├── TalentRiskHeatmap.tsx       # 人才风险热力图
│   │   │   ├── BudgetAllocator.tsx         # 预算分配可视化
│   │   │   └── BoardReportSlides.tsx       # 汇报 Slide 模拟器
│   │   └── hooks/
│   │       └── useExecutiveData.ts
│   │
│   ├── features/shared/        # 跨层共享功能
│   │   ├── pages/
│   │   │   ├── SkillDetailPage.tsx       # Skill 详情页（通用）
│   │   │   ├── IncidentDetailPage.tsx    # Incident 详情页（通用）
│   │   │   └── SearchResultPage.tsx      # 全局搜索结果
│   │   └── components/
│   │       ├── SkillDetailDrawer.tsx     # Skill 详情抽屉
│   │       ├── IncidentTimeline.tsx      # Incident 时间线
│   │       ├── CommentThread.tsx         # 评论/讨论串
│   │       └── GlobalSearch.tsx          # 全局搜索框
│   │
│   ├── stores/                 # Zustand Store
│   │   ├── authStore.ts        # 角色/用户/权限
│   │   ├── uiStore.ts          # 主题、侧边栏折叠、全局 Toast
│   │   ├── skillStore.ts       # Skill 实体缓存
│   │   ├── incidentStore.ts    # Incident 实体缓存
│   │   └── layer2Store.ts      # Team Leader 专用状态
│   │
│   ├── mocks/                  # MSW + Mock 数据
│   │   ├── browser.ts          # MSW worker 启动
│   │   ├── handlers.ts         # 请求拦截处理器
│   │   ├── factories/          # 数据工厂（按领域）
│   │   │   ├── skillFactory.ts
│   │   │   ├── userFactory.ts
│   │   │   ├── incidentFactory.ts
│   │   │   ├── teamFactory.ts
│   │   │   └── reportFactory.ts
│   │   └── seeds/              # 初始种子数据
│   │       └── initialData.ts
│   │
│   ├── lib/
│   │   ├── utils.ts            # cn() 等通用工具
│   │   ├── constants.ts        # 全局常量（状态色、等级定义）
│   │   ├── formatters.ts       # 数字/日期/时长格式化
│   │   └── mockDelay.ts        # Mock 请求延迟模拟
│   │
│   ├── types/
│   │   ├── skill.ts            # Skill 全类型定义
│   │   ├── user.ts             # 用户/成员类型
│   │   ├── incident.ts         # Incident/事故类型
│   │   ├── team.ts             # 团队/组织类型
│   │   ├── report.ts           # 报告/汇报类型
│   │   ├── governance.ts       # 治理/策略类型
│   │   └── api.ts              # API 响应通用包装
│   │
│   ├── hooks/
│   │   ├── useRole.ts          # 当前角色查询
│   │   ├── useTheme.ts         # 主题切换
│   │   └── useDebounce.ts      # 防抖
│   │
│   └── routes/
│       ├── index.tsx           # 路由表定义
│       ├── RouteGuard.tsx      # 角色路由守卫
│       └── routeMeta.ts        # 路由元数据（标题、角色可见性）
│
├── index.html
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
└── package.json
```

---

## 四、Mock 数据层设计

### 4.1 Mock 数据设计原则

- **工厂函数生成**：使用工厂函数（非静态 JSON）保证数据关联性（如 Skill 的创建者必须是 User 列表中的真实用户）。
- **关联完整**：User ↔ Skill ↔ Incident ↔ Team 之间外键关联正确。
- **时序真实**：所有时间戳模拟过去 90 天的合理分布，趋势数据有明确走向（如 MTTR 逐周下降）。
- **种子数据固定**：`initialData.ts` 导出固定的种子数据，保证每次刷新页面看到一致的状态（便于演示）。

### 4.2 核心实体类型定义

以下是前端 TypeScript 类型，Mock 工厂严格遵循这些类型生成数据。

#### User（用户/成员）

```typescript
interface User {
  id: string;                    // "user_001"
  name: string;                  // "王伟"
  handle: string;                // "@wang_wu"
  avatar: string;                // 头像 URL
  role: 'engineer' | 'lead' | 'executive';
  teamId: string;                // 所属团队
  title: string;                 // "Senior SRE"
  joinDate: string;              // ISO 日期
  skillsMastery: MasteryItem[];  // 技能掌握情况
  status: 'online' | 'busy' | 'offline' | 'oncall';
  metrics: UserMetrics;
}

interface MasteryItem {
  domain: string;                // "Oracle性能诊断"
  level: number;                 // 0-100
  trend: 'up' | 'down' | 'flat';
}

interface UserMetrics {
  totalIncidents: number;
  avgMTTR: number;               // 分钟
  skillsCreated: number;
  skillsAdoptedByOthers: number;
}
```

#### Skill（技能原子）

```typescript
type SkillStatus = 'healthy' | 'attention' | 'outdated' | 'archived';
type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

interface Skill {
  id: string;                    // "skill-oracle-slow-query-diag-v3"
  name: string;
  version: number;
  authorId: string;
  teamId: string;
  createdAt: string;
  lastUsedAt: string;
  useCount: number;
  successRate: number;           // 0-1
  avgResolutionTime: number;     // 分钟
  
  classification: {
    domain: string[];            // ["database", "oracle", "performance"]
    scenario: string[];          // ["incident", "optimization"]
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    riskLevel: RiskLevel;
  };
  
  dependencies: {
    requiredSkills: string[];    // Skill ID 列表
    requiredAccess: string[];
  };
  
  content: {
    triggerConditions: string;
    diagnosisSteps: SkillStep[];
    decisionTree?: DecisionNode;
    executionActions: ActionItem[];
    rollbackPlan: string;
  };
  
  governance: {
    approvalStatus: 'draft' | 'pending' | 'approved' | 'rejected';
    reviewerId?: string;
    complianceTags: string[];
    expiryReviewDate: string;
    aiGenerated: boolean;         // 是否由 AI 辅助生成
    aiConfidence?: number;
  };
  
  evolution: {
    parentSkillId?: string;
    changeLog: string;
    deprecationCandidates?: string[];
  };
  
  healthStatus: SkillStatus;
  healthScore: number;           // 0-100
}

interface SkillStep {
  order: number;
  title: string;
  description: string;
  command?: string;              // 可执行命令
  estimatedTime: number;         // 预计耗时（分钟）
  verification: string;          // 如何验证此步骤成功
}

interface ActionItem {
  type: 'command' | 'script' | 'config' | 'manual';
  content: string;
  safetyLevel: RiskLevel;
}

interface DecisionNode {
  condition: string;
  trueBranch?: SkillStep[] | DecisionNode;
  falseBranch?: SkillStep[] | DecisionNode;
}
```

#### Incident（事故/任务）

```typescript
type IncidentPriority = 'P1' | 'P2' | 'P3' | 'P4';
type IncidentStatus = 'open' | 'diagnosing' | 'fixing' | 'verifying' | 'closed';

interface Incident {
  id: string;                    // "INC-2024-0789"
  title: string;
  priority: IncidentPriority;
  status: IncidentStatus;
  createdAt: string;
  resolvedAt?: string;
  mttr?: number;                 // 实际耗时（分钟）
  
  assigneeId: string;
  commanderId?: string;
  teamId: string;
  
  context: {
    alertSource: string;         // "PagerDuty-12345"
    affectedService: string;
    environment: 'prod' | 'staging' | 'dev';
    initialSymptom: string;
  };
  
  skillUsage: SkillUsageLog[];   // 使用了哪些 Skill
  timeline: TimelineEvent[];
  postmortem?: Postmortem;
}

interface SkillUsageLog {
  skillId: string;
  startedAt: string;
  completedAt?: string;
  currentStep: number;
  totalSteps: number;
  success: boolean;
}

interface TimelineEvent {
  timestamp: string;
  type: 'alert' | 'response' | 'diagnosis' | 'decision' | 'action' | 'info' | 'resolution';
  actorId?: string;
  description: string;
  source: 'pagerduty' | 'teams' | 'vscode' | 'manual';
}

interface Postmortem {
  rootCause: string;
  actionItems: string[];
  lessonsLearned: string[];
  skillsCreated: string[];       // 本次产生的 Skill ID
}
```

#### Team（团队）

```typescript
interface Team {
  id: string;
  name: string;                  // "DB-SRE"
  memberIds: string[];
  skillIds: string[];
  
  coverage: DomainCoverage[];    // 各领域覆盖度
  metrics: TeamMetrics;
  schedule: ScheduleSlot[][];    // 排班表
}

interface DomainCoverage {
  domain: string;                // "Oracle高可用(RAC)"
  coverageCount: number;         // 覆盖人数
  totalMembers: number;
  avgDepth: number;              // 平均深度 0-100
  healthStatus: SkillStatus;
}

interface TeamMetrics {
  avgMTTR: number;
  sloAchievement: number;        // 0-1
  incidentCountThisWeek: number;
  skillUsageThisWeek: number;
  newSkillsThisWeek: number;
}
```

#### Organization（组织层数据）

```typescript
interface OrgSnapshot {
  date: string;
  activeSkillCount: number;
  coverageRate: number;          // 核心场景覆盖率
  crossTeamReuseRate: number;    // 跨团队复用率
  avgMTTR: number;
  sloAchievement: number;
  singlePointRisks: number;      // 单点风险领域数
}

interface MaturityAssessment {
  overallLevel: number;          // 1.0 - 5.0
  overallLabel: string;          // "L3 系统化"
  dimensions: MaturityDimension[];
}

interface MaturityDimension {
  name: string;                  // "Skill覆盖度"
  score: number;                 // 0-5
  trend: 'up' | 'down' | 'flat';
  benchmark: 'above' | 'avg' | 'below';
}

interface AIGovernanceReport {
  month: string;
  totalSkills: number;
  aiAssistedCount: number;
  aiOnlyCount: number;           // 纯 AI 生成未经审核
  complianceRate: number;        // 合规率
  pendingReview: number;
  flagged: number;               // 风险标记
}

interface SkillConflict {
  id: string;
  severity: 'critical' | 'minor';
  skillA: { id: string; name: string; teamId: string };
  skillB: { id: string; name: string; teamId: string };
  conflictType: 'logic_contradiction' | 'parameter_mismatch' | 'overlap';
  description: string;
  suggestedAction: string;
}
```

### 4.3 Mock 工厂数据规模

| 实体 | 数量 | 说明 |
|------|------|------|
| User | 18 人 | 3 个团队 × 6 人 |
| Team | 3 个 | DB-SRE / Platform-SRE / Infra-SRE |
| Skill | 312 个 | 覆盖数据库、K8s、网络、监控、安全等 |
| Incident | 120 条 | 过去 90 天分布，含完整时间线 |
| OrgSnapshot | 12 条 | 过去 12 周每周快照 |
| Report | 6 条 | 周报/月报/ROI 报告模板 |
| SkillConflict | 4 组 | 含 2 组严重冲突 |

---

## 五、路由与页面架构

### 5.1 路由表

```typescript
// 路由按角色分组，但统一注册，通过 RouteGuard 控制可见性

const routes = [
  // ========== 公共路由 ==========
  { path: '/', element: <LandingPage />, roles: ['all'] },
  { path: '/login', element: <LoginPage />, roles: ['all'] },
  
  // ========== 第一层：一般使用者 ==========
  { path: '/diagnose', element: <DiagnosePage />, roles: ['engineer', 'lead', 'executive'] },
  { path: '/my-skills', element: <MySkillsPage />, roles: ['engineer', 'lead'] },
  { path: '/snippets', element: <SnippetVaultPage />, roles: ['engineer', 'lead'] },
  { path: '/learning', element: <LearningMapPage />, roles: ['engineer', 'lead'] },
  { path: '/arena', element: <ArenaPage />, roles: ['engineer', 'lead'] },
  { path: '/profile', element: <ProfilePage />, roles: ['engineer', 'lead', 'executive'] },
  
  // ========== 第二层：Team Leader ==========
  { path: '/team', element: <TeamOverviewPage />, roles: ['lead', 'executive'] },
  { path: '/team/radar', element: <SkillRadarPage />, roles: ['lead', 'executive'] },
  { path: '/team/mttr', element: <MTTRAnalysisPage />, roles: ['lead', 'executive'] },
  { path: '/team/members', element: <MembersPage />, roles: ['lead', 'executive'] },
  { path: '/team/schedule', element: <SchedulingPage />, roles: ['lead'] },
  { path: '/team/reports', element: <ReportsPage />, roles: ['lead', 'executive'] },
  
  // ========== 第三层：高管 ==========
  { path: '/executive', element: <ExecutiveDashboardPage />, roles: ['executive'] },
  { path: '/executive/governance', element: <AIGovernancePage />, roles: ['executive'] },
  { path: '/executive/strategy', element: <StrategyAlignPage />, roles: ['executive'] },
  { path: '/executive/planner', element: <OrgPlannerPage />, roles: ['executive'] },
  { path: '/executive/maturity', element: <MaturityAssessmentPage />, roles: ['executive'] },
  { path: '/executive/board-report', element: <BoardReportPage />, roles: ['executive'] },
  
  // ========== 共享路由 ==========
  { path: '/skill/:skillId', element: <SkillDetailPage />, roles: ['all'] },
  { path: '/incident/:incidentId', element: <IncidentDetailPage />, roles: ['all'] },
  { path: '/search', element: <SearchResultPage />, roles: ['all'] },
];
```

### 5.2 路由守卫逻辑

```typescript
// RouteGuard.tsx 伪代码
function RouteGuard({ allowedRoles, children }) {
  const currentRole = useAuthStore(s => s.currentRole);
  const isAllowed = allowedRoles.includes('all') || allowedRoles.includes(currentRole);
  
  if (!isAllowed) {
    // 工程师访问高管页面 → 提示无权限并建议切换角色（Mock 模式下提供切换入口）
    return <RoleMismatchPage currentRole={currentRole} requiredRole={allowedRoles} />;
  }
  return children;
}
```

> **Mock 模式下的角色切换**：Header 中常驻一个 `RoleSwitcher` 下拉框，可在 Engineer / Team Lead / Executive 之间即时切换，页面自动刷新权限和导航菜单。这便于在纯前端演示中一次性展示三层角色。

---

## 六、全局布局与导航系统

### 6.1 AppShell 布局

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Header (64px)                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ [Logo] SkillForge   [🔍全局搜索...]        [🌙] [🔔] [👤] [▼角色切换] │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
├──────────┬──────────────────────────────────────────────────────────────────┤
│          │  Breadcrumb: 团队概览 / 技能雷达                                   │
│ Sidebar  ├──────────────────────────────────────────────────────────────────┤
│ (240px)  │                                                                  │
│          │                                                                  │
│ [导航菜单]│                        Main Content                              │
│          │                        (自适应宽度)                               │
│  首页     │                                                                  │
│  诊断     │                                                                  │
│  ...     │                                                                  │
│          │                                                                  │
│          │                                                                  │
│          │                                                                  │
│ [底部区]  │                                                                  │
│  帮助     │                                                                  │
│  设置     │                                                                  │
├──────────┴──────────────────────────────────────────────────────────────────┤
│  Footer (可选，极简版权信息)                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 侧边栏导航（按角色动态）

| 层级 | 主导航项 | 图标 | 子菜单 |
|------|---------|------|--------|
| **公共** | 首页 / 工作台 | Home | — |
| **第一层** | 智能诊断 | Zap | — |
| | 我的 Skill 工坊 | Wrench | — |
| | 命令片段库 | Terminal | — |
| | 学习地图 | Map | — |
| | 实战演练场 | Swords | — |
| | 个人档案 | User | — |
| **第二层** | 团队概览 | LayoutDashboard | — |
| | 技能雷达 | Target | — |
| | MTTR 分析 | TrendingDown | — |
| | 人员成长 | Users | — |
| | 排班优化 | CalendarDays | — |
| | 汇报材料 | FileText | 周报 / 月报 / 资源请求 |
| **第三层** | 组织仪表板 | BarChart3 | — |
| | AI 治理 | ShieldCheck | 使用全景 / 策略管理 / 冲突检测 |
| | 战略对齐 | GitMerge | — |
| | 能力规划 | Compass | — |
| | 成熟度评估 | Award | — |
| | 集团汇报 | Presentation | — |

---

## 七、第一层：一般使用者页面设计

### 7.1 智能诊断中心（DiagnosePage）

**页面定位**：用户遇到问题时的第一入口。模拟 VS Code 诊断面板的 Web 版本。

**URL**: `/diagnose`

**布局结构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  智能诊断中心                                          [📋 历史诊断记录 ▾]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │    🔍 描述你的问题、粘贴错误日志或输入告警 ID...                    │   │
│  │                                                                     │   │
│  │    [Oracle-prod-01 响应超时 > 5s]  [📎 添加上下文]  [开始诊断 →]   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  快速触发标签: [ORA-04031] [K8s Pod Evicted] [API 5xx] [连接池耗尽] ...   │
│                                                                             │
│  ────────────────────────────────── 或历史记录 ─────────────────────────   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  推荐解决路径（基于上下文 "Oracle-prod-01 response timeout"）        │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ 1. ★ Oracle 慢查询诊断路径                                     │ │   │
│  │  │    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │ │   │
│  │  │    成功率 92%  │  平均耗时 18min  │  置信度高                   │ │   │
│  │  │    创建者: @li_si  │  最近使用: 3天前 by @zhang_san            │ │   │
│  │  │    步骤预览: AWR报告 → Top SQL定位 → 索引优化 → 验证           │ │   │
│  │  │    [查看详细步骤]  [一键复制命令]  [在演练场模拟]              │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ 2. 连接池耗尽排查                                               │ │   │
│  │  │    成功率 78%  │  平均耗时 25min                               │ │   │
│  │  │    ...                                                         │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ 3. 资源瓶颈定位                                                 │ │   │
│  │  │    成功率 65%  │  平均耗时 35min                               │ │   │
│  │  │    ...                                                         │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  💡 团队知识: @li_si 是该领域专家 (处理过12次类似问题)              │   │
│  │     [发起 Teams 通话]  [查看他的 Skill 库]                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**交互细节**：

1. **诊断输入框**：支持多行文本，粘贴错误日志时自动高亮关键错误码（如 `ORA-04031`）。
2. **上下文附件**：点击"添加上下文"展开抽屉，可选当前环境（Prod/Staging）、 affected service。
3. **Skill 链路卡片**：
   - hover 时显示步骤详情 tooltip
   - 点击"查看详细步骤" → 右侧滑出 Drawer，展示完整 Step-by-step 流程
   - 点击"一键复制命令" → 复制首条可执行命令到剪贴板，Toast 提示
   - 每张卡片底部有微型趋势 sparkline（近 30 天成功率走势）
4. **专家信息**：点击专家头像 → 弹出微型 Profile Popover，显示其技能掌握度和在线状态。

**Mock 数据接口**：

```typescript
// GET /api/diagnose?query={text}&context={env}:{service}
interface DiagnoseResponse {
  queryInterpretation: string;   // 系统理解的查询意图
  matchedSkills: MatchedSkill[];
  suggestedExperts: Expert[];
  similarIncidents: Incident[];  // 历史类似事故
}

interface MatchedSkill {
  skill: Skill;
  matchScore: number;            // 匹配分数
  reason: string;                // 为什么推荐
  estimatedTime: number;         // 预估解决时间
}
```

---

### 7.2 我的 Skill 工坊（MySkillsPage）

**页面定位**：个人 Skill 资产的管理中心。展示"我创建的、我使用的、系统为我自动生成的"所有 Skill。

**URL**: `/my-skills`

**布局结构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  我的 Skill 工坊                                        [+ 新建 Skill] [⚙️]  │
├─────────────────────────────────────────────────────────────────────────────┤
│  [全部 ▾] [健康度: 全部 ▾] [排序: 最近使用 ▾]        [🔍 搜索我的 Skill...]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │  总创建    │ │  在用      │ │  草稿待确认│ │  被复用次数│               │
│  │  23        │ │  18        │ │  5 ⚠️      │ │  127       │               │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
│                                                                             │
│  ─────────── 自动生成草稿（待确认） ───────────                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 📝 "Oracle AWR自动分析 + 索引在线重建" (来自 INC-2024-0789)         │   │
│  │    系统检测到你在本次 incident 中使用了新的解决模式                 │   │
│  │    新增: AWR自动拉取脚本、在线重建索引的安全性判断                  │   │
│  │    [一键保存为新 Skill]  [合并到现有 Skill]  [忽略]  [查看操作记录]  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ──────────────── 我的 Skill 列表（网格/列表切换）────────────────        │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐    │
│  │ 🟢 Oracle慢查询诊断 │  │ 🟢 K8s Pod异常退出 │  │ 🟡 连接池调优指南  │    │
│  │    v3.2            │  │    v1.5            │  │    v2.0            │    │
│  │ 使用: 47次         │  │ 使用: 32次         │  │ 使用: 8次          │    │
│  │ 成功率: 89%        │  │ 成功率: 95% → 72%  │  │ 成功率: 78%        │    │
│  │ [查看] [编辑]      │  │ [查看] [编辑] ⚠️   │  │ [查看] [编辑]      │    │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**交互细节**：

1. **顶部 KPI 卡片**：4 个 MetricCard，点击后下方列表自动筛选对应类型。
2. **草稿确认区**：
   - 仅当有未确认草稿时显示。
   - "一键保存" → 直接发布为 approved 状态。
   - "合并到现有 Skill" → 弹出选择框，选择后生成新版本（version + 0.1）。
3. **Skill 卡片**：
   - 健康度通过左上角小圆点色标表示（🟢🟡🔴⚪）。
   - 成功率带趋势箭头（→ 表示下降，↑ 表示提升）。
   - hover 时显示操作按钮（查看/编辑/分享/删除）。
4. **新建 Skill**：点击后打开全屏 Wizard，分步骤引导创建（触发条件 → 诊断步骤 → 执行动作 → 回滚方案 → 治理信息）。

---

### 7.3 命令片段库（SnippetVaultPage）

**页面定位**：团队共享的命令片段的 Web 管理端。模拟 VS Code 中 `sf:` 触发器的 Web 版本。

**URL**: `/snippets`

**核心交互**：

- **搜索框**：输入时实时过滤，支持标签语法 `tag:oracle mttr`。
- **片段卡片**：左侧展示命令预览（Syntax Highlighted），右侧展示元信息（创建者、使用次数、成功率、环境要求）。
- **参数自动填充**：若命令中包含 `{变量}`，卡片展开显示变量说明及自动填充规则。
- **一键复制**：点击即复制到剪贴板。

**Mock 数据接口**：

```typescript
// GET /api/snippets?query={text}&tag={tag}&sort={usage|recent|success}
interface Snippet {
  id: string;
  title: string;
  command: string;
  description: string;
  tags: string[];
  authorId: string;
  useCount: number;
  successRate: number;
  applicableEnv: string[];       // ["Oracle 19c+", "RAC"]
  variables?: SnippetVariable[];
}
```

---

### 7.4 学习地图（LearningMapPage）

**页面定位**：个人技能成长的可视化导航。替代传统"教学平台"，更像"知识地图 GPS"。

**URL**: `/learning`

**布局结构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  我的技能地图                                        [📊 查看详细数据]      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  左侧: 技能领域树                    右侧: 选中领域的详情面板               │
│  ┌───────────────────┐ ┌─────────────────────────────────────────────────┐  │
│  │ ▼ Oracle DB       │ │  Oracle DB 性能诊断                              │  │
│  │   ├─ 基础管理 ████│ │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │   ├─ 性能诊断 ████│ │  你的掌握度: 85%  ████████████████░░░░           │  │
│  │   ├─ 高可用(RAC) █│ │  团队均值: 72%   ██████████████░░░░░░░░         │  │
│  │   └─ 安全审计  ██ │ │  你在团队中的位置: 高于平均 ✓                     │  │
│  │ ▶ Kubernetes      │ │                                                  │  │
│  │   ├─ 基础概念 ████│ │  📈 成长曲线                                      │  │
│  │   ├─ 部署管理 ███ │ │  [近90天趋势折线图]                               │  │
│  │   ├─ 网络/存储 ██ │ │                                                  │  │
│  │   └─ 故障排查  █  │ │  🎯 建议下一步:                                   │  │
│  │ ▶ Linux 系统调优  │ │  掌握度已达85%，下一步建议深入 "RAC 故障切换"     │  │
│  │ ▶ 监控告警配置    │ │  相关 Skill: [Oracle RAC 主备切换] [DataGuard]    │  │
│  │                   │ │                                                  │  │
│  │ 📌 团队需要你提升:│ │  💡 同类角色工程师通常通过此路径成长:             │  │
│  │    K8s网络/存储   │ │    基础管理 → 性能诊断 → 高可用 → 安全审计       │  │
│  │    (团队覆盖仅20%)│ │    预计总时长: 6个月                               │  │
│  └───────────────────┘ └─────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键设计**：

- **掌握度颜色**：0-40% 红色、40-70% 黄色、70-90% 蓝色、90-100% 绿色。
- **团队缺口提示**：若某领域团队覆盖率低于阈值，在左侧树节点旁显示 `⚠️` 图标。
- **路径规划**：点击"建议下一步"中的 Skill → 打开 Skill 详情 Drawer，可直接开始学习/练习。

---

### 7.5 实战演练场（ArenaPage）

**页面定位**：安全环境下的 Skill 模拟执行与对比回放。

**URL**: `/arena`

**页面分区**：

1. **场景列表**：卡片网格展示可用演练场景（如"Oracle ORA-04031 模拟"、"K8s Pod 驱逐排查"），每张卡片带难度标签和预估时长。
2. **演练中界面（点击场景后进入）**：
   - 左侧：模拟终端（Web Terminal UI），可输入命令，系统返回预设的模拟输出。
   - 右侧：Skill 引导面板（与真实诊断中一致），但处于"沙箱模式"——所有步骤执行前有安全确认。
   - 底部：计时器 + 当前步骤进度。
3. **演练报告**：完成后弹出对比报告——你的操作路径 vs 最佳实践路径的差异点高亮标注。

---

### 7.6 个人档案（ProfilePage）

**页面定位**：个人成就、能力档案、贡献历史的展示页。

**URL**: `/profile`

**内容模块**：

- **头部卡片**：头像、姓名、职位、团队、入职时间、当前状态。
- **能力概览**：小型雷达图展示 5-6 个核心领域的掌握度。
- **成就时间线**：按时间倒序展示获得的微成就（如"Skill 被复用 10 次"、"连续 30 天使用诊断"、"某领域专家认证"）。
- **贡献统计**：创建的 Skill 数、被使用的总次数、帮助团队节省的总人时。
- **周报快照**：展示最近 4 周的自动生成的能力快照卡片。

---


---

## 八、第二层：Team Leader 页面设计

### 8.1 团队概览（TeamOverviewPage）

**页面定位**：Team Leader 的"每日首屏"。将原设计中 Teams 推送的"每日脉搏" + "Incident 实时态势" + "Skill 动态" 汇聚为一个 Web 综合看板。

**URL**: `/team`

**布局结构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DB-SRE 团队概览                              [📅 5/15] [⏱️ 自动刷新: 30s]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ 昨夜值班 │ │ 今日排班 │ │ 进行中   │ │ Skill动态│ │ 本周MTTR │         │
│  │ ✅ 平稳  │ │ 日班:张三│ │ P3: 1件  │ │ 使用5次  │ │ 20min ↓│         │
│  │ 无P1/P2  │ │ 夜班:李四│ │ 待根因   │ │ +1草稿   │ │ vs上周   │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
│                                                                             │
│  ───────────────────────── 实时 Incident 态势 ─────────────────────────    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ⚡ INC-0801 (P2) │ DB-prod 响应超时 │ 进行中 22min                    │   │
│  │                                                                     │   │
│  │  响应人: @zhang_san (入职8个月)                                     │   │
│  │  状态: 诊断阶段  │  使用 Skill: "Oracle慢查询诊断" (步骤 2/5)         │   │
│  │                                                                     │   │
│  │  SkillForge 评估:                                                   │   │
│  │  🟢 进展正常 - 预计再需15-20min (该响应人历史同类成功率 85%)         │   │
│  │                                                                     │   │
│  │  [进入频道]  [指派支援]  [查看实时步骤]  [暂不介入]                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ─────────────────────────── 团队技能速览 ─────────────────────────────    │
│                                                                             │
│  高风险单点:                                                                │
│  🔴 Oracle RAC: 仅 @li_si 掌握  →  [立即启动传承计划]                    │
│  🟡 K8s网络策略: 仅 @wang_wu 掌握                                       │
│                                                                             │
│  本周能力变化:                                                              │
│  • @小李 K8s基础运维: 未达标 → 已达标 ✅                                  │
│  • Oracle性能领域团队覆盖率: 50% → 67% ↑                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键交互**：

1. **KPI 行**：5 张 MetricCard，点击后下钻到对应详情页。
2. **Incident 态势卡片**：
   - 若当前无活跃 P1/P2，卡片显示"🟢 当前无活跃高优事故"。
   - 状态灯颜色自动根据系统评估变化：🟢进展正常 / 🟡需要关注（超时均值） / 🔴建议介入（2倍均值且无进展）。
   - "指派支援" → 弹出成员选择器，显示每个人在该 Skill 上的熟练度，方便选最合适的支援者。
3. **单点风险**：点击"启动传承计划" → 打开传承计划 Wizard（选择接班人、设定 deadline、生成 Skill 交接清单）。

---

### 8.2 团队技能雷达（SkillRadarPage）

**页面定位**：团队能力的全景可视化，回答"团队在哪些领域强、哪些领域弱、哪里有单点风险"。

**URL**: `/team/radar`

**布局结构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DB-SRE 技能雷达                                    [📅 时间范围 ▾] [⚙️]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  左侧: 雷达图总览                    右侧: 选中领域详情                     │
│  ┌───────────────────┐ ┌─────────────────────────────────────────────────┐  │
│  │                   │ │  Oracle 高可用(RAC)                              │  │
│  │    [雷达图]       │ │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │                   │ │  覆盖率: 1/6人 (16%)  🔴 严重风险                  │  │
│  │  点击扇区查看详情  │ │  深度: 中 (平均掌握度 55%)                        │  │
│  │                   │ │  健康度: 🟢 (Skill近期有更新和使用)               │  │
│  │                   │ │                                                  │  │
│  └───────────────────┘ │  掌握者列表:                                      │  │
│                        │  ┌─────────────────────────────────────────────┐  │  │
│  图例:                 │  │ @li_si  ████████████████░░░░ 92% 专家        │  │
│  ─ 覆盖率 (人数)       │  │ @wang_wu ████████░░░░░░░░░░░ 50% 可部分覆盖  │  │
│  ─ 深度 (平均熟练度)   │  │ @zhang_san ████░░░░░░░░░░░░░ 25% 初学        │  │
│  ─ 健康度 (颜色)       │  │ 其余3人: 未掌握                                │  │
│                        │  └─────────────────────────────────────────────┐  │
│                        │                                                  │  │
│                        │  ⚠️ 风险分析:                                     │  │
│                        │  如果 @li_si 离职，该领域能力将归零               │  │
│                        │  预计 MTTR 上升: 15min → 60min+                  │  │
│                        │                                                  │  │
│                        │  [启动传承计划]  [安排专项培训]  [查看相关Skill]  │  │
│                        └─────────────────────────────────────────────────┘  │
│                                                                             │
│  ─────────────────────────── 单点风险汇总 ─────────────────────────────    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 领域              │ 唯一掌握者 │ 业务影响    │ 风险等级 │ 操作       │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ Oracle RAC        │ @li_si     │ 核心支付    │ 🔴 极高  │ [传承计划] │   │
│  │ K8s网络策略       │ @wang_wu   │ 容器平台    │ 🟡 高    │ [传承计划] │   │
│  │ 数据库灾备演练    │ @li_si     │ 灾备合规    │ 🔴 极高  │ [传承计划] │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**雷达图设计**：

- **维度**：每个技术领域一个轴（Oracle基础管理、Oracle性能诊断、Oracle RAC、K8s基础运维、K8s故障排查、Linux系统调优、监控告警配置、CI/CD流水线）。
- **多层多边形**：
  - 外圈（浅色填充）：团队理论最大覆盖（100%）。
  - 中圈（实线）：当前覆盖率（掌握该领域的人数占比）。
  - 内圈（虚线）：平均深度（熟练度均值）。
- **颜色映射**：若某领域单点风险，该扇区背景色微红提示。
- **交互**：点击雷达图某一扇区 → 右侧详情面板切换。

---

### 8.3 MTTR 趋势与归因分析（MTTRAnalysisPage）

**页面定位**：回答"MTTR 为什么是这个数字、该怎么改善"。

**URL**: `/team/mttr`

**布局结构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MTTR 趋势与归因分析                         [📅 本月 ▾] [📥 导出报告]      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ 总体MTTR │ │ 检测→响应│ │ 响应→诊断│ │ 诊断→修复│ │ 修复→验证│         │
│  │ 22min ↓  │ │ 3min ↓   │ │ 8min ↓   │ │ 9min →   │ │ 2min →   │         │
│  │ vs上月28  │ │ vs上月4  │ │ vs上月12 │ │ vs上月10 │ │ vs上月2  │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
│                                                                             │
│  [📈 MTTR趋势折线图 - 近12周]                                               │
│                                                                             │
│  ──────────────────────── Skill 贡献归因 ────────────────────────────      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   [柱状图对比]                                                      │   │
│  │   使用Skill辅助: MTTR 18min  ████████████████████                   │   │
│  │   未使用Skill:   MTTR 32min  ████████████████████████████████████   │   │
│  │                                                                     │   │
│  │   Skill 对 MTTR 的因果贡献估计: 降低 38% (双重差分法估算)            │   │
│  │   置信度: 82%                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ──────────────────────── 最大瓶颈识别 ──────────────────────────────      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ "诊断→修复" 阶段占比 41% — 最大改善空间                              │   │
│  │                                                                     │   │
│  │ Top3 耗时场景:                                                      │   │
│  │ 1. K8s网络类问题 (平均诊断15min) — Skill覆盖不足 ⚠️                   │   │
│  │    → [查看缺口详情] [推动补齐Skill]                                  │   │
│  │ 2. Oracle性能类问题 (平均诊断10min) — Skill部分过时 🟡                │   │
│  │    → [查看待更新Skill]                                              │   │
│  │ 3. 跨服务链路追踪 (平均诊断12min) — 无对应Skill 🔴                    │   │
│  │    → [发起Skill创建]                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**图表设计**：

- **MTTR 趋势折线图**：X 轴为周/月，Y 轴为分钟。主线为团队实际 MTTR，虚线为行业基准（DORA Elite/High 分界线）。叠加柱状图表示每周 incident 数量。
- **阶段拆解堆叠面积图**：展示各阶段（检测→响应→诊断→修复→验证）随时间的变化占比。
- **因果归因可视化**：使用"瀑布图"或"对比柱状图"直观展示"有 Skill" vs "无 Skill"的差异。

---

### 8.4 人员管理与成长追踪（MembersPage）

**页面定位**：团队成员列表 + 单人深度成长追踪 + 1:1 会议准备材料。

**URL**: `/team/members`

**布局**：左侧成员列表（带进度条迷你图），右侧选中成员详情面板。

**成员列表行设计**：

```
┌──────────────────────────────────────────────────────────────────────────┐
│ [👤] @小李 (入职3个月)  ████████░░░░ 68%  整体进度                        │
│     Oracle基础 ████████████████ 95% ✅  │  K8s基础 ██████████░░░░ 62% ⏳   │
│     本周: 首次独立处理 Oracle 性能问题 ✨                                │
│     [查看成长追踪] [生成1:1准备材料]                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

**详情面板标签页**：

1. **能力进度**：各领域达标情况的进度条 + 与同期新人对比线。
2. **活动时间线**：最近 30 天的 Skill 使用、incident 处理、学习行为时间线。
3. **1:1 准备材料**（按钮生成）：
   - 自上上次 1:1 以来的数据摘要。
   - "可能的聊天话题"建议（基于数据洞察自动推导）。
4. **传承关系**：该成员作为师傅/徒弟的知识传递网络微型图。

---

### 8.5 排班与技能覆盖（SchedulingPage）

**页面定位**：排班表 + 技能覆盖风险分析的一体化视图。

**URL**: `/team/schedule`

**布局**：

- **上方**：周排班甘特式网格（周一到周日 × 日班/夜班），格子内显示值班人头像。
- **风险分析栏**：排班表下方自动渲染风险卡片。
  - 🔴 高风险：某时段关键领域覆盖低于安全线（如夜班无 Oracle 专家）。
  - 🟡 中风险：单人值班且其某领域熟练度不足。
- **优化建议**："将周三夜班 @小李 和周四夜班 @wang_wu 互换 → Oracle 覆盖风险从🔴降至🟢"，附带"应用建议"按钮（Mock 模式下点击后更新排班表状态）。

---

### 8.6 汇报材料（ReportsPage）

**页面定位**：周报/月报/资源请求报告的自动生成与预览。

**URL**: `/team/reports`

**功能分区**：

1. **报告列表**：本周报、上月月报、资源请求草稿、历史报告归档。
2. **报告预览器（点击后展开）**：
   - 左侧：报告大纲导航（可靠性指标 / 团队效能 / 能力建设 / 下周重点）。
   - 右侧：渲染后的报告内容，所有数据自动填充。
   - 顶部操作栏：[编辑] [导出 Word] [直接发送给 Manager] [分享链接]。
3. **资源请求支持**：独立 Tab，输入新方向名称后展示能力差距分析 + 补齐方案对比（内部培养 vs 外部招聘 vs 混合方案）。

---

## 九、第三层：高管层页面设计

### 9.1 组织级一页纸仪表板（ExecutiveDashboardPage）

**页面定位**：高管每周早上的"第一屏"。30 秒内获取组织技术状态的核心结论。

**URL**: `/executive`

**布局结构（极度压缩）**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  技术组织周状态 │ W20 2026 │ 一页纸摘要                           [📥导出] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  总体状态: 🟢 稳定                                                          │
│                                                                             │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐                  │
│  │ 可靠性          │ │ 能力资产        │ │ 人才风险        │                  │
│  │                 │ │                 │ │                 │                  │
│  │ SLO: 99.96% ✅  │ │ 活跃Skill: 312  │ │ 单点依赖: 7     │                  │
│  │ 目标: 99.9%     │ │ ↑8 vs上周       │ │ ↓1 vs上周       │                  │
│  │                 │ │                 │ │                 │                  │
│  │ P1: 0  P2: 2    │ │ 覆盖率: 73%     │ │ 本月离职: 1     │                  │
│  │ MTTR: 25min ↓19%│ │ 复用率: 22%     │ │ 传承计划已启动  │                  │
│  └────────────────┘ └────────────────┘ └────────────────┘                  │
│                                                                             │
│  ───────────────────────── 需要您关注 ────────────────────────────────      │
│                                                                             │
│  ⚠️ K8s高级运维领域能力缺口持续3周未改善                                   │
│     → 3个团队均报告该领域人才短缺                                          │
│     → 建议: 专项招聘或外部培训投入决策                                       │
│     [查看详细分析] [发起预算申请]                                           │
│                                                                             │
│  [查看各团队明细]                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**设计原则**：

- **一屏原则**：核心内容严格限制在一屏（viewport height）内，无需滚动。
- **红绿灯语言**：所有状态用色标表达，避免高管阅读文字。
- **关注区克制**："需要您关注"仅当真正需要决策时出现，避免狼来了效应。
- **下钻路径**：每个卡片/数字均可点击，进入对应详情页（如 MTTR 数字点击 → MTTR 分析页，Skill 数量点击 → AI 治理页）。

---

### 9.2 AI 治理控制台（AIGovernancePage）

**页面定位**：高管对组织级 AI 使用的全景管控。

**URL**: `/executive/governance`

**标签页结构**：

#### Tab 1: AI 使用全景（月度报告）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  AI Skill 使用月报 │ 2026年5月                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [环形图]  AI辅助生成: 189 (61%)  │  纯人工: 123  │  纯AI未审核: 23 ⚠️    │
│                                                                             │
│  安全状态:                                                                  │
│  🟢 合规 289个 (93%)                                                        │
│  🟡 待审核 18个 (5%)   → 最高风险: 3个涉及生产环境配置变更                  │
│  🔴 需要处理 5个 (2%)   → [查看详情]                                        │
│                                                                             │
│  合规率趋势: [91% → 93% 折线图]                                             │
│                                                                             │
│  本月治理动作: 审核完成34个 │ 修正8个 │ 拒绝/归档3个 │ 自动发布112个        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Tab 2: 治理策略管理

- 策略表格：风险等级（低/中/高/极高） × AI 生成后处理方式（自动发布 / 需确认 / 需审批 / 需安全审核）。
- 内容安全规则开关：硬编码凭证拦截 / IP 脱敏 / 权限检查。
- AI 行为边界声明：AI 可以做什么 / 不可以做什么（文本展示 + 编辑能力）。
- 审计要求配置：保留年限、报告周期、审批层级。

#### Tab 3: Skill 冲突检测

- 冲突列表表格：严重冲突（红色置顶）+ 轻微不一致（可折叠）。
- 每行展示：Skill A vs Skill B 的矛盾描述、影响分析、建议动作。
- "发起协调流程"按钮 → 弹出创建会议/发送通知的 Wizard。

---

### 9.3 战略对齐（StrategyAlignPage）

**页面定位**：将组织目标分解为所需 Skill 能力，展示当前覆盖率与差距。

**URL**: `/executive/strategy`

**核心可视化：桑基图（Sankey Diagram）**

```
左侧: 组织目标                    中间: 能力域                     右侧: 具体Skill
┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
│ P1 MTTR降低30%  │────────────→│ 故障诊断自动化  │────────────→│ 自动日志分析Skill│
│                 │             │                 │────────────→│ 告警关联诊断Skill│
│                 │────────────→│ On-call团队达标 │────────────→│ 值班技能培训    │
│ SLO 99.99%      │             │                 │────────────→│ 专家后备机制    │
│                 │────────────→│ 变更零故障      │────────────→│ 变更预检Skill   │
└─────────────────┘             └─────────────────┘             └─────────────────┘
```

**流量颜色规则**：

- 绿色流量：当前满足度 ≥ 80%。
- 黄色流量：当前满足度 50%-80%。
- 红色流量：当前满足度 < 50%。

**页面分区**：

1. **目标输入区**：高管可输入/选择组织目标，系统自动分解。
2. **桑基图主视觉**：交互式，hover 某条流量显示具体数值（当前覆盖率 / 目标要求 / 差距）。
3. **差距清单**：右侧列表展示所有红色/黄色流量，按缺口大小排序，附带"建议投入"估算。

---

### 9.4 组织能力规划器（OrgPlannerPage）

**页面定位**："往前看"——未来 12 个月的能力规划与 What-If 模拟。

**URL**: `/executive/planner`

**布局**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  12个月能力前瞻 │ 2026 H2 - 2027 H1                        [What-If模拟 ▾] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [表格]                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 能力方向      │ 当前 │ 6个月后 │ 12个月后 │ 投入建议 │ 趋势折线迷你图│   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ K8s/云原生    │ L2.5 │ L3.5    │ L4.0     │ ★★★★★   │ 📈           │   │
│  │ AI Ops        │ L1.0 │ L2.0    │ L3.0     │ ★★★★☆   │ 📈           │   │
│  │ 可观测性(eBPF)│ L0   │ L1.5    │ L2.5     │ ★★★☆☆   │ 📈           │   │
│  │ 安全运维      │ L2.0 │ L2.5    │ L3.0     │ ★★★☆☆   │ 📈           │   │
│  │ 传统DB运维    │ L4.0 │ L4.0    │ L3.5↓    │ ★☆☆☆☆   │ 📉           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  关键里程碑时间线:                                                          │
│  ● Q3-2026: K8s能力覆盖率达80%                                              │
│  ● Q4-2026: AI Ops试点团队达到L2                                            │
│  ● Q1-2027: eBPF可观测性替代30%传统监控                                     │
│  ● Q2-2027: K8s达到L4预测式水平                                             │
│                                                                             │
│  所需资源总览: 当前52人 → 建议58人 (+6 HC)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What-If 模拟器交互**：

- 顶部条件选择器："如果 H2 零招聘" / "如果预算增加 20%" / "如果 K8s 延后 3 个月"。
- 选择后表格数据实时重算，变化的单元格用动画高亮（如从 L3.5 变为 L3.2，数字闪黄）。
- 底部自动生成"风险评估"文本（如"零招聘方案下 AI Ops 方向纯内部培养不现实"）。

---

### 9.5 成熟度评估（MaturityAssessmentPage）

**页面定位**：组织级 SRE 能力成熟度评估（SFM 模型 L1-L5）。

**URL**: `/executive/maturity`

**核心视觉：成熟度阶梯图**

```
      L5 自进化  │████████████░░░░░░░░│  得分: 2.1/5.0
                 │                    │
      L4 预测式  │████████████████░░░░│  得分: 3.8/5.0
                 │                    │
  →   L3 系统化  │████████████████████│  得分: 5.0/5.0  ← 当前所在层
                 │                    │
      L2 积累式  │████████████████░░░░│  得分: 3.5/5.0
                 │                    │
      L1 临时式  │████████████████████│  得分: 5.0/5.0
                 └────────────────────┘
```

**页面内容**：

1. **总体评估**：大字显示当前等级（"L3 系统化 — 得分 3.2/5.0"），附带向 L4 的预计时间。
2. **维度得分雷达图**：5 个维度（Skill覆盖度、知识传承效率、AI治理成熟度、自动化程度、度量与持续改进）。
3. **提升路径**：L3→L4 需要重点突破的方向列表，每个方向带预计投入和预期效果。
4. **与行业基准对比**：横向柱状图展示各维度 vs 行业均值的位置。
5. **团队子评分**：Tab 切换查看各团队的成熟度子评分。

---

### 9.6 集团汇报材料（BoardReportPage）

**页面定位**：自动生成面向集团/董事会的汇报材料，可直接导出为 PPT。

**URL**: `/executive/board-report`

**功能设计**：

1. **Slide 模拟器**：Web 端模拟 PPT 的翻页体验，每页一个核心信息。
   - Page 1: 一句话总结
   - Page 2: 关键成果（可靠性 / 效率 / 人才 / 协同）
   - Page 3: 技术能力资产概览（Skill 总量、增长、健康度、AI 占比）
   - Page 4: 风险与需求（能力缺口、预算请求）
   - Page 5: 下季度目标
2. **每页编辑能力**：点击文字进入轻度编辑（仅改文案，不改数据）。
3. **导出**：右上角 [导出为 PPT] [导出为 PDF] 按钮（Mock 模式下导出为带水印的演示文件或提示"演示模式"）。
4. **数据锁定提示**：所有数据旁有小锁图标，hover 显示"数据来自 SkillForge，自动更新"，强调不可随意编造。

---


---

## 十、通用组件设计

### 10.1 SkillCard（Skill 信息卡片）

**用途**：全站最高频组件，出现在诊断推荐、Skill 列表、冲突检测、传承计划等所有涉及 Skill 展示的位置。

**Props 接口**：

```typescript
interface SkillCardProps {
  skill: Skill;
  variant?: 'compact' | 'default' | 'detailed';
  showHealth?: boolean;          // 是否显示健康度色标
  showActions?: boolean;         // 是否显示操作按钮
  showTrend?: boolean;           // 是否显示近30天成功率趋势 sparkline
  highlightMatch?: string;       // 高亮匹配的文本（来自搜索）
  onClick?: () => void;
  onEdit?: () => void;
  onExecute?: () => void;
}
```

**三种变体**：

- **compact**：单行高度，仅名称 + 成功率 + 使用次数。用于列表内嵌、时间线附件。
- **default**：卡片高度，含名称、作者、成功率、最近使用、步骤预览（前3步）、操作按钮。用于网格列表。
- **detailed**：展开高度，含完整元信息（依赖、治理状态、变更日志、适用环境）。用于详情 Drawer 内和报告引用。

**视觉规范**：

- 左侧 4px 色条表示健康度（green-500 / yellow-500 / red-500 / gray-400）。
- 右上角若 `aiGenerated === true`，显示 `🤖 AI` 小徽章。
- hover 时整体上浮 2px + shadow 加深，暗示可点击。

---

### 10.2 MetricCard（指标概览卡片）

**用途**：Dashboard 顶部 KPI 行、高管一页纸、汇报材料中的关键数字展示。

**Props 接口**：

```typescript
interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;                 // "min" / "%" / "个"
  trend?: 'up' | 'down' | 'flat';
  trendValue?: string;           // "↓12% vs上周"
  status?: 'good' | 'warning' | 'danger' | 'neutral';
  icon?: LucideIcon;
  onClick?: () => void;
  loading?: boolean;
}
```

**视觉规范**：

- 背景：卡片底色根据主题自适应（slate-800 暗色 / white 亮色）。
- 数值字体：text-3xl font-bold tracking-tight。
- 趋势文字：绿色表示改善（MTTR 下降）、红色表示恶化，与趋势方向解耦——由调用方传入语义化的 `status`。
- 底部微型进度条：若该指标有目标值，可传入 `target` 和 `current`，底部渲染细进度条。

---

### 10.3 IncidentBadge（事故等级徽章）

**Props**：`priority: 'P1' | 'P2' | 'P3' | 'P4'`

**色标映射**：

| 等级 | 背景色 | 文字色 | 含义 |
|------|--------|--------|------|
| P1 | red-600 | white | 核心业务中断 |
| P2 | orange-500 | white | 重大功能受损 |
| P3 | yellow-500 | black | 一般问题 |
| P4 | blue-400 | white | 轻微/优化项 |

---

### 10.4 MaturityBadge（成熟度徽章）

**Props**：`level: number` (1.0 - 5.0)

**展示规则**：

- 整数部分决定标签：L1 临时式 / L2 积累式 / L3 系统化 / L4 预测式 / L5 自进化。
- 小数部分决定进度环填充度（如 3.2 = L3 环填充 20% 到 L4）。
- 颜色阶梯：L1 灰 / L2 蓝 / L3 靛蓝 / L4 紫 / L5  emerald。

---

### 10.5 SkillStepper（Skill 步骤引导器）

**用途**：在诊断详情 Drawer 和演练场中，引导用户按步骤执行 Skill。

**Props**：

```typescript
interface SkillStepperProps {
  steps: SkillStep[];
  currentStep: number;           // 当前所在步骤（从1开始）
  onStepComplete?: (stepIndex: number) => void;
  onExecuteCommand?: (command: string) => void;
  readonly?: boolean;            // 只读模式（历史回顾）
}
```

**交互设计**：

- 左侧垂直步骤条（Stepper），已完成步骤打勾、当前步骤高亮脉冲动画、未开始步骤灰显。
- 右侧步骤详情：标题、描述、命令（若存在则渲染为可复制代码块）、预计耗时、验证方式。
- 当前步骤若有命令，显示 [复制命令] [在终端执行（演示）] 按钮。
- 决策树节点：以分支卡片形式展示，用户选择 True/False 后进入对应分支。

---

### 10.6 UserAvatar（用户头像组件）

**Props**：

```typescript
interface UserAvatarProps {
  user: User;
  size?: 'sm' | 'md' | 'lg';
  showStatus?: boolean;          // 右下角在线状态小圆点
  showTooltip?: boolean;         // hover 显示微型 Profile
  onClick?: () => void;
}
```

**状态色标**：在线(green) / 忙碌(red) / 离线(gray) / 值班中(orange 闪烁)。

**微型 Profile Tooltip**：hover 0.5s 后弹出，显示姓名、职位、当前负责领域 Top3、最近活跃时间。

---

### 10.7 TimeRangePicker（时间范围选择器）

**预设选项**：今日 / 本周 / 本月 / 本季度 / 过去7天 / 过去30天 / 过去90天 / 自定义。

**交互**：点击后下拉选择，选择后触发 `onChange(range)`，全局图表/数据自动刷新。

---

### 10.8 GlobalSearch（全局搜索）

**触发方式**：Header 中 🔍 图标 或快捷键 `Ctrl+K`。

**搜索范围**：Skill（名称/描述/标签）、Incident（ID/标题）、User（姓名/handle）、Snippet（命令/标题）。

**结果展示**：Command Palette 风格，分类显示结果，键盘上下选择，回车跳转。

---

## 十一、状态管理与数据流

### 11.1 Zustand Store 拆分

#### authStore（身份与角色）

```typescript
interface AuthState {
  currentUser: User | null;
  currentRole: 'engineer' | 'lead' | 'executive';
  switchRole: (role: 'engineer' | 'lead' | 'executive') => void;
}
```

> Mock 模式下 `currentUser` 固定为种子数据中的 `@wang_wu`，通过 `switchRole` 切换视角时，导航菜单和页面权限即时更新，但用户身份不变（模拟"同一个人在不同角色下看到不同内容"）。

#### uiStore（UI 状态）

```typescript
interface UIState {
  theme: 'dark' | 'light' | 'system';
  sidebarCollapsed: boolean;
  activeTimeRange: TimeRange;    // 全局时间范围，影响所有图表
  toasts: ToastItem[];
  globalSearchOpen: boolean;
  setTheme: (t) => void;
  toggleSidebar: () => void;
  setTimeRange: (r) => void;
  openSearch: () => void;
  addToast: (toast) => void;
}
```

#### skillStore（Skill 实体缓存）

```typescript
interface SkillState {
  skills: Record<string, Skill>;   // 以 id 为键的 Map
  skillList: string[];             // 当前列表中的 ID（支持分页/筛选）
  selectedSkillId: string | null;
  filters: SkillFilter;
  loading: boolean;
  
  fetchSkills: (filters?) => Promise<void>;
  fetchSkillDetail: (id: string) => Promise<void>;
  saveSkillDraft: (draft: Partial<Skill>) => Promise<void>;
  setFilters: (f) => void;
}
```

#### incidentStore（事故实体缓存）

```typescript
interface IncidentState {
  incidents: Record<string, Incident>;
  activeIncidentIds: string[];     // 当前活跃事故
  selectedIncidentId: string | null;
  fetchIncidents: () => Promise<void>;
  fetchIncidentDetail: (id: string) => Promise<void>;
}
```

#### layer2Store（Team Leader 专用状态）

```typescript
interface Layer2State {
  teamPulse: TeamPulse | null;
  skillRadar: SkillRadarData | null;
  mttrAnalysis: MTTRAnalysis | null;
  memberProfiles: Record<string, MemberProfile>;
  schedule: ScheduleData | null;
  reports: Report[];
  
  fetchTeamPulse: () => Promise<void>;
  fetchSkillRadar: () => Promise<void>;
  fetchMTTRAnalysis: (range: TimeRange) => Promise<void>;
  generateReport: (type: 'weekly' | 'monthly') => Promise<void>;
}
```

### 11.2 数据流模式

采用 **"Store 驱动 + React Query 风格预取"** 的混合模式：

```
页面加载
  → React Router loader 调用 MSW API（Mock）
  → 数据写入对应 Zustand Store
  → 组件 mount 时从 Store 读取（无额外请求）
  → 用户交互（筛选/翻页）→ 调用 Store action → MSW API → Store 更新 → 组件重渲染
```

**为什么不用 React Query**：纯 Mock 场景下引入 RQ 增加复杂度，Zustand + 简单的 async action 足够。若后续接真实 API，可在 Store action 中无缝替换为 `fetch` 或迁移至 TanStack Query。

### 11.3 Mock API 端点清单

| 方法 | 端点 | 说明 | 消费页面 |
|------|------|------|---------|
| GET | `/api/me` | 当前用户 | 全局 |
| POST | `/api/auth/switch-role` | 切换角色 | 全局 |
| GET | `/api/diagnose` | 智能诊断 | DiagnosePage |
| GET | `/api/skills` | Skill 列表（分页/筛选） | MySkillsPage |
| GET | `/api/skills/:id` | Skill 详情 | SkillDetailPage |
| POST | `/api/skills` | 创建/保存 Skill | MySkillsPage |
| GET | `/api/snippets` | 命令片段列表 | SnippetVaultPage |
| GET | `/api/users/:id/learning-map` | 个人学习地图 | LearningMapPage |
| GET | `/api/arena/scenarios` | 演练场景列表 | ArenaPage |
| GET | `/api/team/pulse` | 团队每日脉搏 | TeamOverviewPage |
| GET | `/api/team/radar` | 技能雷达数据 | SkillRadarPage |
| GET | `/api/team/mttr` | MTTR 分析数据 | MTTRAnalysisPage |
| GET | `/api/team/members` | 成员列表 | MembersPage |
| GET | `/api/team/members/:id/progress` | 成员成长进度 | MembersPage |
| GET | `/api/team/schedule` | 排班数据 | SchedulingPage |
| GET | `/api/team/reports` | 报告列表 | ReportsPage |
| POST | `/api/team/reports/generate` | 生成报告 | ReportsPage |
| GET | `/api/org/snapshot` | 组织级周状态 | ExecutiveDashboardPage |
| GET | `/api/org/governance` | AI 治理报告 | AIGovernancePage |
| GET | `/api/org/conflicts` | Skill 冲突列表 | AIGovernancePage |
| GET | `/api/org/strategy-align` | 战略对齐数据 | StrategyAlignPage |
| GET | `/api/org/planner` | 能力规划数据 | OrgPlannerPage |
| POST | `/api/org/planner/simulate` | What-If 模拟 | OrgPlannerPage |
| GET | `/api/org/maturity` | 成熟度评估 | MaturityAssessmentPage |
| GET | `/api/org/board-report` | 集团汇报材料 | BoardReportPage |
| GET | `/api/incidents` | Incident 列表 | 全局 |
| GET | `/api/incidents/:id` | Incident 详情 | IncidentDetailPage |
| GET | `/api/search` | 全局搜索 | GlobalSearch |

---

## 十二、交互设计规范

### 12.1 色彩系统

**主色调**：

```css
:root {
  --primary-50: #eff6ff;   --primary-100: #dbeafe;
  --primary-200: #bfdbfe;  --primary-300: #93c5fd;
  --primary-400: #60a5fa;  --primary-500: #3b82f6;
  --primary-600: #2563eb;  --primary-700: #1d4ed8;
  --primary-800: #1e40af;  --primary-900: #1e3a8a;
}
```

**语义色（全站统一，不可重新定义）**：

| 语义 | 色值（Dark） | 色值（Light） | 用途 |
|------|-------------|--------------|------|
| 成功/健康 | emerald-400 | emerald-600 | 正常状态、达标、成功率高 |
| 警告/注意 | yellow-400 | yellow-600 | 需要关注、中等风险、待审核 |
| 危险/严重 | red-400 | red-600 | 高风险、P1、未达标、错误 |
| 信息/中性 | blue-400 | blue-600 | 提示、进行中、一般信息 |
| 归档/缺失 | slate-500 | slate-400 | 已归档、未掌握、无数据 |
| 单点风险 | orange-400 | orange-600 | 唯一掌握者警告 |

**成熟度专属色阶**：

| 等级 | 颜色 |
|------|------|
| L1 | slate-400 |
| L2 | blue-500 |
| L3 | indigo-500 |
| L4 | violet-500 |
| L5 | emerald-500 |

### 12.2 字体与排版

- **数字/数据**：`font-variant-numeric: tabular-nums` 保证等宽，对齐美观。
- **标题层级**：Page 标题 `text-2xl font-bold`，Section 标题 `text-lg font-semibold`，Card 标题 `text-base font-medium`。
- **行高**：正文 `leading-relaxed` (1.625)，紧凑数据 `leading-snug` (1.375)。

### 12.3 动效规范

- **页面切换**：`fade-in` (opacity 0→1, 150ms) + `slide-up` (translateY 8px→0, 200ms)。
- **数据加载**：Skeleton 占位，禁止用 Spinning Loader 占据整个屏幕（可用局部骨架屏）。
- **数字变化**：MetricCard 数值更新时，`spring` 动画（0.3s），从旧值滚动到新值。
- **图表更新**：Recharts 数据变化时启用 `isAnimationActive={true}`，duration 800ms。
- **hover 反馈**：所有可点击卡片 `transition-all duration-200`，hover 时 `translateY(-2px)` + `shadow-lg`。

### 12.4 响应式断点

| 断点 | 宽度 | 布局调整 |
|------|------|---------|
| mobile | < 768px | 侧边栏隐藏为 Drawer，网格单列，高管仪表板垂直堆叠 |
| tablet | 768px - 1024px | 侧边栏可折叠，网格双列，部分图表简化 |
| desktop | 1024px - 1440px | 标准布局，侧边栏固定 240px |
| wide | > 1440px | 内容区 max-width 1400px 居中，避免过宽阅读困难 |

### 12.5 空状态与错误状态

- **空状态**：统一使用 `EmptyState` 组件，包含插图（Lucide 组合图标）、说明文字、建议动作按钮。
  - 例："暂无 Skill 草稿" → "去处理一个 incident，系统会自动为你生成"
- **错误状态**：局部错误用 Inline Alert（卡片内红色提示），全局错误用 Toast 通知。
- **无权限**：`RoleMismatchPage` 提供"切换到正确角色"的一键按钮，而非冷冰冰的 403。

### 12.6 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+K` / `Cmd+K` | 打开全局搜索 |
| `Ctrl+/` | 切换深色/浅色主题 |
| `Ctrl+B` | 折叠/展开侧边栏 |
| `Esc` | 关闭 Drawer / Modal / 搜索 |
| `1/2/3` | 快速切换角色（1=Engineer, 2=Lead, 3=Executive） |

---

## 附录 A：Mock 数据核心示例

以下提供部分种子数据的精确示例，供开发时直接复制到 `mocks/seeds/initialData.ts`。

### A.1 用户种子（3人示例）

```typescript
export const seedUsers: User[] = [
  {
    id: 'user_wang_wu',
    name: '王伟',
    handle: '@wang_wu',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=wangwu',
    role: 'engineer',
    teamId: 'team_db_sre',
    title: 'Senior SRE',
    joinDate: '2021-03-15',
    skillsMastery: [
      { domain: 'Oracle基础管理', level: 95, trend: 'flat' },
      { domain: 'Oracle性能诊断', level: 85, trend: 'up' },
      { domain: 'Oracle高可用(RAC)', level: 50, trend: 'up' },
      { domain: 'K8s故障排查', level: 22, trend: 'up' },
    ],
    status: 'online',
    metrics: { totalIncidents: 147, avgMTTR: 18, skillsCreated: 12, skillsAdoptedByOthers: 89 }
  },
  {
    id: 'user_li_si',
    name: '李四',
    handle: '@li_si',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=lisi',
    role: 'engineer',
    teamId: 'team_db_sre',
    title: 'Staff SRE - Oracle 专家',
    joinDate: '2019-06-01',
    skillsMastery: [
      { domain: 'Oracle基础管理', level: 98, trend: 'flat' },
      { domain: 'Oracle性能诊断', level: 96, trend: 'flat' },
      { domain: 'Oracle高可用(RAC)', level: 94, trend: 'flat' },
      { domain: '数据库灾备演练', level: 91, trend: 'flat' },
    ],
    status: 'oncall',
    metrics: { totalIncidents: 312, avgMTTR: 12, skillsCreated: 34, skillsAdoptedByOthers: 256 }
  },
  {
    id: 'user_lead_zhang',
    name: '张经理',
    handle: '@zhang_mgr',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=zhangmgr',
    role: 'lead',
    teamId: 'team_db_sre',
    title: 'DB-SRE Team Lead',
    joinDate: '2018-01-10',
    skillsMastery: [
      { domain: '团队管理', level: 88, trend: 'flat' },
      { domain: 'Oracle性能诊断', level: 78, trend: 'flat' },
    ],
    status: 'busy',
    metrics: { totalIncidents: 89, avgMTTR: 22, skillsCreated: 8, skillsAdoptedByOthers: 45 }
  },
];
```

### A.2 Skill 种子（2个示例）

```typescript
export const seedSkills: Skill[] = [
  {
    id: 'skill-oracle-slow-query-diag-v3',
    name: 'Oracle 慢查询诊断与优化',
    version: 3.2,
    authorId: 'user_li_si',
    teamId: 'team_db_sre',
    createdAt: '2025-09-15T10:00:00Z',
    lastUsedAt: '2026-05-10T14:32:00Z',
    useCount: 47,
    successRate: 0.89,
    avgResolutionTime: 18,
    classification: {
      domain: ['database', 'oracle', 'performance'],
      scenario: ['incident', 'optimization'],
      difficulty: 'intermediate',
      riskLevel: 'medium',
    },
    dependencies: {
      requiredSkills: ['skill-basic-oracle-admin', 'skill-sql-profiling'],
      requiredAccess: ['oracle-prod-readonly', 'awr-report'],
    },
    content: {
      triggerConditions: 'Oracle DB 响应时间 > SLO阈值 持续5分钟',
      diagnosisSteps: [
        { order: 1, title: '生成 AWR 报告', description: '执行 awrrpt.sql 获取最近1小时快照对比', command: '@?/rdbms/admin/awrrpt.sql', estimatedTime: 3, verification: 'AWR 报告成功生成且包含 SQL Ordered by Elapsed Time' },
        { order: 2, title: '定位 Top SQL', description: '从 AWR 中提取 Elapsed Time 最高的3条 SQL', command: "SELECT sql_id, elapsed_time/1000000 as elapsed_sec FROM v\$sql ORDER BY elapsed_time DESC FETCH FIRST 3 ROWS ONLY;", estimatedTime: 5, verification: '已确定高耗时 SQL 的 sql_id' },
        { order: 3, title: '分析执行计划', description: '使用 DBMS_XPLAN 查看目标 SQL 的执行计划', command: 'SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(:sql_id));', estimatedTime: 5, verification: '确认是否存在全表扫描或索引跳跃' },
        { order: 4, title: '索引优化建议', description: '根据执行计划建议创建或重建索引', command: '-- 示例: CREATE INDEX idx_xxx ON table(column);', estimatedTime: 3, verification: '索引创建成功且执行计划改善' },
        { order: 5, title: '验证与监控', description: '观察 SQL 执行时间是否回到基线', command: '', estimatedTime: 2, verification: 'DB 响应时间恢复至 SLO 以内' },
      ],
      executionActions: [
        { type: 'command', content: 'ALTER INDEX idx_xxx REBUILD ONLINE;', safetyLevel: 'medium' },
      ],
      rollbackPlan: '若索引重建后性能更差，执行 ALTER INDEX idx_xxx REBUILD; 并联系 DBA 回退',
    },
    governance: {
      approvalStatus: 'approved',
      reviewerId: 'user_lead_zhang',
      complianceTags: ['prod-safe', 'no-data-mutation'],
      expiryReviewDate: '2026-11-15',
      aiGenerated: false,
    },
    evolution: {
      parentSkillId: 'skill-oracle-slow-query-diag-v2',
      changeLog: '新增 AWR 自动分析步骤，替换手动 ASH 查询',
      deprecationCandidates: ['skill-manual-ash-query'],
    },
    healthStatus: 'healthy',
    healthScore: 88,
  },
  {
    id: 'skill-k8s-pod-evicted-diag-v1',
    name: 'K8s Pod 被驱逐诊断',
    version: 1.5,
    authorId: 'user_wang_wu',
    teamId: 'team_db_sre',
    createdAt: '2025-12-01T08:00:00Z',
    lastUsedAt: '2026-05-14T09:15:00Z',
    useCount: 32,
    successRate: 0.72,
    avgResolutionTime: 25,
    classification: {
      domain: ['kubernetes', 'troubleshooting'],
      scenario: ['incident'],
      difficulty: 'intermediate',
      riskLevel: 'low',
    },
    dependencies: { requiredSkills: [], requiredAccess: ['kubectl-readonly'] },
    content: {
      triggerConditions: 'K8s Pod 状态为 Evicted',
      diagnosisSteps: [
        { order: 1, title: '查看 Pod Events', description: '使用 kubectl describe 查看被驱逐原因', command: 'kubectl describe pod <pod-name> -n <namespace>', estimatedTime: 2, verification: 'Events 中显示驱逐原因（如 DiskPressure / MemoryPressure）' },
        { order: 2, title: '检查节点资源', description: '查看节点磁盘/内存/CPU 压力状态', command: 'kubectl describe node <node-name>', estimatedTime: 5, verification: '确认节点 Conditions 中是否存在 Pressure=True' },
        { order: 3, title: '根因定位与修复', description: '根据驱逐原因清理资源或调整 limit/request', command: '-- 视根因而定', estimatedTime: 15, verification: '节点压力解除或资源配置合理' },
      ],
      executionActions: [],
      rollbackPlan: 'N/A - 本 Skill 为诊断类，不涉及变更',
    },
    governance: {
      approvalStatus: 'approved',
      reviewerId: 'user_lead_zhang',
      complianceTags: ['read-only'],
      expiryReviewDate: '2026-12-01',
      aiGenerated: true,
      aiConfidence: 0.91,
    },
    evolution: {
      changeLog: 'v1.5 补充了 DiskPressure 场景的专门处理步骤',
    },
    healthStatus: 'attention',
    healthScore: 62,
  },
];
```

### A.3 Incident 种子（1个完整示例）

```typescript
export const seedIncidents: Incident[] = [
  {
    id: 'INC-2024-0789',
    title: 'DB-prod-01 响应超时',
    priority: 'P2',
    status: 'closed',
    createdAt: '2026-05-14T02:15:00Z',
    resolvedAt: '2026-05-14T02:42:00Z',
    mttr: 27,
    assigneeId: 'user_wang_wu',
    commanderId: 'user_lead_zhang',
    teamId: 'team_db_sre',
    context: {
      alertSource: 'PagerDuty-PD-89432',
      affectedService: 'Oracle-prod-01',
      environment: 'prod',
      initialSymptom: '数据库响应时间 > 5s，连接池告警',
    },
    skillUsage: [
      { skillId: 'skill-oracle-slow-query-diag-v3', startedAt: '2026-05-14T02:17:00Z', completedAt: '2026-05-14T02:38:00Z', currentStep: 5, totalSteps: 5, success: true },
    ],
    timeline: [
      { timestamp: '2026-05-14T02:15:00Z', type: 'alert', description: '告警触发: DB-prod-01 响应超时', source: 'pagerduty' },
      { timestamp: '2026-05-14T02:17:00Z', type: 'response', actorId: 'user_wang_wu', description: '@wang_wu 响应告警，开始排查', source: 'pagerduty' },
      { timestamp: '2026-05-14T02:20:00Z', type: 'diagnosis', actorId: 'user_wang_wu', description: '执行 AWR 报告分析', source: 'vscode' },
      { timestamp: '2026-05-14T02:25:00Z', type: 'info', actorId: 'user_wang_wu', description: '在 Teams 中 @li_si 请求协助', source: 'teams' },
      { timestamp: '2026-05-14T02:32:00Z', type: 'diagnosis', actorId: 'user_wang_wu', description: '定位到 Top SQL: sql_id=abc123，执行计划全表扫描', source: 'vscode' },
      { timestamp: '2026-05-14T02:38:00Z', type: 'action', actorId: 'user_wang_wu', description: '执行索引在线重建', source: 'vscode' },
      { timestamp: '2026-05-14T02:42:00Z', type: 'resolution', actorId: 'user_wang_wu', description: '确认 DB 响应时间恢复正常 (< 200ms)', source: 'manual' },
    ],
    postmortem: {
      rootCause: '新上线查询缺少合适索引，导致全表扫描引发 IO 瓶颈',
      actionItems: [
        '为 query_id=abc123 添加索引到部署 checklist',
        '将本次诊断过程 Skill 化为 "AWR快速定位慢查询"',
        '为该类型告警添加自动诊断 runbook',
      ],
      lessonsLearned: ['上线前 SQL 审查需包含执行计划检查', '慢查询告警应直接关联 Skill 推荐'],
      skillsCreated: ['skill-awr-quick-slow-query-v1'],
    },
  },
];
```

---

## 附录 B：开发阶段建议

### Phase 1：骨架与第一层（Week 1-2）

1. 搭建项目脚手架（Vite + React + Tailwind + shadcn/ui）。
2. 配置 MSW，导入种子数据。
3. 实现全局布局（AppShell + Sidebar + Header + RoleSwitcher）。
4. 实现第一层核心页面：诊断中心、我的 Skill 工坊、命令片段库。
5. 实现共享页面：Skill 详情页、Incident 详情页。

### Phase 2：第二层与图表（Week 3-4）

1. 实现团队概览、技能雷达（Recharts 雷达图）、MTTR 分析（折线图 + 柱状图）。
2. 实现人员管理、排班优化。
3. 实现汇报材料生成与预览。

### Phase 3：第三层与高管视图（Week 5-6）

1. 实现一页纸仪表板（极度压缩布局）。
2. 实现 AI 治理控制台（表格 + 策略编辑器）。
3. 实现战略对齐桑基图（可用简化版条形图替代，若桑基图实现成本高）。
4. 实现成熟度评估、集团汇报 Slide 模拟器。

### Phase 4：打磨与响应式（Week 7）

1. 全局响应式适配（tablet + mobile）。
2. 暗色/浅色主题完善。
3. 动效与微交互调优。
4. Mock 数据丰富度补充（确保演示流畅）。

---

*文档版本：v1.0*  
*创建日期：2026-05-15*  
*适用范围：SkillForge Web Dashboard 前端开发*  
*状态：可直接进入编码阶段*


---

## 附录 C：Teams Bot 与 VS Code 插件入口 Mock 设计

> **设计前提**：本附录仅描述 Teams / VS Code 端的**入口级 UI Mock**——即插件如何展示信息、用户如何点击跳转到 Web Dashboard 的对应页面。Web 页面的详细设计已在第七~九章完成，此处通过路由参数建立一一映射关系。
>
> **Mock 原则**：纯前端演示时，Teams / VS Code 的"插件界面"以 Web 页面内的**模拟器面板**形式呈现，方便在一个浏览器标签内完成全链路演示。

---

### C.1 整体入口映射关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        入口层（Teams / VS Code）                             │
│                        ─────────────────────────                             │
│  Teams Bot 卡片 ──点击──┐                                                   │
│  VS Code 侧边栏 ──点击──┼──→ Web Dashboard 对应页面（带 context 参数）      │
│  VS Code 状态栏 ──点击──┤                                                   │
│  告警通知链接   ──点击──┘                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**跳转协议**：所有插件入口统一跳转到 `https://skillforge.example.com/{path}?from={source}&ctx={context}`

| 插件入口场景 | Source 参数 | 跳转 Web 页面 | Context 参数示例 |
|-------------|------------|--------------|-----------------|
| Teams 频道 Bot 推荐 Skill | `teams-channel` | `/diagnose?skillId=xxx` | `incidentId=INC-0801` |
| Teams 专家定位结果 | `teams-expert` | `/profile?userId=xxx` | `domain=Oracle+RAC` |
| Teams 问答捕获确认 | `teams-qa` | `/my-skills?draft=true` | `channelId=xxx` |
| VS Code 诊断面板推荐 | `vscode-diagnose` | `/diagnose?query=xxx` | `errorCode=ORA-04031` |
| VS Code 操作录制提示 | `vscode-recorder` | `/my-skills?tab=drafts` | `incidentId=INC-0789` |
| VS Code 代码内联提示 | `vscode-inline` | `/skill/:id` | `file=postgresql.conf` |
| VS Code Snippet 搜索 | `vscode-snippet` | `/snippets?query=xxx` | `env=Oracle+19c` |
| PagerDuty 告警链接 | `pagerduty` | `/diagnose?alertId=xxx` | `service=DB-prod-01` |

---

### C.2 Teams Bot 入口 Mock

#### C.2.1 演示方式

在 Web Dashboard 中新增一个 **"Teams 模拟器"** 页面（`/simulator/teams`），以聊天界面形式模拟 Teams 频道，展示 SkillForge Bot 的各种卡片消息，点击卡片后直接在当前浏览器内跳转到对应 Web 页面。

#### C.2.2 Incident 频道 Bot 卡片（B1 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  #incident-db-prod-timeout                                     [模拟Teams]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [09:32] @zhang_san: DB 又超时了，响应时间飙到 8s                          │
│                                                                             │
│  [09:32] 🤖 SkillForge (点击展开 ▼)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  检测到相关经验：                                                   │   │
│  │                                                                     │   │
│  │  • Skill "Oracle 慢查询诊断" 可能适用 (成功率 92%)                  │   │
│  │    ──▶ [在 Web 中查看完整步骤]  ←── 点击跳转到 /diagnose?skillId=.. │   │
│  │                                                                     │   │
│  │  • 上次类似问题: INC-2024-0654, 由 @li_si 在 22min 内解决          │   │
│  │    ──▶ [查看历史 incident]  ←── 点击跳转到 /incident/INC-2024-0654 │   │
│  │                                                                     │   │
│  │  • 该问题常见根因: 连接池泄漏(40%) │ 配置错误(35%) │ ...            │   │
│  │                                                                     │   │
│  │  [🚀 一键打开诊断中心]  ←── 点击跳转到 /diagnose?context=oracle     │   │
│  │  [🔕 静默(本次不再提醒)]                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [09:33] @li_si: 看看 AWR，可能是凌晨那个 batch job                      │
│                                                                             │
│  [09:35] @zhang_san: 找到了，sql_id=abc123，全表扫描                     │
│                                                                             │
│  [09:42] @zhang_san: 重建索引后恢复了                                    │
│                                                                             │
│  [09:42] 🤖 SkillForge (点击展开 ▼)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ✓ Incident 已关闭 (耗时 27min)                                     │   │
│  │                                                                     │   │
│  │  检测到一个可复用的解决模式：                                       │   │
│  │  "Oracle AWR自动分析 + 索引在线重建"                                │   │
│  │                                                                     │   │
│  │  [在 Web 中保存为 Skill]  ←── 点击跳转到 /my-skills?draft=INC-0789 │   │
│  │  [忽略]                                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**交互说明**：

- 卡片默认折叠，点击标题展开。这是为了不打扰 Teams 讨论流。
- 所有带下划线的链接都是真实可点击的 Web 跳转。
- "一键打开诊断中心"是大号主按钮，带 `🚀` emoji 强调——这是最高频的转化路径。
- Incident 解决后的自动提示是"无感沉淀"的关键：用户只点一次，就能进入 Web 的 Skill 草稿确认页。

#### C.2.3 专家定位器卡片（B4 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  #general                                                          [Teams]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [14:15] @小李: @SkillForge 谁了解 Oracle RAC 故障切换？                  │
│                                                                             │
│  [14:15] 🤖 SkillForge:                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Oracle RAC 相关领域专家：                                          │   │
│  │                                                                     │   │
│  │  1. 👤 @li_si (DB-SRE)                                              │   │
│  │     解决过 9 次 RAC 相关 incident, 平均 MTTR 15min                  │   │
│  │     创建了 3 个相关 Skill                                           │   │
│  │     当前状态: 在线 ✓ (非值班)                                       │   │
│  │     [查看 li_si 的 Skill 库] ←── 跳转 /profile?userId=user_li_si   │   │
│  │     [在 Web 中发起通话]      ←── 跳转 /profile?userId=user_li_si   │   │
│  │                                                                     │   │
│  │  2. 👤 @chen_qi (Infra)                                             │   │
│  │     解决过 4 次相关 incident                                        │   │
│  │     当前状态: 值班中 ⚠️ (建议非紧急不打扰)                          │   │
│  │                                                                     │   │
│  │  [🔍 查看 Oracle RAC 相关所有 Skill]                                │   │
│  │     ←── 跳转 /search?q=Oracle+RAC&filter=skill                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### C.2.4 问答捕获卡片（B2 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  #sre-help                                                         [Teams]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [11:20] @小李: 怎么查看 K8s pod 被驱逐的原因？                           │
│  [11:22] @大王: kubectl describe pod xxx 看 Events 部分                   │
│  [11:25] @小李: 找到了，是 DiskPressure，谢谢！                           │
│                                                                             │
│  [11:25] 🤖 SkillForge:                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  检测到一次有价值的问答：                                           │   │
│  │                                                                     │   │
│  │  Q: "怎么查看 K8s pod 被驱逐的原因？" (@小李)                       │   │
│  │  A: "kubectl describe pod xxx 看 Events 部分..." (@大王)           │   │
│  │                                                                     │   │
│  │  [在 Web 中保存为 Skill 片段]  ←── 跳转 /my-skills?from=qa&question= │   │
│  │  [忽略]                                                             │   │
│  │                                                                     │   │
│  │  保存后效果: 下次有人问类似问题，Bot 可以直接推荐这个解答           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### C.2.5 每日脉搏私信卡片（F1 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SkillForge Bot (私信)                                             [Teams]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [08:00] 🤖 SkillForge:                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  📊 团队日报 │ 2026-05-15 (Thu) │ DB-SRE Team                       │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  昨夜值班: @wang_wu                                                 │   │
│  │  ✅ 平稳 - 无 P1/P2, 3 个自动恢复告警                               │   │
│  │                                                                     │   │
│  │  Skill 动态:                                                        │   │
│  │  • 昨日团队使用 Skill 5 次, 新增 1 个 Skill 草稿                    │   │
│  │                                                                     │   │
│  │  [📱 在 Web 中打开团队概览]  ←── 跳转 /team?from=daily-pulse       │   │
│  │  [📊 查看 MTTR 趋势]         ←── 跳转 /team/mttr?range=week        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### C.2.6 Incident 实时态势卡片（F2 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SkillForge Bot (私信)                                             [Teams]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [10:45] 🤖 SkillForge ⚡ 实时态势提醒：                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ⚡ INC-0801 (P2) │ DB-prod 响应超时 │ 进行中 22min                  │   │
│  │                                                                     │   │
│  │  响应人: @zhang_san (入职8个月)                                     │   │
│  │                                                                     │   │
│  │  🟡 需要关注 - 已超过该类问题平均解决时间 (15min)                   │   │
│  │     @zhang_san 在该 Skill 上的熟练度: 中等 (65%)                    │   │
│  │                                                                     │   │
│  │  [在 Web 中查看实时态势]  ←── 跳转 /team?focus=INC-0801            │   │
│  │  [查看 zhang_san 的成长追踪] ←── 跳转 /team/members?user=zhang_san │   │
│  │  [指派支援]                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### C.3 VS Code 插件入口 Mock

#### C.3.1 演示方式

在 Web Dashboard 中新增一个 **"VS Code 模拟器"** 页面（`/simulator/vscode`），以 iframe 或 CSS 模拟的 VS Code 界面展示 SkillForge 插件的嵌入效果。点击面板中的按钮/链接后，在当前浏览器打开对应 Web 页面。

#### C.3.2 智能诊断面板（A1 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [文件] [编辑] [选择] ...                              [🔍] [⚙️] [-] [×]  │
├──────────┬──────────────────────────────────────────────────────────────────┤
│          │  src/                                                              │
│  Explorer│    config/                                                         │
│  [×]     │    services/                                                       │
│          │    main.ts                                                         │
│          │                                                                    │
│  🛠️      │  ─────────── Terminal ───────────                                  │
│ SkillForge│  $ sqlplus / as sysdba                                            │
│ [×]      │  ORA-04031: unable to allocate 3896 bytes of shared memory      │
│          │                                                                    │
│  ⚡      │  ┌─────────────────────────────────────────────────────────┐     │
│  检测到  │  │  SkillForge 诊断                              [×]       │     │
│  错误    │  ├─────────────────────────────────────────────────────────┤     │
│          │  │                                                           │     │
│  推荐    │  │  ⚡ 检测到错误: ORA-04031                                │     │
│  解决    │  │                                                           │     │
│  路径    │  │  1. ★ Shared Pool 内存不足诊断                           │     │
│          │  │     成功率: 88% │ 平均: 12min                            │     │
│          │  │     [在 Web 中查看] ←── 跳转 /diagnose?error=ORA-04031  │     │
│          │  │     [一键复制命令]                                       │     │
│          │  │                                                           │     │
│  💡      │  │  2. SGA 动态调优                                         │     │
│  团队    │  │     成功率: 75% │ 平均: 20min                            │     │
│  知识    │  │     [在 Web 中查看]                                      │     │
│          │  │                                                           │     │
│          │  │  💡 @li_si 是该领域专家 (处理过12次类似问题)             │     │
│          │  │     [查看专家档案] ←── 跳转 /profile?userId=user_li_si │     │
│          │  │                                                           │     │
│          │  │  [🔍 打开诊断中心] ←── 跳转 /diagnose?from=vscode      │     │
│          │  └─────────────────────────────────────────────────────────┘     │
│          │                                                                    │
└──────────┴──────────────────────────────────────────────────────────────────┘
```

**交互说明**：

- **自动触发**：当终端检测到 `ORA-` 错误码或 `Exception` 关键字时，侧边栏自动亮灯并弹出诊断面板。
- **右键菜单**：在终端选中错误日志 → 右键 → "SkillForge: Diagnose this" → 打开诊断面板。
- **主按钮**：面板底部的 `[🔍 打开诊断中心]` 是最大转化入口，跳转到 Web 的 `/diagnose` 并携带错误上下文。

#### C.3.3 操作录制与 Skill 生成提示（A2 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ...                                                                        │
│  ─────────── Status Bar ───────────                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  [Ln 12, Col 34]  [UTF-8]  [TypeScript]  [🛠️ ●]  [🔄]  [⏱️ 15min]  │  │
│  │                                           ↑                         │  │
│  │                                SkillForge 正在录制                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  [ Incident 解决后，状态栏弹出通知 ]                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  🛠️ SkillForge                                                      │   │
│  │                                                                     │   │
│  │  检测到一次成功的问题解决 (ORA-04031)                               │   │
│  │  已自动保存为 Skill 草稿                                            │   │
│  │                                                                     │   │
│  │  [在 Web 中查看草稿] ←── 跳转 /my-skills?tab=drafts&incident=0789  │   │
│  │  [不再提示此类]                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**交互说明**：

- 状态栏 `🛠️ ●` 表示正在录制操作序列，hover 显示"已录制 12 条命令"。
- 点击状态栏图标 → 打开 Web 的 `/my-skills?tab=drafts`。
- Incident 解决后的通知 toast 中，"在 Web 中查看草稿"是最强转化入口。

#### C.3.4 上下文感知的代码/配置助手（A3 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  postgresql.conf                                                            │
│  ─────────────────────────────────────────                                  │
│  shared_buffers = 4GB                                                       │
│  ─────────────────────────────────────────                                  │
│  │ 💡 SkillForge                                                          │
│  │ 此值在 Incident INC-2024-0456 中从 2GB 调至 4GB (@wang_wu, 2025-03)   │
│  │ [查看详情] ←── 跳转 /incident/INC-2024-0456                           │
│  │ [查看 Skill "PostgreSQL内存调优"] ←── 跳转 /skill/skill-postgres-mem  │
│  ─────────────────────────────────────────                                  │
│                                                                             │
│  max_connections = 200                                                      │
│  ─────────────────────────────────────────                                  │
│  │ 💡 SkillForge                                                          │
│  │ 团队 Skill 建议此值不超过 CPU 核数×4 (当前主机: 48核, 建议≤192)       │
│  │ [在 Web 中查看调优指南] ←── 跳转 /skill/skill-postgres-conn-tune     │
│  ─────────────────────────────────────────                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**交互说明**：

- 以 **CodeLens / Inline Comment** 形式出现在配置文件/代码下方。
- 每条提示带 1-2 个链接：一个跳转到相关 Incident，一个跳转到对应 Skill 详情页。

#### C.3.5 命令片段库触发（A6 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Terminal                                                                   │
│  ─────────────────────────────────────────                                  │
│  $ sf: check oracle performance                                             │
│       ─────────────────────────────────                                     │
│       │ 🔍 SkillForge Snippets                                            │
│       │                                                                    │
│       │ [1] AWR报告快速生成 (@li_si, 使用47次)                            │
│       │     @?/rdbms/admin/awrrpt.sql                                      │
│       │     [在 Web 中查看] ←── 跳转 /snippets?id=snippet-awr-quick       │
│       │     [一键复制]                                                     │
│       │                                                                    │
│       │ [2] Top 10 SQL by Elapsed Time (@zhang_san, 使用89次)             │
│       │     SELECT sql_id, elapsed_time...                                 │
│       │     [在 Web 中查看] ←── 跳转 /snippets?id=snippet-top-sql         │
│       │     [一键复制]                                                     │
│       │                                                                    │
│       │ [3] 实时会话等待事件分析 (@wang_wu, 使用35次)                      │
│       │     [在 Web 中查看]                                                │
│       │                                                                    │
│       │ [📚 打开片段库] ←── 跳转 /snippets?query=oracle+performance      │
│       └─────────────────────────────────                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### C.3.6 Postmortem 智能助手（A4 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  postmortem-INC-0789.md                                                     │
│  ─────────────────────────────────────────                                  │
│  ## 时间线 (SkillForge 自动生成，请 Review)                                 │
│  ─────────────────────────────────────────                                  │
│  │ 📝 所有时间线事件均来自 SkillForge 记录                               │
│  │ [查看原始记录] ←── 跳转 /incident/INC-0789?tab=timeline               │
│  ─────────────────────────────────────────                                  │
│  - 02:15 - 告警触发: DB-prod-01 响应超时 (来源: PagerDuty)                 │
│  - 02:17 - @wang_wu 响应告警，开始排查 (来源: 操作记录)                    │
│  - 02:20 - 执行 AWR 报告分析 (来源: 终端录制)                              │
│  - 02:25 - 在 Teams 中 @li_si 请求协助 (来源: Teams 记录)                  │
│  - 02:32 - 定位到 Top SQL: sql_id=xxx (来源: 终端录制)                     │
│  - 02:38 - 执行索引重建 (来源: 终端录制)                                   │
│  - 02:42 - 确认恢复正常 (来源: 监控数据)                                   │
│  - 02:45 - 关闭 incident (来源: PagerDuty)                                 │
│                                                                             │
│  ## Action Items (SkillForge 建议，请确认)                                  │
│  ─────────────────────────────────────────                                  │
│  - [ ] 为 query_id=xxx 添加索引到部署 checklist                            │
│  - [ ] 将本次诊断过程 Skill 化为 "AWR快速定位慢查询"                       │
│  │   [在 Web 中编辑 Skill 草稿] ←── 跳转 /my-skills?draft=INC-0789      │
│  - [ ] 为该类型告警添加自动诊断 runbook                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### C.3.7 学习路径导航器（A5 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🛠️ SkillForge: 我的技能地图                                               │
│  ─────────────────────────────────────────                                  │
│  Oracle DB ████████████░░░░ 78%                                             │
│    ├─ 基础管理     ████████████████ 95% ✓                                  │
│    ├─ 性能诊断     ██████████████░░ 85%                                    │
│    ├─ 高可用(RAC)  ████████░░░░░░░░ 50%  ← 建议下一步                      │
│    │              [在 Web 中查看学习路径] ←── 跳转 /learning?domain=oracle-rac
│    └─ 安全审计     ████░░░░░░░░░░░░ 25%                                    │
│                                                                             │
│  Kubernetes ██████░░░░░░░░░░ 40%                                            │
│    ├─ 基础概念     ████████████████ 92% ✓                                  │
│    ├─ 部署管理     ██████████░░░░░░ 65%                                    │
│    ├─ 网络/存储    ████░░░░░░░░░░░░ 30%  ← 团队需要                        │
│    │              [在 Web 中查看缺口] ←── 跳转 /team/radar?highlight=k8s-net
│    └─ 故障排查     ██░░░░░░░░░░░░░░ 15%                                    │
│                                                                             │
│  [📱 打开完整学习地图] ←── 跳转 /learning?from=vscode                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### C.3.8 Incident 上下文切换器（A7 场景）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Status Bar                                                                 │
│  ─────────────────────────────────────────                                  │
│  [Ln 12, Col 34] ...  [🛠️ INC-2024-0789: DB超时 ▼]  [⏱️ 12min]          │
│                           ──────────────────────────                        │
│                           │ 切换到:                                        │
│                           │ ├─ INC-2024-0790: API网关5xx (步骤3/5)        │
│                           │ │   [在 Web 中打开] ←── 跳转 /incident/0790   │
│                           │ ├─ TASK: K8s部署配置编写 (暂停)               │
│                           │ │   [在 Web 中打开任务]                       │
│                           │ └─ + 新建上下文                               │
│                           └──────────────────────────                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### C.4 告警系统集成入口（PagerDuty / 监控系统）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [PagerDuty 告警邮件 / 短信 / 应用推送 Mock]                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🔴 P1 告警: 支付服务不可用                                                 │
│  影响: 全部在线支付交易中断                                                 │
│  时间: 2026-05-15 10:23                                                   │
│                                                                             │
│  [📱 在 SkillForge 中查看诊断建议]                                          │
│     ←── 跳转 /diagnose?alertId=PD-12345&service=payment&priority=P1      │
│                                                                             │
│  [📊 查看团队实时态势]                                                      │
│     ←── 跳转 /team?focus=live-incident&alertId=PD-12345                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### C.5 Web 端对应的落地页设计

当用户从插件点击跳转到 Web 时，目标页面需要**感知来源**并给出对应的上文：

#### 从 Teams 诊断卡片进入 `/diagnose`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  智能诊断中心                                                               │
│  ─────────────────────────────────────────                                  │
│  [💬 来自 Teams #incident-db-prod-timeout]  [清除上下文]                    │
│                                                                             │
│  🔍 描述你的问题...                                                         │
│  [ORA-04031 unable to allocate memory]  ←─ 已自动带入 Teams 中的错误码     │
│                                                                             │
│  [开始诊断 →]                                                               │
│                                                                             │
│  ...（下方展示推荐 Skill 链路，与普通诊断页一致）...                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**设计点**：页面顶部显示一个来源上下文条（Source Context Bar），告知用户"你是从哪进来的"，并预填充对应参数。

#### 从 VS Code 草稿提示进入 `/my-skills?tab=drafts`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  我的 Skill 工坊                                                            │
│  ─────────────────────────────────────────                                  │
│  [🛠️ 来自 VS Code 操作录制 │ Incident INC-0789]  [查看操作记录]            │
│                                                                             │
│  ─────────── 自动生成草稿（来自本次 Incident）───────────                   │
│                                                                             │
│  📝 "Oracle AWR自动分析 + 索引在线重建"                                     │
│     来源: INC-0789 (27min) │ 系统置信度: 89%                               │
│     [一键确认发布]  [在 Web 中编辑完善]  [忽略]                             │
│                                                                             │
│  ...（下方是我的 Skill 列表）...                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### C.6 新增 Web 模拟器页面

为在纯前端环境下完整演示"插件 → Web"的链路，在 Web 端新增两个模拟器页面：

| 页面 | 路由 | 说明 |
|------|------|------|
| Teams 模拟器 | `/simulator/teams` | 模拟 Teams 频道聊天界面，展示 Bot 卡片，点击跳转真实 Web 页面 |
| VS Code 模拟器 | `/simulator/vscode` | 模拟 VS Code 界面（侧边栏 + 终端 + 编辑器），展示插件嵌入效果 |

**Teams 模拟器布局**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [← 返回 Dashboard]  Teams 场景模拟器                              [?]     │
├─────────────────────────────────────────────────────────────────────────────┤
│  左侧频道列表          中间聊天区域                    右侧 Web 预览        │
│  ┌──────────┐ ┌──────────────────────────────┐ ┌────────────────────────┐  │
│  │ #general │ │ @zhang_san: DB 又超时了...   │ │                        │  │
│  │ #inc-... │ │ 🤖 SkillForge (展开 ▼)       │ │  点击左侧卡片中的      │  │
│  │ #sre-help│ │ ...                          │ │  "在 Web 中查看"       │  │
│  │ ...      │ │ ...                          │ │  链接后，这里会        │  │
│  │          │ │ [点击任意链接] ─────────────→│ │  渲染对应的 Web 页面   │  │
│  │          │ │                              │ │                        │  │
│  └──────────┘ └──────────────────────────────┘ └────────────────────────┘  │
│                                                                             │
│  底部场景切换: [Incident频道助手] [专家定位] [问答捕获] [每日脉搏] [态势提醒] │
└─────────────────────────────────────────────────────────────────────────────┘
```

**VS Code 模拟器布局**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [← 返回 Dashboard]  VS Code 场景模拟器                            [?]     │
├─────────────────────────────────────────────────────────────────────────────┤
│  模拟 VS Code 界面                                                          │
│  ┌────────┬────────────────────────────────────────┬─────────────────────┐  │
│  │ Explorer│  editor/postgresql.conf                │ 🛠️ SkillForge      │  │
│  │ ...    │  ─────────────────────────────────    │ 诊断                │  │
│  │        │                                       │ [推荐 Skill 列表]   │  │
│  │        │  shared_buffers = 4GB                 │ [链接] → Web        │  │
│  │        │  │ 💡 SkillForge 提示...            │                     │  │
│  │        │  │ [查看详情] → Web                 │                     │  │
│  │        │                                       │                     │  │
│  │        │  ─────── Terminal ───────             │                     │  │
│  │        │  $ ORA-04031...                       │                     │  │
│  │        │                                       │                     │  │
│  └────────┴────────────────────────────────────────┴─────────────────────┘  │
│                                                                             │
│  底部场景切换: [诊断面板] [操作录制] [代码助手] [片段库] [学习地图] [上下文切换]│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### C.7 路由更新

在原路由表基础上追加：

```typescript
// 插件模拟器
{ path: '/simulator/teams', element: <TeamsSimulatorPage />, roles: ['all'] },
{ path: '/simulator/vscode', element: <VSCodeSimulatorPage />, roles: ['all'] },

// 来源感知参数（已有页面支持 query string）
// /diagnose?from=teams-channel&incidentId=xxx
// /diagnose?from=vscode-diagnose&errorCode=ORA-04031
// /my-skills?tab=drafts&from=vscode-recorder&incidentId=xxx
// /profile?userId=xxx&from=teams-expert
// /skill/:id?from=vscode-inline
// /snippets?query=xxx&from=vscode-snippet
// /team?focus=INC-xxx&from=teams-pulse
// /incident/:id?tab=timeline&from=vscode-postmortem
```

---

*附录 C 版本：v1.0*  
*追加日期：2026-05-15*  
*适用范围：Teams Bot / VS Code 插件入口 Mock + Web 落地页联动设计*
