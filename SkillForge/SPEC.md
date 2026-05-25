# SkillForge Web Dashboard — 实现规范 (SPEC)

> **原则**: 纯前端原型展示，Mock 数据驱动，一栈代码覆盖三层角色视角  
> **技术栈**: React 19 + TypeScript 5 + Vite 6 + Tailwind CSS 4 + shadcn/ui + Zustand + React Router v7 + MSW + Recharts + Lucide React  
> **实现阶段**: 按 Phase 1 → Phase 2 → Phase 3 → Phase 4 推进，每阶段可独立运行演示

---

## 一、架构约束

### 1.1 单仓库结构

```
SkillForge/
├── public/
├── src/
│   ├── main.tsx              # 入口：启动 MSW + React
│   ├── App.tsx               # Provider 汇聚 + 路由表
│   ├── index.css             # Tailwind + CSS 变量
│   │
│   ├── components/ui/        # shadcn/ui 基础组件（CLI 安装）
│   ├── components/layout/    # AppShell, Sidebar, Header, RoleSwitcher, BreadcrumbNav
│   ├── components/charts/    # 图表封装（RadarChart, TrendLine, HeatmapCalendar, GaugeChart）
│   ├── components/shared/    # 跨模块业务组件（SkillCard, MetricCard, UserAvatar, IncidentBadge, MaturityBadge, TimeRangePicker, EmptyState, LoadingOverlay）
│   │
│   ├── features/layer1/pages/     # DiagnosePage, MySkillsPage, SnippetVaultPage, LearningMapPage, ArenaPage, ProfilePage
│   ├── features/layer1/components/# 各页面专属组件
│   ├── features/layer2/pages/     # TeamOverviewPage, SkillRadarPage, MTTRAnalysisPage, MembersPage, SchedulingPage, ReportsPage
│   ├── features/layer2/components/# 各页面专属组件
│   ├── features/layer3/pages/     # ExecutiveDashboardPage, AIGovernancePage, StrategyAlignPage, OrgPlannerPage, MaturityAssessmentPage, BoardReportPage
│   ├── features/layer3/components/# 各页面专属组件
│   ├── features/shared/pages/     # SkillDetailPage, IncidentDetailPage, SearchResultPage, LandingPage, LoginPage, RoleMismatchPage
│   ├── features/shared/components/# SkillDetailDrawer, IncidentTimeline, CommentThread, GlobalSearch
│   │
│   ├── stores/               # Zustand Store：authStore, uiStore, skillStore, incidentStore, layer2Store
│   ├── mocks/                # MSW browser.ts, handlers.ts, factories/*, seeds/initialData.ts
│   ├── lib/                  # utils.ts (cn), constants.ts, formatters.ts, mockDelay.ts
│   ├── types/                # skill.ts, user.ts, incident.ts, team.ts, report.ts, governance.ts, api.ts
│   ├── hooks/                # useRole.ts, useTheme.ts, useDebounce.ts
│   └── routes/               # index.tsx, RouteGuard.tsx, routeMeta.ts
│
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

### 1.2 编码铁律

1. **所有 UI 文本使用中文**，类型/变量/组件名使用英文 PascalCase/camelCase。
2. **Mock 数据必须固定种子** (`initialData.ts`)，确保每次刷新页面状态一致。
3. **角色切换即时生效**：Header 中常驻 `RoleSwitcher`，切换后导航菜单 + 路由权限 + 页面内容同步刷新。
4. **深色模式为默认**，通过 `uiStore` 的 `theme` 控制，Tailwind `dark` class 切换。
5. **所有图表/数字必须附带叙事文本**，禁止裸数据展示。
6. **响应式断点**：mobile(<768px) 侧边栏变 Drawer；tablet(768-1024px) 侧边栏可折叠；desktop(1024-1440px) 标准布局；wide(>1440px) max-width 1400px 居中。

---

## 二、类型系统 (先写 types/)

按设计文档精确实现以下类型，所有 Mock 工厂和组件 Props 必须引用这些类型：

- `types/user.ts`: `User`, `MasteryItem`, `UserMetrics`
- `types/skill.ts`: `Skill`, `SkillStep`, `ActionItem`, `DecisionNode`, `SkillStatus`, `RiskLevel`
- `types/incident.ts`: `Incident`, `SkillUsageLog`, `TimelineEvent`, `Postmortem`, `IncidentPriority`, `IncidentStatus`
- `types/team.ts`: `Team`, `DomainCoverage`, `TeamMetrics`, `ScheduleSlot`
- `types/report.ts`: `Report`, `ReportType`
- `types/governance.ts`: `OrgSnapshot`, `MaturityAssessment`, `MaturityDimension`, `AIGovernanceReport`, `SkillConflict`
- `types/api.ts`: `ApiResponse<T>`, `DiagnoseResponse`, `MatchedSkill`, `Expert`, `Snippet`, `TimeRange`

---

## 三、Mock 数据层 (先写 mocks/)

### 3.1 数据规模（严格按文档）

| 实体 | 数量 | 说明 |
|------|------|------|
| User | 18 人 | 3 团队 × 6 人 |
| Team | 3 个 | DB-SRE / Platform-SRE / Infra-SRE |
| Skill | 312 个 | 覆盖数据库、K8s、网络、监控、安全等 |
| Incident | 120 条 | 过去 90 天分布，含完整时间线 |
| OrgSnapshot | 12 条 | 过去 12 周每周快照 |
| Report | 6 条 | 周报/月报/ROI 报告模板 |
| SkillConflict | 4 组 | 含 2 组严重冲突 |

### 3.2 工厂函数要求

- `mocks/factories/userFactory.ts`: 生成 18 个用户，固定 seed，保证 avatar/姓名/技能分布一致。
- `mocks/factories/skillFactory.ts`: 生成 312 个 Skill，分类到 8 个技术领域，healthStatus 按概率分布。
- `mocks/factories/incidentFactory.ts`: 生成 120 条 incident，时间戳均匀分布在 90 天内，状态按概率分布（70% closed, 其余 open/diagnosing/fixing/verifying）。
- `mocks/factories/teamFactory.ts`: 生成 3 个团队，关联 18 个用户和对应 Skill。
- `mocks/factories/reportFactory.ts`: 生成 6 条报告。
- `mocks/seeds/initialData.ts`: 导出 `seedUsers`, `seedSkills`, `seedIncidents`, `seedTeams`, `seedReports`, `seedConflicts`, `seedOrgSnapshots`。提供文档中的精确示例数据（王伟、李四、张经理 + Oracle 慢查询 Skill + INC-0789）。

### 3.3 MSW Handlers

`mocks/handlers.ts` 拦截以下端点（严格按 11.3 节 API 清单）：

```typescript
// 核心端点（Phase 1 必须实现）
GET  /api/me
POST /api/auth/switch-role
GET  /api/diagnose
GET  /api/skills
GET  /api/skills/:id
GET  /api/snippets
GET  /api/users/:id/learning-map
GET  /api/arena/scenarios
GET  /api/team/pulse
GET  /api/team/radar
GET  /api/team/mttr
GET  /api/team/members
GET  /api/team/schedule
GET  /api/team/reports
GET  /api/org/snapshot
GET  /api/org/governance
GET  /api/org/maturity
GET  /api/incidents
GET  /api/incidents/:id
GET  /api/search
```

每个 handler 必须：
1. 调用 `mockDelay(200-800)` 模拟网络延迟
2. 从 `initialData.ts` 读取种子数据并做简单过滤/排序
3. 返回 `ApiResponse<T>` 标准包装 `{ success: true, data: T }`

---

## 四、状态管理 (stores/)

### 4.1 authStore

```typescript
interface AuthState {
  currentUser: User | null;
  currentRole: 'engineer' | 'lead' | 'executive';
  switchRole: (role: 'engineer' | 'lead' | 'executive') => void;
}
```

- 默认 `currentRole: 'engineer'`
- `switchRole` 后立即更新 `currentUser.role` 并触发页面重渲染

### 4.2 uiStore

```typescript
interface UIState {
  theme: 'dark' | 'light';
  sidebarCollapsed: boolean;
  activeTimeRange: TimeRange;
  globalSearchOpen: boolean;
  setTheme: (t: 'dark' | 'light') => void;
  toggleSidebar: () => void;
  toggleTheme: () => void;
  openSearch: () => void;
  closeSearch: () => void;
}
```

- `theme` 默认 `'dark'`，持久化到 localStorage
- 快捷键 `Ctrl+/` 切换主题，`Ctrl+K` 打开搜索，`Ctrl+B` 折叠侧边栏，`1/2/3` 切换角色

### 4.3 skillStore / incidentStore / layer2Store

按设计文档实现，包含：实体缓存 Map、列表 ID、loading 状态、async action（调用 MSW API）。

---

## 五、全局布局组件 (components/layout/)

### 5.1 AppShell

- 结构：Sidebar(240px) + MainContent(自适应) + Header(64px fixed)
- 背景：暗色 `bg-slate-950`，亮色 `bg-slate-50`
- 内容区 padding: `p-6`

### 5.2 Sidebar

- 按 `currentRole` 动态渲染导航项（设计文档 6.2 节）
- 每项含 Lucide 图标 + 中文标签
- 底部：帮助、设置
- Collapsed 时只显示图标

### 5.3 Header

- 左侧：Logo + 面包屑
- 中间：全局搜索按钮（显示 `Ctrl+K` 快捷键提示）
- 右侧：主题切换 🌙 / 通知铃铛 🔔 / 用户头像 / RoleSwitcher 下拉框

### 5.4 RoleSwitcher

- 下拉选项：工程师 / Team Leader / 高管
- 切换后立即 toast 提示 "已切换至 XXX 视角"

### 5.5 BreadcrumbNav

- 根据当前路由自动解析面包屑

---

## 六、通用共享组件 (components/shared/)

按优先级实现：

1. **MetricCard** (`components/shared/MetricCard.tsx`)
   - Props: `title, value, unit?, trend?, trendValue?, status?, icon?, onClick?, loading?`
   - 数值 `text-3xl font-bold tracking-tight`
   - hover: `translateY(-2px) shadow-lg transition-all duration-200`

2. **SkillCard** (`components/shared/SkillCard.tsx`)
   - variant: `compact` | `default` | `detailed`
   - 左侧 4px 色条表示健康度
   - 右上角 `aiGenerated` 时显示 🤖 AI 徽章
   - hover 上浮 + 阴影加深

3. **SkillStatusBadge** (`components/shared/SkillStatusBadge.tsx`)
   - healthy → emerald / attention → yellow / outdated → red / archived → slate

4. **IncidentBadge** (`components/shared/IncidentBadge.tsx`)
   - P1→red-600 / P2→orange-500 / P3→yellow-500 / P4→blue-400

5. **MaturityBadge** (`components/shared/MaturityBadge.tsx`)
   - level 整数部分决定标签(L1-L5)，小数部分决定进度环填充
   - L1 slate / L2 blue / L3 indigo / L4 violet / L5 emerald

6. **UserAvatar** (`components/shared/UserAvatar.tsx`)
   - 尺寸 sm/md/lg，右下角状态圆点（在线绿/忙碌红/离线灰/值班橙闪烁）
   - hover 0.5s 后弹出微型 Profile Tooltip

7. **TimeRangePicker** (`components/shared/TimeRangePicker.tsx`)
   - 预设：今日/本周/本月/过去7天/过去30天/过去90天

8. **EmptyState** (`components/shared/EmptyState.tsx`)
   - Lucide 组合图标 + 说明文字 + 建议动作按钮

9. **LoadingOverlay** (`components/shared/LoadingOverlay.tsx`)
   - 局部 Skeleton，禁止全屏 spinning loader

---

## 七、路由系统 (routes/)

### 7.1 路由表

```typescript
const routes = [
  { path: '/', element: <LandingPage />, roles: ['all'] },
  { path: '/login', element: <LoginPage />, roles: ['all'] },
  
  // Layer 1
  { path: '/diagnose', element: <DiagnosePage />, roles: ['engineer', 'lead', 'executive'] },
  { path: '/my-skills', element: <MySkillsPage />, roles: ['engineer', 'lead'] },
  { path: '/snippets', element: <SnippetVaultPage />, roles: ['engineer', 'lead'] },
  { path: '/learning', element: <LearningMapPage />, roles: ['engineer', 'lead'] },
  { path: '/arena', element: <ArenaPage />, roles: ['engineer', 'lead'] },
  { path: '/profile', element: <ProfilePage />, roles: ['engineer', 'lead', 'executive'] },
  
  // Layer 2
  { path: '/team', element: <TeamOverviewPage />, roles: ['lead', 'executive'] },
  { path: '/team/radar', element: <SkillRadarPage />, roles: ['lead', 'executive'] },
  { path: '/team/mttr', element: <MTTRAnalysisPage />, roles: ['lead', 'executive'] },
  { path: '/team/members', element: <MembersPage />, roles: ['lead', 'executive'] },
  { path: '/team/schedule', element: <SchedulingPage />, roles: ['lead'] },
  { path: '/team/reports', element: <ReportsPage />, roles: ['lead', 'executive'] },
  
  // Layer 3
  { path: '/executive', element: <ExecutiveDashboardPage />, roles: ['executive'] },
  { path: '/executive/governance', element: <AIGovernancePage />, roles: ['executive'] },
  { path: '/executive/strategy', element: <StrategyAlignPage />, roles: ['executive'] },
  { path: '/executive/planner', element: <OrgPlannerPage />, roles: ['executive'] },
  { path: '/executive/maturity', element: <MaturityAssessmentPage />, roles: ['executive'] },
  { path: '/executive/board-report', element: <BoardReportPage />, roles: ['executive'] },
  
  // Shared
  { path: '/skill/:skillId', element: <SkillDetailPage />, roles: ['all'] },
  { path: '/incident/:incidentId', element: <IncidentDetailPage />, roles: ['all'] },
  { path: '/search', element: <SearchResultPage />, roles: ['all'] },
  
  // Simulators
  { path: '/simulator/teams', element: <TeamsSimulatorPage />, roles: ['all'] },
  { path: '/simulator/vscode', element: <VSCodeSimulatorPage />, roles: ['all'] },
];
```

### 7.2 RouteGuard

- 检查 `currentRole` 是否在 `allowedRoles` 中
- 无权限时渲染 `RoleMismatchPage`，提供"一键切换到正确角色"按钮

---

## 八、实现阶段 (Phase)

### Phase 1: 骨架与第一层 (核心基础)

目标：**可运行的单页应用，包含全局布局 + 第一层核心页面**

1. **项目初始化**
   - `npm create vite@latest . -- --template react-ts`
   - 安装依赖：`react react-dom react-router-dom zustand recharts lucide-react date-fns clsx tailwind-merge @radix-ui/*`
   - 配置 Tailwind CSS 4 + CSS 变量（暗色/亮色）
   - 初始化 shadcn/ui (若 CLI 可用)

2. **类型系统** (`src/types/*`)
   - 全部 7 个类型文件

3. **Mock 数据层** (`src/mocks/*`)
   - `initialData.ts` + 5 个工厂函数
   - `handlers.ts` 核心端点（Phase 1 需要的：me, skills, incidents, snippets, diagnose, team/pulse, org/snapshot）
   - `browser.ts` 启动 MSW

4. **Store** (`src/stores/*`)
   - `authStore`, `uiStore`, `skillStore`, `incidentStore`

5. **全局布局** (`src/components/layout/*`)
   - `AppShell`, `Sidebar`, `Header`, `RoleSwitcher`, `BreadcrumbNav`

6. **通用组件** (`src/components/shared/*`)
   - `MetricCard`, `SkillCard` (default variant), `SkillStatusBadge`, `IncidentBadge`, `UserAvatar`, `EmptyState`, `LoadingOverlay`

7. **第一层页面** (`src/features/layer1/pages/*`)
   - `DiagnosePage`：输入框 + 快速标签 + Skill 推荐卡片列表（含专家信息）
   - `MySkillsPage`：KPI 行 + 草稿确认区 + Skill 网格卡片
   - `SnippetVaultPage`：搜索框 + 片段卡片列表（语法高亮用 `<pre>` + Tailwind）

8. **共享页面** (`src/features/shared/pages/*`)
   - `LandingPage`：产品简介 + 角色选择入口
   - `SkillDetailPage`：Skill 完整信息展示
   - `IncidentDetailPage`：Incident 时间线展示
   - `RoleMismatchPage`

9. **路由系统** (`src/routes/*`)
   - 完整路由表 + `RouteGuard`

**Phase 1 验收标准**：
- 运行 `npm run dev`，页面正常加载，MSW 无报错
- 可切换三种角色，侧边栏导航动态变化
- 可访问 `/diagnose`, `/my-skills`, `/snippets` 并看到 Mock 数据渲染的 UI
- 深色/浅色主题可切换
- 响应式：窗口缩小时侧边栏自动折叠/变 Drawer

---

### Phase 2: 第二层与图表

目标：**Team Leader 视角完整可用，图表可视化上线**

1. **Store 扩展**
   - `layer2Store`

2. **图表组件** (`src/components/charts/*`)
   - `RadarChart`：Recharts RadarChart 封装（团队技能雷达）
   - `TrendLine`：Recharts LineChart + AreaChart 封装（MTTR 趋势）
   - `HeatmapCalendar`：网格热力图（简化版，用于人才风险）
   - `GaugeChart`：进度环（成熟度评估）

3. **通用组件扩展**
   - `SkillCard` 补充 `compact` 和 `detailed` variant
   - `MaturityBadge`
   - `TimeRangePicker`

4. **第二层页面** (`src/features/layer2/pages/*`)
   - `TeamOverviewPage`：5 张 MetricCard + Incident 实时态势卡片 + 单点风险提示
   - `SkillRadarPage`：左侧雷达图 + 右侧领域详情面板 + 单点风险汇总表
   - `MTTRAnalysisPage`：MTTR 趋势折线图 + 阶段拆解 + Skill 贡献归因 + 瓶颈识别
   - `MembersPage`：左侧成员列表（带迷你进度条）+ 右侧详情面板
   - `SchedulingPage`：周排班网格 + 风险分析卡片 + 优化建议
   - `ReportsPage`：报告列表 + 预览器

5. **Mock Handlers 补充**
   - 完成所有 Team Leader 相关端点

**Phase 2 验收标准**：
- 切换为 Team Leader 角色后，所有 `/team/*` 页面可正常访问
- 雷达图、折线图、柱状图正常渲染且有动画
- MTTR 页面数据叙事完整（有"这意味着什么"的文本）

---

### Phase 3: 第三层与高管视图

目标：**高管层一页纸仪表板 + 治理控制台 + 汇报材料**

1. **第三层页面** (`src/features/layer3/pages/*`)
   - `ExecutiveDashboardPage`：**一屏原则**，3 张大卡片（可靠性/能力资产/人才风险）+ 关注区
   - `AIGovernancePage`：Tab 切换（使用全景/策略管理/冲突检测）
   - `StrategyAlignPage`：目标列表 + 简化版桑基图（用条形图+连线模拟，若实现成本高）+ 差距清单
   - `OrgPlannerPage`：12 个月能力表格 + What-If 模拟器 + 里程碑时间线
   - `MaturityAssessmentPage`：成熟度阶梯图 + 雷达图 + 提升路径
   - `BoardReportPage`：Slide 模拟器（翻页组件，5 页核心信息）

2. **第三层组件** (`src/features/layer3/components/*`)
   - `OnePageStatus`, `ExecutiveIncidentBrief`, `ROICard`, `AIPolicyEditor`, `ConflictTable`, `MaturityLadder`, `TalentRiskHeatmap`, `BudgetAllocator`, `BoardReportSlides`

3. **Mock Handlers 补充**
   - 完成所有高管层端点

**Phase 3 验收标准**：
- 切换为 Executive 角色后，`/executive` 一页纸仪表板在一屏内展示完整
- 所有数字可点击下钻到对应详情页
- Slide 模拟器可翻页浏览

---

### Phase 4: 插件模拟器与打磨

目标：**Teams Bot + VS Code 插件模拟器上线，全链路可演示**

1. **模拟器页面** (`src/features/shared/pages/*`)
   - `TeamsSimulatorPage`：左中右三栏（频道列表/聊天区/Web 预览），底部场景切换
   - `VSCodeSimulatorPage`：模拟 VS Code 界面（Explorer/Editor/SkillForge 侧边栏/Terminal），底部场景切换

2. **模拟器组件**
   - 设计文档附录 C 中的所有 Bot 卡片和 VS Code 面板
   - 点击卡片链接在当前浏览器内跳转到真实 Web 页面

3. **来源感知 (Source Context Bar)**
   - `/diagnose`, `/my-skills`, `/profile` 等页面读取 `?from=` 参数
   - 页面顶部渲染来源上下文条，预填充对应参数

4. **全局打磨**
   - 响应式适配（tablet + mobile）
   - 动效：页面切换 fade-in + slide-up (150-200ms)
   - MetricCard 数字 spring 动画
   - 全局快捷键：1/2/3 切换角色
   - 空状态与错误状态统一
   - Mock 数据丰富度补充（确保演示流畅）

**Phase 4 验收标准**：
- 从 Teams 模拟器点击 Bot 卡片链接 → 正确跳转到 Web 页面并带上下文
- 从 VS Code 模拟器点击诊断推荐 → 正确跳转到 `/diagnose`
- 移动端侧边栏变为 Drawer，内容可读
- 全局快捷键响应正常

---

## 九、组件接口规范（核心组件精确 Props）

### MetricCard

```typescript
interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  trend?: 'up' | 'down' | 'flat';
  trendValue?: string;
  status?: 'good' | 'warning' | 'danger' | 'neutral';
  icon?: LucideIcon;
  onClick?: () => void;
  loading?: boolean;
}
```

### SkillCard

```typescript
interface SkillCardProps {
  skill: Skill;
  variant?: 'compact' | 'default' | 'detailed';
  showHealth?: boolean;
  showActions?: boolean;
  showTrend?: boolean;
  highlightMatch?: string;
  onClick?: () => void;
  onEdit?: () => void;
  onExecute?: () => void;
}
```

### SkillStepper

```typescript
interface SkillStepperProps {
  steps: SkillStep[];
  currentStep: number;
  onStepComplete?: (stepIndex: number) => void;
  onExecuteCommand?: (command: string) => void;
  readonly?: boolean;
}
```

### UserAvatar

```typescript
interface UserAvatarProps {
  user: User;
  size?: 'sm' | 'md' | 'lg';
  showStatus?: boolean;
  showTooltip?: boolean;
  onClick?: () => void;
}
```

---

## 十、视觉规范速查

### 色彩

| 语义 | Dark | Light |
|------|------|-------|
| 成功/健康 | emerald-400 | emerald-600 |
| 警告/注意 | yellow-400 | yellow-600 |
| 危险/严重 | red-400 | red-600 |
| 信息/中性 | blue-400 | blue-600 |
| 归档/缺失 | slate-500 | slate-400 |
| 单点风险 | orange-400 | orange-600 |

### 成熟度色阶

| 等级 | 颜色 |
|------|------|
| L1 | slate-400 |
| L2 | blue-500 |
| L3 | indigo-500 |
| L4 | violet-500 |
| L5 | emerald-500 |

### 字体

- 数字：`font-variant-numeric: tabular-nums`
- Page 标题：`text-2xl font-bold`
- Section 标题：`text-lg font-semibold`
- Card 标题：`text-base font-medium`
- 正文：`leading-relaxed`
- 紧凑数据：`leading-snug`

### 动效

- 页面切换：opacity 0→1 (150ms) + translateY 8px→0 (200ms)
- 数据加载：局部 Skeleton
- 数字变化：spring 动画 (0.3s)
- 图表更新：Recharts `isAnimationActive={true}` duration 800ms
- hover：所有可点击卡片 `transition-all duration-200`，hover 时 `translateY(-2px) shadow-lg`

---

## 十一、文件生成顺序

每次新建文件按此顺序依赖：

1. `src/types/*.ts` — 无依赖
2. `src/lib/*.ts` — 依赖 types
3. `src/mocks/seeds/initialData.ts` — 依赖 types
4. `src/mocks/factories/*.ts` — 依赖 types, lib
5. `src/mocks/handlers.ts` — 依赖 factories, types
6. `src/stores/*.ts` — 依赖 types, mocks
7. `src/components/ui/*.tsx` — 无依赖（shadcn）
8. `src/components/shared/*.tsx` — 依赖 types, stores, ui
9. `src/components/charts/*.tsx` — 依赖 types, ui
10. `src/components/layout/*.tsx` — 依赖 shared, stores, ui
11. `src/routes/*.tsx` — 依赖 layout, pages, stores
12. `src/features/*/components/*.tsx` — 依赖 shared, types, stores
13. `src/features/*/pages/*.tsx` — 依赖 components, shared, stores, charts
14. `src/App.tsx` — 依赖 routes, layout, stores
15. `src/main.tsx` — 依赖 App, mocks

---

## 十二、构建与运行

```bash
# 安装
npm install

# 开发
npm run dev

# 构建
npm run build

# 预览
npm run preview
```

> **注意**: 纯前端项目，不需要后端服务。MSW 在开发模式下拦截所有 `/api/*` 请求。
