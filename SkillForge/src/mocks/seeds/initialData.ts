import type { User } from '@/types/user';
import type { Skill } from '@/types/skill';
import type { Incident } from '@/types/incident';
import type { Team } from '@/types/team';
import type { Report } from '@/types/report';
import type { OrgSnapshot, MaturityAssessment, AIGovernanceReport, SkillConflict } from '@/types/governance';

// ==================== Seed Users (18人) ====================
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
      { domain: 'K8s基础运维', level: 65, trend: 'up' },
      { domain: 'K8s故障排查', level: 22, trend: 'up' },
      { domain: 'Linux系统调优', level: 70, trend: 'flat' },
      { domain: '监控告警配置', level: 80, trend: 'flat' },
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
      { domain: 'Linux系统调优', level: 75, trend: 'flat' },
    ],
    status: 'oncall',
    metrics: { totalIncidents: 312, avgMTTR: 12, skillsCreated: 34, skillsAdoptedByOthers: 256 }
  },
  {
    id: 'user_zhang_san',
    name: '张三',
    handle: '@zhang_san',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=zhangsan',
    role: 'engineer',
    teamId: 'team_db_sre',
    title: 'SRE',
    joinDate: '2023-08-01',
    skillsMastery: [
      { domain: 'Oracle基础管理', level: 70, trend: 'up' },
      { domain: 'Oracle性能诊断', level: 55, trend: 'up' },
      { domain: 'Oracle高可用(RAC)', level: 25, trend: 'up' },
      { domain: 'K8s基础运维', level: 40, trend: 'up' },
      { domain: '监控告警配置', level: 60, trend: 'up' },
    ],
    status: 'busy',
    metrics: { totalIncidents: 45, avgMTTR: 28, skillsCreated: 3, skillsAdoptedByOthers: 12 }
  },
  {
    id: 'user_xiao_li',
    name: '小李',
    handle: '@xiao_li',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=xiaoli',
    role: 'engineer',
    teamId: 'team_db_sre',
    title: 'Junior SRE',
    joinDate: '2025-02-01',
    skillsMastery: [
      { domain: 'Oracle基础管理', level: 68, trend: 'up' },
      { domain: 'Oracle性能诊断', level: 35, trend: 'up' },
      { domain: 'K8s基础运维', level: 30, trend: 'up' },
      { domain: 'Linux系统调优', level: 40, trend: 'up' },
    ],
    status: 'online',
    metrics: { totalIncidents: 12, avgMTTR: 35, skillsCreated: 1, skillsAdoptedByOthers: 3 }
  },
  {
    id: 'user_zhao_liu',
    name: '赵六',
    handle: '@zhao_liu',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=zhaoliu',
    role: 'engineer',
    teamId: 'team_db_sre',
    title: 'SRE',
    joinDate: '2022-05-10',
    skillsMastery: [
      { domain: 'Oracle基础管理', level: 80, trend: 'flat' },
      { domain: 'Oracle性能诊断', level: 60, trend: 'flat' },
      { domain: 'K8s基础运维', level: 55, trend: 'up' },
      { domain: '监控告警配置', level: 75, trend: 'flat' },
    ],
    status: 'offline',
    metrics: { totalIncidents: 89, avgMTTR: 22, skillsCreated: 6, skillsAdoptedByOthers: 34 }
  },
  {
    id: 'user_sun_ba',
    name: '孙八',
    handle: '@sun_ba',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=sunba',
    role: 'engineer',
    teamId: 'team_db_sre',
    title: 'SRE',
    joinDate: '2020-11-20',
    skillsMastery: [
      { domain: 'Oracle基础管理', level: 85, trend: 'flat' },
      { domain: 'Oracle性能诊断', level: 70, trend: 'flat' },
      { domain: 'Linux系统调优', level: 88, trend: 'flat' },
      { domain: '监控告警配置', level: 82, trend: 'flat' },
    ],
    status: 'online',
    metrics: { totalIncidents: 156, avgMTTR: 19, skillsCreated: 15, skillsAdoptedByOthers: 67 }
  },
  // Platform-SRE Team
  {
    id: 'user_chen_qi',
    name: '陈七',
    handle: '@chen_qi',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=chenqi',
    role: 'engineer',
    teamId: 'team_platform_sre',
    title: 'Senior SRE - K8s 专家',
    joinDate: '2020-03-15',
    skillsMastery: [
      { domain: 'K8s基础运维', level: 95, trend: 'flat' },
      { domain: 'K8s故障排查', level: 92, trend: 'flat' },
      { domain: 'K8s网络/存储', level: 85, trend: 'flat' },
      { domain: 'Linux系统调优', level: 80, trend: 'flat' },
      { domain: '监控告警配置', level: 78, trend: 'flat' },
    ],
    status: 'online',
    metrics: { totalIncidents: 198, avgMTTR: 15, skillsCreated: 22, skillsAdoptedByOthers: 145 }
  },
  {
    id: 'user_wu_jiu',
    name: '吴九',
    handle: '@wu_jiu',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=wujiu',
    role: 'engineer',
    teamId: 'team_platform_sre',
    title: 'SRE',
    joinDate: '2022-09-01',
    skillsMastery: [
      { domain: 'K8s基础运维', level: 78, trend: 'up' },
      { domain: 'K8s故障排查', level: 65, trend: 'up' },
      { domain: 'K8s网络/存储', level: 55, trend: 'up' },
      { domain: 'Linux系统调优', level: 60, trend: 'flat' },
    ],
    status: 'busy',
    metrics: { totalIncidents: 67, avgMTTR: 24, skillsCreated: 5, skillsAdoptedByOthers: 23 }
  },
  {
    id: 'user_zhou_shi',
    name: '周十',
    handle: '@zhou_shi',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=zhoushi',
    role: 'engineer',
    teamId: 'team_platform_sre',
    title: 'Junior SRE',
    joinDate: '2024-06-01',
    skillsMastery: [
      { domain: 'K8s基础运维', level: 55, trend: 'up' },
      { domain: 'K8s故障排查', level: 35, trend: 'up' },
      { domain: 'K8s网络/存储', level: 25, trend: 'up' },
      { domain: 'Linux系统调优', level: 40, trend: 'up' },
    ],
    status: 'online',
    metrics: { totalIncidents: 15, avgMTTR: 38, skillsCreated: 1, skillsAdoptedByOthers: 2 }
  },
  {
    id: 'user_zheng_shiyi',
    name: '郑十一',
    handle: '@zheng_shiyi',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=zhengshiyi',
    role: 'engineer',
    teamId: 'team_platform_sre',
    title: 'SRE',
    joinDate: '2021-07-10',
    skillsMastery: [
      { domain: 'K8s基础运维', level: 85, trend: 'flat' },
      { domain: 'K8s故障排查', level: 75, trend: 'flat' },
      { domain: 'K8s网络/存储', level: 70, trend: 'flat' },
      { domain: '监控告警配置', level: 80, trend: 'flat' },
    ],
    status: 'oncall',
    metrics: { totalIncidents: 123, avgMTTR: 20, skillsCreated: 10, skillsAdoptedByOthers: 56 }
  },
  {
    id: 'user_qian_shier',
    name: '钱十二',
    handle: '@qian_shier',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=qianshier',
    role: 'engineer',
    teamId: 'team_platform_sre',
    title: 'SRE',
    joinDate: '2023-01-15',
    skillsMastery: [
      { domain: 'K8s基础运维', level: 70, trend: 'up' },
      { domain: 'K8s故障排查', level: 60, trend: 'up' },
      { domain: 'Linux系统调优', level: 55, trend: 'flat' },
    ],
    status: 'offline',
    metrics: { totalIncidents: 42, avgMTTR: 26, skillsCreated: 4, skillsAdoptedByOthers: 18 }
  },
  {
    id: 'user_feng_shisan',
    name: '冯十三',
    handle: '@feng_shisan',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=fengshisan',
    role: 'engineer',
    teamId: 'team_platform_sre',
    title: 'SRE',
    joinDate: '2019-12-01',
    skillsMastery: [
      { domain: 'K8s基础运维', level: 90, trend: 'flat' },
      { domain: 'K8s故障排查', level: 88, trend: 'flat' },
      { domain: 'K8s网络/存储', level: 82, trend: 'flat' },
      { domain: 'Linux系统调优', level: 85, trend: 'flat' },
    ],
    status: 'online',
    metrics: { totalIncidents: 178, avgMTTR: 16, skillsCreated: 18, skillsAdoptedByOthers: 112 }
  },
  // Infra-SRE Team
  {
    id: 'user_huang_shisi',
    name: '黄十四',
    handle: '@huang_shisi',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=huangshisi',
    role: 'engineer',
    teamId: 'team_infra_sre',
    title: 'Senior SRE - 网络专家',
    joinDate: '2018-04-01',
    skillsMastery: [
      { domain: 'Linux系统调优', level: 96, trend: 'flat' },
      { domain: '监控告警配置', level: 92, trend: 'flat' },
      { domain: 'K8s网络/存储', level: 80, trend: 'flat' },
    ],
    status: 'online',
    metrics: { totalIncidents: 234, avgMTTR: 14, skillsCreated: 28, skillsAdoptedByOthers: 189 }
  },
  {
    id: 'user_xu_shiwu',
    name: '徐十五',
    handle: '@xu_shiwu',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=xushiwu',
    role: 'engineer',
    teamId: 'team_infra_sre',
    title: 'SRE',
    joinDate: '2022-01-10',
    skillsMastery: [
      { domain: 'Linux系统调优', level: 75, trend: 'up' },
      { domain: '监控告警配置', level: 80, trend: 'flat' },
      { domain: 'K8s基础运维', level: 50, trend: 'up' },
    ],
    status: 'busy',
    metrics: { totalIncidents: 78, avgMTTR: 23, skillsCreated: 7, skillsAdoptedByOthers: 29 }
  },
  {
    id: 'user_he_shiliu',
    name: '何十六',
    handle: '@he_shiliu',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=heshiliu',
    role: 'engineer',
    teamId: 'team_infra_sre',
    title: 'Junior SRE',
    joinDate: '2024-09-01',
    skillsMastery: [
      { domain: 'Linux系统调优', level: 45, trend: 'up' },
      { domain: '监控告警配置', level: 50, trend: 'up' },
      { domain: 'K8s基础运维', level: 30, trend: 'up' },
    ],
    status: 'online',
    metrics: { totalIncidents: 8, avgMTTR: 42, skillsCreated: 0, skillsAdoptedByOthers: 0 }
  },
  {
    id: 'user_lin_shiqi',
    name: '林十七',
    handle: '@lin_shiqi',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=linshiqi',
    role: 'engineer',
    teamId: 'team_infra_sre',
    title: 'SRE',
    joinDate: '2021-11-20',
    skillsMastery: [
      { domain: 'Linux系统调优', level: 82, trend: 'flat' },
      { domain: '监控告警配置', level: 85, trend: 'flat' },
      { domain: 'Oracle基础管理', level: 40, trend: 'up' },
    ],
    status: 'offline',
    metrics: { totalIncidents: 95, avgMTTR: 21, skillsCreated: 8, skillsAdoptedByOthers: 41 }
  },
  {
    id: 'user_guo_shiba',
    name: '郭十八',
    handle: '@guo_shiba',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=guoshiba',
    role: 'engineer',
    teamId: 'team_infra_sre',
    title: 'SRE',
    joinDate: '2020-08-15',
    skillsMastery: [
      { domain: 'Linux系统调优', level: 88, trend: 'flat' },
      { domain: '监控告警配置', level: 90, trend: 'flat' },
      { domain: 'K8s故障排查', level: 60, trend: 'up' },
    ],
    status: 'oncall',
    metrics: { totalIncidents: 134, avgMTTR: 17, skillsCreated: 13, skillsAdoptedByOthers: 78 }
  },
  // Leads
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
      { domain: 'K8s基础运维', level: 60, trend: 'flat' },
    ],
    status: 'busy',
    metrics: { totalIncidents: 89, avgMTTR: 22, skillsCreated: 8, skillsAdoptedByOthers: 45 }
  },
  {
    id: 'user_lead_li',
    name: '李经理',
    handle: '@li_mgr',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=limgr',
    role: 'lead',
    teamId: 'team_platform_sre',
    title: 'Platform-SRE Team Lead',
    joinDate: '2017-06-01',
    skillsMastery: [
      { domain: '团队管理', level: 90, trend: 'flat' },
      { domain: 'K8s故障排查', level: 85, trend: 'flat' },
    ],
    status: 'online',
    metrics: { totalIncidents: 112, avgMTTR: 20, skillsCreated: 10, skillsAdoptedByOthers: 67 }
  },
  {
    id: 'user_lead_wang',
    name: '王经理',
    handle: '@wang_mgr',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=wangmgr',
    role: 'lead',
    teamId: 'team_infra_sre',
    title: 'Infra-SRE Team Lead',
    joinDate: '2016-03-15',
    skillsMastery: [
      { domain: '团队管理', level: 92, trend: 'flat' },
      { domain: 'Linux系统调优', level: 80, trend: 'flat' },
    ],
    status: 'online',
    metrics: { totalIncidents: 145, avgMTTR: 18, skillsCreated: 12, skillsAdoptedByOthers: 89 }
  },
  // Executive
  {
    id: 'user_vp_tech',
    name: '技术VP',
    handle: '@vp_tech',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=vptech',
    role: 'executive',
    teamId: 'team_db_sre',
    title: 'VP of Engineering',
    joinDate: '2015-01-01',
    skillsMastery: [],
    status: 'online',
    metrics: { totalIncidents: 0, avgMTTR: 0, skillsCreated: 0, skillsAdoptedByOthers: 0 }
  },
];

// ==================== Seed Skills (精选示例 + 生成312个的辅助数据) ====================
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
        { order: 2, title: '定位 Top SQL', description: '从 AWR 中提取 Elapsed Time 最高的3条 SQL', command: "SELECT sql_id, elapsed_time/1000000 as elapsed_sec FROM v$sql ORDER BY elapsed_time DESC FETCH FIRST 3 ROWS ONLY;", estimatedTime: 5, verification: '已确定高耗时 SQL 的 sql_id' },
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

// 生成更多 Skill
const skillTemplates = [
  { name: 'Oracle AWR 自动分析报告', domain: ['database', 'oracle', 'performance'], difficulty: 'beginner' as const, risk: 'low' as const },
  { name: 'Oracle RAC 故障切换诊断', domain: ['database', 'oracle', 'ha'], difficulty: 'advanced' as const, risk: 'high' as const },
  { name: 'Oracle 连接池耗尽排查', domain: ['database', 'oracle', 'performance'], difficulty: 'intermediate' as const, risk: 'medium' as const },
  { name: 'K8s Deployment 滚动更新故障', domain: ['kubernetes', 'troubleshooting'], difficulty: 'intermediate' as const, risk: 'medium' as const },
  { name: 'K8s Service 网络不通诊断', domain: ['kubernetes', 'network'], difficulty: 'intermediate' as const, risk: 'medium' as const },
  { name: 'K8s 存储卷挂载失败排查', domain: ['kubernetes', 'storage'], difficulty: 'intermediate' as const, risk: 'medium' as const },
  { name: 'Linux 高负载排查', domain: ['linux', 'performance'], difficulty: 'intermediate' as const, risk: 'low' as const },
  { name: 'Linux 内存泄漏定位', domain: ['linux', 'performance'], difficulty: 'advanced' as const, risk: 'medium' as const },
  { name: 'Prometheus 告警规则调优', domain: ['monitoring', 'prometheus'], difficulty: 'intermediate' as const, risk: 'low' as const },
  { name: 'Grafana 仪表盘配置', domain: ['monitoring', 'grafana'], difficulty: 'beginner' as const, risk: 'low' as const },
  { name: 'CI/CD 流水线故障诊断', domain: ['cicd', 'troubleshooting'], difficulty: 'intermediate' as const, risk: 'low' as const },
  { name: '数据库备份恢复验证', domain: ['database', 'backup'], difficulty: 'intermediate' as const, risk: 'high' as const },
];

const authors = ['user_wang_wu', 'user_li_si', 'user_zhang_san', 'user_chen_qi', 'user_huang_shisi', 'user_feng_shisan', 'user_guo_shiba', 'user_lead_zhang', 'user_lead_li', 'user_lead_wang'];
const teams = ['team_db_sre', 'team_platform_sre', 'team_infra_sre'];
const healthStatuses: Array<'healthy' | 'attention' | 'outdated' | 'archived'> = ['healthy', 'healthy', 'healthy', 'attention', 'attention', 'outdated', 'archived'];

for (let i = 0; i < 310; i++) {
  const tpl = skillTemplates[i % skillTemplates.length];
  const author = authors[i % authors.length];
  const team = teams[i % teams.length];
  const health = healthStatuses[Math.floor(Math.random() * healthStatuses.length)];
  const version = 1 + Math.floor(Math.random() * 5);
  const useCount = Math.floor(Math.random() * 100);
  const successRate = 0.5 + Math.random() * 0.45;
  const aiGen = Math.random() > 0.7;
  
  seedSkills.push({
    id: `skill-generated-${i}`,
    name: `${tpl.name} v${version}`,
    version: version,
    authorId: author,
    teamId: team,
    createdAt: `2025-${String((i % 12) + 1).padStart(2, '0')}-15T10:00:00Z`,
    lastUsedAt: `2026-${String((i % 5) + 1).padStart(2, '0')}-${String((i % 28) + 1).padStart(2, '0')}T${String((i % 24)).padStart(2, '0')}:00:00Z`,
    useCount,
    successRate: Math.round(successRate * 100) / 100,
    avgResolutionTime: 10 + Math.floor(Math.random() * 40),
    classification: {
      domain: tpl.domain,
      scenario: ['incident'],
      difficulty: tpl.difficulty,
      riskLevel: tpl.risk,
    },
    dependencies: {
      requiredSkills: [],
      requiredAccess: [],
    },
    content: {
      triggerConditions: `触发条件示例 ${i}`,
      diagnosisSteps: [
        { order: 1, title: '初步检查', description: '检查系统状态', estimatedTime: 5, verification: '确认问题现象' },
      ],
      executionActions: [],
      rollbackPlan: '回滚方案示例',
    },
    governance: {
      approvalStatus: 'approved',
      reviewerId: 'user_lead_zhang',
      complianceTags: [],
      expiryReviewDate: '2026-12-31',
      aiGenerated: aiGen,
      aiConfidence: aiGen ? Math.round((0.7 + Math.random() * 0.25) * 100) / 100 : undefined,
    },
    evolution: {
      changeLog: `v${version} 更新`,
    },
    healthStatus: health,
    healthScore: health === 'healthy' ? 80 + Math.floor(Math.random() * 20) : health === 'attention' ? 50 + Math.floor(Math.random() * 20) : 20 + Math.floor(Math.random() * 20),
  });
}

// ==================== Seed Incidents ====================
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

// 生成更多 incidents
const priorities: Array<'P1' | 'P2' | 'P3' | 'P4'> = ['P1', 'P2', 'P3', 'P3', 'P3', 'P4', 'P4'];
const statuses: Array<'open' | 'diagnosing' | 'fixing' | 'verifying' | 'closed'> = ['closed', 'closed', 'closed', 'closed', 'closed', 'closed', 'closed', 'diagnosing', 'fixing', 'open'];

for (let i = 0; i < 119; i++) {
  const day = Math.floor(Math.random() * 90);
  const date = new Date('2026-02-15');
  date.setDate(date.getDate() + day);
  const dateStr = date.toISOString().slice(0, 10);
  const priority = priorities[Math.floor(Math.random() * priorities.length)];
  const status = statuses[Math.floor(Math.random() * statuses.length)];
  const assignee = seedUsers[Math.floor(Math.random() * 18)].id;
  const mttr = status === 'closed' ? 10 + Math.floor(Math.random() * 50) : undefined;
  
  seedIncidents.push({
    id: `INC-2026-${String(1000 + i).slice(1)}`,
    title: `模拟事故 ${i + 1}`,
    priority,
    status,
    createdAt: `${dateStr}T${String(Math.floor(Math.random() * 24)).padStart(2, '0')}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}:00Z`,
    resolvedAt: status === 'closed' ? `${dateStr}T${String(Math.floor(Math.random() * 24)).padStart(2, '0')}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}:00Z` : undefined,
    mttr,
    assigneeId: assignee,
    teamId: seedUsers.find(u => u.id === assignee)?.teamId || 'team_db_sre',
    context: {
      alertSource: `PagerDuty-PD-${80000 + i}`,
      affectedService: `Service-${i % 10}`,
      environment: ['prod', 'staging', 'dev'][i % 3] as 'prod' | 'staging' | 'dev',
      initialSymptom: `症状描述 ${i}`,
    },
    skillUsage: [],
    timeline: [
      { timestamp: `${dateStr}T00:00:00Z`, type: 'alert', description: '告警触发', source: 'pagerduty' },
    ],
  });
}

// ==================== Seed Teams ====================
export const seedTeams: Team[] = [
  {
    id: 'team_db_sre',
    name: 'DB-SRE',
    memberIds: seedUsers.filter(u => u.teamId === 'team_db_sre').map(u => u.id),
    skillIds: seedSkills.filter(s => s.teamId === 'team_db_sre').map(s => s.id),
    coverage: [
      { domain: 'Oracle基础管理', coverageCount: 5, totalMembers: 6, avgDepth: 80, healthStatus: 'healthy' },
      { domain: 'Oracle性能诊断', coverageCount: 4, totalMembers: 6, avgDepth: 65, healthStatus: 'healthy' },
      { domain: 'Oracle高可用(RAC)', coverageCount: 1, totalMembers: 6, avgDepth: 55, healthStatus: 'attention' },
      { domain: 'K8s基础运维', coverageCount: 3, totalMembers: 6, avgDepth: 45, healthStatus: 'healthy' },
      { domain: 'K8s故障排查', coverageCount: 2, totalMembers: 6, avgDepth: 30, healthStatus: 'attention' },
      { domain: 'Linux系统调优', coverageCount: 4, totalMembers: 6, avgDepth: 60, healthStatus: 'healthy' },
      { domain: '监控告警配置', coverageCount: 4, totalMembers: 6, avgDepth: 70, healthStatus: 'healthy' },
    ],
    metrics: {
      avgMTTR: 22,
      sloAchievement: 0.996,
      incidentCountThisWeek: 5,
      skillUsageThisWeek: 23,
      newSkillsThisWeek: 2,
    },
    schedule: [
      [{ userId: 'user_wang_wu', shift: 'day' }, { userId: 'user_li_si', shift: 'night' }],
      [{ userId: 'user_zhang_san', shift: 'day' }, { userId: 'user_zhao_liu', shift: 'night' }],
      [{ userId: 'user_sun_ba', shift: 'day' }, { userId: 'user_xiao_li', shift: 'night' }],
      [{ userId: 'user_li_si', shift: 'day' }, { userId: 'user_wang_wu', shift: 'night' }],
      [{ userId: 'user_zhao_liu', shift: 'day' }, { userId: 'user_zhang_san', shift: 'night' }],
      [{ userId: 'user_xiao_li', shift: 'day' }, { userId: 'user_sun_ba', shift: 'night' }],
      [{ userId: 'user_wang_wu', shift: 'day' }, { userId: 'user_li_si', shift: 'night' }],
    ],
  },
  {
    id: 'team_platform_sre',
    name: 'Platform-SRE',
    memberIds: seedUsers.filter(u => u.teamId === 'team_platform_sre').map(u => u.id),
    skillIds: seedSkills.filter(s => s.teamId === 'team_platform_sre').map(s => s.id),
    coverage: [
      { domain: 'K8s基础运维', coverageCount: 6, totalMembers: 6, avgDepth: 78, healthStatus: 'healthy' },
      { domain: 'K8s故障排查', coverageCount: 5, totalMembers: 6, avgDepth: 72, healthStatus: 'healthy' },
      { domain: 'K8s网络/存储', coverageCount: 3, totalMembers: 6, avgDepth: 55, healthStatus: 'attention' },
      { domain: 'Linux系统调优', coverageCount: 4, totalMembers: 6, avgDepth: 65, healthStatus: 'healthy' },
      { domain: '监控告警配置', coverageCount: 3, totalMembers: 6, avgDepth: 60, healthStatus: 'healthy' },
    ],
    metrics: {
      avgMTTR: 18,
      sloAchievement: 0.998,
      incidentCountThisWeek: 3,
      skillUsageThisWeek: 31,
      newSkillsThisWeek: 3,
    },
    schedule: [],
  },
  {
    id: 'team_infra_sre',
    name: 'Infra-SRE',
    memberIds: seedUsers.filter(u => u.teamId === 'team_infra_sre').map(u => u.id),
    skillIds: seedSkills.filter(s => s.teamId === 'team_infra_sre').map(s => s.id),
    coverage: [
      { domain: 'Linux系统调优', coverageCount: 5, totalMembers: 6, avgDepth: 82, healthStatus: 'healthy' },
      { domain: '监控告警配置', coverageCount: 5, totalMembers: 6, avgDepth: 80, healthStatus: 'healthy' },
      { domain: 'K8s基础运维', coverageCount: 2, totalMembers: 6, avgDepth: 35, healthStatus: 'attention' },
      { domain: 'K8s故障排查', coverageCount: 1, totalMembers: 6, avgDepth: 30, healthStatus: 'attention' },
    ],
    metrics: {
      avgMTTR: 20,
      sloAchievement: 0.997,
      incidentCountThisWeek: 4,
      skillUsageThisWeek: 19,
      newSkillsThisWeek: 1,
    },
    schedule: [],
  },
];

// ==================== Seed Reports ====================
export const seedReports: Report[] = [
  {
    id: 'report-weekly-20',
    title: 'DB-SRE 团队周报 W20',
    type: 'weekly',
    createdAt: '2026-05-15T08:00:00Z',
    authorId: 'user_lead_zhang',
    teamId: 'team_db_sre',
    summary: '本周团队处理 5 个 incident，MTTR 平均 22min，新增 2 个 Skill。',
    sections: [
      { id: 's1', title: '可靠性指标', content: 'SLO 达成率 99.6%，P1: 0, P2: 2, P3: 3', metrics: { mttr: 22, slo: 0.996 } },
      { id: 's2', title: '团队效能', content: 'Skill 使用 23 次，新创建 2 个，1 个草稿待确认', metrics: { skillUsage: 23, newSkills: 2 } },
      { id: 's3', title: '能力建设', content: '小李 Oracle 基础管理达标，Oracle RAC 仍为单点风险', metrics: {} },
      { id: 's4', title: '下周重点', content: '推进 RAC 传承计划，完成 K8s 网络策略 Skill 补齐', metrics: {} },
    ],
  },
  {
    id: 'report-monthly-05',
    title: 'Platform-SRE 团队月报 5月',
    type: 'monthly',
    createdAt: '2026-05-01T08:00:00Z',
    authorId: 'user_lead_li',
    teamId: 'team_platform_sre',
    summary: '5 月团队整体表现优异，SLO 达成率 99.8%，MTTR 降至 18min。',
    sections: [
      { id: 's1', title: '可靠性指标', content: 'SLO 99.8%，incident 总数 12，无 P1', metrics: {} },
      { id: 's2', title: '团队效能', content: 'Skill 使用 120 次，新增 10 个', metrics: {} },
      { id: 's3', title: '能力建设', content: 'K8s 网络/存储覆盖度从 40% 提升至 50%', metrics: {} },
    ],
  },
];

// ==================== Seed Org Snapshots ====================
export const seedOrgSnapshots: OrgSnapshot[] = Array.from({ length: 12 }, (_, i) => {
  const date = new Date('2026-02-15');
  date.setDate(date.getDate() + i * 7);
  return {
    date: date.toISOString().slice(0, 10),
    activeSkillCount: 280 + i * 3,
    coverageRate: 0.65 + i * 0.008,
    crossTeamReuseRate: 0.15 + i * 0.006,
    avgMTTR: 32 - i * 0.8,
    sloAchievement: 0.985 + i * 0.002,
    singlePointRisks: 9 - Math.floor(i / 3),
  };
});

// ==================== Seed Maturity Assessment ====================
export const seedMaturityAssessment: MaturityAssessment = {
  overallLevel: 3.2,
  overallLabel: 'L3 系统化',
  dimensions: [
    { name: 'Skill覆盖度', score: 3.5, trend: 'up', benchmark: 'avg' },
    { name: '知识传承效率', score: 2.8, trend: 'flat', benchmark: 'below' },
    { name: 'AI治理成熟度', score: 3.0, trend: 'up', benchmark: 'avg' },
    { name: '自动化程度', score: 3.4, trend: 'up', benchmark: 'above' },
    { name: '度量与持续改进', score: 3.3, trend: 'flat', benchmark: 'avg' },
  ],
};

// ==================== Seed AI Governance Report ====================
export const seedAIGovernanceReport: AIGovernanceReport = {
  month: '2026-05',
  totalSkills: 312,
  aiAssistedCount: 189,
  aiOnlyCount: 23,
  complianceRate: 0.93,
  pendingReview: 18,
  flagged: 5,
};

// ==================== Seed Skill Conflicts ====================
export const seedSkillConflicts: SkillConflict[] = [
  {
    id: 'conflict-1',
    severity: 'critical',
    skillA: { id: 'skill-oracle-slow-query-diag-v3', name: 'Oracle 慢查询诊断与优化', teamId: 'team_db_sre' },
    skillB: { id: 'skill-generated-5', name: 'Oracle AWR 自动分析报告 v2', teamId: 'team_db_sre' },
    conflictType: 'logic_contradiction',
    description: '两个 Skill 在索引重建步骤上建议相反：一个建议 ONLINE，一个建议 OFFLINE',
    suggestedAction: '召开评审会议统一索引重建标准',
  },
  {
    id: 'conflict-2',
    severity: 'critical',
    skillA: { id: 'skill-k8s-pod-evicted-diag-v1', name: 'K8s Pod 被驱逐诊断', teamId: 'team_db_sre' },
    skillB: { id: 'skill-generated-8', name: 'K8s Deployment 滚动更新故障 v3', teamId: 'team_platform_sre' },
    conflictType: 'parameter_mismatch',
    description: 'Pod 驱逐后重启的参数配置与 Deployment 更新策略不一致',
    suggestedAction: '协调两个团队统一 Pod 重启参数',
  },
  {
    id: 'conflict-3',
    severity: 'minor',
    skillA: { id: 'skill-generated-15', name: 'Linux 高负载排查 v2', teamId: 'team_infra_sre' },
    skillB: { id: 'skill-generated-22', name: 'Linux 内存泄漏定位 v1', teamId: 'team_infra_sre' },
    conflictType: 'overlap',
    description: '两个 Skill 的诊断步骤 1-2 高度重叠，造成重复',
    suggestedAction: '合并公共步骤为通用检查清单',
  },
  {
    id: 'conflict-4',
    severity: 'minor',
    skillA: { id: 'skill-generated-30', name: 'Prometheus 告警规则调优 v4', teamId: 'team_infra_sre' },
    skillB: { id: 'skill-generated-31', name: 'Grafana 仪表盘配置 v2', teamId: 'team_platform_sre' },
    conflictType: 'overlap',
    description: '告警阈值在 Grafana 和 Prometheus 中分别定义，维护困难',
    suggestedAction: '统一告警阈值管理至单一来源',
  },
];
