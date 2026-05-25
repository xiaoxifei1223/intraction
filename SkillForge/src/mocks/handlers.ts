import { http, HttpResponse } from 'msw';
import { mockDelay } from '@/lib/mockDelay';
import {
  seedUsers,
  seedSkills,
  seedIncidents,
  seedTeams,
  seedReports,
  seedOrgSnapshots,
  seedMaturityAssessment,
  seedAIGovernanceReport,
  seedSkillConflicts,
} from './seeds/initialData';

function apiResponse<T>(data: T) {
  return HttpResponse.json({ success: true, data });
}

export const handlers = [
  // Auth
  http.get('/api/me', async () => {
    await mockDelay();
    return apiResponse(seedUsers.find((u) => u.id === 'user_wang_wu'));
  }),

  http.post('/api/auth/switch-role', async ({ request }) => {
    await mockDelay(100, 300);
    const body = (await request.json()) as { role: string };
    const user = seedUsers.find((u) => u.id === 'user_wang_wu');
    if (user) {
      user.role = body.role as 'engineer' | 'lead' | 'executive';
    }
    return apiResponse(user);
  }),

  // Diagnose
  http.get('/api/diagnose', async ({ request }) => {
    await mockDelay();
    const url = new URL(request.url);
    const query = url.searchParams.get('query') || '';
    const matchedSkills = seedSkills
      .filter((s) => s.name.includes(query) || query === '')
      .slice(0, 5)
      .map((skill) => ({
        skill,
        matchScore: Math.round(0.6 + Math.random() * 0.35),
        reason: `基于 ${skill.classification.domain.join('/')} 领域匹配`,
        estimatedTime: skill.avgResolutionTime,
      }));
    const experts = seedUsers
      .filter((u) => u.role === 'engineer')
      .slice(0, 2)
      .map((user) => ({
        user,
        relevantIncidents: Math.floor(Math.random() * 15),
        avgMTTR: user.metrics.avgMTTR,
        relatedSkills: Math.floor(Math.random() * 8),
      }));
    return apiResponse({
      queryInterpretation: query || '未提供查询',
      matchedSkills,
      suggestedExperts: experts,
      similarIncidents: seedIncidents.slice(0, 3),
    });
  }),

  // Skills
  http.get('/api/skills', async ({ request }) => {
    await mockDelay();
    const url = new URL(request.url);
    const search = url.searchParams.get('query') || '';
    const filtered = seedSkills.filter(
      (s) => s.name.includes(search) || s.classification.domain.some((d) => d.includes(search))
    );
    return apiResponse(filtered);
  }),

  http.get('/api/skills/:id', async ({ params }) => {
    await mockDelay();
    const skill = seedSkills.find((s) => s.id === params.id);
    return apiResponse(skill);
  }),

  // Snippets
  http.get('/api/snippets', async ({ request }) => {
    await mockDelay();
    const url = new URL(request.url);
    const query = url.searchParams.get('query') || '';
    const snippets = seedSkills.slice(0, 20).map((s) => ({
      id: `snippet-${s.id}`,
      title: s.name,
      command: s.content.diagnosisSteps[0]?.command || 'echo "No command"',
      description: s.content.triggerConditions,
      tags: s.classification.domain,
      authorId: s.authorId,
      useCount: s.useCount,
      successRate: s.successRate,
      applicableEnv: s.classification.domain,
    }));
    const filtered = snippets.filter(
      (s) => s.title.includes(query) || s.tags.some((t) => t.includes(query))
    );
    return apiResponse(filtered);
  }),

  // Learning Map
  http.get('/api/users/:id/learning-map', async ({ params }) => {
    await mockDelay();
    const user = seedUsers.find((u) => u.id === params.id);
    return apiResponse({
      userId: params.id,
      domains: user?.skillsMastery || [],
      suggestions: ['Oracle RAC 故障切换', 'K8s 网络策略配置', 'Prometheus 高级告警'],
    });
  }),

  // Arena
  http.get('/api/arena/scenarios', async () => {
    await mockDelay();
    return apiResponse([
      { id: 'scenario-1', title: 'Oracle ORA-04031 模拟', difficulty: 'intermediate', estimatedTime: 20, description: 'Shared Pool 内存不足诊断' },
      { id: 'scenario-2', title: 'K8s Pod 驱逐排查', difficulty: 'beginner', estimatedTime: 15, description: '节点资源压力导致 Pod 被驱逐' },
      { id: 'scenario-3', title: 'API 网关 5xx 故障', difficulty: 'advanced', estimatedTime: 30, description: '链路追踪定位服务故障' },
    ]);
  }),

  // Team
  http.get('/api/team/pulse', async () => {
    await mockDelay();
    return apiResponse({
      team: seedTeams[0],
      activeIncidents: seedIncidents.filter((i) => i.status !== 'closed').slice(0, 3),
      skillUsageToday: 5,
      newDrafts: 1,
      mttrThisWeek: 22,
      mttrLastWeek: 28,
      highRiskDomains: [
        { domain: 'Oracle RAC', singleOwner: '@li_si', risk: 'critical' as const },
        { domain: 'K8s网络策略', singleOwner: '@chen_qi', risk: 'high' as const },
      ],
    });
  }),

  http.get('/api/team/radar', async () => {
    await mockDelay();
    return apiResponse({
      team: seedTeams[0],
      radarData: seedTeams[0].coverage.map((c) => ({
        domain: c.domain,
        coverage: Math.round((c.coverageCount / c.totalMembers) * 100),
        depth: c.avgDepth,
        health: c.healthStatus,
      })),
    });
  }),

  http.get('/api/team/mttr', async () => {
    await mockDelay();
    const weeks = Array.from({ length: 12 }, (_, i) => ({
      week: `W${i + 9}`,
      mttr: 30 - i * 0.8 + Math.random() * 4,
      incidentCount: Math.floor(Math.random() * 8) + 2,
      withSkill: 18 + Math.random() * 5,
      withoutSkill: 32 + Math.random() * 6,
    }));
    return apiResponse({
      overallMTTR: 22,
      phases: {
        detect: 3,
        response: 8,
        diagnose: 9,
        fix: 2,
      },
      weeklyTrend: weeks,
      topBottlenecks: [
        { phase: '诊断→修复', percentage: 41, topScenario: 'K8s网络类问题', avgTime: 15 },
        { phase: '响应→诊断', percentage: 36, topScenario: 'Oracle性能类问题', avgTime: 10 },
      ],
    });
  }),

  http.get('/api/team/members', async () => {
    await mockDelay();
    return apiResponse(seedUsers.filter((u) => u.teamId === 'team_db_sre' && u.role === 'engineer'));
  }),

  http.get('/api/team/schedule', async () => {
    await mockDelay();
    return apiResponse(seedTeams[0].schedule);
  }),

  http.get('/api/team/reports', async () => {
    await mockDelay();
    return apiResponse(seedReports);
  }),

  // Org / Executive
  http.get('/api/org/snapshot', async () => {
    await mockDelay();
    const latest = seedOrgSnapshots[seedOrgSnapshots.length - 1];
    const prev = seedOrgSnapshots[seedOrgSnapshots.length - 2];
    return apiResponse({
      ...latest,
      trend: {
        activeSkillCount: latest.activeSkillCount - prev.activeSkillCount,
        coverageRate: latest.coverageRate - prev.coverageRate,
        singlePointRisks: latest.singlePointRisks - prev.singlePointRisks,
      },
      alerts: [
        { type: 'warning' as const, message: 'K8s高级运维领域能力缺口持续3周未改善', suggestion: '建议专项招聘或外部培训投入' },
      ],
    });
  }),

  http.get('/api/org/governance', async () => {
    await mockDelay();
    return apiResponse({
      report: seedAIGovernanceReport,
      conflicts: seedSkillConflicts,
      policies: [
        { riskLevel: 'low' as const, action: '自动发布' },
        { riskLevel: 'medium' as const, action: '需确认' },
        { riskLevel: 'high' as const, action: '需审批' },
        { riskLevel: 'critical' as const, action: '需安全审核' },
      ],
    });
  }),

  http.get('/api/org/maturity', async () => {
    await mockDelay();
    return apiResponse(seedMaturityAssessment);
  }),

  http.get('/api/org/strategy-align', async () => {
    await mockDelay();
    return apiResponse({
      goals: [
        { id: 'g1', name: 'MTTR降低30%', target: 0.7, current: 0.55 },
        { id: 'g2', name: 'SLO 99.99%', target: 0.9999, current: 0.996 },
        { id: 'g3', name: '零重大变更故障', target: 1.0, current: 0.85 },
      ],
      flows: [
        { from: 'MTTR降低30%', to: '故障诊断自动化', value: 80, status: 'good' as const },
        { from: 'MTTR降低30%', to: 'On-call团队达标', value: 60, status: 'warning' as const },
        { from: 'SLO 99.99%', to: '变更零故障', value: 85, status: 'good' as const },
      ],
    });
  }),

  http.get('/api/org/planner', async () => {
    await mockDelay();
    return apiResponse({
      directions: [
        { name: 'K8s/云原生', current: 2.5, target6m: 3.5, target12m: 4.0, investment: 5 },
        { name: 'AI Ops', current: 1.0, target6m: 2.0, target12m: 3.0, investment: 4 },
        { name: '可观测性(eBPF)', current: 0, target6m: 1.5, target12m: 2.5, investment: 3 },
        { name: '安全运维', current: 2.0, target6m: 2.5, target12m: 3.0, investment: 3 },
        { name: '传统DB运维', current: 4.0, target6m: 4.0, target12m: 3.5, investment: 1 },
      ],
      milestones: [
        { date: 'Q3-2026', event: 'K8s能力覆盖率达80%' },
        { date: 'Q4-2026', event: 'AI Ops试点团队达到L2' },
        { date: 'Q1-2027', event: 'eBPF可观测性替代30%传统监控' },
        { date: 'Q2-2027', event: 'K8s达到L4预测式水平' },
      ],
      headcount: { current: 52, recommended: 58, delta: 6 },
    });
  }),

  http.get('/api/org/board-report', async () => {
    await mockDelay();
    return apiResponse({
      slides: [
        { title: '一句话总结', content: '组织技术状态稳定，MTTR 持续下降，人才风险可控。' },
        { title: '关键成果', content: '可靠性 99.96% / 效率提升 19% / 人才传承计划启动 2 个' },
        { title: '技术能力资产', content: '活跃 Skill 312 个，增长 8 个，健康度 93%。' },
        { title: '风险与需求', content: 'K8s高级运维缺口需决策，建议增加 6 HC。' },
        { title: '下季度目标', content: 'K8s 覆盖率达 80%，AI Ops 试点团队达 L2。' },
      ],
    });
  }),

  // Incidents
  http.get('/api/incidents', async ({ request }) => {
    await mockDelay();
    const url = new URL(request.url);
    const status = url.searchParams.get('status');
    let incidents = seedIncidents;
    if (status) {
      incidents = incidents.filter((i) => i.status === status);
    }
    return apiResponse(incidents.slice(0, 50));
  }),

  http.get('/api/incidents/:id', async ({ params }) => {
    await mockDelay();
    const incident = seedIncidents.find((i) => i.id === params.id);
    return apiResponse(incident);
  }),

  // Search
  http.get('/api/search', async ({ request }) => {
    await mockDelay();
    const url = new URL(request.url);
    const q = url.searchParams.get('q') || '';
    const skills = seedSkills.filter((s) => s.name.includes(q)).slice(0, 5);
    const incidents = seedIncidents.filter((i) => i.title.includes(q) || i.id.includes(q)).slice(0, 5);
    const users = seedUsers.filter((u) => u.name.includes(q) || u.handle.includes(q)).slice(0, 5);
    return apiResponse({ skills, incidents, users });
  }),
];
