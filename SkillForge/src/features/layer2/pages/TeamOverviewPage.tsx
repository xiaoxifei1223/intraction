import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Calendar, Clock, AlertCircle, Zap, ArrowDown, CheckCircle } from 'lucide-react';
import { MetricCard } from '@/components/shared/MetricCard';
import { IncidentBadge } from '@/components/shared/IncidentBadge';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { useLayer2Store } from '@/stores/layer2Store';

export function TeamOverviewPage() {
  const navigate = useNavigate();
  const { teamPulse, fetchTeamPulse, loading } = useLayer2Store();

  useEffect(() => {
    fetchTeamPulse();
  }, []);

  if (loading) return <LoadingOverlay />;
  if (!teamPulse) return <LoadingOverlay />;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">DB-SRE 团队概览</h1>
        <div className="flex items-center gap-3 text-sm text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1"><Calendar size={14} /> 5/15</span>
          <span className="flex items-center gap-1"><Clock size={14} /> 自动刷新: 30s</span>
        </div>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
        <MetricCard title="昨夜值班" value="✅ 平稳" unit="" status="good" trend="flat" trendValue="无P1/P2" />
        <MetricCard title="今日排班" value="日班:张三" unit="" status="neutral" />
        <MetricCard title="进行中" value="P3: 1件" unit="" status="warning" />
        <MetricCard title="Skill动态" value="使用5次" unit="+1草稿" status="good" trend="up" trendValue="+1 vs昨日" />
        <MetricCard
          title="本周MTTR"
          value={teamPulse.team.metrics.avgMTTR}
          unit="min"
          status="good"
          trend="down"
          trendValue="↓ vs上周28"
          onClick={() => navigate('/team/mttr')}
        />
      </div>

      {/* Active Incidents */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">实时 Incident 态势</h2>
        {teamPulse.activeIncidents.length > 0 ? (
          <div className="space-y-4">
            {teamPulse.activeIncidents.map((incident: any) => (
              <div key={incident.id} className="border border-slate-200 dark:border-slate-800 rounded-lg p-4">
                <div className="flex items-center gap-3 mb-3">
                  <IncidentBadge priority={incident.priority} />
                  <span className="font-medium text-slate-900 dark:text-slate-100">{incident.id}</span>
                  <span className="text-sm text-slate-500 dark:text-slate-400">{incident.title}</span>
                  <span className="ml-auto text-xs text-slate-400 dark:text-slate-500">进行中 22min</span>
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                  响应人: @zhang_san (入职8个月) · 状态: 诊断阶段 · 使用 Skill: "Oracle慢查询诊断" (步骤 2/5)
                </div>
                <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400 mb-3">
                  <CheckCircle size={14} /> 🟢 进展正常 - 预计再需15-20min (该响应人历史同类成功率 85%)
                </div>
                <div className="flex items-center gap-2">
                  <button className="px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                    进入频道
                  </button>
                  <button className="px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                    指派支援
                  </button>
                  <button className="px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                    查看实时步骤
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 py-4">
            <CheckCircle size={18} /> 🟢 当前无活跃高优事故
          </div>
        )}
      </div>

      {/* Risk & Changes */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">高风险单点</h2>
          <div className="space-y-3">
            {teamPulse.highRiskDomains?.map((risk: any, i: number) => (
              <div key={i} className="flex items-center justify-between py-2 border-b border-slate-100 dark:border-slate-800 last:border-0">
                <div className="flex items-center gap-2">
                  <AlertCircle size={16} className={risk.risk === 'critical' ? 'text-red-500' : 'text-yellow-500'} />
                  <span className="text-sm text-slate-700 dark:text-slate-300">{risk.domain}: 仅 {risk.singleOwner} 掌握</span>
                </div>
                <button className="text-xs px-2 py-1 rounded bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors">
                  启动传承计划
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">本周能力变化</h2>
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300">
              <CheckCircle size={14} className="text-emerald-500" />
              @小李 K8s基础运维: 未达标 → 已达标 ✅
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300">
              <ArrowDown size={14} className="text-emerald-500 rotate-180" />
              Oracle性能领域团队覆盖率: 50% → 67% ↑
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300">
              <Zap size={14} className="text-yellow-500" />
              新增 2 个 Skill 草稿待确认
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
