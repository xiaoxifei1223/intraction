import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Download, CheckCircle, AlertTriangle, TrendingUp, TrendingDown, ArrowRight } from 'lucide-react';
import { MetricCard } from '@/components/shared/MetricCard';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';

export function ExecutiveDashboardPage() {
  const navigate = useNavigate();
  const [snapshot, setSnapshot] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/org/snapshot')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setSnapshot(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingOverlay />;
  if (!snapshot) return <LoadingOverlay />;

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">技术组织周状态</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">W20 2026 · 一页纸摘要</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
          <Download size={16} /> 导出
        </button>
      </div>

      {/* Overall Status */}
      <div className="flex items-center gap-2 mb-6">
        <CheckCircle size={20} className="text-emerald-500" />
        <span className="text-lg font-medium text-emerald-700 dark:text-emerald-400">🟢 稳定</span>
      </div>

      {/* Three Pillars */}
      <div className="grid md:grid-cols-3 gap-4 mb-8">
        <div
          onClick={() => navigate('/team/mttr')}
          className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 hover:shadow-lg hover:-translate-y-1 transition-all cursor-pointer"
        >
          <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-4">可靠性</h2>
          <div className="mb-4">
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">{(snapshot.sloAchievement * 100).toFixed(2)}%</div>
            <div className="text-sm text-slate-500 dark:text-slate-400">SLO 达成</div>
          </div>
          <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400 mb-2">
            <CheckCircle size={14} /> 目标: 99.9%
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400">
            P1: 0 &nbsp; P2: 2
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">MTTR: {snapshot.avgMTTR}min</span>
              <span className="text-xs text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                <TrendingDown size={12} /> 19%
              </span>
            </div>
          </div>
        </div>

        <div
          onClick={() => navigate('/executive/governance')}
          className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 hover:shadow-lg hover:-translate-y-1 transition-all cursor-pointer"
        >
          <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-4">能力资产</h2>
          <div className="mb-4">
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">{snapshot.activeSkillCount}</div>
            <div className="text-sm text-slate-500 dark:text-slate-400">活跃 Skill</div>
          </div>
          <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400 mb-2">
            <TrendingUp size={14} /> +{snapshot.trend?.activeSkillCount || 8} vs上周
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400">
            覆盖率: {(snapshot.coverageRate * 100).toFixed(0)}%
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800">
            <div className="text-sm text-slate-600 dark:text-slate-400">
              复用率: {(snapshot.crossTeamReuseRate * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        <div
          onClick={() => navigate('/team/radar')}
          className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 hover:shadow-lg hover:-translate-y-1 transition-all cursor-pointer"
        >
          <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-4">人才风险</h2>
          <div className="mb-4">
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">{snapshot.singlePointRisks}</div>
            <div className="text-sm text-slate-500 dark:text-slate-400">单点依赖</div>
          </div>
          <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400 mb-2">
            <TrendingDown size={14} /> -1 vs上周
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400">
            本月离职: 1
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800">
            <div className="text-sm text-emerald-600 dark:text-emerald-400">
              传承计划已启动
            </div>
          </div>
        </div>
      </div>

      {/* Alerts */}
      {snapshot.alerts && snapshot.alerts.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-yellow-800 dark:text-yellow-300 mb-4 flex items-center gap-2">
            <AlertTriangle size={18} /> 需要您关注
          </h2>
          {snapshot.alerts.map((alert: any, i: number) => (
            <div key={i} className="mb-4 last:mb-0">
              <p className="text-sm text-yellow-700 dark:text-yellow-400 font-medium mb-1">⚠️ {alert.message}</p>
              <p className="text-xs text-yellow-600 dark:text-yellow-500 mb-2">{alert.suggestion}</p>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => navigate('/executive/planner')}
                  className="text-xs px-3 py-1.5 rounded-md bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 hover:bg-yellow-200 dark:hover:bg-yellow-900/50 transition-colors"
                >
                  查看详细分析
                </button>
                <button
                  onClick={() => navigate('/executive/planner')}
                  className="text-xs px-3 py-1.5 rounded-md bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 hover:bg-yellow-200 dark:hover:bg-yellow-900/50 transition-colors"
                >
                  发起预算申请
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
