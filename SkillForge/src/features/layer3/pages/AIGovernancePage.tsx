import { useEffect, useState } from 'react';
import { ShieldCheck, Bot, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { useAuthStore } from '@/stores/authStore';

export function AIGovernancePage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'policy' | 'conflicts'>('overview');

  useEffect(() => {
    fetch('/api/org/governance')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setData(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingOverlay />;
  if (!data) return <LoadingOverlay />;

  const report = data.report;
  const conflicts = data.conflicts;

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">AI 治理控制台</h1>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 mb-6 border-b border-slate-200 dark:border-slate-800">
        {[
          { key: 'overview', label: '使用全景' },
          { key: 'policy', label: '策略管理' },
          { key: 'conflicts', label: '冲突检测' },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.key
                ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <div className="space-y-6">
          <div className="grid md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 text-center">
              <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">{report.aiAssistedCount}</div>
              <div className="text-sm text-slate-500 dark:text-slate-400">AI辅助生成</div>
              <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">{(report.aiAssistedCount / report.totalSkills * 100).toFixed(0)}%</div>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 text-center">
              <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">{report.totalSkills - report.aiAssistedCount - report.aiOnlyCount}</div>
              <div className="text-sm text-slate-500 dark:text-slate-400">纯人工</div>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 text-center">
              <div className="text-3xl font-bold text-red-600 dark:text-red-400">{report.aiOnlyCount}</div>
              <div className="text-sm text-slate-500 dark:text-slate-400">纯AI未审核 ⚠️</div>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 text-center">
              <div className="text-3xl font-bold text-emerald-600 dark:text-emerald-400">{(report.complianceRate * 100).toFixed(0)}%</div>
              <div className="text-sm text-slate-500 dark:text-slate-400">合规率</div>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">安全状态</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm text-emerald-700 dark:text-emerald-400">
                <CheckCircle size={16} /> 🟢 合规 {report.totalSkills - report.pendingReview - report.flagged} 个 ({((report.totalSkills - report.pendingReview - report.flagged) / report.totalSkills * 100).toFixed(0)}%)
              </div>
              <div className="flex items-center gap-2 text-sm text-yellow-700 dark:text-yellow-400">
                <Clock size={16} /> 🟡 待审核 {report.pendingReview} 个 ({(report.pendingReview / report.totalSkills * 100).toFixed(0)}%)
              </div>
              <div className="flex items-center gap-2 text-sm text-red-700 dark:text-red-400">
                <AlertTriangle size={16} /> 🔴 需要处理 {report.flagged} 个 ({(report.flagged / report.totalSkills * 100).toFixed(0)}%)
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'policy' && (
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">治理策略</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <th className="pb-2 font-medium">风险等级</th>
                <th className="pb-2 font-medium">处理方式</th>
              </tr>
            </thead>
            <tbody>
              {data.policies.map((policy: any, i: number) => (
                <tr key={i} className="border-b border-slate-100 dark:border-slate-800 last:border-0">
                  <td className="py-3 text-slate-900 dark:text-slate-100 capitalize">{policy.riskLevel}</td>
                  <td className="py-3 text-slate-600 dark:text-slate-400">{policy.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {activeTab === 'conflicts' && (
        <div className="space-y-4">
          {conflicts.map((conflict: any) => (
            <div
              key={conflict.id}
              className={`bg-white dark:bg-slate-900 rounded-xl border p-6 ${
                conflict.severity === 'critical'
                  ? 'border-red-200 dark:border-red-800'
                  : 'border-slate-200 dark:border-slate-800'
              }`}
            >
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle size={16} className={conflict.severity === 'critical' ? 'text-red-500' : 'text-yellow-500'} />
                <span className={`text-sm font-medium ${conflict.severity === 'critical' ? 'text-red-600 dark:text-red-400' : 'text-yellow-600 dark:text-yellow-400'}`}>
                  {conflict.severity === 'critical' ? '严重冲突' : '轻微不一致'}
                </span>
              </div>
              <h3 className="font-medium text-slate-900 dark:text-slate-100 mb-2">
                {conflict.skillA.name} vs {conflict.skillB.name}
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">{conflict.description}</p>
              <div className="text-sm text-slate-500 dark:text-slate-400 mb-3">
                建议: {conflict.suggestedAction}
              </div>
              <button className="px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                发起协调流程
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
