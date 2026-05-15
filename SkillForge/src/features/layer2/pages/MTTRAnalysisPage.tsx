import { useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { MetricCard } from '@/components/shared/MetricCard';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { useLayer2Store } from '@/stores/layer2Store';
import { ArrowDown, AlertTriangle, TrendingDown, Clock } from 'lucide-react';

export function MTTRAnalysisPage() {
  const { mttrAnalysis, fetchMTTRAnalysis, loading } = useLayer2Store();

  useEffect(() => {
    fetchMTTRAnalysis();
  }, []);

  if (loading || !mttrAnalysis) return <LoadingOverlay />;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">MTTR 趋势与归因分析</h1>
        <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
          📥 导出报告
        </button>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
        <MetricCard title="总体MTTR" value={mttrAnalysis.overallMTTR} unit="min" status="good" trend="down" trendValue="vs上月28" icon={TrendingDown} />
        <MetricCard title="检测→响应" value={mttrAnalysis.phases.detect} unit="min" status="good" trend="down" trendValue="vs上月4" icon={Clock} />
        <MetricCard title="响应→诊断" value={mttrAnalysis.phases.response} unit="min" status="good" trend="down" trendValue="vs上月12" icon={Clock} />
        <MetricCard title="诊断→修复" value={mttrAnalysis.phases.diagnose} unit="min" status="neutral" trend="flat" trendValue="vs上月10" icon={Clock} />
        <MetricCard title="修复→验证" value={mttrAnalysis.phases.fix} unit="min" status="neutral" trend="flat" trendValue="vs上月2" icon={Clock} />
      </div>

      {/* Trend Chart */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">MTTR 趋势 - 近12周</h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mttrAnalysis.weeklyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="week" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f1f5f9' }}
              />
              <Legend />
              <Line type="monotone" dataKey="mttr" name="团队MTTR" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6' }} />
              <Line type="monotone" dataKey="withSkill" name="使用Skill" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981' }} />
              <Line type="monotone" dataKey="withoutSkill" name="未使用Skill" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Skill Attribution */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Skill 贡献归因</h2>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[{ name: '对比', 使用Skill: 18, 未使用Skill: 32 }]} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f1f5f9' }} />
                <Legend />
                <Bar dataKey="使用Skill" fill="#10b981" radius={[0, 4, 4, 0]} />
                <Bar dataKey="未使用Skill" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-4">
            Skill 对 MTTR 的因果贡献估计: <strong className="text-emerald-600 dark:text-emerald-400">降低 38%</strong> (双重差分法估算)
            <br />
            置信度: 82%
          </p>
        </div>

        {/* Bottleneck */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">最大瓶颈识别</h2>
          <div className="bg-yellow-50 dark:bg-yellow-900/10 rounded-lg p-4 mb-4">
            <p className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
              "诊断→修复" 阶段占比 41% — 最大改善空间
            </p>
          </div>
          <div className="space-y-3">
            {mttrAnalysis.topBottlenecks?.map((b: any, i: number) => (
              <div key={i} className="border-b border-slate-100 dark:border-slate-800 last:border-0 pb-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-medium text-slate-800 dark:text-slate-200">
                    {i + 1}. {b.topScenario}
                  </span>
                  <AlertTriangle size={14} className="text-yellow-500" />
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">
                  平均诊断 {b.avgTime}min · Skill覆盖不足
                </div>
                <button className="text-xs text-primary-600 dark:text-primary-400 hover:underline">
                  查看缺口详情 →
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
