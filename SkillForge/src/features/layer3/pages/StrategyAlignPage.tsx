import { useEffect, useState } from 'react';
import { GitMerge, ArrowRight, AlertTriangle } from 'lucide-react';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';

export function StrategyAlignPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/org/strategy-align')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setData(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingOverlay />;
  if (!data) return <LoadingOverlay />;

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">战略对齐</h1>
      </div>

      {/* Goals */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">组织目标</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {data.goals.map((goal: any) => (
            <div key={goal.id} className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4">
              <div className="font-medium text-slate-900 dark:text-slate-100 mb-2">{goal.name}</div>
              <div className="text-sm text-slate-500 dark:text-slate-400 mb-2">
                当前: {(goal.current * 100).toFixed(0)}% / 目标: {(goal.target * 100).toFixed(0)}%
              </div>
              <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${goal.current >= goal.target ? 'bg-emerald-500' : goal.current >= goal.target * 0.7 ? 'bg-yellow-500' : 'bg-red-500'}`}
                  style={{ width: `${Math.min(goal.current / goal.target * 100, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Simplified Sankey */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
          <GitMerge size={18} /> 指标-能力映射
        </h2>
        <div className="space-y-4">
          {data.flows.map((flow: any, i: number) => (
            <div key={i} className="flex items-center gap-4">
              <div className="w-32 text-sm font-medium text-slate-900 dark:text-slate-100 text-right">{flow.from}</div>
              <ArrowRight size={16} className={
                flow.status === 'good' ? 'text-emerald-500' : flow.status === 'warning' ? 'text-yellow-500' : 'text-red-500'
              } />
              <div className="flex-1 bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-700 dark:text-slate-300">{flow.to}</span>
                  <span className={`text-xs font-medium ${
                    flow.status === 'good' ? 'text-emerald-600 dark:text-emerald-400' : 'text-yellow-600 dark:text-yellow-400'
                  }`}>
                    满足度 {flow.value}%
                  </span>
                </div>
                <div className="h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden mt-2">
                  <div
                    className={`h-full rounded-full ${flow.status === 'good' ? 'bg-emerald-500' : 'bg-yellow-500'}`}
                    style={{ width: `${flow.value}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Gap List */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
          <AlertTriangle size={18} className="text-yellow-500" /> 差距清单
        </h2>
        <div className="space-y-3">
          {data.flows.filter((f: any) => f.status !== 'good').map((flow: any, i: number) => (
            <div key={i} className="flex items-center justify-between py-3 border-b border-slate-100 dark:border-slate-800 last:border-0">
              <div>
                <div className="text-sm font-medium text-slate-800 dark:text-slate-200">{flow.from} → {flow.to}</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">当前满足度 {flow.value}%，目标需达到 80%</div>
              </div>
              <button className="text-xs px-3 py-1.5 rounded-md bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 hover:bg-primary-100 dark:hover:bg-primary-900/30 transition-colors">
                查看投入建议
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
