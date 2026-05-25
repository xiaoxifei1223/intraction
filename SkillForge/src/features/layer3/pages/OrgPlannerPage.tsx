import { useEffect, useState } from 'react';
import { Compass, Star, Users, AlertTriangle } from 'lucide-react';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';

export function OrgPlannerPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [scenario, setScenario] = useState<'normal' | 'no-hire' | 'increase-budget'>('normal');

  useEffect(() => {
    fetch('/api/org/planner')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setData(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingOverlay />;
  if (!data) return <LoadingOverlay />;

  const modifier = scenario === 'no-hire' ? -0.3 : scenario === 'increase-budget' ? 0.3 : 0;

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">12个月能力前瞻</h1>
        <select
          value={scenario}
          onChange={(e) => setScenario(e.target.value as any)}
          className="px-3 py-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm text-slate-700 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-primary-500"
        >
          <option value="normal">正常招聘</option>
          <option value="no-hire">H2 零招聘</option>
          <option value="increase-budget">预算增加 20%</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <th className="pb-3 font-medium">能力方向</th>
                <th className="pb-3 font-medium">当前</th>
                <th className="pb-3 font-medium">6个月后</th>
                <th className="pb-3 font-medium">12个月后</th>
                <th className="pb-3 font-medium">投入建议</th>
              </tr>
            </thead>
            <tbody>
              {data.directions.map((dir: any, i: number) => {
                const target12 = Math.max(0, Math.min(5, dir.target12m + modifier));
                return (
                  <tr key={i} className="border-b border-slate-100 dark:border-slate-800 last:border-0">
                    <td className="py-3 text-slate-900 dark:text-slate-100 font-medium">{dir.name}</td>
                    <td className="py-3 text-slate-600 dark:text-slate-400">L{dir.current}</td>
                    <td className="py-3 text-slate-600 dark:text-slate-400">L{dir.target6m}</td>
                    <td className={`py-3 font-medium ${scenario !== 'normal' ? 'text-yellow-600 dark:text-yellow-400' : 'text-slate-900 dark:text-slate-100'}`}>
                      L{target12.toFixed(1)}
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1">
                        {Array.from({ length: 5 }, (_, j) => (
                          <Star
                            key={j}
                            size={12}
                            className={j < dir.investment ? 'text-yellow-500 fill-yellow-500' : 'text-slate-300 dark:text-slate-600'}
                          />
                        ))}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Milestones */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">关键里程碑</h2>
          <div className="space-y-4">
            {data.milestones.map((m: any, i: number) => (
              <div key={i} className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-primary-500 mt-2 shrink-0" />
                <div>
                  <div className="text-sm font-medium text-slate-800 dark:text-slate-200">{m.date}</div>
                  <div className="text-sm text-slate-500 dark:text-slate-400">{m.event}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
            <Users size={18} /> 所需资源
          </h2>
          <div className="text-center py-4">
            <div className="text-4xl font-bold text-slate-900 dark:text-slate-100">
              {data.headcount.current} → {data.headcount.recommended}
            </div>
            <div className="text-sm text-slate-500 dark:text-slate-400 mt-2">
              建议增加 <span className="text-emerald-600 dark:text-emerald-400 font-medium">+{data.headcount.delta} HC</span>
            </div>
          </div>
          {scenario === 'no-hire' && (
            <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/10 rounded-lg p-3 flex items-start gap-2">
              <AlertTriangle size={16} className="text-yellow-500 mt-0.5" />
              <p className="text-sm text-yellow-700 dark:text-yellow-400">
                零招聘方案下，AI Ops 方向纯内部培养不现实，建议调整预期。
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
