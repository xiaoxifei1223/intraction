import { useEffect, useState } from 'react';
import { Award, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';

export function MaturityAssessmentPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/org/maturity')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setData(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingOverlay />;
  if (!data) return <LoadingOverlay />;

  const levelInt = Math.floor(data.overallLevel);
  const levelFrac = data.overallLevel - levelInt;

  const maturityLabels = ['L1 临时式', 'L2 积累式', 'L3 系统化', 'L4 预测式', 'L5 自进化'];
  const maturityColors = ['bg-slate-400', 'bg-blue-500', 'bg-indigo-500', 'bg-violet-500', 'bg-emerald-500'];

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">成熟度评估</h1>
      </div>

      {/* Overall */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-slate-500 dark:text-slate-400 mb-2">当前成熟度等级</div>
            <div className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
              {data.overallLabel}
            </div>
            <div className="text-lg text-slate-600 dark:text-slate-400">
              得分 {data.overallLevel}/5.0
            </div>
          </div>
          <div className="w-24 h-24">
            <svg viewBox="0 0 36 36" className="w-full h-full">
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="#334155"
                strokeWidth="3"
              />
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="#6366f1"
                strokeWidth="3"
                strokeDasharray={`${(data.overallLevel / 5) * 100}, 100`}
              />
            </svg>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Ladder */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">成熟度阶梯</h2>
          <div className="space-y-3">
            {[5, 4, 3, 2, 1].map((level) => (
              <div key={level} className="flex items-center gap-3">
                <div className="w-16 text-sm text-slate-500 dark:text-slate-400 text-right">{maturityLabels[level - 1]}</div>
                <div className="flex-1 h-6 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${maturityColors[level - 1]} ${level === levelInt ? 'opacity-100' : level < levelInt ? 'opacity-60' : 'opacity-20'}`}
                    style={{ width: level === levelInt ? `${levelFrac * 100}%` : level < levelInt ? '100%' : '0%' }}
                  />
                </div>
                <div className="w-12 text-xs text-slate-400 dark:text-slate-500">
                  {level === levelInt ? `${(levelFrac * 100).toFixed(0)}%` : level < levelInt ? '100%' : '0%'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Radar */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">维度得分</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={data.dimensions}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 5]} tick={{ fill: '#64748b', fontSize: 10 }} />
                <Radar dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Dimensions Detail */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">维度详情</h2>
        <div className="space-y-4">
          {data.dimensions.map((dim: any, i: number) => (
            <div key={i} className="flex items-center justify-between py-3 border-b border-slate-100 dark:border-slate-800 last:border-0">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-slate-800 dark:text-slate-200">{dim.name}</span>
                {dim.trend === 'up' && <TrendingUp size={14} className="text-emerald-500" />}
                {dim.trend === 'down' && <TrendingDown size={14} className="text-red-500" />}
                {dim.trend === 'flat' && <Minus size={14} className="text-slate-400" />}
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm text-slate-600 dark:text-slate-400">{dim.score}/5.0</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  dim.benchmark === 'above' ? 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400' :
                  dim.benchmark === 'avg' ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400' :
                  'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-600 dark:text-yellow-400'
                }`}>
                  {dim.benchmark === 'above' ? '高于平均' : dim.benchmark === 'avg' ? '平均' : '低于平均'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
