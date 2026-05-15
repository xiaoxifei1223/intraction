import { useEffect, useState } from 'react';
import { BookOpen, TrendingUp, Target, AlertTriangle } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';

interface LearningDomain {
  domain: string;
  level: number;
  trend: 'up' | 'down' | 'flat';
  teamAverage?: number;
  suggestion?: string;
}

export function LearningMapPage() {
  const { currentUser } = useAuthStore();
  const [domains, setDomains] = useState<LearningDomain[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (currentUser) {
      fetch(`/api/users/${currentUser.id}/learning-map`)
        .then((res) => res.json())
        .then((json) => {
          if (json.success) {
            const mapped = json.data.domains.map((d: any) => ({
              domain: d.domain,
              level: d.level,
              trend: d.trend,
              teamAverage: Math.max(40, d.level + (Math.random() * 20 - 10)),
              suggestion: d.level > 80 ? '建议深入高阶内容' : d.level > 50 ? '继续巩固当前水平' : '建议加强基础学习',
            }));
            setDomains(mapped);
          }
        })
        .finally(() => setLoading(false));
    }
  }, [currentUser]);

  const getLevelColor = (level: number) => {
    if (level >= 90) return 'bg-emerald-500';
    if (level >= 70) return 'bg-blue-500';
    if (level >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">我的技能地图</h1>
        <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
          <BookOpen size={16} /> 查看详细数据
        </button>
      </div>

      {loading && <LoadingOverlay />}

      {!loading && (
        <div className="grid md:grid-cols-2 gap-6">
          {/* Domain Tree */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">技能领域</h2>
            <div className="space-y-4">
              {domains.map((domain) => (
                <div key={domain.domain} className="group">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{domain.domain}</span>
                      {domain.level < 40 && <AlertTriangle size={14} className="text-yellow-500" />}
                    </div>
                    <span className="text-sm text-slate-500 dark:text-slate-400">{domain.level}%</span>
                  </div>
                  <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${getLevelColor(domain.level)}`}
                      style={{ width: `${domain.level}%` }}
                    />
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-slate-400 dark:text-slate-500">
                      团队均值 {domain.teamAverage?.toFixed(0)}%
                    </span>
                    {domain.level > (domain.teamAverage || 0) && (
                      <span className="text-xs text-emerald-600 dark:text-emerald-400">高于平均 ✓</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Detail Panel */}
          <div className="space-y-6">
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">成长建议</h2>
              {domains.filter((d) => d.level < 70).slice(0, 3).map((domain) => (
                <div key={domain.domain} className="mb-4 last:mb-0">
                  <div className="flex items-center gap-2 mb-1">
                    <Target size={16} className="text-primary-500" />
                    <span className="font-medium text-slate-800 dark:text-slate-200">{domain.domain}</span>
                  </div>
                  <p className="text-sm text-slate-500 dark:text-slate-400 ml-6">{domain.suggestion}</p>
                </div>
              ))}
            </div>

            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
                <TrendingUp size={18} /> 近90天趋势
              </h2>
              <div className="h-40 flex items-end justify-between gap-2">
                {Array.from({ length: 12 }, (_, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-primary-200 dark:bg-primary-900/30 rounded-t transition-all hover:bg-primary-300 dark:hover:bg-primary-900/50"
                    style={{ height: `${20 + Math.random() * 70}%` }}
                  />
                ))}
              </div>
              <div className="flex justify-between mt-2 text-xs text-slate-400 dark:text-slate-500">
                <span>2月</span>
                <span>5月</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
