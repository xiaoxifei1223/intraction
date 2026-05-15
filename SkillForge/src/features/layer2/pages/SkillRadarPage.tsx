import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Users } from 'lucide-react';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { useLayer2Store } from '@/stores/layer2Store';

export function SkillRadarPage() {
  const navigate = useNavigate();
  const { skillRadar, fetchSkillRadar, loading } = useLayer2Store();
  const [selectedDomain, setSelectedDomain] = useState<any>(null);

  useEffect(() => {
    fetchSkillRadar();
  }, []);

  if (loading || !skillRadar) return <LoadingOverlay />;

  const radarData = skillRadar.radarData || [];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">DB-SRE 技能雷达</h1>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Radar Chart */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-4">团队能力全景</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="domain" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                <Radar name="覆盖率" dataKey="coverage" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Radar name="深度" dataKey="depth" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Domain Detail */}
        <div className="space-y-4">
          {radarData.map((domain: any) => (
            <button
              key={domain.domain}
              onClick={() => setSelectedDomain(domain)}
              className="w-full text-left bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 hover:shadow-md transition-all"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-slate-900 dark:text-slate-100">{domain.domain}</span>
                {domain.coverage < 30 && <AlertTriangle size={16} className="text-red-500" />}
              </div>
              <div className="flex items-center gap-4 text-sm text-slate-500 dark:text-slate-400">
                <span>覆盖率: {domain.coverage}%</span>
                <span>深度: {domain.depth}</span>
                <span className={domain.health === 'healthy' ? 'text-emerald-500' : 'text-yellow-500'}>
                  {domain.health === 'healthy' ? '🟢 健康' : '🟡 需关注'}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Single Point Risk Table */}
      <div className="mt-8 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">单点风险汇总</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <th className="pb-2 font-medium">领域</th>
                <th className="pb-2 font-medium">唯一掌握者</th>
                <th className="pb-2 font-medium">业务影响</th>
                <th className="pb-2 font-medium">风险等级</th>
                <th className="pb-2 font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-slate-100 dark:border-slate-800">
                <td className="py-3 text-slate-900 dark:text-slate-100">Oracle RAC</td>
                <td className="py-3 text-slate-600 dark:text-slate-400">@li_si</td>
                <td className="py-3 text-slate-600 dark:text-slate-400">核心支付</td>
                <td className="py-3"><span className="text-red-600 dark:text-red-400 font-medium">🔴 极高</span></td>
                <td className="py-3">
                  <button className="text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                    传承计划
                  </button>
                </td>
              </tr>
              <tr>
                <td className="py-3 text-slate-900 dark:text-slate-100">K8s网络策略</td>
                <td className="py-3 text-slate-600 dark:text-slate-400">@chen_qi</td>
                <td className="py-3 text-slate-600 dark:text-slate-400">容器平台</td>
                <td className="py-3"><span className="text-yellow-600 dark:text-yellow-400 font-medium">🟡 高</span></td>
                <td className="py-3">
                  <button className="text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                    传承计划
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
