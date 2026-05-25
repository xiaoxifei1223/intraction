import { useEffect, useState } from 'react';
import { Swords, Clock, BarChart3, Play, Trophy } from 'lucide-react';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { EmptyState } from '@/components/shared/EmptyState';

interface Scenario {
  id: string;
  title: string;
  difficulty: string;
  estimatedTime: number;
  description: string;
}

export function ArenaPage() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeScenario, setActiveScenario] = useState<Scenario | null>(null);

  useEffect(() => {
    fetch('/api/arena/scenarios')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setScenarios(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  const getDifficultyColor = (d: string) => {
    switch (d) {
      case 'beginner': return 'bg-emerald-100 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-400';
      case 'intermediate': return 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400';
      case 'advanced': return 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400';
      default: return 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-400';
    }
  };

  if (loading) return <LoadingOverlay />;

  if (activeScenario) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">{activeScenario.title}</h1>
            <p className="text-slate-500 dark:text-slate-400">{activeScenario.description}</p>
          </div>
          <button
            onClick={() => setActiveScenario(null)}
            className="px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            返回场景列表
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-slate-900 rounded-xl p-4 font-mono text-sm text-green-400 h-96 overflow-y-auto">
            <div className="text-slate-500 mb-2">$ sqlplus / as sysdba</div>
            <div className="text-red-400 mb-2">ORA-04031: unable to allocate 3896 bytes of shared memory</div>
            <div className="text-slate-500 mb-2">$ _</div>
            <div className="text-xs text-slate-600 mt-4">[模拟终端 - 输入命令进行演练]</div>
          </div>

          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 mb-4">Skill 引导面板</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3 p-3 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
                <div className="w-6 h-6 rounded-full bg-primary-500 text-white flex items-center justify-center text-xs font-bold">1</div>
                <span className="text-sm text-slate-700 dark:text-slate-300">生成 AWR 报告</span>
              </div>
              <div className="flex items-center gap-3 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg opacity-50">
                <div className="w-6 h-6 rounded-full bg-slate-400 text-white flex items-center justify-center text-xs font-bold">2</div>
                <span className="text-sm text-slate-600 dark:text-slate-400">定位 Top SQL</span>
              </div>
            </div>
            <div className="mt-6 flex items-center justify-between text-sm text-slate-500 dark:text-slate-400">
              <span className="flex items-center gap-1"><Clock size={14} /> 已用时 5min</span>
              <span>步骤 1/5</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">实战演练场</h1>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        {scenarios.map((scenario) => (
          <button
            key={scenario.id}
            onClick={() => setActiveScenario(scenario)}
            className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 text-left hover:shadow-lg hover:-translate-y-1 transition-all"
          >
            <div className="flex items-center justify-between mb-3">
              <Swords size={24} className="text-primary-500" />
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getDifficultyColor(scenario.difficulty)}`}>
                {scenario.difficulty === 'beginner' ? '初级' : scenario.difficulty === 'intermediate' ? '中级' : '高级'}
              </span>
            </div>
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 mb-2">{scenario.title}</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">{scenario.description}</p>
            <div className="flex items-center gap-3 text-xs text-slate-400 dark:text-slate-500">
              <span className="flex items-center gap-1"><Clock size={12} /> {scenario.estimatedTime}min</span>
              <span className="flex items-center gap-1"><BarChart3 size={12} /> 多步骤</span>
            </div>
            <div className="mt-4 flex items-center gap-2 text-primary-600 dark:text-primary-400 text-sm font-medium">
              <Play size={14} /> 开始演练
            </div>
          </button>
        ))}
      </div>

      <div className="mt-8 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
          <Trophy size={18} className="text-yellow-500" /> 最近战绩
        </h2>
        <div className="space-y-3">
          {[
            { scenario: 'Oracle 慢查询诊断', score: 92, date: '2026-05-10' },
            { scenario: 'K8s Pod 驱逐排查', score: 78, date: '2026-05-08' },
          ].map((record, i) => (
            <div key={i} className="flex items-center justify-between py-2 border-b border-slate-100 dark:border-slate-800 last:border-0">
              <div>
                <div className="font-medium text-slate-800 dark:text-slate-200">{record.scenario}</div>
                <div className="text-xs text-slate-400 dark:text-slate-500">{record.date}</div>
              </div>
              <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{record.score}分</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
