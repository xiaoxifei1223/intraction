import { useState } from 'react';
import { Search, Paperclip, Zap, Clock, User, TrendingUp, Copy, Play } from 'lucide-react';
import { SkillCard } from '@/components/shared/SkillCard';
import { UserAvatar } from '@/components/shared/UserAvatar';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { EmptyState } from '@/components/shared/EmptyState';
import { useAuthStore } from '@/stores/authStore';
import type { DiagnoseResponse } from '@/types/api';

const quickTags = ['ORA-04031', 'K8s Pod Evicted', 'API 5xx', '连接池耗尽', '响应超时'];

export function DiagnosePage() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<DiagnoseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const { currentUser } = useAuthStore();

  const handleDiagnose = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/diagnose?query=${encodeURIComponent(query)}`);
      const json = await res.json();
      if (json.success) setResult(json.data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">智能诊断中心</h1>
      </div>

      {/* Input Area */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <div className="relative">
          <Search className="absolute left-4 top-4 text-slate-400" size={20} />
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="描述你的问题、粘贴错误日志或输入告警 ID..."
            className="w-full pl-12 pr-4 py-3 bg-slate-50 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 resize-none h-24 focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        </div>
        <div className="flex items-center justify-between mt-3">
          <button className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300">
            <Paperclip size={16} /> 添加上下文
          </button>
          <button
            onClick={handleDiagnose}
            disabled={!query.trim() || loading}
            className="flex items-center gap-2 px-5 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Zap size={16} /> 开始诊断 →
          </button>
        </div>
      </div>

      {/* Quick Tags */}
      <div className="flex flex-wrap items-center gap-2 mb-6">
        <span className="text-sm text-slate-500 dark:text-slate-400">快速触发:</span>
        {quickTags.map((tag) => (
          <button
            key={tag}
            onClick={() => setQuery(tag)}
            className="px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            {tag}
          </button>
        ))}
      </div>

      {/* Results */}
      {loading && <LoadingOverlay />}

      {result && !loading && (
        <div className="space-y-6 animate-page-enter">
          {result.queryInterpretation && (
            <div className="text-sm text-slate-500 dark:text-slate-400">
              系统理解: <span className="text-slate-700 dark:text-slate-300">{result.queryInterpretation}</span>
            </div>
          )}

          <div>
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
              推荐解决路径（基于上下文）
            </h3>
            <div className="space-y-4">
              {result.matchedSkills?.map((matched, idx) => (
                <div
                  key={idx}
                  className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 hover:shadow-md transition-all"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2">
                        {idx === 0 && <span className="text-yellow-500">★</span>}
                        <h4 className="font-semibold text-slate-900 dark:text-slate-100">{matched.skill.name}</h4>
                      </div>
                      <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{matched.reason}</p>
                    </div>
                    <span className="text-xs px-2 py-1 rounded-full bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400 font-medium">
                      匹配度 {(matched.matchScore * 100).toFixed(0)}%
                    </span>
                  </div>

                  <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400 mb-3">
                    <span className="flex items-center gap-1"><TrendingUp size={14} /> 成功率 {(matched.skill.successRate * 100).toFixed(0)}%</span>
                    <span className="flex items-center gap-1"><Clock size={14} /> 平均 {matched.skill.avgResolutionTime}min</span>
                    <span className="flex items-center gap-1"><Zap size={14} /> 预估 {matched.estimatedTime}min</span>
                  </div>

                  <div className="text-sm text-slate-500 dark:text-slate-400 mb-3">
                    步骤预览: {matched.skill.content.diagnosisSteps.slice(0, 3).map((s) => s.title).join(' → ')}
                  </div>

                  <div className="flex items-center gap-2">
                    <button className="px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                      查看详细步骤
                    </button>
                    <button
                      onClick={() => {
                        const cmd = matched.skill.content.diagnosisSteps[0]?.command;
                        if (cmd) navigator.clipboard.writeText(cmd);
                      }}
                      className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                    >
                      <Copy size={14} /> 一键复制命令
                    </button>
                    <button className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                      <Play size={14} /> 在演练场模拟
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {result.suggestedExperts && result.suggestedExperts.length > 0 && (
            <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-5">
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">💡 团队知识</h3>
              <div className="flex items-center gap-3">
                {result.suggestedExperts.map((expert, idx) => (
                  <div key={idx} className="flex items-center gap-3 bg-white dark:bg-slate-900 rounded-lg p-3 border border-slate-200 dark:border-slate-800">
                    <UserAvatar user={expert.user} size="sm" />
                    <div>
                      <div className="text-sm font-medium">{expert.user.name}</div>
                      <div className="text-xs text-slate-500 dark:text-slate-400">
                        处理过 {expert.relevantIncidents} 次类似问题 · 平均 MTTR {expert.avgMTTR}min
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {!result && !loading && (
        <EmptyState
          title="开始诊断"
          description="输入问题描述或错误日志，系统将为你推荐最佳解决路径"
        />
      )}
    </div>
  );
}
