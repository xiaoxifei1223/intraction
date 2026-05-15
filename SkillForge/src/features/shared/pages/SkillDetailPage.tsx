import { useParams } from 'react-router-dom';
import { useEffect } from 'react';
import { Bot, Clock, User, Tag, Shield, ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useSkillStore } from '@/stores/skillStore';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { EmptyState } from '@/components/shared/EmptyState';
import { SkillStatusBadge } from '@/components/shared/SkillStatusBadge';

export function SkillDetailPage() {
  const { skillId } = useParams<{ skillId: string }>();
  const navigate = useNavigate();
  const { skills, fetchSkillDetail, loading } = useSkillStore();
  const skill = skillId ? skills[skillId] : null;

  useEffect(() => {
    if (skillId && !skills[skillId]) {
      fetchSkillDetail(skillId);
    }
  }, [skillId]);

  if (loading) return <LoadingOverlay />;
  if (!skill) return <EmptyState title="Skill 不存在" description="找不到对应的 Skill 信息" />;

  return (
    <div className="max-w-4xl mx-auto">
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 mb-4"
      >
        <ArrowLeft size={16} /> 返回
      </button>

      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">{skill.name}</h1>
            <div className="flex items-center gap-3">
              <SkillStatusBadge status={skill.healthStatus} />
              <span className="text-sm text-slate-500 dark:text-slate-400">v{skill.version}</span>
              {skill.governance.aiGenerated && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 text-xs font-medium">
                  <Bot size={12} /> AI 生成
                </span>
              )}
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">{(skill.successRate * 100).toFixed(0)}%</div>
            <div className="text-sm text-slate-500 dark:text-slate-400">成功率</div>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">使用次数</div>
            <div className="text-lg font-semibold">{skill.useCount}</div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">平均解决时间</div>
            <div className="text-lg font-semibold">{skill.avgResolutionTime}min</div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">健康评分</div>
            <div className="text-lg font-semibold">{skill.healthScore}</div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">难度</div>
            <div className="text-lg font-semibold">{skill.classification.difficulty}</div>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
              <Tag size={16} /> 分类
            </h3>
            <div className="flex flex-wrap gap-2">
              {skill.classification.domain.map((d) => (
                <span key={d} className="px-2 py-1 rounded-md bg-slate-100 dark:bg-slate-800 text-xs text-slate-600 dark:text-slate-400">
                  {d}
                </span>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
              <Clock size={16} /> 触发条件
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
              {skill.content.triggerConditions}
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">诊断步骤</h3>
            <div className="space-y-3">
              {skill.content.diagnosisSteps.map((step) => (
                <div key={step.order} className="flex gap-4 bg-slate-50 dark:bg-slate-800 rounded-lg p-4">
                  <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center shrink-0">
                    <span className="text-sm font-semibold text-primary-700 dark:text-primary-300">{step.order}</span>
                  </div>
                  <div className="flex-1">
                    <h4 className="font-medium text-slate-900 dark:text-slate-100 mb-1">{step.title}</h4>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">{step.description}</p>
                    {step.command && (
                      <pre className="text-xs bg-slate-900 text-slate-100 rounded-md p-3 overflow-x-auto">
                        {step.command}
                      </pre>
                    )}
                    <div className="text-xs text-slate-400 dark:text-slate-500 mt-2">
                      预计 {step.estimatedTime}min · 验证: {step.verification}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {skill.content.rollbackPlan && (
            <div>
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
                <Shield size={16} /> 回滚方案
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
                {skill.content.rollbackPlan}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
