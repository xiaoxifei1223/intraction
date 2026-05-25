import { useNavigate } from 'react-router-dom';
import { Eye, Edit, Play, Copy, Bot } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Skill } from '@/types/skill';
import { SkillStatusBadge } from './SkillStatusBadge';

interface SkillCardProps {
  skill: Skill;
  variant?: 'compact' | 'default' | 'detailed';
  showHealth?: boolean;
  showActions?: boolean;
  onClick?: () => void;
}

export function SkillCard({
  skill,
  variant = 'default',
  showHealth = true,
  showActions = true,
  onClick,
}: SkillCardProps) {
  const navigate = useNavigate();

  const healthColors: Record<string, string> = {
    healthy: 'border-l-emerald-500',
    attention: 'border-l-yellow-500',
    outdated: 'border-l-red-500',
    archived: 'border-l-slate-400',
  };

  if (variant === 'compact') {
    return (
      <div
        onClick={() => onClick?.() || navigate(`/skill/${skill.id}`)}
        className="flex items-center gap-3 px-3 py-2 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 border-l-4 cursor-pointer hover:shadow-md transition-all"
        style={{ borderLeftColor: 'var(--health-color)' }}
      >
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium truncate text-slate-900 dark:text-slate-100">{skill.name}</div>
          <div className="text-xs text-slate-500 dark:text-slate-400">
            成功率 {(skill.successRate * 100).toFixed(0)}% · 使用 {skill.useCount} 次
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 border-l-4 transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5',
        healthColors[skill.healthStatus] || 'border-l-slate-400',
        onClick && 'cursor-pointer'
      )}
      onClick={() => onClick?.() || navigate(`/skill/${skill.id}`)}
    >
      <div className="p-5">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            {showHealth && <SkillStatusBadge status={skill.healthStatus} />}
            <h3 className="font-semibold text-slate-900 dark:text-slate-100">{skill.name}</h3>
          </div>
          {skill.governance.aiGenerated && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 text-xs font-medium">
              <Bot size={12} /> AI
            </span>
          )}
        </div>

        <div className="text-sm text-slate-500 dark:text-slate-400 mb-3 line-clamp-2">
          {skill.content.triggerConditions}
        </div>

        <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400 mb-3">
          <span>v{skill.version}</span>
          <span>成功率 {(skill.successRate * 100).toFixed(0)}%</span>
          <span>使用 {skill.useCount} 次</span>
          <span>平均 {skill.avgResolutionTime}min</span>
        </div>

        {variant === 'detailed' && (
          <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400 mb-3">
            <div>领域: {skill.classification.domain.join(', ')}</div>
            <div>场景: {skill.classification.scenario.join(', ')}</div>
            <div>难度: {skill.classification.difficulty}</div>
            <div>治理状态: {skill.governance.approvalStatus}</div>
            {skill.governance.aiConfidence && (
              <div>AI 置信度: {(skill.governance.aiConfidence * 100).toFixed(0)}%</div>
            )}
          </div>
        )}

        {showActions && (
          <div className="flex items-center gap-2 pt-3 border-t border-slate-100 dark:border-slate-800">
            <button
              onClick={(e) => {
                e.stopPropagation();
                navigate(`/skill/${skill.id}`);
              }}
              className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-sm hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
            >
              <Eye size={14} /> 查看
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
              }}
              className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-sm hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
            >
              <Edit size={14} /> 编辑
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                const cmd = skill.content.diagnosisSteps[0]?.command;
                if (cmd) navigator.clipboard.writeText(cmd);
              }}
              className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-sm hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
            >
              <Copy size={14} /> 复制命令
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
