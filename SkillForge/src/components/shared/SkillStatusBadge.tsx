import { cn } from '@/lib/utils';
import type { SkillStatus } from '@/types/skill';

interface SkillStatusBadgeProps {
  status: SkillStatus;
  size?: 'sm' | 'md';
}

const statusConfig: Record<SkillStatus, { label: string; color: string }> = {
  healthy: { label: '健康', color: 'bg-emerald-500' },
  attention: { label: '需关注', color: 'bg-yellow-500' },
  outdated: { label: '已过时', color: 'bg-red-500' },
  archived: { label: '已归档', color: 'bg-slate-400' },
};

export function SkillStatusBadge({ status, size = 'sm' }: SkillStatusBadgeProps) {
  const config = statusConfig[status];
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn('rounded-full', config.color, size === 'sm' ? 'w-2 h-2' : 'w-2.5 h-2.5')} />
      <span className="text-xs text-slate-500 dark:text-slate-400">{config.label}</span>
    </span>
  );
}
