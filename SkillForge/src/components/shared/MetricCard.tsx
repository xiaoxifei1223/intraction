import type { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  trend?: 'up' | 'down' | 'flat';
  trendValue?: string;
  status?: 'good' | 'warning' | 'danger' | 'neutral';
  icon?: LucideIcon;
  onClick?: () => void;
  loading?: boolean;
}

export function MetricCard({
  title,
  value,
  unit,
  trend,
  trendValue,
  status = 'neutral',
  icon: Icon,
  onClick,
  loading,
}: MetricCardProps) {
  const statusColors = {
    good: 'text-emerald-600 dark:text-emerald-400',
    warning: 'text-yellow-600 dark:text-yellow-400',
    danger: 'text-red-600 dark:text-red-400',
    neutral: 'text-slate-600 dark:text-slate-400',
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 animate-pulse">
        <div className="h-4 w-20 bg-slate-200 dark:bg-slate-800 rounded mb-3" />
        <div className="h-8 w-24 bg-slate-200 dark:bg-slate-800 rounded" />
      </div>
    );
  }

  return (
    <div
      onClick={onClick}
      className={cn(
        'bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5',
        onClick && 'cursor-pointer'
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-sm text-slate-500 dark:text-slate-400">{title}</span>
        {Icon && <Icon size={18} className="text-slate-400 dark:text-slate-500" />}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
          {value}
        </span>
        {unit && <span className="text-sm text-slate-500 dark:text-slate-400">{unit}</span>}
      </div>
      {trend && (
        <div className={cn('flex items-center gap-1 mt-2 text-sm', statusColors[status])}>
          {trend === 'up' && <TrendingUp size={14} />}
          {trend === 'down' && <TrendingDown size={14} />}
          {trend === 'flat' && <Minus size={14} />}
          <span>{trendValue}</span>
        </div>
      )}
    </div>
  );
}
