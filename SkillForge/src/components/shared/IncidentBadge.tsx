import { cn } from '@/lib/utils';
import type { IncidentPriority } from '@/types/incident';

interface IncidentBadgeProps {
  priority: IncidentPriority;
}

const priorityConfig: Record<IncidentPriority, { bg: string; text: string }> = {
  P1: { bg: 'bg-red-600', text: 'text-white' },
  P2: { bg: 'bg-orange-500', text: 'text-white' },
  P3: { bg: 'bg-yellow-500', text: 'text-black' },
  P4: { bg: 'bg-blue-400', text: 'text-white' },
};

export function IncidentBadge({ priority }: IncidentBadgeProps) {
  const config = priorityConfig[priority];
  return (
    <span className={cn('inline-flex items-center px-2 py-0.5 rounded text-xs font-bold', config.bg, config.text)}>
      {priority}
    </span>
  );
}
