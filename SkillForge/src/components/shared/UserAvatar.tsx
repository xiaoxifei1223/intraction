import { cn } from '@/lib/utils';
import type { User } from '@/types/user';

interface UserAvatarProps {
  user: User;
  size?: 'sm' | 'md' | 'lg';
  showStatus?: boolean;
}

const sizeClasses = {
  sm: 'w-6 h-6',
  md: 'w-8 h-8',
  lg: 'w-12 h-12',
};

const statusColors = {
  online: 'bg-emerald-500',
  busy: 'bg-red-500',
  offline: 'bg-slate-400',
  oncall: 'bg-orange-500',
};

export function UserAvatar({ user, size = 'md', showStatus = true }: UserAvatarProps) {
  return (
    <div className="relative inline-block">
      <img
        src={user.avatar}
        alt={user.name}
        className={cn('rounded-full bg-slate-200 dark:bg-slate-700 object-cover', sizeClasses[size])}
      />
      {showStatus && (
        <span
          className={cn(
            'absolute bottom-0 right-0 rounded-full border-2 border-white dark:border-slate-900',
            size === 'sm' ? 'w-2 h-2' : size === 'md' ? 'w-2.5 h-2.5' : 'w-3.5 h-3.5',
            statusColors[user.status]
          )}
        />
      )}
    </div>
  );
}
