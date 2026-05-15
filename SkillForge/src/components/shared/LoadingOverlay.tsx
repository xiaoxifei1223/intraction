import { cn } from '@/lib/utils';

interface LoadingOverlayProps {
  className?: string;
}

export function LoadingOverlay({ className }: LoadingOverlayProps) {
  return (
    <div className={cn('flex items-center justify-center py-8', className)}>
      <div className="w-8 h-8 border-2 border-primary-300 border-t-primary-600 rounded-full animate-spin" />
    </div>
  );
}
