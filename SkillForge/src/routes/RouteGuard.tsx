import { Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';

interface RouteGuardProps {
  allowedRoles: string[];
  children: React.ReactNode;
}

export function RouteGuard({ allowedRoles, children }: RouteGuardProps) {
  const { currentRole } = useAuthStore();
  const isAllowed = allowedRoles.includes('all') || allowedRoles.includes(currentRole);

  if (!isAllowed) {
    return <Navigate to="/unauthorized" replace />;
  }

  return <>{children}</>;
}
