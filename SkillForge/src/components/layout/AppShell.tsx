import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { cn } from '@/lib/utils';
import { useUIStore } from '@/stores/uiStore';

export function AppShell() {
  const { sidebarCollapsed } = useUIStore();

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 overflow-hidden">
      <Sidebar />
      <div className={cn('flex-1 flex flex-col transition-all duration-300', sidebarCollapsed ? 'ml-16' : 'ml-60')}>
        <Header />
        <main className="flex-1 overflow-auto p-6 animate-page-enter">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
