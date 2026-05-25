import { useNavigate, useLocation } from 'react-router-dom';
import { Search, Bell, Moon, Sun, Command } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useUIStore } from '@/stores/uiStore';
import { useAuthStore } from '@/stores/authStore';
import { RoleSwitcher } from './RoleSwitcher';
import { BreadcrumbNav } from './BreadcrumbNav';

export function Header() {
  const { toggleTheme, theme, openSearch } = useUIStore();
  const { currentUser } = useAuthStore();
  const navigate = useNavigate();

  return (
    <header className="h-16 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-6 shrink-0">
      <div className="flex items-center gap-4">
        <BreadcrumbNav />
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={openSearch}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 text-sm hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
        >
          <Search size={16} />
          <span className="hidden sm:inline">全局搜索</span>
          <kbd className="hidden md:inline-flex items-center gap-1 px-1.5 py-0.5 text-xs rounded bg-slate-200 dark:bg-slate-700">
            <Command size={12} />K
          </kbd>
        </button>

        <button
          onClick={toggleTheme}
          className="p-2 rounded-md text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>

        <button className="p-2 rounded-md text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors relative">
          <Bell size={18} />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full" />
        </button>

        <RoleSwitcher />

        {currentUser && (
          <button
            onClick={() => navigate('/profile')}
            className="flex items-center gap-2 ml-2"
          >
            <img
              src={currentUser.avatar}
              alt={currentUser.name}
              className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-700"
            />
            <span className="hidden md:inline text-sm font-medium">{currentUser.name}</span>
          </button>
        )}
      </div>
    </header>
  );
}
