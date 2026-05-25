import { useNavigate, useLocation } from 'react-router-dom';
import {
  Home,
  Zap,
  Wrench,
  Terminal,
  Map,
  Swords,
  User,
  LayoutDashboard,
  Target,
  TrendingDown,
  Users,
  CalendarDays,
  FileText,
  BarChart3,
  ShieldCheck,
  GitMerge,
  Compass,
  Award,
  Presentation,
  HelpCircle,
  Settings,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAuthStore } from '@/stores/authStore';
import { useUIStore } from '@/stores/uiStore';

interface NavItem {
  label: string;
  path: string;
  icon: React.ElementType;
  roles: string[];
}

const navItems: NavItem[] = [
  { label: '工作台', path: '/', icon: Home, roles: ['all'] },
  { label: '智能诊断', path: '/diagnose', icon: Zap, roles: ['engineer', 'lead', 'executive'] },
  { label: '我的 Skill 工坊', path: '/my-skills', icon: Wrench, roles: ['engineer', 'lead'] },
  { label: '命令片段库', path: '/snippets', icon: Terminal, roles: ['engineer', 'lead'] },
  { label: '学习地图', path: '/learning', icon: Map, roles: ['engineer', 'lead'] },
  { label: '实战演练场', path: '/arena', icon: Swords, roles: ['engineer', 'lead'] },
  { label: '个人档案', path: '/profile', icon: User, roles: ['engineer', 'lead', 'executive'] },
  { label: '团队概览', path: '/team', icon: LayoutDashboard, roles: ['lead', 'executive'] },
  { label: '技能雷达', path: '/team/radar', icon: Target, roles: ['lead', 'executive'] },
  { label: 'MTTR 分析', path: '/team/mttr', icon: TrendingDown, roles: ['lead', 'executive'] },
  { label: '人员成长', path: '/team/members', icon: Users, roles: ['lead', 'executive'] },
  { label: '排班优化', path: '/team/schedule', icon: CalendarDays, roles: ['lead'] },
  { label: '汇报材料', path: '/team/reports', icon: FileText, roles: ['lead', 'executive'] },
  { label: '组织仪表板', path: '/executive', icon: BarChart3, roles: ['executive'] },
  { label: 'AI 治理', path: '/executive/governance', icon: ShieldCheck, roles: ['executive'] },
  { label: '战略对齐', path: '/executive/strategy', icon: GitMerge, roles: ['executive'] },
  { label: '能力规划', path: '/executive/planner', icon: Compass, roles: ['executive'] },
  { label: '成熟度评估', path: '/executive/maturity', icon: Award, roles: ['executive'] },
  { label: '集团汇报', path: '/executive/board-report', icon: Presentation, roles: ['executive'] },
];

export function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { currentRole } = useAuthStore();
  const { sidebarCollapsed, toggleSidebar } = useUIStore();

  const visibleItems = navItems.filter(
    (item) => item.roles.includes('all') || item.roles.includes(currentRole)
  );

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 h-full bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col transition-all duration-300 z-40',
        sidebarCollapsed ? 'w-16' : 'w-60'
      )}
    >
      <div className="h-16 flex items-center justify-between px-4 border-b border-slate-200 dark:border-slate-800">
        {!sidebarCollapsed && (
          <span className="font-bold text-lg text-primary-600 dark:text-primary-400 truncate">
            SkillForge
          </span>
        )}
        <button
          onClick={toggleSidebar}
          className="p-1 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto py-4 space-y-1 px-2">
        {visibleItems.map((item) => {
          const active = location.pathname === item.path;
          return (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={cn(
                'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                active
                  ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
              )}
              title={sidebarCollapsed ? item.label : undefined}
            >
              <item.icon size={20} />
              {!sidebarCollapsed && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      <div className="border-t border-slate-200 dark:border-slate-800 py-4 px-2 space-y-1">
        <button
          onClick={() => navigate('/simulator/teams')}
          className={cn(
            'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800',
            location.pathname === '/simulator/teams' && 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
          )}
          title={sidebarCollapsed ? 'Teams 模拟器' : undefined}
        >
          <Presentation size={20} />
          {!sidebarCollapsed && <span>Teams 模拟器</span>}
        </button>
        <button
          onClick={() => navigate('/simulator/vscode')}
          className={cn(
            'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800',
            location.pathname === '/simulator/vscode' && 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
          )}
          title={sidebarCollapsed ? 'VS Code 模拟器' : undefined}
        >
          <Terminal size={20} />
          {!sidebarCollapsed && <span>VS Code 模拟器</span>}
        </button>
        <button
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          title={sidebarCollapsed ? '帮助' : undefined}
        >
          <HelpCircle size={20} />
          {!sidebarCollapsed && <span>帮助</span>}
        </button>
        <button
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          title={sidebarCollapsed ? '设置' : undefined}
        >
          <Settings size={20} />
          {!sidebarCollapsed && <span>设置</span>}
        </button>
      </div>
    </aside>
  );
}
