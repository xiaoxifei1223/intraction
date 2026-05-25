import { useLocation } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';
import { cn } from '@/lib/utils';

const pathMap: Record<string, string> = {
  '/': '工作台',
  '/diagnose': '智能诊断',
  '/my-skills': '我的 Skill 工坊',
  '/snippets': '命令片段库',
  '/learning': '学习地图',
  '/arena': '实战演练场',
  '/profile': '个人档案',
  '/team': '团队概览',
  '/team/radar': '技能雷达',
  '/team/mttr': 'MTTR 分析',
  '/team/members': '人员成长',
  '/team/schedule': '排班优化',
  '/team/reports': '汇报材料',
  '/executive': '组织仪表板',
  '/executive/governance': 'AI 治理',
  '/executive/strategy': '战略对齐',
  '/executive/planner': '能力规划',
  '/executive/maturity': '成熟度评估',
  '/executive/board-report': '集团汇报',
  '/simulator/teams': 'Teams 模拟器',
  '/simulator/vscode': 'VS Code 模拟器',
};

export function BreadcrumbNav() {
  const location = useLocation();
  const label = pathMap[location.pathname] || '页面';

  return (
    <nav className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
      <Home size={14} />
      <ChevronRight size={14} />
      <span className="font-medium text-slate-800 dark:text-slate-200">{label}</span>
    </nav>
  );
}
