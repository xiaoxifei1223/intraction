import { useEffect, useState } from 'react';
import { Plus, Filter, Search, Wrench, CheckCircle, AlertTriangle, Archive, TrendingUp, TrendingDown } from 'lucide-react';
import { SkillCard } from '@/components/shared/SkillCard';
import { MetricCard } from '@/components/shared/MetricCard';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import type { Skill } from '@/types/skill';

export function MySkillsPage() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');

  useEffect(() => {
    setLoading(true);
    fetch('/api/skills')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setSkills(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  const filtered = skills.filter((s) => s.name.includes(search));

  const totalCreated = skills.filter((s) => s.authorId === 'user_wang_wu').length;
  const activeSkills = skills.filter((s) => s.healthStatus === 'healthy').length;
  const drafts = skills.filter((s) => s.governance.approvalStatus === 'draft').length;
  const totalReuse = skills.reduce((sum, s) => sum + s.useCount, 0);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">我的 Skill 工坊</h1>
        <button className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
          <Plus size={16} /> 新建 Skill
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <MetricCard title="总创建" value={totalCreated} icon={Wrench} status="good" />
        <MetricCard title="在用" value={activeSkills} icon={CheckCircle} status="good" />
        <MetricCard
          title="草稿待确认"
          value={drafts}
          icon={AlertTriangle}
          status={drafts > 0 ? 'warning' : 'good'}
          trend={drafts > 0 ? 'up' : undefined}
          trendValue={drafts > 0 ? '有未确认项' : undefined}
        />
        <MetricCard title="被复用次数" value={totalReuse} icon={Archive} status="good" />
      </div>

      {/* Draft Alert */}
      {drafts > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800 rounded-xl p-4 mb-6">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-medium text-yellow-800 dark:text-yellow-300 mb-1">📝 自动生成草稿（待确认）</h3>
              <p className="text-sm text-yellow-700 dark:text-yellow-400">
                "Oracle AWR自动分析 + 索引在线重建" (来自 INC-2024-0789)
              </p>
              <p className="text-xs text-yellow-600 dark:text-yellow-500 mt-1">
                系统检测到你在本次 incident 中使用了新的解决模式
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button className="px-3 py-1.5 rounded-md bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 text-sm hover:bg-yellow-200 dark:hover:bg-yellow-900/50 transition-colors">
                一键保存
              </button>
              <button className="px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-sm hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                忽略
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex items-center gap-3 mb-6">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-2.5 text-slate-400" size={16} />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索我的 Skill..."
            className="w-full pl-10 pr-4 py-2 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        </div>
        <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
          <Filter size={16} /> 筛选
        </button>
      </div>

      {/* Skill Grid */}
      {loading && <LoadingOverlay />}

      {!loading && (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.slice(0, 24).map((skill) => (
            <SkillCard key={skill.id} skill={skill} />
          ))}
        </div>
      )}

      {!loading && filtered.length === 0 && (
        <div className="text-center py-12 text-slate-500 dark:text-slate-400">
          未找到匹配的 Skill
        </div>
      )}
    </div>
  );
}
