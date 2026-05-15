import { useSearchParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Search } from 'lucide-react';
import { SkillCard } from '@/components/shared/SkillCard';
import { IncidentBadge } from '@/components/shared/IncidentBadge';
import { UserAvatar } from '@/components/shared/UserAvatar';
import { EmptyState } from '@/components/shared/EmptyState';
import type { Skill } from '@/types/skill';
import type { Incident } from '@/types/incident';
import type { User } from '@/types/user';

export function SearchResultPage() {
  const [searchParams] = useSearchParams();
  const q = searchParams.get('q') || '';
  const [results, setResults] = useState<{ skills: Skill[]; incidents: Incident[]; users: User[] } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (q) {
      setLoading(true);
      fetch(`/api/search?q=${encodeURIComponent(q)}`)
        .then((res) => res.json())
        .then((json) => {
          if (json.success) setResults(json.data);
        })
        .finally(() => setLoading(false));
    }
  }, [q]);

  if (!q) {
    return (
      <EmptyState
        title="请输入搜索内容"
        description="在上方搜索框输入关键词，或按 Ctrl+K 打开全局搜索"
      />
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-primary-300 border-t-primary-600 rounded-full animate-spin" />
      </div>
    );
  }

  if (!results || (results.skills.length === 0 && results.incidents.length === 0 && results.users.length === 0)) {
    return <EmptyState title="未找到结果" description={`没有找到与 "${q}" 相关的内容`} />;
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
        <Search size={20} /> "{q}" 的搜索结果
      </h2>

      {results.skills.length > 0 && (
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">Skill ({results.skills.length})</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {results.skills.map((skill) => (
              <SkillCard key={skill.id} skill={skill} />
            ))}
          </div>
        </div>
      )}

      {results.incidents.length > 0 && (
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">Incident ({results.incidents.length})</h3>
          <div className="space-y-3">
            {results.incidents.map((incident) => (
              <a
                key={incident.id}
                href={`/incident/${incident.id}`}
                className="block bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-4 hover:shadow-md transition-all"
              >
                <div className="flex items-center gap-3 mb-2">
                  <IncidentBadge priority={incident.priority} />
                  <span className="font-medium text-slate-900 dark:text-slate-100">{incident.id}</span>
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-400">{incident.title}</p>
              </a>
            ))}
          </div>
        </div>
      )}

      {results.users.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">用户 ({results.users.length})</h3>
          <div className="grid md:grid-cols-3 gap-4">
            {results.users.map((user) => (
              <a
                key={user.id}
                href={`/profile?userId=${user.id}`}
                className="flex items-center gap-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-4 hover:shadow-md transition-all"
              >
                <UserAvatar user={user} size="md" />
                <div>
                  <div className="font-medium text-slate-900 dark:text-slate-100">{user.name}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">{user.handle}</div>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
