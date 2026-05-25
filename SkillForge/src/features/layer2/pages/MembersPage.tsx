import { useEffect, useState } from 'react';
import { Users, TrendingUp, FileText, Network } from 'lucide-react';
import { UserAvatar } from '@/components/shared/UserAvatar';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { useLayer2Store } from '@/stores/layer2Store';
import type { User } from '@/types/user';

export function MembersPage() {
  const { members, fetchMembers, loading } = useLayer2Store();
  const [selectedMember, setSelectedMember] = useState<User | null>(null);

  useEffect(() => {
    fetchMembers();
  }, []);

  if (loading) return <LoadingOverlay />;

  const member = selectedMember || members[0];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">人员管理与成长追踪</h1>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {/* Member List */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 max-h-[600px] overflow-y-auto">
          <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">团队成员</h2>
          <div className="space-y-3">
            {members.map((m: User) => (
              <button
                key={m.id}
                onClick={() => setSelectedMember(m)}
                className={`w-full text-left p-3 rounded-lg border transition-all ${
                  member?.id === m.id
                    ? 'border-primary-300 dark:border-primary-700 bg-primary-50 dark:bg-primary-900/10'
                    : 'border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800'
                }`}
              >
                <div className="flex items-center gap-3">
                  <UserAvatar user={m} size="sm" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">{m.name}</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">{m.handle}</div>
                  </div>
                </div>
                <div className="mt-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 dark:text-slate-500 mb-1">
                    <span>整体进度</span>
                    <span>{Math.round(m.skillsMastery.reduce((s, i) => s + i.level, 0) / m.skillsMastery.length)}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary-500 rounded-full"
                      style={{ width: `${Math.round(m.skillsMastery.reduce((s, i) => s + i.level, 0) / m.skillsMastery.length)}%` }}
                    />
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Detail Panel */}
        <div className="md:col-span-2 space-y-6">
          {member && (
            <>
              <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
                <div className="flex items-start gap-4 mb-6">
                  <UserAvatar user={member} size="lg" />
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">{member.name}</h2>
                    <p className="text-slate-500 dark:text-slate-400">{member.title} · {member.handle}</p>
                    <div className="flex items-center gap-4 mt-2 text-sm text-slate-500 dark:text-slate-400">
                      <span>入职: {member.joinDate}</span>
                      <span>Incident: {member.metrics.totalIncidents}</span>
                      <span>MTTR: {member.metrics.avgMTTR}min</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">能力进度</h3>
                  {member.skillsMastery.map((skill: { domain: string; level: number }) => (
                    <div key={skill.domain}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-slate-600 dark:text-slate-400">{skill.domain}</span>
                        <span className="text-xs text-slate-500 dark:text-slate-400">{skill.level}%</span>
                      </div>
                      <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            skill.level >= 80 ? 'bg-emerald-500' : skill.level >= 50 ? 'bg-blue-500' : 'bg-yellow-500'
                          }`}
                          style={{ width: `${skill.level}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <button className="flex items-center justify-center gap-2 p-4 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 hover:shadow-md transition-all">
                  <FileText size={18} className="text-primary-500" />
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">生成 1:1 准备材料</span>
                </button>
                <button className="flex items-center justify-center gap-2 p-4 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 hover:shadow-md transition-all">
                  <Network size={18} className="text-emerald-500" />
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">查看传承关系</span>
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
