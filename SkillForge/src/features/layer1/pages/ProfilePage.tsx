import { useAuthStore } from '@/stores/authStore';
import { UserAvatar } from '@/components/shared/UserAvatar';
import { Award, Wrench, Clock, TrendingUp, Star, Calendar } from 'lucide-react';

export function ProfilePage() {
  const { currentUser } = useAuthStore();

  if (!currentUser) {
    return <div className="text-center py-20 text-slate-500 dark:text-slate-400">请先登录</div>;
  }

  const achievements = [
    { title: 'Skill 被复用 10 次', date: '2026-04-15', icon: Star },
    { title: '连续 30 天使用诊断', date: '2026-03-20', icon: Calendar },
    { title: 'Oracle 性能诊断专家认证', date: '2026-02-10', icon: Award },
    { title: '创建第 10 个 Skill', date: '2026-01-05', icon: Wrench },
  ];

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header Card */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <div className="flex items-start gap-6">
          <UserAvatar user={currentUser} size="lg" />
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">{currentUser.name}</h1>
            <p className="text-slate-500 dark:text-slate-400">{currentUser.handle} · {currentUser.title}</p>
            <div className="flex items-center gap-4 mt-3 text-sm text-slate-500 dark:text-slate-400">
              <span>团队: DB-SRE</span>
              <span>入职: {currentUser.joinDate}</span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-emerald-500" /> 在线
              </span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">{currentUser.metrics.totalIncidents}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">处理 Incident</div>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5">
          <div className="flex items-center gap-2 mb-2">
            <Wrench size={18} className="text-primary-500" />
            <span className="text-sm text-slate-500 dark:text-slate-400">创建 Skill</span>
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">{currentUser.metrics.skillsCreated}</div>
        </div>
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp size={18} className="text-emerald-500" />
            <span className="text-sm text-slate-500 dark:text-slate-400">被复用次数</span>
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">{currentUser.metrics.skillsAdoptedByOthers}</div>
        </div>
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5">
          <div className="flex items-center gap-2 mb-2">
            <Clock size={18} className="text-blue-500" />
            <span className="text-sm text-slate-500 dark:text-slate-400">平均 MTTR</span>
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">{currentUser.metrics.avgMTTR}min</div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Radar Chart Placeholder */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">能力概览</h2>
          <div className="space-y-3">
            {currentUser.skillsMastery.slice(0, 6).map((skill) => (
              <div key={skill.domain}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-slate-700 dark:text-slate-300">{skill.domain}</span>
                  <span className="text-sm text-slate-500 dark:text-slate-400">{skill.level}%</span>
                </div>
                <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-500 rounded-full transition-all duration-500"
                    style={{ width: `${skill.level}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Achievement Timeline */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">成就时间线</h2>
          <div className="space-y-4">
            {achievements.map((achievement, i) => (
              <div key={i} className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-yellow-50 dark:bg-yellow-900/20 flex items-center justify-center shrink-0">
                  <achievement.icon size={16} className="text-yellow-600 dark:text-yellow-400" />
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-800 dark:text-slate-200">{achievement.title}</div>
                  <div className="text-xs text-slate-400 dark:text-slate-500">{achievement.date}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
