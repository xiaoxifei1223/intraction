import { useEffect } from 'react';
import { CalendarDays, AlertTriangle, CheckCircle, Lightbulb } from 'lucide-react';
import { UserAvatar } from '@/components/shared/UserAvatar';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { useLayer2Store } from '@/stores/layer2Store';
import { seedUsers } from '@/mocks/seeds/initialData';

const weekDays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];

export function SchedulingPage() {
  const { schedule, fetchSchedule, loading } = useLayer2Store();

  useEffect(() => {
    fetchSchedule();
  }, []);

  if (loading || !schedule) return <LoadingOverlay />;

  const getUser = (userId: string) => seedUsers.find((u) => u.id === userId);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">排班与技能覆盖</h1>
      </div>

      {/* Schedule Grid */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left">
                <th className="pb-4 text-sm text-slate-500 dark:text-slate-400 font-medium">班次</th>
                {weekDays.map((day) => (
                  <th key={day} className="pb-4 text-sm text-slate-500 dark:text-slate-400 font-medium text-center">
                    {day}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-3 text-sm font-medium text-slate-700 dark:text-slate-300">日班</td>
                {schedule.map((day: any[], idx: number) => (
                  <td key={idx} className="py-3 text-center">
                    {day
                      .filter((s) => s.shift === 'day')
                      .map((s) => {
                        const user = getUser(s.userId);
                        return user ? (
                          <div key={s.userId} className="flex flex-col items-center gap-1">
                            <UserAvatar user={user} size="sm" showStatus={false} />
                            <span className="text-xs text-slate-500 dark:text-slate-400">{user.name}</span>
                          </div>
                        ) : null;
                      })}
                  </td>
                ))}
              </tr>
              <tr className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-3 text-sm font-medium text-slate-700 dark:text-slate-300">夜班</td>
                {schedule.map((day: any[], idx: number) => (
                  <td key={idx} className="py-3 text-center">
                    {day
                      .filter((s) => s.shift === 'night')
                      .map((s) => {
                        const user = getUser(s.userId);
                        return user ? (
                          <div key={s.userId} className="flex flex-col items-center gap-1">
                            <UserAvatar user={user} size="sm" showStatus={false} />
                            <span className="text-xs text-slate-500 dark:text-slate-400">{user.name}</span>
                          </div>
                        ) : null;
                      })}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Risk Analysis */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-800 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle size={18} className="text-red-500" />
            <h2 className="text-lg font-semibold text-red-800 dark:text-red-300">风险分析</h2>
          </div>
          <div className="space-y-3">
            <div className="flex items-start gap-2 text-sm text-red-700 dark:text-red-400">
              <span>🔴</span>
              <span>周三夜班无 Oracle 专家覆盖 — 若发生数据库故障，MTTR 可能延长至 60min+</span>
            </div>
            <div className="flex items-start gap-2 text-sm text-yellow-700 dark:text-yellow-400">
              <span>🟡</span>
              <span>周六日班 @小李 K8s 熟练度不足，建议安排简单告警处理</span>
            </div>
          </div>
        </div>

        <div className="bg-emerald-50 dark:bg-emerald-900/10 border border-emerald-200 dark:border-emerald-800 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Lightbulb size={18} className="text-emerald-500" />
            <h2 className="text-lg font-semibold text-emerald-800 dark:text-emerald-300">优化建议</h2>
          </div>
          <div className="space-y-3">
            <div className="flex items-start gap-2 text-sm text-emerald-700 dark:text-emerald-400">
              <CheckCircle size={14} className="mt-0.5" />
              <span>将周三夜班 @小李 和周四夜班 @wang_wu 互换 → Oracle 覆盖风险从 🔴 降至 🟢</span>
            </div>
            <button className="mt-2 px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm hover:bg-emerald-700 transition-colors">
              应用建议
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
