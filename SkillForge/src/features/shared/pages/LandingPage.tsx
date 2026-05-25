import { useNavigate } from 'react-router-dom';
import { Zap, Users, BarChart3 } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';

export function LandingPage() {
  const navigate = useNavigate();
  const { switchRole } = useAuthStore();

  const enterAs = (role: 'engineer' | 'lead' | 'executive') => {
    switchRole(role);
    if (role === 'engineer') navigate('/diagnose');
    else if (role === 'lead') navigate('/team');
    else navigate('/executive');
  };

  return (
    <div className="max-w-4xl mx-auto py-12">
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-4">
          SkillForge
        </h1>
        <p className="text-lg text-slate-500 dark:text-slate-400">
          技能锻造平台 — 将团队经验转化为可复用的数字资产
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <button
          onClick={() => enterAs('engineer')}
          className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-8 text-center hover:shadow-lg hover:-translate-y-1 transition-all group"
        >
          <div className="w-16 h-16 rounded-full bg-primary-50 dark:bg-primary-900/20 flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
            <Zap size={28} className="text-primary-600 dark:text-primary-400" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">工程师</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            智能诊断、Skill 管理、学习地图、实战演练
          </p>
        </button>

        <button
          onClick={() => enterAs('lead')}
          className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-8 text-center hover:shadow-lg hover:-translate-y-1 transition-all group"
        >
          <div className="w-16 h-16 rounded-full bg-indigo-50 dark:bg-indigo-900/20 flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
            <Users size={28} className="text-indigo-600 dark:text-indigo-400" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">Team Leader</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            团队概览、技能雷达、MTTR 分析、人员成长
          </p>
        </button>

        <button
          onClick={() => enterAs('executive')}
          className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-8 text-center hover:shadow-lg hover:-translate-y-1 transition-all group"
        >
          <div className="w-16 h-16 rounded-full bg-emerald-50 dark:bg-emerald-900/20 flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
            <BarChart3 size={28} className="text-emerald-600 dark:text-emerald-400" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">高管</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            组织仪表板、AI 治理、战略对齐、集团汇报
          </p>
        </button>
      </div>

      <div className="mt-12 text-center">
        <p className="text-sm text-slate-400 dark:text-slate-500">
          纯前端演示原型 · Mock 数据驱动 · 角色切换即时生效
        </p>
      </div>
    </div>
  );
}
