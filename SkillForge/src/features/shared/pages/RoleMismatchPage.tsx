import { useNavigate } from 'react-router-dom';
import { ShieldAlert, ArrowLeft } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';

export function RoleMismatchPage() {
  const navigate = useNavigate();
  const { currentRole, switchRole } = useAuthStore();

  const suggestions = {
    engineer: { label: '工程师', path: '/diagnose' },
    lead: { label: 'Team Leader', path: '/team' },
    executive: { label: '高管', path: '/executive' },
  };

  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="w-20 h-20 rounded-full bg-yellow-50 dark:bg-yellow-900/20 flex items-center justify-center mb-6">
        <ShieldAlert size={40} className="text-yellow-600 dark:text-yellow-400" />
      </div>
      <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
        当前角色无法访问此页面
      </h2>
      <p className="text-slate-500 dark:text-slate-400 mb-6">
        你当前的角色是 <strong>{suggestions[currentRole]?.label}</strong>，该页面需要更高权限。
      </p>
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
        >
          <ArrowLeft size={16} /> 返回
        </button>
        {currentRole !== 'lead' && (
          <button
            onClick={() => {
              switchRole('lead');
              navigate('/team');
            }}
            className="px-4 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700 transition-colors"
          >
            切换为 Team Leader
          </button>
        )}
        {currentRole !== 'executive' && (
          <button
            onClick={() => {
              switchRole('executive');
              navigate('/executive');
            }}
            className="px-4 py-2 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition-colors"
          >
            切换为高管
          </button>
        )}
      </div>
    </div>
  );
}
