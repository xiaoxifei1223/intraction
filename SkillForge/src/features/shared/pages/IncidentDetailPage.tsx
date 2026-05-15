import { useParams } from 'react-router-dom';
import { useEffect } from 'react';
import { ArrowLeft, Clock, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useIncidentStore } from '@/stores/incidentStore';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { EmptyState } from '@/components/shared/EmptyState';
import { IncidentBadge } from '@/components/shared/IncidentBadge';
import { formatDate } from '@/lib/formatters';

export function IncidentDetailPage() {
  const { incidentId } = useParams<{ incidentId: string }>();
  const navigate = useNavigate();
  const { incidents, fetchIncidentDetail, loading } = useIncidentStore();
  const incident = incidentId ? incidents[incidentId] : null;

  useEffect(() => {
    if (incidentId && !incidents[incidentId]) {
      fetchIncidentDetail(incidentId);
    }
  }, [incidentId]);

  if (loading) return <LoadingOverlay />;
  if (!incident) return <EmptyState title="Incident 不存在" description="找不到对应的事故信息" />;

  return (
    <div className="max-w-4xl mx-auto">
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 mb-4"
      >
        <ArrowLeft size={16} /> 返回
      </button>

      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">{incident.id}</h1>
              <IncidentBadge priority={incident.priority} />
              <span className={
                incident.status === 'closed'
                  ? 'px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400 text-xs font-medium'
                  : 'px-2 py-0.5 rounded-full bg-yellow-50 dark:bg-yellow-900/20 text-yellow-600 dark:text-yellow-400 text-xs font-medium'
              }>
                {incident.status}
              </span>
            </div>
            <p className="text-lg text-slate-700 dark:text-slate-300">{incident.title}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">受影响服务</div>
            <div className="text-sm font-semibold">{incident.context.affectedService}</div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">环境</div>
            <div className="text-sm font-semibold">{incident.context.environment}</div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">创建时间</div>
            <div className="text-sm font-semibold">{formatDate(incident.createdAt)}</div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">MTTR</div>
            <div className="text-sm font-semibold">{incident.mttr ? `${incident.mttr}min` : '-'}</div>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
            <Clock size={16} /> 时间线
          </h3>
          <div className="space-y-4">
            {incident.timeline.map((event, idx) => (
              <div key={idx} className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-3 h-3 rounded-full bg-primary-500" />
                  {idx < incident.timeline.length - 1 && (
                    <div className="w-0.5 h-full bg-slate-200 dark:bg-slate-800 mt-1" />
                  )}
                </div>
                <div className="pb-4">
                  <div className="text-xs text-slate-400 dark:text-slate-500 mb-1">
                    {formatDate(event.timestamp)} · {event.source}
                  </div>
                  <p className="text-sm text-slate-700 dark:text-slate-300">{event.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {incident.postmortem && (
          <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-800">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <AlertTriangle size={16} /> Postmortem
            </h3>
            <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4 space-y-3">
              <div>
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">根因</div>
                <p className="text-sm text-slate-700 dark:text-slate-300">{incident.postmortem.rootCause}</p>
              </div>
              <div>
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Action Items</div>
                <ul className="list-disc list-inside text-sm text-slate-700 dark:text-slate-300">
                  {incident.postmortem.actionItems.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
