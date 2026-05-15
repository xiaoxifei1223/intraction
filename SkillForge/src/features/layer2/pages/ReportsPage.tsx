import { useEffect, useState } from 'react';
import { FileText, Download, Send, Share2, ChevronRight } from 'lucide-react';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { EmptyState } from '@/components/shared/EmptyState';
import { useLayer2Store } from '@/stores/layer2Store';

export function ReportsPage() {
  const { reports, fetchReports, loading } = useLayer2Store();
  const [selectedReport, setSelectedReport] = useState<any>(null);

  useEffect(() => {
    fetchReports();
  }, []);

  if (loading) return <LoadingOverlay />;

  if (selectedReport) {
    return (
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => setSelectedReport(null)}
          className="text-sm text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 mb-4"
        >
          ← 返回报告列表
        </button>

        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">{selectedReport.title}</h1>
            <div className="flex items-center gap-2">
              <button className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                <Download size={14} /> 导出
              </button>
              <button className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                <Send size={14} /> 发送
              </button>
              <button className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                <Share2 size={14} /> 分享
              </button>
            </div>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-slate-600 dark:text-slate-400 mb-6">{selectedReport.summary}</p>
            {selectedReport.sections?.map((section: any) => (
              <div key={section.id} className="mb-6 pb-6 border-b border-slate-100 dark:border-slate-800 last:border-0">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">{section.title}</h3>
                <p className="text-slate-600 dark:text-slate-400">{section.content}</p>
                {section.metrics && Object.keys(section.metrics).length > 0 && (
                  <div className="grid grid-cols-3 gap-4 mt-4">
                    {Object.entries(section.metrics).map(([key, value]) => (
                      <div key={key} className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{key}</div>
                        <div className="text-lg font-semibold text-slate-900 dark:text-slate-100">{String(value)}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">汇报材料</h1>
      </div>

      <div className="space-y-3">
        {reports.map((report: any) => (
          <button
            key={report.id}
            onClick={() => setSelectedReport(report)}
            className="w-full text-left bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 hover:shadow-md transition-all flex items-center justify-between"
          >
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-lg bg-primary-50 dark:bg-primary-900/20 flex items-center justify-center">
                <FileText size={20} className="text-primary-600 dark:text-primary-400" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">{report.title}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">{report.type === 'weekly' ? '周报' : '月报'} · {report.createdAt?.slice(0, 10)}</p>
              </div>
            </div>
            <ChevronRight size={18} className="text-slate-400 dark:text-slate-500" />
          </button>
        ))}
      </div>

      {reports.length === 0 && <EmptyState title="暂无报告" description="报告将在此生成和归档" />}
    </div>
  );
}
