import { useEffect, useState } from 'react';
import { Presentation, ChevronLeft, ChevronRight, Download, Lock } from 'lucide-react';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';

export function BoardReportPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [currentSlide, setCurrentSlide] = useState(0);

  useEffect(() => {
    fetch('/api/org/board-report')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setData(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingOverlay />;
  if (!data) return <LoadingOverlay />;

  const slides = data.slides;

  const nextSlide = () => setCurrentSlide((prev) => Math.min(prev + 1, slides.length - 1));
  const prevSlide = () => setCurrentSlide((prev) => Math.max(prev - 1, 0));

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Presentation size={24} /> 集团汇报材料
        </h1>
        <div className="flex items-center gap-2">
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
            <Download size={16} /> 导出 PPT
          </button>
        </div>
      </div>

      {/* Slide Viewer */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        {/* Slide Header */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50">
          <div className="text-sm text-slate-500 dark:text-slate-400">
            Slide {currentSlide + 1} / {slides.length}
          </div>
          <div className="flex items-center gap-1 text-xs text-slate-400 dark:text-slate-500">
            <Lock size={12} /> 数据来自 SkillForge，自动更新
          </div>
        </div>

        {/* Slide Content */}
        <div className="p-12 min-h-[400px] flex flex-col items-center justify-center text-center">
          <div className="text-sm text-slate-400 dark:text-slate-500 mb-4 uppercase tracking-wider">
            SkillForge 集团汇报
          </div>
          <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-6">
            {slides[currentSlide].title}
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl leading-relaxed">
            {slides[currentSlide].content}
          </p>

          {currentSlide === 1 && (
            <div className="grid grid-cols-4 gap-6 mt-8 w-full max-w-2xl">
              {[
                { label: '可靠性', value: '99.96%', color: 'text-emerald-600 dark:text-emerald-400' },
                { label: '效率提升', value: '19%', color: 'text-blue-600 dark:text-blue-400' },
                { label: '人才传承', value: '2个', color: 'text-indigo-600 dark:text-indigo-400' },
                { label: 'Skill增长', value: '+8', color: 'text-violet-600 dark:text-violet-400' },
              ].map((item) => (
                <div key={item.label}>
                  <div className={`text-2xl font-bold ${item.color}`}>{item.value}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">{item.label}</div>
                </div>
              ))}
            </div>
          )}

          {currentSlide === 2 && (
            <div className="grid grid-cols-3 gap-6 mt-8 w-full max-w-xl">
              {[
                { label: '活跃 Skill', value: '312' },
                { label: '健康度', value: '93%' },
                { label: 'AI 占比', value: '61%' },
              ].map((item) => (
                <div key={item.label} className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4">
                  <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">{item.value}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">{item.label}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200 dark:border-slate-800">
          <button
            onClick={prevSlide}
            disabled={currentSlide === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft size={16} /> 上一页
          </button>
          <div className="flex items-center gap-2">
            {slides.map((_: any, i: number) => (
              <button
                key={i}
                onClick={() => setCurrentSlide(i)}
                className={`w-2 h-2 rounded-full transition-colors ${
                  i === currentSlide ? 'bg-primary-500' : 'bg-slate-300 dark:bg-slate-600'
                }`}
              />
            ))}
          </div>
          <button
            onClick={nextSlide}
            disabled={currentSlide === slides.length - 1}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            下一页 <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
