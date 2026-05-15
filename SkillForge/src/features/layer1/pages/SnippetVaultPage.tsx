import { useEffect, useState } from 'react';
import { Search, Copy, Tag, Terminal } from 'lucide-react';
import { LoadingOverlay } from '@/components/shared/LoadingOverlay';
import { EmptyState } from '@/components/shared/EmptyState';
import type { Snippet } from '@/types/api';

export function SnippetVaultPage() {
  const [snippets, setSnippets] = useState<Snippet[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');

  useEffect(() => {
    setLoading(true);
    fetch('/api/snippets')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) setSnippets(json.data);
      })
      .finally(() => setLoading(false));
  }, []);

  const filtered = snippets.filter(
    (s) => s.title.includes(search) || s.tags.some((t) => t.includes(search))
  );

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">命令片段库</h1>
      </div>

      <div className="relative mb-6">
        <Search className="absolute left-4 top-3 text-slate-400" size={18} />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索片段，支持 tag:oracle mttr 语法..."
          className="w-full pl-12 pr-4 py-3 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500"
        />
      </div>

      {loading && <LoadingOverlay />}

      {!loading && (
        <div className="space-y-4">
          {filtered.map((snippet) => (
            <div
              key={snippet.id}
              className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 hover:shadow-md transition-all"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-slate-900 dark:text-slate-100">{snippet.title}</h3>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{snippet.description}</p>
                </div>
                <button
                  onClick={() => navigator.clipboard.writeText(snippet.command)}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                >
                  <Copy size={14} /> 复制
                </button>
              </div>

              <pre className="bg-slate-900 text-slate-100 rounded-lg p-3 overflow-x-auto text-sm font-mono mb-3">
                {snippet.command}
              </pre>

              <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1"><Terminal size={12} /> 使用 {snippet.useCount} 次</span>
                  <span>成功率 {(snippet.successRate * 100).toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  {snippet.tags.map((tag) => (
                    <span key={tag} className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800">
                      <Tag size={10} /> {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && filtered.length === 0 && (
        <EmptyState title="未找到片段" description="尝试其他关键词或标签" />
      )}
    </div>
  );
}
