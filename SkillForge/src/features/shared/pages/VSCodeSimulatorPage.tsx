import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';

function LinkButton({ href, children }: { href: string; children: React.ReactNode }) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate(href)}
      className="text-primary-600 dark:text-primary-400 hover:underline text-xs inline-flex items-center gap-1"
    >
      {children}
    </button>
  );
}

export function VSCodeSimulatorPage() {
  const [scenario, setScenario] = useState<'diagnose' | 'record' | 'inline' | 'snippet' | 'learning'>('diagnose');

  const panelContent = {
    diagnose: (
      <div className="p-4 space-y-3">
        <div className="text-sm font-medium text-slate-900 dark:text-slate-100">⚡ 检测到错误: ORA-04031</div>
        <div className="space-y-2">
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-sm font-medium mb-1">1. ★ Shared Pool 内存不足诊断</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">成功率: 88% │ 平均: 12min</div>
            <div className="mt-2 space-x-2">
              <LinkButton href="/diagnose?error=ORA-04031">在 Web 中查看</LinkButton>
              <LinkButton href="/snippets?id=snippet-ora-04031">一键复制命令</LinkButton>
            </div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
            <div className="text-sm font-medium mb-1">2. SGA 动态调优</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">成功率: 75% │ 平均: 20min</div>
          </div>
        </div>
        <div className="text-xs text-slate-500 dark:text-slate-400 pt-2">
          💡 @li_si 是该领域专家 (处理过12次类似问题)<br />
          <LinkButton href="/profile?userId=user_li_si">查看专家档案</LinkButton>
        </div>
        <LinkButton href="/diagnose?from=vscode">🔍 打开诊断中心</LinkButton>
      </div>
    ),
    record: (
      <div className="p-4">
        <div className="text-sm text-slate-600 dark:text-slate-400 mb-3">
          🛠️ 正在录制操作序列 (已录制 12 条命令)
        </div>
        <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3 space-y-2">
          <div className="text-sm font-medium">检测到一次成功的问题解决 (ORA-04031)</div>
          <div className="text-xs text-slate-500 dark:text-slate-400">已自动保存为 Skill 草稿</div>
          <LinkButton href="/my-skills?tab=drafts&incident=0789">在 Web 中查看草稿</LinkButton>
        </div>
      </div>
    ),
    inline: (
      <div className="p-4 space-y-3">
        <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">postgresql.conf</div>
          <div className="text-sm font-mono">shared_buffers = 4GB</div>
          <div className="text-xs text-yellow-600 dark:text-yellow-400 mt-1">
            💡 此值在 Incident INC-2024-0456 中从 2GB 调至 4GB<br />
            <LinkButton href="/incident/INC-2024-0456">查看详情</LinkButton>
          </div>
        </div>
      </div>
    ),
    snippet: (
      <div className="p-4 space-y-2">
        <div className="text-sm text-slate-600 dark:text-slate-400">sf: check oracle performance</div>
        <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
          <div className="text-sm font-medium">[1] AWR报告快速生成</div>
          <div className="text-xs font-mono text-slate-500 dark:text-slate-400 mt-1">@?/rdbms/admin/awrrpt.sql</div>
          <div className="mt-2 space-x-2">
            <LinkButton href="/snippets?id=snippet-awr-quick">在 Web 中查看</LinkButton>
          </div>
        </div>
        <LinkButton href="/snippets?query=oracle+performance">📚 打开片段库</LinkButton>
      </div>
    ),
    learning: (
      <div className="p-4 space-y-3">
        <div className="text-sm font-medium">🛠️ SkillForge: 我的技能地图</div>
        <div className="space-y-2 text-sm">
          <div>Oracle DB ████████████░░░░ 78%</div>
          <div className="pl-4 text-xs text-slate-500 dark:text-slate-400">
            ├─ 高可用(RAC) ████████░░░░░░░░ 50% ← 建议下一步<br />
            <LinkButton href="/learning?domain=oracle-rac">在 Web 中查看学习路径</LinkButton>
          </div>
        </div>
        <LinkButton href="/learning?from=vscode">📱 打开完整学习地图</LinkButton>
      </div>
    ),
  };

  const scenarios: { key: typeof scenario; label: string }[] = [
    { key: 'diagnose', label: '诊断面板' },
    { key: 'record', label: '操作录制' },
    { key: 'inline', label: '代码助手' },
    { key: 'snippet', label: '片段库' },
    { key: 'learning', label: '学习地图' },
  ];

  return (
    <div className="h-[calc(100vh-140px)] flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">VS Code 场景模拟器</h2>
        <span className="text-xs text-slate-400 dark:text-slate-500">点击链接在当前浏览器内跳转到真实 Web 页面</span>
      </div>

      <div className="flex-1 bg-slate-900 rounded-xl border border-slate-700 overflow-hidden flex text-slate-100">
        {/* Explorer */}
        <div className="w-48 border-r border-slate-700 bg-slate-800 p-2">
          <div className="text-xs font-semibold text-slate-400 mb-2">EXPLORER</div>
          <div className="text-xs text-slate-300 py-0.5">src/</div>
          <div className="text-xs text-slate-400 pl-3 py-0.5">config/</div>
          <div className="text-xs text-slate-400 pl-3 py-0.5">services/</div>
          <div className="text-xs text-slate-400 pl-3 py-0.5">main.ts</div>
        </div>

        {/* Editor */}
        <div className="flex-1 flex flex-col border-r border-slate-700">
          <div className="h-8 bg-slate-800 border-b border-slate-700 flex items-center px-3 text-xs text-slate-300">
            postgresql.conf
          </div>
          <div className="flex-1 p-4 font-mono text-xs text-slate-300 space-y-1">
            <div>shared_buffers = 4GB</div>
            <div className="text-yellow-400">│ 💡 SkillForge 提示...</div>
            <div>max_connections = 200</div>
            <div className="mt-4 text-slate-500">──────── Terminal ────────</div>
            <div className="text-red-400">$ ORA-04031: unable to allocate 3896 bytes of shared memory</div>
          </div>
        </div>

        {/* SkillForge Panel */}
        <div className="w-72 bg-slate-800 border-l border-slate-700 flex flex-col">
          <div className="h-8 border-b border-slate-700 flex items-center px-3 text-xs font-semibold text-slate-300">
            🛠️ SkillForge
          </div>
          <div className="flex-1 overflow-y-auto">
            {panelContent[scenario]}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 mt-4">
        <span className="text-sm text-slate-500 dark:text-slate-400">场景切换:</span>
        {scenarios.map((btn) => (
          <button
            key={btn.key}
            onClick={() => setScenario(btn.key)}
            className={cn(
              'px-3 py-1.5 rounded-md text-sm transition-colors',
              scenario === btn.key
                ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
            )}
          >
            {btn.label}
          </button>
        ))}
      </div>
    </div>
  );
}
