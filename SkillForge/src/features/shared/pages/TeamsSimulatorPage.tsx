import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bot, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';

function BotCard({ children }: { children: React.ReactNode; title?: string }) {
  const [expanded, setExpanded] = useState(true);
  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 my-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-300"
      >
        <span className="flex items-center gap-2">
          <Bot size={16} className="text-purple-500" /> SkillForge
        </span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {expanded && <div className="px-4 pb-3">{children}</div>}
    </div>
  );
}

function LinkButton({ href, children }: { href: string; children: React.ReactNode }) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate(href)}
      className="text-primary-600 dark:text-primary-400 hover:underline text-sm inline-flex items-center gap-1"
    >
      {children} <ExternalLink size={12} />
    </button>
  );
}

export function TeamsSimulatorPage() {
  const [scenario, setScenario] = useState<'incident' | 'expert' | 'qa' | 'pulse' | 'alert'>('incident');

  const scenarios = {
    incident: (
      <>
        <div className="text-sm text-slate-700 dark:text-slate-300 mb-1">
          <strong>@zhang_san:</strong> DB 又超时了，响应时间飙到 8s
        </div>
        <BotCard>
          <div className="space-y-2 text-sm">
            <p className="text-slate-600 dark:text-slate-400">检测到相关经验：</p>
            <ul className="space-y-2">
              <li>
                • Skill "Oracle 慢查询诊断" 可能适用 (成功率 92%)<br />
                <LinkButton href="/diagnose?skillId=skill-oracle-slow-query-diag-v3">在 Web 中查看完整步骤</LinkButton>
              </li>
              <li>
                • 上次类似问题: INC-2024-0654, 由 @li_si 在 22min 内解决<br />
                <LinkButton href="/incident/INC-2024-0654">查看历史 incident</LinkButton>
              </li>
            </ul>
            <div className="pt-2">
              <LinkButton href="/diagnose?context=oracle">🚀 一键打开诊断中心</LinkButton>
            </div>
          </div>
        </BotCard>
        <div className="text-sm text-slate-700 dark:text-slate-300 mb-1">
          <strong>@zhang_san:</strong> 重建索引后恢复了
        </div>
        <BotCard>
          <div className="space-y-2 text-sm">
            <p className="text-slate-600 dark:text-slate-400">✓ Incident 已关闭 (耗时 27min)</p>
            <p>检测到一个可复用的解决模式："Oracle AWR自动分析 + 索引在线重建"</p>
            <LinkButton href="/my-skills?draft=INC-0789">在 Web 中保存为 Skill</LinkButton>
          </div>
        </BotCard>
      </>
    ),
    expert: (
      <>
        <div className="text-sm text-slate-700 dark:text-slate-300 mb-1">
          <strong>@小李:</strong> @SkillForge 谁了解 Oracle RAC 故障切换？
        </div>
        <BotCard title="专家定位">
          <div className="space-y-3 text-sm">
            <div>
              <div className="font-medium text-slate-900 dark:text-slate-100">👤 @li_si (DB-SRE)</div>
              <div className="text-slate-500 dark:text-slate-400">解决过 9 次 RAC 相关 incident, 平均 MTTR 15min</div>
              <LinkButton href="/profile?userId=user_li_si">查看 li_si 的 Skill 库</LinkButton>
            </div>
            <LinkButton href="/search?q=Oracle+RAC&filter=skill">🔍 查看 Oracle RAC 相关所有 Skill</LinkButton>
          </div>
        </BotCard>
      </>
    ),
    qa: (
      <>
        <div className="text-sm text-slate-700 dark:text-slate-300 mb-1">
          <strong>@小李:</strong> 怎么查看 K8s pod 被驱逐的原因？
        </div>
        <div className="text-sm text-slate-700 dark:text-slate-300 mb-1">
          <strong>@大王:</strong> kubectl describe pod xxx 看 Events 部分
        </div>
        <BotCard>
          <div className="space-y-2 text-sm">
            <p>检测到一次有价值的问答：</p>
            <p className="text-slate-600 dark:text-slate-400">Q: "怎么查看 K8s pod 被驱逐的原因？" (@小李)</p>
            <p className="text-slate-600 dark:text-slate-400">A: "kubectl describe pod xxx 看 Events 部分..." (@大王)</p>
            <LinkButton href="/my-skills?from=qa&question=k8s-pod-evicted">在 Web 中保存为 Skill 片段</LinkButton>
          </div>
        </BotCard>
      </>
    ),
    pulse: (
      <BotCard>
        <div className="space-y-2 text-sm">
          <p className="font-medium">📊 团队日报 │ 2026-05-15 │ DB-SRE Team</p>
          <p>昨夜值班: @wang_wu</p>
          <p>✅ 平稳 - 无 P1/P2, 3 个自动恢复告警</p>
          <p>Skill 动态: 昨日团队使用 Skill 5 次, 新增 1 个 Skill 草稿</p>
          <div className="pt-2 space-x-3">
            <LinkButton href="/team?from=daily-pulse">📱 在 Web 中打开团队概览</LinkButton>
            <LinkButton href="/team/mttr?range=week">📊 查看 MTTR 趋势</LinkButton>
          </div>
        </div>
      </BotCard>
    ),
    alert: (
      <BotCard>
        <div className="space-y-2 text-sm">
          <p className="font-medium text-yellow-600 dark:text-yellow-400">⚡ INC-0801 (P2) │ DB-prod 响应超时 │ 进行中 22min</p>
          <p>响应人: @zhang_san (入职8个月)</p>
          <p>🟡 需要关注 - 已超过该类问题平均解决时间 (15min)</p>
          <div className="pt-2 space-x-3">
            <LinkButton href="/team?focus=INC-0801">在 Web 中查看实时态势</LinkButton>
            <LinkButton href="/team/members?user=zhang_san">查看成长追踪</LinkButton>
          </div>
        </div>
      </BotCard>
    ),
  };

  const scenarioButtons: { key: typeof scenario; label: string }[] = [
    { key: 'incident', label: 'Incident频道' },
    { key: 'expert', label: '专家定位' },
    { key: 'qa', label: '问答捕获' },
    { key: 'pulse', label: '每日脉搏' },
    { key: 'alert', label: '态势提醒' },
  ];

  return (
    <div className="h-[calc(100vh-140px)] flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Teams 场景模拟器</h2>
        <span className="text-xs text-slate-400 dark:text-slate-500">点击 Bot 卡片中的链接可跳转到真实 Web 页面</span>
      </div>

      <div className="flex-1 bg-slate-50 dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden flex">
        {/* Channel list */}
        <div className="w-48 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3">
          <div className="text-xs font-semibold text-slate-400 dark:text-slate-500 mb-2">频道</div>
          {['#general', '#incident-db', '#sre-help', '#alerts'].map((ch) => (
            <div key={ch} className="text-sm text-slate-600 dark:text-slate-400 py-1 px-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
              {ch}
            </div>
          ))}
        </div>

        {/* Chat area */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {scenarios[scenario]}
          </div>
        </div>
      </div>

      {/* Scenario switcher */}
      <div className="flex items-center gap-2 mt-4">
        <span className="text-sm text-slate-500 dark:text-slate-400">场景切换:</span>
        {scenarioButtons.map((btn) => (
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
