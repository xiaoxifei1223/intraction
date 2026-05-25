export const STATUS_COLORS = {
  healthy: { dark: 'emerald-400', light: 'emerald-600', bg: 'bg-emerald-500' },
  attention: { dark: 'yellow-400', light: 'yellow-600', bg: 'bg-yellow-500' },
  outdated: { dark: 'red-400', light: 'red-600', bg: 'bg-red-500' },
  archived: { dark: 'slate-500', light: 'slate-400', bg: 'bg-slate-400' },
} as const;

export const PRIORITY_COLORS = {
  P1: { bg: 'bg-red-600', text: 'text-white' },
  P2: { bg: 'bg-orange-500', text: 'text-white' },
  P3: { bg: 'bg-yellow-500', text: 'text-black' },
  P4: { bg: 'bg-blue-400', text: 'text-white' },
} as const;

export const MATURITY_COLORS = [
  'slate-400',
  'blue-500',
  'indigo-500',
  'violet-500',
  'emerald-500',
] as const;

export const MATURITY_LABELS = [
  'L1 临时式',
  'L2 积累式',
  'L3 系统化',
  'L4 预测式',
  'L5 自进化',
] as const;

export const DOMAINS = [
  'Oracle基础管理',
  'Oracle性能诊断',
  'Oracle高可用(RAC)',
  'K8s基础运维',
  'K8s故障排查',
  'K8s网络/存储',
  'Linux系统调优',
  '监控告警配置',
] as const;
