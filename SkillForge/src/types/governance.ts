export interface OrgSnapshot {
  date: string;
  activeSkillCount: number;
  coverageRate: number;
  crossTeamReuseRate: number;
  avgMTTR: number;
  sloAchievement: number;
  singlePointRisks: number;
}

export interface MaturityDimension {
  name: string;
  score: number;
  trend: 'up' | 'down' | 'flat';
  benchmark: 'above' | 'avg' | 'below';
}

export interface MaturityAssessment {
  overallLevel: number;
  overallLabel: string;
  dimensions: MaturityDimension[];
}

export interface AIGovernanceReport {
  month: string;
  totalSkills: number;
  aiAssistedCount: number;
  aiOnlyCount: number;
  complianceRate: number;
  pendingReview: number;
  flagged: number;
}

export interface SkillConflict {
  id: string;
  severity: 'critical' | 'minor';
  skillA: { id: string; name: string; teamId: string };
  skillB: { id: string; name: string; teamId: string };
  conflictType: 'logic_contradiction' | 'parameter_mismatch' | 'overlap';
  description: string;
  suggestedAction: string;
}
