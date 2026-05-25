import type { SkillStatus } from './skill';

export interface DomainCoverage {
  domain: string;
  coverageCount: number;
  totalMembers: number;
  avgDepth: number;
  healthStatus: SkillStatus;
}

export interface TeamMetrics {
  avgMTTR: number;
  sloAchievement: number;
  incidentCountThisWeek: number;
  skillUsageThisWeek: number;
  newSkillsThisWeek: number;
}

export interface ScheduleSlot {
  userId: string;
  shift: 'day' | 'night';
}

export interface Team {
  id: string;
  name: string;
  memberIds: string[];
  skillIds: string[];
  coverage: DomainCoverage[];
  metrics: TeamMetrics;
  schedule: ScheduleSlot[][];
}
