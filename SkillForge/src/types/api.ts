import type { Skill } from './skill';
import type { User } from './user';
import type { Incident } from './incident';

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

export interface MatchedSkill {
  skill: Skill;
  matchScore: number;
  reason: string;
  estimatedTime: number;
}

export interface Expert {
  user: User;
  relevantIncidents: number;
  avgMTTR: number;
  relatedSkills: number;
}

export interface DiagnoseResponse {
  queryInterpretation: string;
  matchedSkills: MatchedSkill[];
  suggestedExperts: Expert[];
  similarIncidents: Incident[];
}

export interface Snippet {
  id: string;
  title: string;
  command: string;
  description: string;
  tags: string[];
  authorId: string;
  useCount: number;
  successRate: number;
  applicableEnv: string[];
}

export interface TimeRange {
  label: string;
  startDate: string;
  endDate: string;
}
