export type IncidentPriority = 'P1' | 'P2' | 'P3' | 'P4';
export type IncidentStatus = 'open' | 'diagnosing' | 'fixing' | 'verifying' | 'closed';

export interface SkillUsageLog {
  skillId: string;
  startedAt: string;
  completedAt?: string;
  currentStep: number;
  totalSteps: number;
  success: boolean;
}

export interface TimelineEvent {
  timestamp: string;
  type: 'alert' | 'response' | 'diagnosis' | 'decision' | 'action' | 'info' | 'resolution';
  actorId?: string;
  description: string;
  source: 'pagerduty' | 'teams' | 'vscode' | 'manual';
}

export interface Postmortem {
  rootCause: string;
  actionItems: string[];
  lessonsLearned: string[];
  skillsCreated: string[];
}

export interface Incident {
  id: string;
  title: string;
  priority: IncidentPriority;
  status: IncidentStatus;
  createdAt: string;
  resolvedAt?: string;
  mttr?: number;
  assigneeId: string;
  commanderId?: string;
  teamId: string;
  context: {
    alertSource: string;
    affectedService: string;
    environment: 'prod' | 'staging' | 'dev';
    initialSymptom: string;
  };
  skillUsage: SkillUsageLog[];
  timeline: TimelineEvent[];
  postmortem?: Postmortem;
}
