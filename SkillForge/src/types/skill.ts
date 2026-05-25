export type SkillStatus = 'healthy' | 'attention' | 'outdated' | 'archived';
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

export interface SkillStep {
  order: number;
  title: string;
  description: string;
  command?: string;
  estimatedTime: number;
  verification: string;
}

export interface ActionItem {
  type: 'command' | 'script' | 'config' | 'manual';
  content: string;
  safetyLevel: RiskLevel;
}

export interface DecisionNode {
  condition: string;
  trueBranch?: SkillStep[] | DecisionNode;
  falseBranch?: SkillStep[] | DecisionNode;
}

export interface Skill {
  id: string;
  name: string;
  version: number;
  authorId: string;
  teamId: string;
  createdAt: string;
  lastUsedAt: string;
  useCount: number;
  successRate: number;
  avgResolutionTime: number;
  classification: {
    domain: string[];
    scenario: string[];
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    riskLevel: RiskLevel;
  };
  dependencies: {
    requiredSkills: string[];
    requiredAccess: string[];
  };
  content: {
    triggerConditions: string;
    diagnosisSteps: SkillStep[];
    decisionTree?: DecisionNode;
    executionActions: ActionItem[];
    rollbackPlan: string;
  };
  governance: {
    approvalStatus: 'draft' | 'pending' | 'approved' | 'rejected';
    reviewerId?: string;
    complianceTags: string[];
    expiryReviewDate: string;
    aiGenerated: boolean;
    aiConfidence?: number;
  };
  evolution: {
    parentSkillId?: string;
    changeLog: string;
    deprecationCandidates?: string[];
  };
  healthStatus: SkillStatus;
  healthScore: number;
}
