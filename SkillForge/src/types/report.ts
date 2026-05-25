export type ReportType = 'weekly' | 'monthly' | 'roi' | 'resource';

export interface Report {
  id: string;
  title: string;
  type: ReportType;
  createdAt: string;
  authorId: string;
  teamId?: string;
  summary: string;
  sections: ReportSection[];
}

export interface ReportSection {
  id: string;
  title: string;
  content: string;
  metrics?: Record<string, string | number>;
}
