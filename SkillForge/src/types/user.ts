export interface MasteryItem {
  domain: string;
  level: number;
  trend: 'up' | 'down' | 'flat';
}

export interface UserMetrics {
  totalIncidents: number;
  avgMTTR: number;
  skillsCreated: number;
  skillsAdoptedByOthers: number;
}

export interface User {
  id: string;
  name: string;
  handle: string;
  avatar: string;
  role: 'engineer' | 'lead' | 'executive';
  teamId: string;
  title: string;
  joinDate: string;
  skillsMastery: MasteryItem[];
  status: 'online' | 'busy' | 'offline' | 'oncall';
  metrics: UserMetrics;
}
