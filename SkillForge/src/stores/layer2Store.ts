import { create } from 'zustand';

interface Layer2State {
  teamPulse: any | null;
  skillRadar: any | null;
  mttrAnalysis: any | null;
  members: any[];
  schedule: any | null;
  reports: any[];
  loading: boolean;
  fetchTeamPulse: () => Promise<void>;
  fetchSkillRadar: () => Promise<void>;
  fetchMTTRAnalysis: () => Promise<void>;
  fetchMembers: () => Promise<void>;
  fetchSchedule: () => Promise<void>;
  fetchReports: () => Promise<void>;
}

export const useLayer2Store = create<Layer2State>((set) => ({
  teamPulse: null,
  skillRadar: null,
  mttrAnalysis: null,
  members: [],
  schedule: null,
  reports: [],
  loading: false,
  fetchTeamPulse: async () => {
    set({ loading: true });
    const res = await fetch('/api/team/pulse');
    const json = await res.json();
    if (json.success) set({ teamPulse: json.data, loading: false });
  },
  fetchSkillRadar: async () => {
    set({ loading: true });
    const res = await fetch('/api/team/radar');
    const json = await res.json();
    if (json.success) set({ skillRadar: json.data, loading: false });
  },
  fetchMTTRAnalysis: async () => {
    set({ loading: true });
    const res = await fetch('/api/team/mttr');
    const json = await res.json();
    if (json.success) set({ mttrAnalysis: json.data, loading: false });
  },
  fetchMembers: async () => {
    set({ loading: true });
    const res = await fetch('/api/team/members');
    const json = await res.json();
    if (json.success) set({ members: json.data, loading: false });
  },
  fetchSchedule: async () => {
    set({ loading: true });
    const res = await fetch('/api/team/schedule');
    const json = await res.json();
    if (json.success) set({ schedule: json.data, loading: false });
  },
  fetchReports: async () => {
    set({ loading: true });
    const res = await fetch('/api/team/reports');
    const json = await res.json();
    if (json.success) set({ reports: json.data, loading: false });
  },
}));
