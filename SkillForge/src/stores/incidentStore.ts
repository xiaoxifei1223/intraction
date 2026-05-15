import { create } from 'zustand';
import type { Incident } from '@/types/incident';

interface IncidentState {
  incidents: Record<string, Incident>;
  activeIncidentIds: string[];
  selectedIncidentId: string | null;
  loading: boolean;
  fetchIncidents: (status?: string) => Promise<void>;
  fetchIncidentDetail: (id: string) => Promise<void>;
  setSelectedIncident: (id: string | null) => void;
}

export const useIncidentStore = create<IncidentState>((set) => ({
  incidents: {},
  activeIncidentIds: [],
  selectedIncidentId: null,
  loading: false,
  fetchIncidents: async (status) => {
    set({ loading: true });
    const url = status ? `/api/incidents?status=${status}` : '/api/incidents';
    const res = await fetch(url);
    const json = await res.json();
    if (json.success) {
      const list = json.data as Incident[];
      const map: Record<string, Incident> = {};
      list.forEach((i) => (map[i.id] = i));
      set({
        incidents: map,
        activeIncidentIds: list.filter((i) => i.status !== 'closed').map((i) => i.id),
        loading: false,
      });
    }
  },
  fetchIncidentDetail: async (id) => {
    set({ loading: true });
    const res = await fetch(`/api/incidents/${id}`);
    const json = await res.json();
    if (json.success) {
      const incident = json.data as Incident;
      set((state) => ({
        incidents: { ...state.incidents, [incident.id]: incident },
        selectedIncidentId: incident.id,
        loading: false,
      }));
    }
  },
  setSelectedIncident: (id) => set({ selectedIncidentId: id }),
}));
