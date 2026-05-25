import { create } from 'zustand';
import type { Skill } from '@/types/skill';

interface SkillState {
  skills: Record<string, Skill>;
  skillList: string[];
  selectedSkillId: string | null;
  loading: boolean;
  fetchSkills: (query?: string) => Promise<void>;
  fetchSkillDetail: (id: string) => Promise<void>;
  setSelectedSkill: (id: string | null) => void;
}

export const useSkillStore = create<SkillState>((set) => ({
  skills: {},
  skillList: [],
  selectedSkillId: null,
  loading: false,
  fetchSkills: async (query = '') => {
    set({ loading: true });
    const res = await fetch(`/api/skills?query=${encodeURIComponent(query)}`);
    const json = await res.json();
    if (json.success) {
      const list = json.data as Skill[];
      const map: Record<string, Skill> = {};
      list.forEach((s) => (map[s.id] = s));
      set({ skills: map, skillList: list.map((s) => s.id), loading: false });
    }
  },
  fetchSkillDetail: async (id) => {
    set({ loading: true });
    const res = await fetch(`/api/skills/${id}`);
    const json = await res.json();
    if (json.success) {
      const skill = json.data as Skill;
      set((state) => ({
        skills: { ...state.skills, [skill.id]: skill },
        selectedSkillId: skill.id,
        loading: false,
      }));
    }
  },
  setSelectedSkill: (id) => set({ selectedSkillId: id }),
}));
