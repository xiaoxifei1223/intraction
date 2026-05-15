import { create } from 'zustand';
import type { User } from '@/types/user';

interface AuthState {
  currentUser: User | null;
  currentRole: 'engineer' | 'lead' | 'executive';
  setUser: (user: User | null) => void;
  switchRole: (role: 'engineer' | 'lead' | 'executive') => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  currentUser: null,
  currentRole: 'engineer',
  setUser: (user) => set({ currentUser: user }),
  switchRole: (role) =>
    set((state) => ({
      currentRole: role,
      currentUser: state.currentUser ? { ...state.currentUser, role } : null,
    })),
}));
