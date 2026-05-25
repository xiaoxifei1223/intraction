import { create } from 'zustand';

interface UIState {
  theme: 'dark' | 'light';
  sidebarCollapsed: boolean;
  globalSearchOpen: boolean;
  setTheme: (t: 'dark' | 'light') => void;
  toggleTheme: () => void;
  toggleSidebar: () => void;
  openSearch: () => void;
  closeSearch: () => void;
}

function getSavedTheme(): 'dark' | 'light' {
  try {
    const t = localStorage.getItem('theme') as 'dark' | 'light' | null;
    if (t === 'dark' || t === 'light') return t;
  } catch {
    // localStorage might be disabled
  }
  return 'dark';
}

function applyThemeClass(theme: 'dark' | 'light') {
  if (typeof document !== 'undefined') {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }
}

const savedTheme = getSavedTheme();
applyThemeClass(savedTheme);

export const useUIStore = create<UIState>((set) => ({
  theme: savedTheme,
  sidebarCollapsed: false,
  globalSearchOpen: false,
  setTheme: (t) => {
    try {
      localStorage.setItem('theme', t);
    } catch {
      // ignore
    }
    applyThemeClass(t);
    set({ theme: t });
  },
  toggleTheme: () =>
    set((state) => {
      const t = state.theme === 'dark' ? 'light' : 'dark';
      try {
        localStorage.setItem('theme', t);
      } catch {
        // ignore
      }
      applyThemeClass(t);
      return { theme: t };
    }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  openSearch: () => set({ globalSearchOpen: true }),
  closeSearch: () => set({ globalSearchOpen: false }),
}));
