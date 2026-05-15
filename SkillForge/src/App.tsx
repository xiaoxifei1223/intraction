import { BrowserRouter } from 'react-router-dom';
import { AppRoutes } from './routes';
import { useEffect, useState } from 'react';
import { useAuthStore } from './stores/authStore';

function App() {
  const { setUser } = useAuthStore();
  const [mswReady, setMswReady] = useState(false);

  useEffect(() => {
    // Start MSW in development
    if (import.meta.env.DEV) {
      import('./mocks/browser')
        .then(({ worker }) => worker.start({ onUnhandledRequest: 'bypass' }))
        .then(() => {
          console.log('[MSW] Mocking enabled');
          setMswReady(true);
        })
        .catch((err) => {
          console.error('[MSW] Failed to start:', err);
          setMswReady(true); // Render anyway
        });
    } else {
      setMswReady(true);
    }
  }, []);

  useEffect(() => {
    if (!mswReady) return;
    fetch('/api/me')
      .then((res) => res.json())
      .then((json) => {
        if (json.success) {
          setUser(json.data);
        }
      })
      .catch((err) => {
        console.error('Failed to fetch user:', err);
      });
  }, [mswReady, setUser]);

  if (!mswReady) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-950 text-slate-100">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary-300 border-t-primary-600 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-slate-400">正在初始化...</p>
        </div>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}

export default App;
