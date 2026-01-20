
import React, { useState, useEffect } from 'react';
import PredictionTab from './components/PredictionTab';
import RankingTab from './components/RankingTab';
import VisualizationTab from './components/VisualizationTab';
import { TabType } from './types';
import { APP_NAME } from './constants';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>(TabType.PREDICTION);
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    // Check if the user has a preferred theme in localStorage or system settings
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('theme');
      if (saved === 'dark' || saved === 'light') return saved;
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'light';
  });

  // Sync theme state with the document root class
  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 transition-colors duration-500 text-slate-900 dark:text-slate-100 relative overflow-hidden">
      
      {/* Cosmic Background Pattern (Visible in Dark Mode) */}
      <div className="fixed inset-0 pointer-events-none opacity-0 dark:opacity-40 transition-opacity duration-1000">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-900/20 via-slate-950 to-slate-950"></div>
        <div className="absolute inset-0" style={{ backgroundImage: 'radial-gradient(circle, #fff 1px, transparent 1px)', backgroundSize: '100px 100px' }}></div>
        <div className="absolute inset-0" style={{ backgroundImage: 'radial-gradient(circle, #fff 0.5px, transparent 0.5px)', backgroundSize: '40px 40px', opacity: 0.3 }}></div>
      </div>

      {/* Header Navigation */}
      <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-tr from-blue-600 to-cyan-400 rounded-xl flex items-center justify-center font-black text-xl text-white shadow-lg shadow-blue-500/20 rotate-3">
              A
            </div>
            <div>
              <h1 className="text-2xl font-black tracking-tighter bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-400 bg-clip-text text-transparent">
                {APP_NAME}
              </h1>
              <p className="text-[10px] uppercase tracking-widest font-bold text-blue-500 dark:text-blue-400 leading-none">AI Exoplanet Intelligence</p>
            </div>
          </div>
          
          <nav className="hidden md:flex items-center bg-slate-100 dark:bg-slate-800/50 p-1 rounded-full border border-slate-200 dark:border-slate-700">
            <button 
              onClick={() => setActiveTab(TabType.PREDICTION)}
              className={`px-6 py-2 rounded-full text-sm font-bold transition-all ${activeTab === TabType.PREDICTION ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
            >
              Prediction
            </button>
            <button 
              onClick={() => setActiveTab(TabType.RANKING)}
              className={`px-6 py-2 rounded-full text-sm font-bold transition-all ${activeTab === TabType.RANKING ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
            >
              Ranking
            </button>
            <button 
              onClick={() => setActiveTab(TabType.VISUALIZATION)}
              className={`px-6 py-2 rounded-full text-sm font-bold transition-all ${activeTab === TabType.VISUALIZATION ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
            >
              Visualization
            </button>
          </nav>

          <div className="flex items-center gap-4">
            <button
              onClick={toggleTheme}
              className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 transition-all flex items-center justify-center border border-slate-200 dark:border-slate-700 group"
              aria-label="Toggle Theme"
            >
              {theme === 'light' ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-slate-600 group-hover:rotate-12 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-400 group-hover:rotate-90 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow relative z-10">
        <div className="animate-in fade-in duration-700">
          {activeTab === TabType.PREDICTION && <PredictionTab />}
          {activeTab === TabType.RANKING && <RankingTab />}
          {activeTab === TabType.VISUALIZATION && <VisualizationTab theme={theme} />}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-800 py-10 text-center relative z-10">
        <div className="max-w-4xl mx-auto px-6">
          <p className="text-slate-400 dark:text-slate-500 text-xs font-medium tracking-widest uppercase">
            © 2024 AstroHabit Intelligence • Data from NASA Exoplanet Archive
          </p>
          <div className="mt-4 flex justify-center gap-6">
             <div className="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]"></div>
             <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.6)]"></div>
             <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]"></div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
