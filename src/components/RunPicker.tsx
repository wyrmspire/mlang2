import React, { useEffect, useState } from 'react';
import { api } from '../api/client';

export const RunPicker: React.FC<{ onSelect: (id: string) => void }> = ({ onSelect }) => {
  const [runs, setRuns] = useState<string[]>([]);
  const [confirmClear, setConfirmClear] = useState(false);

  const refreshRuns = () => {
    api.getRuns().then(setRuns);
  };

  useEffect(() => {
    refreshRuns();
  }, []);

  const handleClearAll = async () => {
    if (!confirmClear) {
      setConfirmClear(true);
      setTimeout(() => setConfirmClear(false), 3000); // Reset after 3s
      return;
    }

    try {
      await api.clearAllRuns(); // We need to add this method to the client first
      setRuns([]);
      onSelect(''); // Clear selection
      setConfirmClear(false);
    } catch (e) {
      console.error('Failed to clear runs:', e);
    }
  };

  return (
    <div className="bg-slate-900/40 rounded-lg p-4 border border-slate-800/60 backdrop-blur-sm shadow-sm">
      <div className="flex justify-between items-center mb-2">
        <label className="text-[10px] text-slate-500 font-bold uppercase tracking-widest flex items-center gap-1">
          <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
          Select Session
        </label>
        <button
          onClick={refreshRuns}
          className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white px-2 py-0.5 rounded transition-colors"
          title="Refresh List"
        >
          REFRESH
        </button>
      </div>
      <div className="flex gap-2">
        <div className="relative flex-1 group">
            <select
            className="w-full appearance-none bg-slate-950 border border-slate-800 rounded-md text-xs p-2.5 text-slate-200 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all cursor-pointer hover:border-slate-700"
            onChange={(e) => onSelect(e.target.value)}
            >
            <option value="">-- Choose Run --</option>
            {runs.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none text-slate-500 group-hover:text-slate-300">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
            </div>
        </div>

        <button
          onClick={handleClearAll}
          className={`px-3 rounded-md text-xs font-bold border transition-all duration-200 ${confirmClear
              ? 'bg-red-500 text-white border-red-400 shadow-[0_0_10px_rgba(239,68,68,0.5)]'
              : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700 hover:text-red-400 hover:border-slate-600'
            }`}
          title="Clear all experiments"
        >
          {confirmClear ? '??' : '🗑'}
        </button>
      </div>
    </div>
  );
};
