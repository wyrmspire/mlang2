import React, { useEffect, useState } from 'react';
import { api } from '../api/client';

export const RunPicker: React.FC<{ onSelect: (id: string) => void }> = ({ onSelect }) => {
  const [runs, setRuns] = useState<string[]>([]);
  const [confirmClear, setConfirmClear] = useState(false);
  const [loading, setLoading] = useState(false);

  const refreshRuns = async () => {
    setLoading(true);
    try {
        const data = await api.getRuns();
        setRuns(data);
    } catch(e) {
        console.error(e);
    } finally {
        setLoading(false);
    }
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
      await api.clearAllRuns();
      setRuns([]);
      onSelect(''); // Clear selection
      setConfirmClear(false);
    } catch (e) {
      console.error('Failed to clear runs:', e);
    }
  };

  return (
    <div className="bg-slate-900/40 rounded-lg p-3 border border-slate-800/60 backdrop-blur-sm shadow-sm">
      <div className="flex justify-between items-center mb-2">
        <label className="text-[10px] text-slate-500 font-bold uppercase tracking-widest flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>
          Session Control
        </label>
        <button
          onClick={refreshRuns}
          className={`text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white px-2 py-1 rounded transition-colors flex items-center gap-1 ${loading ? 'opacity-50 cursor-wait' : ''}`}
          title="Refresh List"
          disabled={loading}
        >
          <svg className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
          SYNC
        </button>
      </div>

      <div className="flex gap-2">
        <div className="relative flex-1 group">
            <select
            className="w-full appearance-none bg-slate-950 border border-slate-800 rounded-md text-xs pl-3 pr-8 py-2.5 text-slate-200 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all cursor-pointer hover:border-slate-700 truncate font-mono"
            onChange={(e) => onSelect(e.target.value)}
            disabled={loading}
            >
            <option value="">-- Select Session --</option>
            {runs.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none text-slate-500 group-hover:text-blue-400 transition-colors">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l4-4 4 4m0 6l-4 4-4-4"></path></svg>
            </div>
        </div>

        <button
          onClick={handleClearAll}
          className={`px-3 rounded-md text-xs font-bold border transition-all duration-200 flex items-center justify-center ${confirmClear
              ? 'bg-red-500 text-white border-red-400 shadow-[0_0_10px_rgba(239,68,68,0.5)] w-16'
              : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700 hover:text-red-400 hover:border-slate-600 aspect-square'
            }`}
          title="Clear all experiments"
        >
          {confirmClear ? (
              <span className="animate-pulse">CONFIRM</span>
          ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
          )}
        </button>
      </div>

      {runs.length > 0 && (
          <div className="mt-2 text-[9px] text-slate-600 text-right font-mono">
              {runs.length} sessions stored
          </div>
      )}
    </div>
  );
};
