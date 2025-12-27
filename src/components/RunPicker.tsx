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
    <div className="p-4 border-b border-slate-700 bg-slate-800">
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs text-slate-400 font-bold uppercase">Select Run</label>
        <button
          onClick={refreshRuns}
          className="text-xs text-blue-400 hover:text-white"
          title="Refresh List"
        >
          ↻
        </button>
      </div>
      <div className="flex gap-2">
        <select
          className="flex-1 bg-slate-900 border border-slate-700 rounded text-sm p-2 text-slate-200 focus:outline-none focus:border-blue-500"
          onChange={(e) => onSelect(e.target.value)}
        >
          <option value="">-- Choose Run --</option>
          {runs.map(r => <option key={r} value={r}>{r}</option>)}
        </select>

        <button
          onClick={handleClearAll}
          className={`px-2 py-1 rounded text-xs font-bold border ${confirmClear
              ? 'bg-red-600 text-white border-red-500 animate-pulse'
              : 'bg-slate-700 text-slate-400 border-slate-600 hover:text-red-400'
            }`}
          title="Clear all experiments"
        >
          {confirmClear ? 'CONFIRM?' : '🗑'}
        </button>
      </div>
    </div>
  );
};