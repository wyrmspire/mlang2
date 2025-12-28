import React from 'react';

interface NavigatorProps {
  mode: 'DECISION' | 'TRADE';
  setMode: (m: 'DECISION' | 'TRADE') => void;
  index: number;
  setIndex: (i: number) => void;
  maxIndex: number;
}

export const Navigator: React.FC<NavigatorProps> = ({ mode, setMode, index, setIndex, maxIndex }) => {
  return (
    <div className="bg-slate-900/40 rounded-lg p-1 border border-slate-800/60 backdrop-blur-sm">
      
      {/* Mode Toggle */}
      <div className="flex bg-slate-950/80 rounded-lg p-1 mb-4 shadow-inner">
        <button 
          onClick={() => setMode('DECISION')}
          className={`flex-1 flex flex-col items-center justify-center py-2.5 rounded-md transition-all duration-300 ${
            mode === 'DECISION'
              ? 'bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-900/20'
              : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
          }`}
        >
          <span className="text-[10px] font-bold uppercase tracking-widest">Decisions</span>
          <span className="text-[9px] opacity-60 font-mono mt-0.5">Scan Points</span>
        </button>
        <button 
          onClick={() => setMode('TRADE')}
          className={`flex-1 flex flex-col items-center justify-center py-2.5 rounded-md transition-all duration-300 ${
             mode === 'TRADE'
              ? 'bg-gradient-to-br from-emerald-600 to-teal-600 text-white shadow-md shadow-emerald-900/20'
              : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
          }`}
        >
          <span className="text-[10px] font-bold uppercase tracking-widest">Trades</span>
          <span className="text-[9px] opacity-60 font-mono mt-0.5">Executions</span>
        </button>
      </div>

      {/* Navigation Controls */}
      <div className="px-3 pb-3">
        <div className="flex justify-between items-end mb-3">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
             {mode === 'DECISION' ? 'Scan Index' : 'Trade Index'}
          </span>
          <div className="font-mono text-xs">
            <span className="text-white font-bold">{index}</span>
            <span className="text-slate-600 mx-1">/</span>
            <span className="text-slate-500">{maxIndex}</span>
          </div>
        </div>
        
        <div className="flex gap-2 mb-4">
          <button 
            onClick={() => setIndex(Math.max(0, index - 1))}
            disabled={index <= 0}
            className="flex-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:hover:bg-slate-800 text-slate-300 py-2 rounded-md border border-slate-700 hover:border-slate-600 transition-all active:scale-95 group relative overflow-hidden"
          >
             <span className="relative z-10 flex items-center justify-center gap-1 text-xs font-bold">
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
                Prev
             </span>
          </button>
          <button 
            onClick={() => setIndex(Math.min(maxIndex, index + 1))}
            disabled={index >= maxIndex}
            className="flex-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:hover:bg-slate-800 text-slate-300 py-2 rounded-md border border-slate-700 hover:border-slate-600 transition-all active:scale-95 group relative overflow-hidden"
          >
             <span className="relative z-10 flex items-center justify-center gap-1 text-xs font-bold">
                Next
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
             </span>
          </button>
        </div>

        <div className="relative h-6 flex items-center mb-2 group">
            <div className="absolute w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                   className="h-full bg-blue-500 transition-all duration-100"
                   style={{ width: `${(index / (maxIndex || 1)) * 100}%` }}
                ></div>
            </div>
            <input
              type="range"
              min="0"
              max={maxIndex}
              value={index}
              onChange={(e) => setIndex(parseInt(e.target.value))}
              className="absolute w-full h-6 opacity-0 cursor-pointer z-10"
            />
            <div
              className="absolute w-3 h-3 bg-white rounded-full shadow-lg pointer-events-none transition-all duration-100 transform -translate-x-1.5 border border-slate-300 group-hover:scale-125"
              style={{ left: `${(index / (maxIndex || 1)) * 100}%` }}
            ></div>
        </div>
      </div>

      <div className="px-3 pb-3">
          <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-3.5 w-3.5 text-slate-600 group-focus-within:text-blue-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                type="text"
                placeholder="Jump to ID..."
                className="w-full bg-slate-950 border border-slate-800 rounded-md pl-9 pr-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all"
              />
          </div>
      </div>

    </div>
  );
};
