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
    <div className="bg-slate-900/40 rounded-lg p-4 border border-slate-800/60 backdrop-blur-sm">
      
      {/* Mode Toggle */}
      <div className="flex bg-slate-950 rounded-lg p-1.5 mb-5 shadow-inner border border-slate-800">
        <button 
          onClick={() => setMode('DECISION')}
          className={`flex-1 text-[11px] uppercase tracking-wider py-2 rounded-md font-bold transition-all duration-200 ${mode === 'DECISION' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-500 hover:text-slate-300'}`}
        >
          Decisions
        </button>
        <button 
          onClick={() => setMode('TRADE')}
          className={`flex-1 text-[11px] uppercase tracking-wider py-2 rounded-md font-bold transition-all duration-200 ${mode === 'TRADE' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-500 hover:text-slate-300'}`}
        >
          Trades
        </button>
      </div>

      {/* Navigation Controls */}
      <div>
        <div className="flex justify-between items-center mb-3">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            Index <span className="text-white font-mono bg-slate-800 px-1.5 py-0.5 rounded ml-1">{index}</span> <span className="text-slate-600">/</span> <span className="font-mono text-slate-400">{maxIndex}</span>
          </span>
        </div>
        
        <div className="flex gap-2 mb-4">
          <button 
            onClick={() => setIndex(Math.max(0, index - 1))}
            className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-200 py-2 rounded-md text-xs font-bold border border-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:border-slate-600 active:transform active:scale-95"
            disabled={index <= 0}
          >
            ← Prev
          </button>
          <button 
            onClick={() => setIndex(Math.min(maxIndex, index + 1))}
            className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-200 py-2 rounded-md text-xs font-bold border border-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:border-slate-600 active:transform active:scale-95"
            disabled={index >= maxIndex}
          >
            Next →
          </button>
        </div>

        <div className="relative h-4 flex items-center mb-2">
            <input
              type="range"
              min="0"
              max={maxIndex}
              value={index}
              onChange={(e) => setIndex(parseInt(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500 hover:accent-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500/30"
            />
        </div>
      </div>

      <div className="relative group">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <svg className="h-3 w-3 text-slate-500 group-focus-within:text-blue-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input 
            type="text" 
            placeholder="Search ID..." 
            className="w-full bg-slate-950 border border-slate-800 rounded-md pl-8 pr-2 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all"
          />
      </div>

    </div>
  );
};
