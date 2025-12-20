import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { VizDecision, ContinuousData, UIAction } from '../types/viz';
import { RunPicker } from '../components/RunPicker';
import { CandleChart } from '../components/CandleChart';
import { ChatAgent } from '../components/ChatAgent';

/**
 * SCAN Lane - Read-only analysis and strategy discovery
 * 
 * Features:
 * - Analyze patterns in historical data
 * - Chat with agent to explore signals
 * - Run new strategies (creates data, switches to TRAIN mode)
 * - No live trading, no learning - pure analysis
 */
export const ScanPage: React.FC = () => {
  const [currentRun, setCurrentRun] = useState<string>('');
  const [decisions, setDecisions] = useState<VizDecision[]>([]);
  const [continuousData, setContinuousData] = useState<ContinuousData | null>(null);
  const [index, setIndex] = useState<number>(0);

  // Load continuous contract data
  useEffect(() => {
    if (decisions.length > 0) {
      const timestamps = decisions
        .map(d => d.timestamp)
        .filter((t): t is string => !!t)
        .sort();

      if (timestamps.length > 0) {
        const startDate = timestamps[0];
        const endDate = timestamps[timestamps.length - 1];
        
        api.getContinuousContract(startDate, endDate).then(data => {
          setContinuousData(data);
        }).catch(err => {
          console.error('Failed to load continuous data:', err);
        });
      }
    }
  }, [decisions]);

  // Load run data
  useEffect(() => {
    if (currentRun) {
      api.getDecisions(currentRun).then(d => {
        setDecisions(d);
        setIndex(0);
      });
    }
  }, [currentRun]);

  const activeDecision = decisions[index] || null;

  const handleAgentAction = async (action: UIAction) => {
    switch (action.type) {
      case 'SET_INDEX':
        setIndex(action.payload);
        break;
      case 'LOAD_RUN':
        setCurrentRun(action.payload);
        break;
      case 'RUN_STRATEGY':
        try {
          const result = await api.runStrategy(action.payload);
          if (result.success && result.run_id) {
            setCurrentRun(result.run_id);
          }
        } catch (e) {
          console.error('Failed to run strategy:', e);
        }
        break;
      default:
        console.warn('Unknown action:', action);
    }
  };

  return (
    <div className="flex h-full w-full bg-slate-900 overflow-hidden">
      {/* LEFT SIDEBAR */}
      <div className="w-80 flex flex-col border-r border-slate-700 bg-slate-800">
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-700">
          <div>
            <h1 className="font-bold text-white text-lg">SCAN Mode</h1>
            <p className="text-xs text-slate-400">Strategy Discovery</p>
          </div>
        </div>

        <RunPicker onSelect={setCurrentRun} />

        {/* Navigation */}
        <div className="p-4 border-b border-slate-700">
          <div className="flex items-center justify-between mb-2">
            <button
              onClick={() => setIndex(Math.max(0, index - 1))}
              disabled={index <= 0}
              className="bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 text-white px-3 py-1 rounded text-sm"
            >
              ← Prev
            </button>
            <span className="text-xs text-slate-400">
              {index + 1} / {decisions.length}
            </span>
            <button
              onClick={() => setIndex(Math.min(decisions.length - 1, index + 1))}
              disabled={index >= decisions.length - 1}
              className="bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 text-white px-3 py-1 rounded text-sm"
            >
              Next →
            </button>
          </div>
        </div>

        {/* Scanner Stats */}
        <div className="p-4 border-b border-slate-700">
          <div className="text-xs text-slate-400 mb-2">Scanner Analysis</div>
          {decisions.length > 0 ? (
            <div className="space-y-2">
              <div className="bg-slate-900 rounded p-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">Signals:</span>
                  <span className="text-white">{decisions.length}</span>
                </div>
              </div>
              {activeDecision && (
                <div className="bg-slate-900 rounded p-2 text-xs">
                  <div className="text-blue-400 mb-1">{activeDecision.scanner_id}</div>
                  <div className="text-slate-400 text-xs">
                    {activeDecision.timestamp?.slice(0, 19) || '-'}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-xs text-slate-500">
              No signals loaded. Run a strategy or select a run.
            </div>
          )}
        </div>

        {/* CHAT AGENT */}
        <div className="flex-1 min-h-0">
          <ChatAgent
            runId={currentRun || 'none'}
            currentIndex={index}
            currentMode="DECISION"
            onAction={handleAgentAction}
          />
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="flex-1 flex flex-col">
        {/* CHART AREA */}
        <div className="flex-1 relative bg-slate-900">
          <CandleChart
            continuousData={continuousData}
            decisions={decisions}
            activeDecision={activeDecision}
            trade={null}
            trades={[]}
          />
          
          {/* Info Overlay */}
          {activeDecision && (
            <div className="absolute top-4 left-4 bg-slate-800/80 backdrop-blur px-3 py-2 rounded border border-slate-700 text-xs shadow-lg pointer-events-none z-20">
              <div className="font-mono text-white">{activeDecision?.timestamp}</div>
              <div className="text-blue-400 font-bold">{activeDecision?.scanner_id || 'unknown'}</div>
              <div className="text-slate-400">Signal {activeDecision?.index + 1}</div>
            </div>
          )}
        </div>

        {/* SIGNAL DETAILS */}
        <div className="h-72 border-t border-slate-700 bg-slate-800 overflow-auto p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase mb-2">Signal Details</h3>
          {activeDecision ? (
            <div className="space-y-3">
              <div className="bg-slate-900 rounded p-3">
                <div className="text-xs text-slate-400 mb-2">Scanner Context</div>
                <pre className="text-xs text-green-400 font-mono">
                  {JSON.stringify(activeDecision.scanner_context, null, 2)}
                </pre>
              </div>
              
              <div className="bg-slate-900 rounded p-3">
                <div className="text-xs text-slate-400 mb-2">Market State</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Price:</span>
                    <span className="text-white">${activeDecision.current_price.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">ATR:</span>
                    <span className="text-white">{activeDecision.atr.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Action:</span>
                    <span className="text-white">{activeDecision.action}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-500 text-center py-8">
              No signal selected
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
