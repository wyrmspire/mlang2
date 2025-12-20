import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import { VizDecision, VizTrade, ContinuousData } from '../types/viz';
import { RunPicker } from '../components/RunPicker';
import { CandleChart } from '../components/CandleChart';
import { DetailsPanel } from '../components/DetailsPanel';
import { ReplayControls } from '../components/ReplayControls';

/**
 * REPLAY Lane - Real-time simulation and playback
 * 
 * Features:
 * - Simulate model execution bar-by-bar
 * - Replay existing strategies with controls
 * - View OCO zones as they evolve
 * - Track simulated account state
 */
export const ReplayPage: React.FC = () => {
  const [currentRun, setCurrentRun] = useState<string>('');
  const [decisions, setDecisions] = useState<VizDecision[]>([]);
  const [trades, setTrades] = useState<VizTrade[]>([]);
  const [continuousData, setContinuousData] = useState<ContinuousData | null>(null);
  const [index, setIndex] = useState<number>(0);
  const [isReplaying, setIsReplaying] = useState<boolean>(false);

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
      Promise.all([
        api.getDecisions(currentRun),
        api.getTrades(currentRun)
      ]).then(([d, t]) => {
        setDecisions(d);
        setTrades(t);
        setIndex(0);
      });
    }
  }, [currentRun]);

  const activeDecision = decisions[index] || null;
  const activeTrade = activeDecision 
    ? trades.find(t => t.decision_id === activeDecision.decision_id) || null 
    : null;

  const handleReplayStart = useCallback(() => {
    setIsReplaying(true);
  }, []);

  const handleReplayEnd = useCallback(() => {
    setIsReplaying(false);
  }, []);

  return (
    <div className="flex h-full w-full bg-slate-900 overflow-hidden">
      {/* LEFT SIDEBAR */}
      <div className="w-80 flex flex-col border-r border-slate-700 bg-slate-800">
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-700">
          <div>
            <h1 className="font-bold text-white text-lg">REPLAY Mode</h1>
            <p className="text-xs text-slate-400">Real-time Simulation</p>
          </div>
        </div>

        <RunPicker onSelect={setCurrentRun} />

        {/* Replay Controls */}
        <div className="p-4 border-b border-slate-700">
          <ReplayControls
            maxIndex={decisions.length - 1}
            currentIndex={index}
            onIndexChange={setIndex}
            onReplayStart={handleReplayStart}
            onReplayEnd={handleReplayEnd}
          />
        </div>

        {/* Manual Navigation */}
        <div className="p-4 border-b border-slate-700">
          <div className="text-xs text-slate-400 mb-2">Manual Control</div>
          <div className="flex items-center justify-between">
            <button
              onClick={() => setIndex(Math.max(0, index - 1))}
              disabled={index <= 0 || isReplaying}
              className="bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 text-white px-3 py-1 rounded text-sm"
            >
              ← Step Back
            </button>
            <span className="text-xs text-slate-400">
              {index + 1} / {decisions.length}
            </span>
            <button
              onClick={() => setIndex(Math.min(decisions.length - 1, index + 1))}
              disabled={index >= decisions.length - 1 || isReplaying}
              className="bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 text-white px-3 py-1 rounded text-sm"
            >
              Step Forward →
            </button>
          </div>
        </div>

        {/* Replay Stats */}
        <div className="flex-1 overflow-auto p-4">
          {activeDecision && (
            <div className="space-y-2 text-sm">
              <div className="bg-slate-900 rounded p-3">
                <div className="text-xs text-slate-400 mb-2">Current State</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Time:</span>
                    <span className="text-white font-mono">
                      {activeDecision.timestamp?.slice(11, 19) || '-'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Price:</span>
                    <span className="text-white">
                      ${activeDecision.current_price.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">ATR:</span>
                    <span className="text-white">
                      {activeDecision.atr.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>

              {activeTrade && (
                <div className="bg-slate-900 rounded p-3">
                  <div className="text-xs text-slate-400 mb-2">Active Trade</div>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Direction:</span>
                      <span className={activeTrade.direction === 'LONG' ? 'text-green-400' : 'text-red-400'}>
                        {activeTrade.direction}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Entry:</span>
                      <span className="text-white">
                        ${activeTrade.entry_price.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">PnL:</span>
                      <span className={activeTrade.pnl_dollars >= 0 ? 'text-green-400' : 'text-red-400'}>
                        ${activeTrade.pnl_dollars.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
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
            trade={activeTrade}
            trades={trades}
          />
          
          {/* Replay Status Overlay */}
          {isReplaying && (
            <div className="absolute top-4 right-4 bg-green-900/80 backdrop-blur px-4 py-2 rounded border border-green-500 text-sm shadow-lg z-20">
              <div className="text-green-400 font-bold">▶ REPLAYING</div>
            </div>
          )}
        </div>

        {/* DETAILS PANEL */}
        <div className="h-72 border-t border-slate-700 bg-slate-800 overflow-auto">
          <DetailsPanel decision={activeDecision} trade={activeTrade} />
        </div>
      </div>
    </div>
  );
};
