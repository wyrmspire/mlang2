import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { VizDecision, VizTrade } from '../types/viz';
import { RunPicker } from '../components/RunPicker';
import { CandleChart } from '../components/CandleChart';
import { DetailsPanel } from '../components/DetailsPanel';

/**
 * TRAIN Lane - For training data analysis and model development
 * 
 * Features:
 * - View historical training runs
 * - Analyze decision distributions
 * - Inspect counterfactual labels
 * - Review model outputs and predictions
 */
export const TrainPage: React.FC = () => {
  const [currentRun, setCurrentRun] = useState<string>('');
  const [decisions, setDecisions] = useState<VizDecision[]>([]);
  const [trades, setTrades] = useState<VizTrade[]>([]);
  const [index, setIndex] = useState<number>(0);

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

  return (
    <div className="flex h-full w-full bg-slate-900 overflow-hidden">
      {/* LEFT SIDEBAR */}
      <div className="w-80 flex flex-col border-r border-slate-700 bg-slate-800">
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-700">
          <div>
            <h1 className="font-bold text-white text-lg">TRAIN Mode</h1>
            <p className="text-xs text-slate-400">Training Data Analysis</p>
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

        {/* Stats */}
        <div className="flex-1 overflow-auto p-4">
          <div className="space-y-2 text-sm">
            <div className="bg-slate-900 rounded p-3">
              <div className="text-xs text-slate-400 mb-2">Dataset Stats</div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-400">Decisions:</span>
                  <span className="text-white">{decisions.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Trades:</span>
                  <span className="text-white">{trades.length}</span>
                </div>
                {decisions.length > 0 && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Trade Rate:</span>
                    <span className="text-white">
                      {((trades.length / decisions.length) * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            </div>

            {activeDecision && (
              <div className="bg-slate-900 rounded p-3">
                <div className="text-xs text-slate-400 mb-2">Current Decision</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Scanner:</span>
                    <span className="text-blue-400">{activeDecision.scanner_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Action:</span>
                    <span className="text-white">{activeDecision.action}</span>
                  </div>
                  {activeDecision.skip_reason && (
                    <div className="flex justify-between">
                      <span className="text-slate-400">Skip Reason:</span>
                      <span className="text-yellow-400">{activeDecision.skip_reason}</span>
                    </div>
                  )}
                  {activeDecision.cf_outcome && (
                    <div className="flex justify-between">
                      <span className="text-slate-400">CF Outcome:</span>
                      <span className={activeDecision.cf_outcome === 'WIN' ? 'text-green-400' : 'text-red-400'}>
                        {activeDecision.cf_outcome}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="flex-1 flex flex-col">
        {/* CHART AREA */}
        <div className="flex-1 relative bg-slate-900">
          <CandleChart
            continuousData={null}
            decisions={decisions}
            activeDecision={activeDecision}
            trade={activeTrade}
            trades={trades}
          />
        </div>

        {/* DETAILS PANEL */}
        <div className="h-72 border-t border-slate-700 bg-slate-800 overflow-auto">
          <DetailsPanel decision={activeDecision} trade={activeTrade} />
        </div>
      </div>
    </div>
  );
};
