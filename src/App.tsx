import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { api } from './api/client';
import { VizDecision, VizTrade, UIAction, ContinuousData } from './types/viz';
import { RunPicker } from './components/RunPicker';
import { Navigator } from './components/Navigator';
import { CandleChart } from './components/CandleChart';
import { DetailsPanel } from './components/DetailsPanel';
import { ChatAgent } from './components/ChatAgent';
import { UnifiedReplayView } from './components/UnifiedReplayView';
import { StatsPanel } from './components/StatsPanel';
import { LabPage } from './components/LabPage';

type PageType = 'trade' | 'lab';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageType>('trade');
  const [currentRun, setCurrentRun] = useState<string>('');
  const [mode, setMode] = useState<'DECISION' | 'TRADE'>('DECISION');
  const [index, setIndex] = useState<number>(0);
  const [showRawData, setShowRawData] = useState<boolean>(false);
  const [showSimulation, setShowSimulation] = useState<boolean>(false);
  const [simulationMode, setSimulationMode] = useState<'SIMULATION' | 'YFINANCE'>('SIMULATION');


  const [continuousData, setContinuousData] = useState<ContinuousData | null>(null);
  const [continuousLoading, setContinuousLoading] = useState<boolean>(true);

  const [decisions, setDecisions] = useState<VizDecision[]>([]);
  const [trades, setTrades] = useState<VizTrade[]>([]);

  // Load continuous contract data - reload when decisions change to match date range
  useEffect(() => {
    setContinuousLoading(true);

    // Calculate date range from decisions if available
    let startDate: string | undefined;
    let endDate: string | undefined;

    if (decisions.length > 0) {
      const timestamps = decisions
        .map(d => d.timestamp)
        .filter((t): t is string => !!t)
        .sort();

      if (timestamps.length > 0) {
        startDate = timestamps[0];
        endDate = timestamps[timestamps.length - 1];
      }
    }

    api.getContinuousContract(startDate, endDate).then(data => {
      setContinuousData(data);
      setContinuousLoading(false);
    }).catch(err => {
      console.error('Failed to load continuous data:', err);
      setContinuousLoading(false);
    });
  }, [decisions]);

  // Load run-specific data
  useEffect(() => {
    if (currentRun) {
      Promise.all([
        api.getDecisions(currentRun),
        api.getTrades(currentRun)
      ]).then(([d, t]) => {
        setDecisions(d);
        setTrades(t);
        setIndex(0); // Reset index on run change
      });
    }
  }, [currentRun]);

  // Derived State
  const activeDecision = useMemo(() => {
    if (mode === 'DECISION') {
      return decisions.find(d => d.index === index) || decisions[index] || null;
    } else {
      const trade = trades.find(t => t.index === index);
      return trade ? decisions.find(d => d.decision_id === trade.decision_id) || null : null;
    }
  }, [mode, index, decisions, trades]);

  const activeTrade = useMemo(() => {
    if (mode === 'TRADE') {
      return trades.find(t => t.index === index) || null;
    } else {
      // Try to find a trade linked to this decision
      return activeDecision ? trades.find(t => t.decision_id === activeDecision.decision_id) || null : null;
    }
  }, [mode, index, trades, activeDecision]);

  // Agent Action Handler
  const handleAgentAction = async (action: UIAction) => {
    switch (action.type) {
      case 'SET_INDEX':
        setIndex(action.payload);
        break;
      case 'SET_MODE':
        setMode(action.payload);
        setIndex(0);
        break;
      case 'LOAD_RUN':
        setCurrentRun(action.payload);
        break;
      case 'RUN_STRATEGY':
        try {
          // Notify user
          console.log("Running strategy...", action.payload);
          const result = await api.runStrategy(action.payload);
          if (result.success && result.run_id) {
            setCurrentRun(result.run_id);
            // Optionally switch to Decision mode to see results
            setMode('DECISION');
          } else {
            console.error("Strategy run failed:", result.error);
          }
        } catch (e) {
          console.error('Failed to run strategy:', e);
        }
        break;

      case 'START_REPLAY':
        setSimulationMode('SIMULATION');
        setShowSimulation(true);
        break;

      case 'TRAIN_FROM_SCAN':
        try {
          console.log("Training from scan...", action.payload);
          // We need to add this method to client.ts first, but for now we can fetch directly or ignore
          // Assuming api.trainFromScan exists or we add it. 
          // For now, let's just log it.
          alert("Training started in background (check console)");
        } catch (e) {
          console.error(e);
        }
        break;

      default:
        console.warn('Unknown action:', action);
    }
  };

  const maxIndex = mode === 'DECISION' ? decisions.length - 1 : trades.length - 1;

  // If Lab page is active, render it instead
  if (currentPage === 'lab') {
    return (
      <div className="flex flex-col h-screen w-full bg-slate-900">
        {/* Page Navigation */}
        <div className="h-12 flex items-center gap-4 px-4 bg-slate-800 border-b border-slate-700">
          <button
            onClick={() => setCurrentPage('trade')}
            className="text-slate-400 hover:text-white px-3 py-1"
          >
            Trade View
          </button>
          <button
            onClick={() => setCurrentPage('lab')}
            className="text-white bg-blue-600 px-3 py-1 rounded"
          >
            🔬 Lab
          </button>
        </div>
        <div className="flex-1">
          <LabPage
            onLoadRun={(runId: string) => {
              setCurrentRun(runId);
              setCurrentPage('trade');
            }}
          />
        </div>
      </div>
    );
  }

  // Trade View (default)
  return (
    <div className="flex h-screen w-full bg-slate-900 overflow-hidden">

      {/* LEFT SIDEBAR */}
      <div className="w-80 flex flex-col border-r border-slate-700 bg-slate-800">
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <h1 className="font-bold text-white text-lg">Trade Viz</h1>
            <button
              onClick={() => setCurrentPage('lab')}
              className="bg-green-600 hover:bg-green-500 text-white text-xs px-2 py-1 rounded"
            >
              🔬 Lab
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setSimulationMode('YFINANCE');
                setShowSimulation(true);
              }}
              className="bg-red-600 hover:bg-red-500 text-white text-xs px-2 py-1 rounded animate-pulse font-bold"
              title="Open Live Trading Dashboard"
            >
              🔴 LIVE
            </button>
            <button
              onClick={() => {
                setSimulationMode('SIMULATION');
                setShowSimulation(true);
              }}
              className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1 rounded"
            >
              ▶ Replay
            </button>
          </div>
        </div>

        <RunPicker onSelect={setCurrentRun} />

        <Navigator
          mode={mode}
          setMode={setMode}
          index={index}
          setIndex={setIndex}
          maxIndex={Math.max(0, maxIndex)}
        />



        <div className="flex-1 overflow-auto p-2">
          {continuousLoading ? (
            <div className="p-4 text-sm text-slate-400 text-center">
              <p>Loading market data...</p>
            </div>
          ) : !currentRun ? (
            <div className="p-4 text-sm text-slate-400 text-center">
              <p>Select a run above to load overlays.</p>
              <p className="mt-2 text-xs text-slate-500">
                Chart shows continuous contract data.
              </p>
            </div>
          ) : (
            <div className="p-4 text-xs text-slate-500 text-center">
              <div>📊 {continuousData?.count?.toLocaleString() || 0} bars loaded</div>
              <div className="mt-1">📍 {decisions.length} decisions, {trades.length} trades</div>
            </div>
          )}

          {/* Raw Data Toggle */}
          <div className="mt-2 px-2">
            <button
              onClick={() => setShowRawData(!showRawData)}
              className="w-full text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 py-2 px-3 rounded"
            >
              {showRawData ? 'Hide Raw Data' : 'Show Raw Data'}
            </button>
          </div>
        </div>

        {/* CHAT AGENT (Bottom Left) */}
        <div className="h-1/3 min-h-[300px]">
          <ChatAgent
            runId={currentRun || 'none'}
            currentIndex={index}
            currentMode={mode}
            onAction={handleAgentAction}
          />
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* STATS PANEL - Shows scan results: P&L, win rate, drawdown etc. */}
        <StatsPanel decisions={decisions} startingBalance={50000} />

        {/* CHART AREA - Now always shows continuous data */}
        <div className="flex-1 relative bg-slate-900">
          <CandleChart
            continuousData={continuousData}
            decisions={decisions}
            activeDecision={activeDecision}
            trade={activeTrade}
            trades={trades}
            defaultShowAllTrades={true}
          />

          {/* Floating Info Overlay */}
          {activeDecision && (
            <div className="absolute top-4 left-4 bg-slate-800/80 backdrop-blur px-3 py-2 rounded border border-slate-700 text-xs shadow-lg pointer-events-none z-20">
              <div className="font-mono text-white">{activeDecision?.timestamp}</div>
              <div className="text-blue-400 font-bold">{activeDecision?.scanner_id || 'unknown'}</div>
              <div className="text-slate-400">Index: {activeDecision?.index}</div>
              {activeDecision?.scanner_context?.direction && (
                <div className={`font-bold ${activeDecision.scanner_context.direction === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>
                  {activeDecision.scanner_context.direction}
                </div>
              )}
            </div>
          )}
        </div>

        {/* RAW DATA PANEL or DETAILS PANEL */}
        <div className="h-48 border-t border-slate-700 bg-slate-800 overflow-auto flex-shrink-0">
          {showRawData ? (
            <div className="p-4">
              <h3 className="text-xs font-bold text-slate-400 uppercase mb-2">Raw Decision Data</h3>
              <pre className="text-xs text-green-400 bg-slate-900 p-3 rounded overflow-auto max-h-56 font-mono">
                {activeDecision ? JSON.stringify(activeDecision, null, 2) : 'No data selected'}
              </pre>
            </div>
          ) : (
            <DetailsPanel decision={activeDecision} trade={activeTrade} />
          )}
        </div>

      </div>

      {/* UNIFIED REPLAY OVERLAY */}
      {showSimulation && (
        <UnifiedReplayView
          onClose={() => setShowSimulation(false)}
          runId={currentRun}
          initialMode={simulationMode}
          lastTradeTimestamp={
            // Use last decision timestamp (decisions always exist, trades may be empty)
            decisions.length > 0
              ? decisions[decisions.length - 1].timestamp || undefined
              : undefined
          }
        />
      )}

    </div>
  );
};

export default App;