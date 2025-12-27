import React, { useState, useEffect, useMemo, useRef } from 'react';
import { api } from './api/client';
import { VizDecision, VizTrade, UIAction, ContinuousData } from './types/viz';
import { RunPicker } from './components/RunPicker';
import { Navigator } from './components/Navigator';
import { CandleChart } from './components/CandleChart';
import { DetailsPanel } from './components/DetailsPanel';
import { ChatAgent } from './components/ChatAgent';
import { LiveSessionView } from './components/LiveSessionView';
import { StatsPanel } from './components/StatsPanel';
import { LabPage } from './components/LabPage';

type PageType = 'trade' | 'lab';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageType>('trade');
  const [currentRun, setCurrentRun] = useState<string>('');
  const [mode, setMode] = useState<'DECISION' | 'TRADE'>('DECISION');
  const [index, setIndex] = useState<number>(0);
  const [showSimulation, setShowSimulation] = useState<boolean>(false);
  const [simulationMode, setSimulationMode] = useState<'SIMULATION' | 'YFINANCE'>('SIMULATION');

  const [continuousData, setContinuousData] = useState<ContinuousData | null>(null);
  const [continuousLoading, setContinuousLoading] = useState<boolean>(true);

  const [decisions, setDecisions] = useState<VizDecision[]>([]);
  const [trades, setTrades] = useState<VizTrade[]>([]);

  // Layout State
  const [chatHeight, setChatHeight] = useState<number>(300);
  const isResizingRef = useRef(false);

  // Load continuous contract data
  useEffect(() => {
    setContinuousLoading(true);

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
      return activeDecision ? trades.find(t => t.decision_id === activeDecision.decision_id) || null : null;
    }
  }, [mode, index, trades, activeDecision]);

  const maxIndex = mode === 'DECISION' ? decisions.length - 1 : trades.length - 1;

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
          console.log("Running strategy...", action.payload);
          const result = await api.runStrategy(action.payload);
          if (result.success && result.run_id) {
            setCurrentRun(result.run_id);
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
        alert("Training started in background (check console)");
        break;
      default:
        console.warn('Unknown action:', action);
    }
  };

  // Resizing Logic
  const startResizing = () => {
    isResizingRef.current = true;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', stopResizing);
    document.body.style.userSelect = 'none'; // Prevent selection while dragging
  };

  const stopResizing = () => {
    isResizingRef.current = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', stopResizing);
    document.body.style.userSelect = '';
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isResizingRef.current) return;
    const newHeight = window.innerHeight - e.clientY;
    // Constrain height (min 100px, max 80% of screen)
    const constrained = Math.max(100, Math.min(newHeight, window.innerHeight * 0.8));
    setChatHeight(constrained);
  };

  // If Lab page is active, render it instead
  if (currentPage === 'lab') {
    return (
      <div className="flex flex-col h-screen w-full bg-slate-900 overflow-hidden">
        <div className="h-12 flex items-center gap-4 px-4 bg-slate-800 border-b border-slate-700 shrink-0">
          <button
            onClick={() => setCurrentPage('trade')}
            className="text-slate-400 hover:text-white px-3 py-1"
          >
            Back to Trade View
          </button>
        </div>
        <div className="flex-1 overflow-hidden min-h-0">
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

      {/* LEFT SIDEBAR - Expanded Width */}
      <div className="w-96 flex flex-col border-r border-slate-700 bg-slate-800 shrink-0">

        {/* Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-700 shrink-0">
          <div className="flex items-center gap-3">
            <h1 className="font-bold text-white text-lg">Trade Viz</h1>
            <button
              onClick={() => setCurrentPage('lab')}
              className="bg-green-600 hover:bg-green-500 text-white text-xs px-2 py-1 rounded"
            >
              🔬 Lab
            </button>
          </div>
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

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-4">

          <RunPicker onSelect={setCurrentRun} />

          <Navigator
            mode={mode}
            setMode={setMode}
            index={index}
            setIndex={setIndex}
            maxIndex={Math.max(0, maxIndex)}
          />

          {!currentRun ? (
            <div className="p-4 text-sm text-slate-400 text-center border border-dashed border-slate-700 rounded">
              <p>Select a run above to see details.</p>
            </div>
          ) : (
            <>
              {/* Details Panel - Stays in Sidebar */}
              <div className="border border-slate-700 rounded overflow-hidden">
                <div className="bg-slate-700 px-3 py-1 text-xs font-bold text-slate-300 uppercase">
                  Decision Context
                </div>
                <div className="bg-slate-900">
                  <DetailsPanel decision={activeDecision} trade={activeTrade} />
                </div>
              </div>

              <div className="text-xs text-slate-500 text-center mt-2">
                📊 {continuousData?.count?.toLocaleString() || 0} bars loaded<br />
                📍 {decisions.length} decisions, {trades.length} trades
              </div>
            </>
          )}
        </div>
      </div>

      {/* MAIN CONTENT - Vertical Layout */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative">

        {/* Stats Panel (Moved Back to Top of Main Content) */}
        {currentRun && (
          <div className="shrink-0 border-b border-slate-700 bg-slate-800">
            <StatsPanel decisions={decisions} startingBalance={50000} />
          </div>
        )}

        {/* Chart Top (Flex Grow) */}
        <div className="flex-1 min-h-0 relative bg-slate-900">
          <CandleChart
            continuousData={continuousData}
            decisions={decisions}
            activeDecision={activeDecision}
            trade={activeTrade}
            trades={trades}
          />

          {/* Floating Info Overlay (Over Chart) */}
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

        {/* Resizer Handle */}
        <div
          className="h-2 bg-slate-800 hover:bg-blue-600 cursor-row-resize shrink-0 flex items-center justify-center transition-colors border-y border-slate-700"
          onMouseDown={startResizing}
        >
          <div className="w-12 h-1 bg-slate-600 rounded-full" />
        </div>

        {/* Chat Bottom (Fixed Height) */}
        <div style={{ height: chatHeight }} className="shrink-0 bg-slate-800">
          <ChatAgent
            runId={currentRun || 'none'}
            currentIndex={index}
            currentMode={mode}
            onAction={handleAgentAction}
          />
        </div>

      </div>

      {/* UNIFIED REPLAY OVERLAY */}
      {showSimulation && (
        <LiveSessionView
          onClose={() => setShowSimulation(false)}
          runId={currentRun}
          initialMode={simulationMode}
          lastTradeTimestamp={
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