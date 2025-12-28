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
import ExperimentsView from './components/ExperimentsView';
import { IndicatorSettingsPanel } from './components/IndicatorSettings';
import { DEFAULT_INDICATOR_SETTINGS, type IndicatorSettings } from './features/chart_indicators';

type PageType = 'trade' | 'lab' | 'experiments';

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

  // Indicator Settings State
  const [indicatorSettings, setIndicatorSettings] = useState<IndicatorSettings>(DEFAULT_INDICATOR_SETTINGS);

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
      <div className="flex flex-col h-screen w-full bg-slate-900 overflow-hidden text-slate-100 font-sans">
        <div className="h-14 flex items-center gap-4 px-6 bg-slate-850/80 backdrop-blur border-b border-slate-800 shrink-0">
          <button
            onClick={() => setCurrentPage('trade')}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors px-3 py-1.5 rounded-md hover:bg-slate-800"
          >
            <span>←</span> Back to Trade View
          </button>
        </div>
        <div className="flex-1 overflow-hidden min-h-0 bg-slate-900">
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

  // If Experiments page is active
  if (currentPage === 'experiments') {
    return (
      <div className="flex flex-col h-screen w-full bg-slate-900 overflow-hidden text-slate-100 font-sans">
        <div className="h-14 flex items-center gap-4 px-6 bg-slate-850/80 backdrop-blur border-b border-slate-800 shrink-0">
          <button
            onClick={() => setCurrentPage('trade')}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors px-3 py-1.5 rounded-md hover:bg-slate-800"
          >
            <span>←</span> Back to Trade View
          </button>
        </div>
        <div className="flex-1 overflow-hidden min-h-0 bg-slate-900">
          <ExperimentsView
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
    <div className="flex h-screen w-full bg-slate-900 text-slate-100 font-sans overflow-hidden">

      {/* LEFT SIDEBAR - Expanded Width */}
      <div className="w-96 flex flex-col border-r border-slate-800 bg-slate-950 shrink-0 shadow-xl z-20">

        {/* Header */}
        <div className="h-16 flex items-center justify-between px-6 border-b border-slate-800 shrink-0 bg-slate-950">
          <div className="flex items-center gap-4">
            <h1 className="font-bold text-white text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-400">Trade Viz</h1>
            <div className="flex gap-2">
              <button
                onClick={() => setCurrentPage('lab')}
                className="text-slate-400 hover:text-green-400 hover:bg-green-400/10 transition-all p-2 rounded-md"
                title="Lab"
              >
                🔬
              </button>
              <button
                onClick={() => setCurrentPage('experiments')}
                className="text-slate-400 hover:text-blue-400 hover:bg-blue-400/10 transition-all p-2 rounded-md"
                title="Experiments"
              >
                🧪
              </button>
            </div>
          </div>
          <button
            onClick={() => {
              setSimulationMode('SIMULATION');
              setShowSimulation(true);
            }}
            className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-lg shadow-purple-900/20 transition-all transform hover:scale-105"
            title="Replay"
          >
            ▶
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-6 custom-scrollbar">

          <RunPicker onSelect={setCurrentRun} />

          <Navigator
            mode={mode}
            setMode={setMode}
            index={index}
            setIndex={setIndex}
            maxIndex={Math.max(0, maxIndex)}
          />

          {!currentRun ? (
            <div className="p-8 text-sm text-slate-500 text-center border border-dashed border-slate-800 rounded-lg bg-slate-900/50">
              <p>Select a run above to see details.</p>
            </div>
          ) : (
            <>
              {/* Details Panel - Stays in Sidebar */}
              <div className="border border-slate-800 rounded-lg overflow-hidden shadow-sm bg-slate-900/50">
                <div className="bg-slate-800/50 px-4 py-2 text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                  Decision Context
                </div>
                <div className="bg-slate-900/30">
                  <DetailsPanel decision={activeDecision} trade={activeTrade} />
                </div>
              </div>

              <div className="flex justify-between items-center text-[10px] text-slate-600 px-2 font-mono uppercase tracking-wider">
                <span>📊 {continuousData?.count?.toLocaleString() || 0} bars</span>
                <span>📍 {decisions.length} decisions / {trades.length} trades</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* MAIN CONTENT - Vertical Layout */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative bg-slate-900">

        {/* Stats Panel (Moved Back to Top of Main Content) */}
        {currentRun && (
          <div className="shrink-0 border-b border-slate-800 bg-slate-900 z-10 shadow-sm">
            <StatsPanel decisions={decisions} startingBalance={50000} />
          </div>
        )}

        {/* Chart Top (Flex Grow) */}
        <div className="flex-1 min-h-0 relative bg-slate-900">
          {/* Indicator Settings - Top LEFT of chart */}
          <div className="absolute top-2 left-2 z-30">
            <IndicatorSettingsPanel settings={indicatorSettings} onChange={setIndicatorSettings} />
          </div>

          <CandleChart
            continuousData={continuousData}
            decisions={decisions}
            activeDecision={activeDecision}
            trade={activeTrade}
            trades={trades}
            indicatorSettings={indicatorSettings}
          />
        </div>

        {/* Resizer Handle */}
        <div
          className="h-1 bg-slate-950 hover:bg-blue-500/50 cursor-row-resize shrink-0 flex items-center justify-center transition-all duration-300 border-y border-slate-800 relative group z-30"
          onMouseDown={startResizing}
        >
          <div className="w-16 h-1 bg-slate-700 rounded-full group-hover:bg-blue-400 transition-colors opacity-50 group-hover:opacity-100" />
        </div>

        {/* Chat Bottom (Fixed Height) */}
        <div style={{ height: chatHeight }} className="shrink-0 bg-slate-950 border-t border-slate-800 shadow-[0_-4px_20px_-5px_rgba(0,0,0,0.3)] z-20">
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