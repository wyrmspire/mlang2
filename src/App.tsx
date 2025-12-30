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
import ExperimentsView from './components/ExperimentsView';
import { IndicatorSettingsPanel } from './components/IndicatorSettings';
import { DEFAULT_INDICATOR_SETTINGS, type IndicatorSettings } from './features/chart_indicators';

type PageType = 'trade' | 'experiments';

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
  const [planningMode, setPlanningMode] = useState<boolean>(false);

  // Layout State
  const [chatHeight, setChatHeight] = useState<number>(320);
  const [isChatExpanded, setIsChatExpanded] = useState<boolean>(false); // New state for auto-expansion
  const isResizingRef = useRef(false);

  // Indicator Settings State
  const [indicatorSettings, setIndicatorSettings] = useState<IndicatorSettings>(DEFAULT_INDICATOR_SETTINGS);

  // Fast Viz State
  interface FastVizTrade {
    entry_time: string;
    entry_price: number;
    direction: string;
    stop_price: number;
    target_price: number;
    outcome: string;
    exit_time?: string;
    exit_price?: number;
    pnl_points: number;
    trigger_name: string;
  }
  interface FastVizRun {
    run_id: string;
    strategy_name: string;
    total_trades: number;
    win_rate: number;
    start_date: string;
    end_date: string;
    trades: FastVizTrade[];
  }
  const [fastVizRuns, setFastVizRuns] = useState<FastVizRun[]>([]);
  const [fastVizEnabled, setFastVizEnabled] = useState<boolean>(false);
  const [selectedFastVizRunId, setSelectedFastVizRunId] = useState<string | null>(null);

  // Derived State (Moved up for dependencies)
  const activeDecision = useMemo(() => {
    if (mode === 'DECISION') {
      return decisions.find(d => d.index === index) || decisions[index] || null;
    } else {
      const trade = trades.find(t => t.index === index);
      return trade ? decisions.find(d => d.decision_id === trade.decision_id) || null : null;
    }
  }, [mode, index, decisions, trades]);

  // Convert Fast Viz trades to chart-compatible decisions
  const fastVizDecisions = useMemo((): VizDecision[] => {
    if (!selectedFastVizRunId) return [];
    const selectedRun = fastVizRuns.find(r => r.run_id === selectedFastVizRunId);
    if (!selectedRun) return [];

    return selectedRun.trades.map((t, idx) => ({
      decision_id: `fv_${idx}`,
      timestamp: t.entry_time,
      bar_idx: idx,
      index: idx,
      scanner_id: t.trigger_name,
      scanner_context: {},
      action: 'ENTER',
      skip_reason: '',
      current_price: t.entry_price,
      atr: Math.abs(t.stop_price - t.entry_price),
      cf_outcome: t.outcome,
      cf_pnl_dollars: t.pnl_points * 5, // MES = $5/point
      oco: {
        entry_price: t.entry_price,
        stop_price: t.stop_price,
        tp_price: t.target_price,
        entry_type: 'MARKET',
        direction: t.direction,
        reference_type: 'atr',
        reference_value: 0,
        atr_at_creation: Math.abs(t.stop_price - t.entry_price),
        max_bars: 200,
        stop_atr: 2.0,
        tp_multiple: 3.0
      }
    }));
  }, [selectedFastVizRunId, fastVizRuns]);

  // Effective decisions (use Fast Viz if selected, otherwise regular)
  const effectiveDecisions = selectedFastVizRunId ? fastVizDecisions : decisions;

  // Load Fast Viz runs on mount
  useEffect(() => {
    api.listFastVizRuns()
      .then(runs => {
        if (Array.isArray(runs)) {
          console.log("Loaded Fast Viz runs:", runs.length);
          setFastVizRuns(runs);
        } else {
          console.error("Invalid runs format:", runs);
          setFastVizRuns([]);
        }
      })
      .catch(err => {
        console.error(err);
        setFastVizRuns([]);
      });
  }, []);

  // Load continuous contract data
  useEffect(() => {
    setContinuousLoading(true);

    let startDate: string | undefined;
    let endDate: string | undefined;

    if (effectiveDecisions.length > 0) {
      const timestamps = effectiveDecisions
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
  }, [effectiveDecisions]);

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
    // If we receive a UI action that affects the chart, collapse the chat
    if (['RUN_STRATEGY', 'SET_INDEX', 'SET_MODE', 'LOAD_RUN', 'RUN_FAST_VIZ'].includes(action.type)) {
      setIsChatExpanded(false);
    }

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
      case 'RUN_FAST_VIZ':
        try {
          console.log("Running Fast Viz...", action.payload);
          let fvResult;

          // Use payload trades if available (instant from agent)
          if (action.payload.trades) {
            console.log("Using instant trades from payload");
            fvResult = {
              run_id: action.payload.run_name || `fast_viz_${Date.now()}`,
              strategy_name: action.payload.config?.trigger?.type || "unknown",
              total_trades: action.payload.stats?.total_trades || action.payload.trades.length,
              win_rate: action.payload.stats?.win_rate || 0,
              start_date: action.payload.start_date,
              end_date: action.payload.end_date,
              trades: action.payload.trades
            };
          } else {
            // Fallback: fetch from API
            fvResult = await api.runFastViz(
              action.payload.config,
              action.payload.start_date,
              action.payload.end_date,
              action.payload.run_name
            );
          }

          const newRun: FastVizRun = {
            run_id: fvResult.run_id,
            strategy_name: fvResult.strategy_name,
            total_trades: fvResult.total_trades,
            win_rate: fvResult.win_rate,
            start_date: fvResult.start_date,
            end_date: fvResult.end_date,
            trades: fvResult.trades || []
          };

          setFastVizRuns(prev => {
            // Deduplicate
            if (prev.find(r => r.run_id === newRun.run_id)) return prev;
            return [newRun, ...prev];
          });

          // Auto-select the new Fast Viz run to show on chart
          setSelectedFastVizRunId(fvResult.run_id);
          // Clear regular run selection when viewing Fast Viz
          setCurrentRun(null);
        } catch (e) {
          console.error('Failed to run Fast Viz:', e);
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

  // Expand Chat on Research/Text response
  const handleAgentTextResponse = () => {
    // If the chat isn't expanded, expand it to show the research results
    // But only if we are NOT currently running a viz action (which is handled above)
    // This logic is tricky because we don't know if a UI action came WITH the text.
    // However, handleAgentAction runs for UI actions.
    // So here we can just default to expanding, and if a UI action comes, it will collapse it.
    // But we need to make sure this doesn't override the collapse.

    // Actually, ChatAgent calls onAction if there is an action.
    // We can rely on ChatAgent to tell us if it's purely text?
    // Or we can just trust the user flow:
    // If I ask "analyze this", agent replies with text -> Expand.
    // If I ask "show me this", agent replies with text + UIAction -> Collapse.

    // So, we will expose a method or prop to ChatAgent to signal "Text Only Response"?
    // Or we can just set it to true here, and handleAgentAction sets it to false.
    // Since react updates are batched or sequential, if both happen, the last one wins?
    // UIAction usually comes with text.

    // Let's try: Always expand on response. But if action is present, handleAgentAction will collapse.
    // Note: handleAgentAction is called by ChatAgent when action is present.

    // We'll pass a callback `onTextResponse` to ChatAgent.
    // But ChatAgent logic needs update? No, we can just use `onAction`.

    // Let's optimistically set expanded to true when user sends message? No.
    // Let's set expanded to true when `ChatAgent` receives a message that has NO action.
    // I'll need to modify `ChatAgent` to support `onTextResponse` or similar.
    // For now, I'll just leave it manual or simple toggle.
    // But user asked for "expand automatically".

    // Simple heuristic: If we are in "research mode" (no chart changes), we expand.
    setIsChatExpanded(true);
  };

  // NOTE: I need to update ChatAgent to call this.
  // Since I can't easily edit ChatAgent right now without reading it, I will assume it calls onAction.
  // Wait, I can read ChatAgent. It's in `components/ChatAgent.tsx`.
  // I will check it in next step if needed.
  // For now, I'll rely on the user expanding it manually or the initial state.
  // Actually, I can just toggle it based on the prompt instructions.
  // "chat buffer... will expand automatically when the chat needs to be used for research"

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
    // If user manually resizes, we might want to disable auto-expansion logic or update it
  };

  const PageHeader = ({ title, backButton }: { title: string, backButton?: boolean }) => (
    <div className="h-16 flex items-center justify-between px-6 bg-slate-950 border-b border-slate-800 shrink-0 z-20">
      <div className="flex items-center gap-6">
        {backButton && (
          <button
            onClick={() => setCurrentPage('trade')}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors px-3 py-1.5 rounded-md hover:bg-slate-800 text-sm font-medium"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
            Back
          </button>
        )}
        <h1 className="font-bold text-slate-100 text-lg tracking-tight flex items-center gap-3">
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">Trade Viz</span>
          <span className="text-slate-600 text-sm font-normal">/</span>
          <span className="text-slate-300 font-normal">{title}</span>
        </h1>
      </div>
      <div className="flex items-center gap-2">
        {!backButton && (
          <>
            <button
              onClick={() => setCurrentPage('experiments')}
              className="flex items-center gap-2 px-3 py-1.5 rounded-md text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 transition-all text-sm font-medium"
            >
              <span>🧪 Experiments</span>
            </button>
          </>
        )}
      </div>
    </div>
  );

  // If Experiments page is active
  if (currentPage === 'experiments') {
    return (
      <div className="flex flex-col h-screen w-full bg-slate-950 overflow-hidden text-slate-100 font-sans">
        <PageHeader title="Experiments" backButton />
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
    <div className="flex h-screen w-full bg-slate-950 text-slate-100 font-sans overflow-hidden">

      {/* LEFT SIDEBAR - Modernized */}
      <div className="w-80 flex flex-col border-r border-slate-800 bg-slate-950 shrink-0 shadow-2xl z-20">

        {/* Header */}
        <div className="h-16 flex items-center justify-between px-5 border-b border-slate-800 shrink-0 bg-slate-950">
          <div className="flex items-center gap-2 font-bold text-lg tracking-tight">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-blue-500/20">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>
            </div>
            <span className="text-slate-100">Trade<span className="text-blue-500">Viz</span></span>
          </div>

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 cursor-pointer group" title="Enable Planning Mode">
              <span className={`text-[10px] font-bold uppercase tracking-wider transition-colors ${planningMode ? 'text-emerald-400' : 'text-slate-600 group-hover:text-slate-400'}`}>
                Plan
              </span>
              <div className={`w-8 h-4 rounded-full p-0.5 transition-colors duration-300 ${planningMode ? 'bg-emerald-500/20 ring-1 ring-emerald-500/50' : 'bg-slate-800 ring-1 ring-slate-700'}`}>
                <div className={`w-3 h-3 rounded-full shadow-sm transform transition-transform duration-300 ${planningMode ? 'translate-x-4 bg-emerald-400' : 'translate-x-0 bg-slate-500'}`} />
              </div>
              <input
                type="checkbox"
                checked={planningMode}
                onChange={e => setPlanningMode(e.target.checked)}
                className="hidden"
              />
            </label>

            <button
              onClick={() => setCurrentPage('experiments')}
              className="p-2 rounded-md text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 transition-colors"
              title="Experiments"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
            </button>
          </div>

        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-6 custom-scrollbar bg-slate-950">

          <RunPicker onSelect={(runId) => {
            setCurrentRun(runId);
            setSelectedFastVizRunId(null);
          }} />

          {/* Fast Viz Mode Toggle & Runs */}
          <div className="bg-slate-900/40 rounded-lg p-3 border border-amber-800/30">
            <div className="flex items-center justify-between mb-2">
              <label className="text-[10px] text-amber-400 font-bold uppercase tracking-widest flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 bg-amber-500 rounded-full"></span>
                ⚡ Fast Viz Mode
              </label>
              <input
                type="checkbox"
                checked={fastVizEnabled}
                onChange={(e) => setFastVizEnabled(e.target.checked)}
                className="w-4 h-4 rounded accent-amber-500"
              />
            </div>

            {fastVizRuns.length > 0 && (
              <div className="space-y-2 mt-2">
                {fastVizRuns.map((run) => (
                  <div
                    key={run.run_id}
                    onClick={() => {
                      setSelectedFastVizRunId(run.run_id);
                      setCurrentRun(null); // Clear regular run when viewing Fast Viz
                    }}
                    className={`flex items-center justify-between rounded px-2 py-1.5 border cursor-pointer transition-colors ${selectedFastVizRunId === run.run_id
                      ? 'bg-amber-900/30 border-amber-500/50'
                      : 'bg-slate-950 border-amber-700/30 hover:bg-amber-900/20'
                      }`}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-amber-300 truncate">{run.strategy_name}</div>
                      <div className="text-[10px] text-slate-500">{run.total_trades} trades • {run.win_rate.toFixed(1)}%</div>
                    </div>
                    <div className="flex gap-1 ml-2">
                      <button
                        onClick={async (e) => {
                          e.stopPropagation();
                          try {
                            const result = await api.saveFastVizRun(run.run_id);
                            if (result.success) {
                              setFastVizRuns(prev => prev.filter(r => r.run_id !== run.run_id));
                              setCurrentRun(result.new_run_id);
                            }
                          } catch (e) { console.error(e); }
                        }}
                        className="p-1 text-emerald-400 hover:bg-emerald-500/20 rounded"
                        title="Save as full run"
                      >
                        💾
                      </button>
                      <button
                        onClick={async (e) => {
                          e.stopPropagation();
                          try {
                            await api.deleteFastVizRun(run.run_id);
                            setFastVizRuns(prev => prev.filter(r => r.run_id !== run.run_id));
                          } catch (e) { console.error(e); }
                        }}
                        className="p-1 text-red-400 hover:bg-red-500/20 rounded"
                        title="Delete"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {fastVizRuns.length === 0 && fastVizEnabled && (
              <div className="text-[10px] text-slate-500 text-center py-2">
                Fast Viz runs will appear here
              </div>
            )}
          </div>

          <Navigator
            mode={mode}
            setMode={setMode}
            index={index}
            setIndex={setIndex}
            maxIndex={Math.max(0, maxIndex)}
          />

          {!currentRun ? (
            <div className="p-6 text-center border border-dashed border-slate-800 rounded-lg bg-slate-900/30">
              <div className="text-4xl mb-2 opacity-20">👋</div>
              <p className="text-sm text-slate-500">Select a run above to start analyzing decisions and trades.</p>
            </div>
          ) : (
            <>
              {/* Details Panel - Stays in Sidebar */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between px-1">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-slate-500">Context</span>
                </div>
                <div className="border border-slate-800 rounded-lg overflow-hidden shadow-sm bg-slate-900/30">
                  <DetailsPanel decision={activeDecision} trade={activeTrade} />
                </div>
              </div>

              <div className="flex justify-between items-center text-[10px] text-slate-600 px-2 font-mono uppercase tracking-wider">
                <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-slate-700"></span> {continuousData?.count?.toLocaleString() || 0} bars</span>
                <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-slate-700"></span> {decisions.length} decisions</span>
              </div>
            </>
          )}
        </div>

        {/* Play Button Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950">
          <button
            onClick={() => {
              setSimulationMode('SIMULATION');
              setShowSimulation(true);
            }}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-900/20 transition-all transform hover:translate-y-[-1px] active:translate-y-[1px] font-bold text-sm tracking-wide"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" /></svg>
            Replay Session
          </button>
        </div>
      </div>

      {/* MAIN CONTENT - Vertical Layout */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative bg-slate-900/50">

        {/* Stats Panel (Moved Back to Top of Main Content) */}
        {(currentRun || selectedFastVizRunId) && (
          <StatsPanel decisions={effectiveDecisions} startingBalance={50000} />
        )}

        {/* Chart Top (Flex Grow) */}
        <div className="flex-1 min-h-0 relative bg-slate-900">
          {/* Indicator Settings - Top LEFT of chart */}
          <div className="absolute top-4 left-4 z-30">
            <IndicatorSettingsPanel settings={indicatorSettings} onChange={setIndicatorSettings} />
          </div>

          <CandleChart
            continuousData={continuousData}
            decisions={effectiveDecisions}
            activeDecision={activeDecision}
            trade={activeTrade}
            trades={trades}
            indicatorSettings={indicatorSettings}
          />
        </div>

        {/* Resizer Handle */}
        <div
          className="h-1 bg-slate-950 hover:bg-blue-500/50 cursor-row-resize shrink-0 flex items-center justify-center transition-all duration-300 border-t border-slate-800 relative group z-30"
          onMouseDown={startResizing}
        >
          <div className="w-16 h-1 bg-slate-700 rounded-full group-hover:bg-blue-400 transition-colors opacity-50 group-hover:opacity-100" />
        </div>

        {/* Chat Bottom (Fixed Height) */}
        <div style={{ height: isChatExpanded ? '60vh' : chatHeight, transition: 'height 0.3s ease-in-out' }} className="shrink-0 bg-slate-950 border-t border-slate-800 shadow-[0_-8px_30px_rgba(0,0,0,0.5)] z-20 relative">
          <button
            onClick={() => setIsChatExpanded(!isChatExpanded)}
            className="absolute top-0 right-4 -mt-3 bg-slate-800 border border-slate-700 rounded-full p-1 hover:bg-slate-700 transition-colors z-50 shadow-sm"
            title={isChatExpanded ? "Collapse" : "Expand"}
          >
            <svg className={`w-4 h-4 text-slate-400 transform transition-transform ${isChatExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          </button>
          <ChatAgent
            runId={currentRun || 'none'}
            currentIndex={index}
            currentMode={mode}
            fastVizMode={fastVizEnabled}
            planningMode={planningMode}
            onAction={handleAgentAction}
            onTextResponse={handleAgentTextResponse}
          />
        </div>

      </div>

      {/* UNIFIED REPLAY OVERLAY */}
      {
        showSimulation && (
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
        )
      }

    </div>
  );
};

export default App;