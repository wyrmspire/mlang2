import React, { useState, useCallback, useRef, useEffect } from 'react';
import { CandleChart } from './CandleChart';
import { VizTrade, VizDecision, VizOCO } from '../types/viz';
import { api } from '../api/client';

interface SimulationViewProps {
    onClose: () => void;
    runId?: string;
    lastTradeTimestamp?: string; // ISO timestamp of last trade in strategy
}

interface BarData {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
}

interface SimEvent {
    type: string;
    data?: any;
}

/**
 * Simulation View 2.0 - Interactive Trading Lab
 * 
 * Backend-owned simulation using the new /sim/* API.
 * The backend runs the market simulation (OMS), frontend displays state.
 * 
 * Features:
 * - Strategy selection and configuration
 * - Play/Pause/Step controls
 * - Live event log showing OMS activity
 * - Chart rendering with pending orders and filled trades
 * - Mid-stream parameter updates
 */
export const SimulationView: React.FC<SimulationViewProps> = ({
    onClose,
    runId,
    lastTradeTimestamp
}) => {
    const [isRunning, setIsRunning] = useState(false);
    const [speed, setSpeed] = useState(200); // ms per bar
    const [bars, setBars] = useState<BarData[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [status, setStatus] = useState('Ready');
    const [ocoState, setOcoState] = useState<{ entry: number, stop: number, tp: number, startTime: number } | null>(null);
    const [triggers, setTriggers] = useState(0);
    const [wins, setWins] = useState(0);
    const [losses, setLosses] = useState(0);
    const [startIndex, setStartIndex] = useState(0);

    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const allBarsRef = useRef<BarData[]>([]);
    const ocoRef = useRef<{ entry: number, stop: number, tp: number, startTime: number } | null>(null);

    // Load continuous contract data starting AFTER the last trade
    useEffect(() => {
        const loadData = async () => {
            setStatus('Loading data...');
            try {
                // Try both ports
                for (const port of [8000, 8001]) {
                    try {
                        // Build query params - if we have lastTradeTimestamp, start from there
                        const params = new URLSearchParams();
                        params.set('timeframe', '1m');

                        // Use last trade timestamp - start 2 days BEFORE for context
                        // This allows seeing the setup leading into where we start simulating
                        if (lastTradeTimestamp) {
                            const lastTradeDate = new Date(lastTradeTimestamp);
                            // Start 2 days before the last trade for context
                            const startDate = new Date(lastTradeDate.getTime() - 2 * 24 * 60 * 60 * 1000);
                            // End 2 weeks after the last trade
                            const endDate = new Date(lastTradeDate.getTime() + 14 * 24 * 60 * 60 * 1000);
                            params.set('start', startDate.toISOString());
                            params.set('end', endDate.toISOString());
                        }

                        const res = await fetch(`http://localhost:${port}/market/continuous?${params}`);
                        if (res.ok) {
                            const data = await res.json();
                            // Convert to our format
                            const loadedBars: BarData[] = data.bars.map((b: any) => ({
                                time: new Date(b.time).getTime() / 1000,
                                open: b.open,
                                high: b.high,
                                low: b.low,
                                close: b.close,
                            }));

                            allBarsRef.current = loadedBars;

                            // Find the bar closest to lastTradeTimestamp to start playback there
                            // The 2 days before show as pre-existing context
                            let simStartIdx = 0;
                            if (lastTradeTimestamp) {
                                const lastTradeTime = new Date(lastTradeTimestamp).getTime() / 1000;
                                simStartIdx = loadedBars.findIndex(b => b.time >= lastTradeTime);
                                if (simStartIdx === -1) simStartIdx = 0;
                            }
                            setStartIndex(simStartIdx);

                            if (loadedBars.length > 0) {
                                const firstBar = new Date(loadedBars[0].time * 1000).toISOString();
                                const lastBar = new Date(loadedBars[loadedBars.length - 1].time * 1000).toISOString();
                                setStatus(`Loaded ${loadedBars.length} bars (${simStartIdx} pre-context): ${firstBar.slice(0, 10)} to ${lastBar.slice(0, 10)}`);
                            } else {
                                setStatus('No data available for this period');
                            }

                            console.log('Simulation data:', {
                                startParam: lastTradeTimestamp,
                                barsLoaded: loadedBars.length,
                                simStartIdx,
                                firstBar: loadedBars[0]?.time ? new Date(loadedBars[0].time * 1000).toISOString() : 'none',
                                lastBar: loadedBars.length > 0 ? new Date(loadedBars[loadedBars.length - 1].time * 1000).toISOString() : 'none'
                            });
                            return;
                        }
                    } catch { }
                }
                setStatus('Failed to load data');
            } catch (e) {
                setStatus(`Error: ${e}`);
            }
        };
        loadData();
    }, [lastTradeTimestamp, runId]);

    const [completedTrades, setCompletedTrades] = useState<VizTrade[]>([]);
    const [completedDecisions, setCompletedDecisions] = useState<VizDecision[]>([]);
    const completedTradesRef = useRef<VizTrade[]>([]);
    const completedDecisionsRef = useRef<VizDecision[]>([]);

    // Start simulation
    const startSimulation = useCallback(() => {
        if (allBarsRef.current.length === 0) {
            setStatus('No data loaded');
            return;
        }

        setIsRunning(true);

        // Resume from current index if we're already part way through
        let startIdx = startIndex;
        if (currentIndex > startIndex && bars.length > 0) {
            startIdx = currentIndex + 1;
            setStatus(`Resuming from bar ${startIdx}...`);
        } else {
            // Reset if starting fresh
            setCurrentIndex(startIndex);
            setBars([]);
            setOcoState(null);
            ocoRef.current = null;
            setTriggers(0);
            setWins(0);
            setLosses(0);
            setCompletedTrades([]);
            setCompletedDecisions([]);
            completedTradesRef.current = [];
            completedDecisionsRef.current = [];
            setStatus(`Running from bar ${startIndex}...`);
        }

        // Add bars one at a time, starting from startIndex
        let idx = startIdx;
        intervalRef.current = setInterval(() => {
            if (idx >= allBarsRef.current.length) {
                stopSimulation();
                setStatus('Completed');
                return;
            }

            const bar = allBarsRef.current[idx];
            setBars(prev => [...prev, bar]);
            setCurrentIndex(idx);

            // Check if OCO resolved FIRST (using ref for current state)
            if (ocoRef.current) {
                let outcome = '';
                let price = 0;

                if (bar.low <= ocoRef.current.stop) {
                    // Stop hit - LOSS
                    outcome = 'LOSS';
                    price = ocoRef.current.stop; // Assume filled at stop
                    setLosses(prev => prev + 1);
                } else if (bar.high >= ocoRef.current.tp) {
                    // TP hit - WIN
                    outcome = 'WIN';
                    price = ocoRef.current.tp;
                    setWins(prev => prev + 1);
                }

                if (outcome) {
                    // Create finished trade records for persistence
                    const tradeId = `sim_trade_${ocoRef.current.startTime}`;
                    const decisionId = `sim_dec_${ocoRef.current.startTime}`;

                    const decision: VizDecision = {
                        decision_id: decisionId,
                        timestamp: new Date(ocoRef.current.startTime * 1000).toISOString(),
                        bar_idx: 0, index: 0, scanner_id: 'sim', scanner_context: {},
                        action: 'OCO', skip_reason: '', current_price: ocoRef.current.entry,
                        atr: 0, cf_outcome: outcome, cf_pnl_dollars: 0,
                        oco: {
                            entry_price: ocoRef.current.entry,
                            stop_price: ocoRef.current.stop,
                            tp_price: ocoRef.current.tp,
                            entry_type: 'MARKET', direction: 'LONG', reference_type: '',
                            reference_value: 0, atr_at_creation: 0, max_bars: 100,
                            stop_atr: 0, tp_multiple: 0
                        }
                    };

                    const trade: VizTrade = {
                        trade_id: tradeId, decision_id: decisionId, index: 0,
                        direction: 'LONG', size: 1,
                        entry_time: new Date(ocoRef.current.startTime * 1000).toISOString(),
                        entry_bar: 0, entry_price: ocoRef.current.entry,
                        exit_time: new Date(bar.time * 1000).toISOString(),
                        exit_bar: 0, exit_price: price,
                        exit_reason: outcome === 'WIN' ? 'TP' : 'SL',
                        outcome: outcome, pnl_points: 0, pnl_dollars: 0,
                        r_multiple: 0, bars_held: 0, mae: 0, mfe: 0, fills: []
                    };

                    completedDecisionsRef.current.push(decision);
                    completedTradesRef.current.push(trade);
                    setCompletedDecisions([...completedDecisionsRef.current]);
                    setCompletedTrades([...completedTradesRef.current]);

                    ocoRef.current = null;
                    setOcoState(null);
                }
            }

            // Simple trigger logic: random for demo, replace with actual model
            // Only trigger if no active OCO
            if (!ocoRef.current && Math.random() < 0.01) { // 1% chance per bar
                const entry = bar.close;
                const stop = entry - 5;
                const tp = entry + 10;
                const newOco = { entry, stop, tp, startTime: bar.time };
                ocoRef.current = newOco;
                setOcoState(newOco);
                setTriggers(prev => prev + 1);
                console.log('OCO TRIGGERED:', newOco);
            }

            idx++;
        }, speed);
    }, [speed, startIndex]);

    // Stop simulation
    const stopSimulation = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsRunning(false);
        setStatus('Stopped');
    }, []);

    // Cleanup
    useEffect(() => {
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, []);

    return (
        <div className="fixed inset-0 bg-slate-900 z-50 flex flex-col">
            {/* Header */}
            <div className="h-14 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-4">
                <h1 className="text-white font-bold">Forward Simulation</h1>
                <button
                    onClick={onClose}
                    className="text-slate-400 hover:text-white"
                >
                    ✕ Close
                </button>
            </div>

            {/* Main content */}
            <div className="flex-1 flex">
                {/* Chart - using real CandleChart */}
                <div className="flex-1 flex flex-col min-h-0">
                    <div className="flex-1 min-h-[400px]">
                        <CandleChart
                            continuousData={bars.length > 0 ? {
                                timeframe: '1m',
                                count: bars.length,
                                bars: bars.map(b => ({
                                    time: new Date(b.time * 1000).toISOString(),
                                    open: b.open,
                                    high: b.high,
                                    low: b.low,
                                    close: b.close,
                                    volume: 0
                                }))
                            } : null}
                            decisions={completedDecisions}
                            activeDecision={null}
                            trade={null}
                            trades={completedTrades}
                            simulationOco={ocoState}
                            forceShowAllTrades={true} /* Force show completed trades */
                        />
                    </div>
                    <div className="p-2 text-xs text-slate-400">
                        Bars loaded: {bars.length} | Playing from bar: {currentIndex}
                    </div>
                </div>

                {/* Controls sidebar */}
                <div className="w-80 bg-slate-800 border-l border-slate-700 p-4">
                    <h2 className="text-sm font-bold text-blue-400 uppercase mb-4">Controls</h2>

                    {/* Speed */}
                    <div className="mb-4">
                        <label className="text-xs text-slate-400">Speed (ms per bar)</label>
                        <select
                            value={speed}
                            onChange={e => setSpeed(parseInt(e.target.value))}
                            disabled={isRunning}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                        >
                            <option value={500}>Slow (500ms)</option>
                            <option value={200}>Normal (200ms)</option>
                            <option value={100}>Fast (100ms)</option>
                            <option value={50}>Very Fast (50ms)</option>
                            <option value={10}>Max (10ms)</option>
                        </select>
                    </div>

                    {/* Play/Stop */}
                    <div className="mb-4">
                        {!isRunning ? (
                            <button
                                onClick={startSimulation}
                                className="w-full bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-4 rounded"
                            >
                                ▶ Start Simulation
                            </button>
                        ) : (
                            <button
                                onClick={stopSimulation}
                                className="w-full bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-4 rounded"
                            >
                                ■ Stop
                            </button>
                        )}
                    </div>

                    {/* Status */}
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-slate-400">Status:</span>
                            <span className="text-white">{status}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Bars:</span>
                            <span className="text-white">{bars.length}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Triggers:</span>
                            <span className="text-yellow-400">{triggers}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Wins:</span>
                            <span className="text-green-400">{wins}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Losses:</span>
                            <span className="text-red-400">{losses}</span>
                        </div>
                        {ocoState && (
                            <div className="bg-yellow-900/50 border border-yellow-600 rounded p-2 mt-4">
                                <div className="text-yellow-400 font-bold text-xs">OCO Active</div>
                                <div className="text-xs text-slate-300 mt-1">
                                    Entry: ${ocoState.entry.toFixed(2)}<br />
                                    Stop: ${ocoState.stop.toFixed(2)}<br />
                                    TP: ${ocoState.tp.toFixed(2)}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
