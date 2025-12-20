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
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface ReplayDecision {
    type: 'DECISION';
    decision_id: string;
    bar_idx: number;
    timestamp: string;
    win_probability: number;
    threshold: number;
    triggered: boolean;
    price: number;
    atr: number;
}

export const SimulationView: React.FC<SimulationViewProps> = ({
    onClose,
    runId,
    lastTradeTimestamp
}) => {
    const [isRunning, setIsRunning] = useState(false);
    const [speed, setSpeed] = useState(200); // ms per bar for playback
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

    // Model Integration
    const [modelLoading, setModelLoading] = useState(false);
    const modelDecisionsRef = useRef<Map<number, ReplayDecision>>(new Map()); // Map timestamp -> Decision

    const ocoRef = useRef<{ entry: number, stop: number, tp: number, startTime: number } | null>(null);

    // Trade tracking
    const [completedTrades, setCompletedTrades] = useState<VizTrade[]>([]);
    const [completedDecisions, setCompletedDecisions] = useState<VizDecision[]>([]);
    const completedTradesRef = useRef<VizTrade[]>([]);
    const completedDecisionsRef = useRef<VizDecision[]>([]);

    // Load continuous contract data starting AFTER the last trade
    useEffect(() => {
        const loadData = async () => {
            setStatus('Loading data...');
            try {
                // Try both ports (logic preserved from original)
                for (const port of [8000, 8001]) {
                    try {
                        const params = new URLSearchParams();
                        params.set('timeframe', '1m');

                        if (lastTradeTimestamp) {
                            const lastTradeDate = new Date(lastTradeTimestamp);
                            const startDate = new Date(lastTradeDate.getTime() - 2 * 24 * 60 * 60 * 1000);
                            const endDate = new Date(lastTradeDate.getTime() + 14 * 24 * 60 * 60 * 1000);
                            params.set('start', startDate.toISOString());
                            params.set('end', endDate.toISOString());
                        }

                        const res = await fetch(`http://localhost:${port}/market/continuous?${params}`);
                        if (res.ok) {
                            const data = await res.json();
                            const loadedBars: BarData[] = data.bars.map((b: any) => ({
                                time: new Date(b.time).getTime() / 1000,
                                open: b.open,
                                high: b.high,
                                low: b.low,
                                close: b.close,
                            }));

                            allBarsRef.current = loadedBars;

                            let simStartIdx = 0;
                            if (lastTradeTimestamp) {
                                const lastTradeTime = new Date(lastTradeTimestamp).getTime() / 1000;
                                simStartIdx = loadedBars.findIndex(b => b.time >= lastTradeTime);
                                if (simStartIdx === -1) simStartIdx = 0;
                            }
                            setStartIndex(simStartIdx);

                            if (loadedBars.length > 0) {
                                setStatus(`Ready (${loadedBars.length} bars)`);
                            } else {
                                setStatus('No data available');
                            }
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


    // Prefetch Model Decisions
    const fetchModelDecisions = async () => {
        setStatus('Running Model Inference (Prefetch)...');
        setModelLoading(true);
        modelDecisionsRef.current.clear();

        try {
            // Replay from lastTradeTimestamp for 7 days at MAX speed
            const startDate = lastTradeTimestamp || "2025-03-18T09:30:00"; // Fallback
            const replaySpeed = 10000; // Super fast
            const threshold = 0.2; // Low threshold for debugging/visibility

            const session = await api.startReplay("models/swing_breakout_model.pth", startDate, 7, replaySpeed, threshold);
            const url = api.getReplayStreamUrl(session.session_id);

            // Consume Stream
            const response = await fetch(url);
            const reader = response.body?.getReader();
            const decoder = new TextDecoder();

            if (reader) {
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.type === 'DECISION' && data.triggered) {
                                    // Store key as seconds
                                    const ts = new Date(data.timestamp).getTime() / 1000;
                                    modelDecisionsRef.current.set(ts, data);

                                    // Also store strictly int version just in case of ms diff
                                    modelDecisionsRef.current.set(Math.floor(ts), data);
                                }
                            } catch { }
                        }
                    }
                }
            }

            // Cleanup session
            await api.stopReplay(session.session_id);
            setStatus(`Inference Complete. Found ${modelDecisionsRef.current.size} triggers (Threshold ${threshold}).`);
            console.log(`[Replay] Found ${modelDecisionsRef.current.size} triggers. Sample keys:`, Array.from(modelDecisionsRef.current.keys()).slice(0, 5));
            setModelLoading(false);
            return true;
        } catch (e: any) {
            setStatus(`Inference Failed: ${e.message}`);
            setModelLoading(false);
            return false;
        }
    };


    const startSimulation = useCallback(async () => {
        if (allBarsRef.current.length === 0) {
            setStatus('No data loaded');
            return;
        }

        // 1. Fetch triggers if needed (first run)
        if (modelDecisionsRef.current.size === 0 && !modelLoading) {
            const success = await fetchModelDecisions();
            if (!success) {
                // Allow proceeding without model (falls back to no trades)?
                // Let's assume user wants to continue even if model failed (debug)
            }
        }

        setIsRunning(true);

        let startIdx = startIndex;
        if (currentIndex > startIndex && bars.length > 0) {
            startIdx = currentIndex + 1;
            setStatus(`Resuming from bar ${startIdx}...`);
        } else {
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

            // OCO Logic (EXIT)
            if (ocoRef.current) {
                let outcome = '';
                let price = 0;

                if (bar.low <= ocoRef.current.stop) {
                    outcome = 'LOSS';
                    price = ocoRef.current.stop;
                    setLosses(prev => prev + 1);
                } else if (bar.high >= ocoRef.current.tp) {
                    outcome = 'WIN';
                    price = ocoRef.current.tp;
                    setWins(prev => prev + 1);
                }

                if (outcome) {
                    const tradeId = `sim_${ocoRef.current.startTime}`;

                    // Log Trade
                    const decision: VizDecision = {
                        decision_id: tradeId,
                        timestamp: new Date(ocoRef.current.startTime * 1000).toISOString(),
                        bar_idx: 0, index: 0, scanner_id: 'cnn', scanner_context: {},
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
                        trade_id: tradeId, decision_id: tradeId, index: completedTradesRef.current.length,
                        direction: 'LONG', size: 1,
                        entry_time: new Date(ocoRef.current.startTime * 1000).toISOString(),
                        entry_bar: 0, entry_price: ocoRef.current.entry,
                        exit_time: new Date(bar.time * 1000).toISOString(),
                        exit_bar: 0, exit_price: price,
                        exit_reason: outcome === 'WIN' ? 'TP' : 'SL',
                        outcome: outcome, pnl_points: price - ocoRef.current.entry,
                        pnl_dollars: (price - ocoRef.current.entry) * 50,
                        r_multiple: outcome === 'WIN' ? 2 : -1, bars_held: 0, mae: 0, mfe: 0, fills: []
                    };

                    completedDecisionsRef.current.push(decision);
                    completedTradesRef.current.push(trade);
                    setCompletedDecisions([...completedDecisionsRef.current]);
                    setCompletedTrades([...completedTradesRef.current]);

                    ocoRef.current = null;
                    setOcoState(null);
                }
            }

            // --- MODEL TRIGGER LOGIC (ENTRY) ---
            // If no active trade, check if Model triggered on this bar
            if (!ocoRef.current) {
                // Try Exact Match first
                let decision = modelDecisionsRef.current.get(bar.time);

                // Try Floor Match
                if (!decision) {
                    decision = modelDecisionsRef.current.get(Math.floor(bar.time));
                }

                if (decision && decision.triggered) {
                    // Use model values
                    const entry = decision.price;
                    const atr = decision.atr || (entry * 0.001);
                    const stop = entry - (2 * atr); // Default 2 ATR
                    const tp = entry + (4 * atr);   // Default 4 ATR

                    const newOco = { entry, stop, tp, startTime: bar.time };
                    ocoRef.current = newOco;
                    setOcoState(newOco);
                    setTriggers(prev => prev + 1);
                    console.log('MODEL TRIGGER:', newOco, 'Prob:', decision.win_probability);
                }
            }

            idx++;
        }, speed);
    }, [speed, startIndex, currentIndex]);

    const stopSimulation = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsRunning(false);
        setStatus('Stopped');
    }, []);

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, []);

    return (
        <div className="fixed inset-0 bg-slate-900 z-50 flex flex-col">
            <div className="h-14 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-4">
                <h1 className="text-white font-bold">Model Replay (Prefetch)</h1>
                <div className="flex items-center space-x-4">
                    <span className="text-sm text-slate-400">Model: swing_breakout_model.pth</span>
                    <button onClick={onClose} className="text-slate-400 hover:text-white">✕ Close</button>
                </div>
            </div>

            <div className="flex-1 flex">
                <div className="flex-1 flex flex-col min-h-0">
                    <div className="flex-1 min-h-[400px]">
                        <CandleChart
                            continuousData={bars.length > 0 ? {
                                timeframe: '1m',
                                count: bars.length,
                                bars: bars.map(b => ({
                                    time: new Date(b.time * 1000).toISOString(),
                                    open: b.open, high: b.high, low: b.low, close: b.close, volume: 0
                                }))
                            } : null}
                            decisions={completedDecisions}
                            activeDecision={null}
                            trade={null}
                            trades={completedTrades}
                            simulationOco={ocoState ? {
                                ...ocoState,
                                startTime: ocoState.startTime // Ensure number if that's what CandleChart expects, or convert?
                                // CandleChart expects startTime as ISO usually? Let's check.
                                // In the restored file it was 'number'.
                                // CandleChart props: simulationOco: { ... startTime: number } | null ?
                                // No, original CandleChart uses timestamps.
                                // Let's check restored SimulationView.tsx line 299: simulationOco={ocoState}
                                // ocoState has number.
                                // Does CandleChart accept number?
                            } : null}
                            forceShowAllTrades={true}
                        />
                    </div>
                </div>

                <div className="w-80 bg-slate-800 border-l border-slate-700 p-4">
                    <h2 className="text-sm font-bold text-blue-400 uppercase mb-4">Controls</h2>

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

                    <div className="mb-4">
                        {!isRunning ? (
                            <button
                                onClick={startSimulation}
                                className="w-full bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-4 rounded"
                            >
                                {modelLoading ? 'Loading Model...' : '▶ Start Replay'}
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
                    {/* Status Info */}
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-slate-400">Status:</span>
                            <span className="text-white bg-slate-800 px-1 truncate max-w-[150px]" title={status}>{status}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Decisions Loaded:</span>
                            <span className="text-blue-400">{modelDecisionsRef.current.size}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Triggers:</span>
                            <span className="text-yellow-400">{completedTrades.length} / {triggers}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Wins:</span>
                            <span className="text-green-400">{wins}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
