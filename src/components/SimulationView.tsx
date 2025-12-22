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
    const [ocoState, setOcoState] = useState<{ entry: number, stop: number, tp: number, startTime: number, direction: 'LONG' | 'SHORT' } | null>(null);
    const [triggers, setTriggers] = useState(0);
    const [wins, setWins] = useState(0);
    const [losses, setLosses] = useState(0);
    const [startIndex, setStartIndex] = useState(0);

    // Trade Settings
    const [entryType, setEntryType] = useState<'MARKET' | 'LIMIT'>('MARKET');
    const [stopAtr, setStopAtr] = useState(2.0);   // ATR multiples for stop
    const [tpAtr, setTpAtr] = useState(4.0);       // ATR multiples for target
    const [threshold, setThreshold] = useState(0.35); // CNN trigger threshold

    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const allBarsRef = useRef<BarData[]>([]);

    // Model Integration
    const [modelLoading, setModelLoading] = useState(false);
    const modelDecisionsRef = useRef<Map<number, ReplayDecision>>(new Map()); // Map timestamp -> Decision

    const ocoRef = useRef<{ entry: number, stop: number, tp: number, startTime: number, direction: 'LONG' | 'SHORT' } | null>(null);

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

            const session = await api.startReplay("models/ifvg_4class_cnn.pth", startDate, 7, replaySpeed, threshold);
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

        // Real-time inference - no prefetch needed

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
                const isLong = ocoRef.current.direction === 'LONG';

                // Direction-aware exit logic
                if (isLong) {
                    if (bar.low <= ocoRef.current.stop) {
                        outcome = 'LOSS';
                        price = ocoRef.current.stop;
                        setLosses(prev => prev + 1);
                    } else if (bar.high >= ocoRef.current.tp) {
                        outcome = 'WIN';
                        price = ocoRef.current.tp;
                        setWins(prev => prev + 1);
                    }
                } else {
                    // SHORT: stop is above entry, TP is below
                    if (bar.high >= ocoRef.current.stop) {
                        outcome = 'LOSS';
                        price = ocoRef.current.stop;
                        setLosses(prev => prev + 1);
                    } else if (bar.low <= ocoRef.current.tp) {
                        outcome = 'WIN';
                        price = ocoRef.current.tp;
                        setWins(prev => prev + 1);
                    }
                }

                if (outcome) {
                    const tradeId = `sim_${ocoRef.current.startTime}`;

                    // Calculate duration
                    const barsHeld = Math.max(1, Math.round((bar.time - ocoRef.current.startTime) / 60));

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
                            entry_type: 'MARKET', direction: ocoRef.current.direction, reference_type: '',
                            reference_value: 0, atr_at_creation: 0, max_bars: 100,
                            stop_atr: 0, tp_multiple: 0
                        },
                        oco_results: {
                            simulation: {
                                bars_held: barsHeld,
                                outcome: outcome,
                                pnl_dollars: (isLong ? (price - ocoRef.current.entry) : (ocoRef.current.entry - price)) * 50
                            }
                        }
                    };

                    const trade: VizTrade = {
                        trade_id: tradeId, decision_id: tradeId, index: completedTradesRef.current.length,
                        direction: ocoRef.current.direction, size: 1,
                        entry_time: new Date(ocoRef.current.startTime * 1000).toISOString(),
                        entry_bar: 0, entry_price: ocoRef.current.entry,
                        exit_time: new Date(bar.time * 1000).toISOString(),
                        exit_bar: 0, exit_price: price,
                        exit_reason: outcome === 'WIN' ? 'TP' : 'SL',
                        outcome: outcome, pnl_points: isLong ? (price - ocoRef.current.entry) : (ocoRef.current.entry - price),
                        pnl_dollars: (isLong ? (price - ocoRef.current.entry) : (ocoRef.current.entry - price)) * 50,
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
            // If no active trade, call /infer every 5 bars
            if (!ocoRef.current && idx % 5 === 0 && idx >= 60) {
                // Build window of last 30 bars (model was trained on 30)
                const windowBars = allBarsRef.current.slice(Math.max(0, idx - 29), idx + 1);

                // Calculate ATR from recent bars (simple: avg of high-low)
                const recentBars = allBarsRef.current.slice(Math.max(0, idx - 13), idx + 1);
                const avgRange = recentBars.reduce((sum, b) => sum + (b.high - b.low), 0) / recentBars.length;
                const atr = avgRange || (bar.close * 0.001);

                // Try both ports
                const tryInfer = async (port: number) => {
                    try {
                        const res = await fetch(`http://localhost:${port}/infer`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                bars: windowBars.map(b => ({
                                    open: b.open,
                                    high: b.high,
                                    low: b.low,
                                    close: b.close,
                                    volume: b.volume || 0
                                })),
                                model_path: 'models/ifvg_4class_cnn.pth',
                                threshold: threshold
                            })
                        });
                        return await res.json();
                    } catch {
                        return null;
                    }
                };

                // Call inference (try port 8000, then 8001)
                tryInfer(8000).then(result => {
                    if (!result) return tryInfer(8001);
                    return result;
                }).then(result => {
                    if (result?.triggered && result.direction !== 'NONE' && !ocoRef.current) {
                        // FIXED: Use current bar from allBarsRef at current idx, not stale closure
                        const currentIdx = idx - 1; // idx was already incremented by interval
                        const currentBar = allBarsRef.current[currentIdx] || allBarsRef.current[allBarsRef.current.length - 1];
                        const entry = currentBar.close; // Market entry at CURRENT close
                        const isLong = result.direction === 'LONG';
                        const stop = isLong ? entry - (stopAtr * atr) : entry + (stopAtr * atr);
                        const tp = isLong ? entry + (tpAtr * atr) : entry - (tpAtr * atr);

                        const newOco = {
                            entry,
                            stop,
                            tp,
                            startTime: currentBar.time, // Use CURRENT bar time
                            direction: result.direction as 'LONG' | 'SHORT'
                        };
                        ocoRef.current = newOco;
                        setOcoState(newOco);
                        setTriggers(prev => prev + 1);
                        console.log(`CNN TRIGGER: ${result.direction} @ ${entry.toFixed(2)}, Stop: ${stop.toFixed(2)}, TP: ${tp.toFixed(2)}, Prob: ${result.probability}`);
                    }
                }).catch(e => console.error('Infer error:', e));
            }

            idx++;
        }, speed);
    }, [speed, startIndex, currentIndex, stopAtr, tpAtr, threshold]);

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
                    <span className="text-sm text-slate-400">Model: ifvg_4class_cnn.pth</span>
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

                    {/* Trade Settings */}
                    <h2 className="text-sm font-bold text-green-400 uppercase mb-2 mt-6">Trade Settings</h2>

                    <div className="mb-3">
                        <label className="text-xs text-slate-400">Entry Type</label>
                        <select
                            value={entryType}
                            onChange={e => setEntryType(e.target.value as 'MARKET' | 'LIMIT')}
                            disabled={isRunning}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                        >
                            <option value="MARKET">Market Order</option>
                            <option value="LIMIT">Limit Order (future)</option>
                        </select>
                    </div>

                    <div className="mb-3">
                        <label className="text-xs text-slate-400">CNN Threshold</label>
                        <input
                            type="number"
                            step="0.01"
                            min="0.1"
                            max="0.9"
                            value={threshold}
                            onChange={e => setThreshold(parseFloat(e.target.value) || 0.35)}
                            disabled={isRunning}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                        />
                        <span className="text-xs text-slate-500">Higher = fewer triggers</span>
                    </div>

                    <div className="mb-3">
                        <label className="text-xs text-slate-400">Stop Loss (ATR ×)</label>
                        <input
                            type="number"
                            step="0.5"
                            min="0.5"
                            max="10"
                            value={stopAtr}
                            onChange={e => setStopAtr(parseFloat(e.target.value) || 2)}
                            disabled={isRunning}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                        />
                    </div>

                    <div className="mb-4">
                        <label className="text-xs text-slate-400">Take Profit (ATR ×)</label>
                        <input
                            type="number"
                            step="0.5"
                            min="0.5"
                            max="20"
                            value={tpAtr}
                            onChange={e => setTpAtr(parseFloat(e.target.value) || 4)}
                            disabled={isRunning}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                        />
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
                            <span className="text-slate-400">Triggers:</span>
                            <span className="text-yellow-400">{completedTrades.length} / {triggers}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Wins:</span>
                            <span className="text-green-400">{wins}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Losses:</span>
                            <span className="text-red-400">{losses}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
