import React, { useState, useCallback, useRef, useEffect } from 'react';
import { CandleChart } from './CandleChart';
import { VizTrade, VizDecision } from '../types/viz';
import { api } from '../api/client';

interface UnifiedReplayViewProps {
    onClose: () => void;
    runId?: string;
    lastTradeTimestamp?: string;
}

interface BarData {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

type DataSourceMode = 'SIMULATION' | 'YFINANCE';
type PlaybackState = 'STOPPED' | 'PLAYING' | 'PAUSED';

export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
    onClose,
    runId,
    lastTradeTimestamp
}) => {
    // Data Source
    const [dataSourceMode, setDataSourceMode] = useState<DataSourceMode>('SIMULATION');

    // Playback State
    const [playbackState, setPlaybackState] = useState<PlaybackState>('STOPPED');
    const [speed, setSpeed] = useState(200); // ms per bar
    const [bars, setBars] = useState<BarData[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [startIndex, setStartIndex] = useState(0);
    const [status, setStatus] = useState('Ready');

    // Model/Scanner Selection - Now with enable checkboxes (OFF by default)
    const [useCnnModel, setUseCnnModel] = useState(false);       // OFF by default
    const [usePatternScanner, setUsePatternScanner] = useState(false); // OFF by default
    const [selectedModel, setSelectedModel] = useState('models/ifvg_4class_cnn.pth');
    const [selectedScanner, setSelectedScanner] = useState('ifvg');
    const [availableModels, setAvailableModels] = useState<string[]>([
        'models/ifvg_4class_cnn.pth',
        'models/ifvg_cnn.pth',
        'models/best_model.pth'
    ]);

    // Entry Configuration (sent to backend)
    const [entryType, setEntryType] = useState<'market' | 'limit'>('market');
    const [stopMethod, setStopMethod] = useState<'atr' | 'swing' | 'fixed_bars'>('atr');
    const [tpMethod, setTpMethod] = useState<'atr' | 'r_multiple'>('atr');

    // OCO State
    const [ocoState, setOcoState] = useState<{
        entry: number;
        stop: number;
        tp: number;
        startTime: number;
        direction: 'LONG' | 'SHORT';
    } | null>(null);

    // Trade Settings
    const [threshold, setThreshold] = useState(0.35);
    const [stopAtr, setStopAtr] = useState(2.0);
    const [tpAtr, setTpAtr] = useState(4.0);

    // Trade Tracking
    const [triggers, setTriggers] = useState(0);
    const [wins, setWins] = useState(0);
    const [losses, setLosses] = useState(0);
    const [completedTrades, setCompletedTrades] = useState<VizTrade[]>([]);
    const [completedDecisions, setCompletedDecisions] = useState<VizDecision[]>([]);

    // YFinance specific
    const [ticker, setTicker] = useState('MES=F');
    const [yfinanceDays, setYfinanceDays] = useState(7);

    // Refs
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const allBarsRef = useRef<BarData[]>([]);
    const ocoRef = useRef<typeof ocoState>(null);
    const completedTradesRef = useRef<VizTrade[]>([]);
    const completedDecisionsRef = useRef<VizDecision[]>([]);
    const eventSourceRef = useRef<EventSource | null>(null);
    const dataSourceModeRef = useRef<DataSourceMode>('SIMULATION');

    // Load data based on selected mode
    useEffect(() => {
        dataSourceModeRef.current = dataSourceMode;
        loadData();
    }, [dataSourceMode, lastTradeTimestamp, runId, ticker, yfinanceDays]);

    const loadData = async () => {
        setStatus('Loading data...');
        try {
            if (dataSourceMode === 'SIMULATION') {
                await loadSimulationData();
            } else {
                await loadYFinanceData();
            }
        } catch (e: any) {
            setStatus(`Error: ${e.message}`);
        }
    };

    const loadSimulationData = async () => {
        // Load from continuous contract JSON
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
                        volume: b.volume || 0
                    }));

                    allBarsRef.current = loadedBars;

                    let simStartIdx = 0;
                    if (lastTradeTimestamp) {
                        const lastTradeTime = new Date(lastTradeTimestamp).getTime() / 1000;
                        simStartIdx = loadedBars.findIndex(b => b.time >= lastTradeTime);
                        if (simStartIdx === -1) simStartIdx = 0;
                    }
                    setStartIndex(simStartIdx);
                    setCurrentIndex(simStartIdx);
                    setStatus(`Ready (${loadedBars.length} bars from JSON)`);
                    return;
                }
            } catch { }
        }
        setStatus('Failed to load simulation data');
    };

    const loadYFinanceData = async () => {
        // Start live YFinance session
        try {
            // Close any existing connection
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }

            // Reset state for fresh start
            allBarsRef.current = [];
            setBars([]);
            setCurrentIndex(0);
            setStartIndex(0);
            setOcoState(null);
            ocoRef.current = null;
            setTriggers(0);
            setWins(0);
            setLosses(0);
            setCompletedTrades([]);
            setCompletedDecisions([]);
            completedTradesRef.current = [];
            completedDecisionsRef.current = [];

            const session = await api.startLiveReplay(ticker, selectedScanner, yfinanceDays, 10.0, {
                entry_type: entryType,
                stop_method: stopMethod,
                tp_method: tpMethod,
                stop_atr: stopAtr,
                tp_atr: tpAtr
            });
            setStatus(`YFinance session started: ${session.session_id}`);

            // Connect to SSE stream
            const es = new EventSource(`http://localhost:8000/replay/stream/${session.session_id}`);
            eventSourceRef.current = es;

            es.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'BAR') {
                        const bar: BarData = {
                            time: new Date(data.timestamp).getTime() / 1000,
                            open: data.open,
                            high: data.high,
                            low: data.low,
                            close: data.close,
                            volume: data.volume || 0
                        };
                        allBarsRef.current.push(bar);
                        setBars([...allBarsRef.current]);
                        setCurrentIndex(allBarsRef.current.length - 1);

                        // Process OCO exits (SL/TP hit checks)
                        processBar(bar, allBarsRef.current.length - 1);
                    } else if (data.type === 'OCO_OPEN' || (data.type === 'DECISION' && data.triggered)) {
                        // Backend triggered a trade entry
                        const newOco = {
                            entry: data.entry_price || data.price,
                            stop: data.stop_price,
                            tp: data.tp_price,
                            startTime: new Date(data.timestamp || Date.now()).getTime() / 1000,
                            direction: data.direction as 'LONG' | 'SHORT'
                        };
                        ocoRef.current = newOco;
                        setOcoState(newOco);
                        setTriggers(prev => prev + 1);
                    } else if (data.type === 'STATUS') {
                        setStatus(data.message || 'YFinance streaming...');
                    } else if (data.type === 'STREAM_END') {
                        setStatus(`Stream ended (code: ${data.exit_code})`);
                        es.close();
                        eventSourceRef.current = null;
                    } else if (data.type === 'ERROR') {
                        setStatus(`Stream error: ${data.message}`);
                        es.close();
                        eventSourceRef.current = null;
                    }
                } catch (parseErr) {
                    console.error('SSE parse error:', parseErr, event.data);
                }
            };

            es.onerror = (err) => {
                console.error('SSE connection error:', err);
                setStatus('YFinance stream error - check console');
                es.close();
                eventSourceRef.current = null;
            };

            setStatus(`YFinance mode: ${ticker} (streaming...)`);
        } catch (e: any) {
            setStatus(`YFinance error: ${e.message}`);
        }
    };

    const handlePlayPause = useCallback(() => {
        if (playbackState === 'PLAYING') {
            // Pause
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            setPlaybackState('PAUSED');
            setStatus('Paused');
        } else {
            // Play or Resume
            startPlayback();
        }
    }, [playbackState, currentIndex, startIndex]);

    const handleStop = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setPlaybackState('STOPPED');
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
        setStatus('Stopped');
    }, [startIndex]);

    const handleRewind = useCallback(() => {
        // Rewind by 100 bars or to start
        const newIndex = Math.max(startIndex, currentIndex - 100);
        setCurrentIndex(newIndex);

        // If playing, update bars
        if (playbackState === 'PLAYING' || playbackState === 'PAUSED') {
            setBars(allBarsRef.current.slice(startIndex, newIndex + 1));
        }

        setStatus(`Rewound to bar ${newIndex}`);
    }, [currentIndex, startIndex, playbackState]);

    const handleFastForward = useCallback(() => {
        // Fast forward by 100 bars or to end
        const newIndex = Math.min(allBarsRef.current.length - 1, currentIndex + 100);
        setCurrentIndex(newIndex);

        if (playbackState === 'PLAYING' || playbackState === 'PAUSED') {
            setBars(allBarsRef.current.slice(startIndex, newIndex + 1));
        }

        setStatus(`Fast forwarded to bar ${newIndex}`);
    }, [currentIndex, startIndex, playbackState]);

    const handleSeek = useCallback((index: number) => {
        setCurrentIndex(index);
        if (playbackState === 'PLAYING' || playbackState === 'PAUSED') {
            setBars(allBarsRef.current.slice(startIndex, index + 1));
        }
    }, [startIndex, playbackState]);

    const startPlayback = useCallback(() => {
        if (allBarsRef.current.length === 0) {
            setStatus('No data loaded');
            return;
        }

        setPlaybackState('PLAYING');

        let idx = playbackState === 'PAUSED' ? currentIndex : startIndex;
        if (playbackState !== 'PAUSED') {
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
        }

        setStatus('Playing...');

        intervalRef.current = setInterval(() => {
            if (idx >= allBarsRef.current.length) {
                handleStop();
                setStatus('Completed');
                return;
            }

            const bar = allBarsRef.current[idx];
            setBars(prev => [...prev, bar]);
            setCurrentIndex(idx);

            // Process OCO exits and model triggers
            processBar(bar, idx);

            idx++;
        }, speed);
    }, [speed, startIndex, currentIndex, playbackState, handleStop]);

    const processBar = (bar: BarData, idx: number) => {
        // OCO Exit Logic
        if (ocoRef.current) {
            let outcome = '';
            let price = 0;
            const isLong = ocoRef.current.direction === 'LONG';

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
                const barsHeld = Math.max(1, Math.round((bar.time - ocoRef.current.startTime) / 60));

                const decision: VizDecision = {
                    decision_id: tradeId,
                    timestamp: new Date(ocoRef.current.startTime * 1000).toISOString(),
                    bar_idx: 0, index: 0, scanner_id: selectedScanner, scanner_context: {},
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

        // Model Trigger Logic (Entry) - Only for SIMULATION mode with CNN enabled
        // In YFinance mode, the backend (run_live_mode.py) handles strategy triggering
        if (dataSourceModeRef.current === 'SIMULATION' && useCnnModel && !ocoRef.current && idx % 5 === 0 && idx >= 60) {
            const windowBars = allBarsRef.current.slice(Math.max(0, idx - 29), idx + 1);
            const recentBars = allBarsRef.current.slice(Math.max(0, idx - 13), idx + 1);
            const avgRange = recentBars.reduce((sum, b) => sum + (b.high - b.low), 0) / recentBars.length;
            const atr = avgRange || (bar.close * 0.001);

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
                            model_path: selectedModel,
                            threshold: threshold
                        })
                    });
                    return await res.json();
                } catch {
                    return null;
                }
            };

            tryInfer(8000).then(result => {
                if (!result) return tryInfer(8001);
                return result;
            }).then(result => {
                if (result?.triggered && result.direction !== 'NONE' && !ocoRef.current) {
                    const entry = bar.close;
                    const isLong = result.direction === 'LONG';
                    const stop = isLong ? entry - (stopAtr * atr) : entry + (stopAtr * atr);
                    const tp = isLong ? entry + (tpAtr * atr) : entry - (tpAtr * atr);

                    const newOco = {
                        entry,
                        stop,
                        tp,
                        startTime: bar.time,
                        direction: result.direction as 'LONG' | 'SHORT'
                    };
                    ocoRef.current = newOco;
                    setOcoState(newOco);
                    setTriggers(prev => prev + 1);
                }
            }).catch(e => console.error('Infer error:', e));
        }
    };

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        };
    }, []);

    return (
        <div className="fixed inset-0 bg-slate-900 z-50 flex flex-col">
            {/* Header */}
            <div className="h-14 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-4">
                <h1 className="text-white font-bold">Unified Replay Mode</h1>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-400">Data Source:</span>
                        <select
                            value={dataSourceMode}
                            onChange={e => setDataSourceMode(e.target.value as DataSourceMode)}
                            disabled={playbackState === 'PLAYING'}
                            className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                        >
                            <option value="SIMULATION">Simulation (JSON)</option>
                            <option value="YFINANCE">YFinance (API)</option>
                        </select>
                    </div>
                    <button onClick={onClose} className="text-slate-400 hover:text-white">✕ Close</button>
                </div>
            </div>

            <div className="flex-1 flex">
                {/* Main Chart Area */}
                <div className="flex-1 flex flex-col min-h-0">
                    <div className="flex-1 min-h-[400px]">
                        <CandleChart
                            continuousData={bars.length > 0 ? {
                                timeframe: '1m',
                                count: bars.length,
                                bars: bars.map(b => ({
                                    time: new Date(b.time * 1000).toISOString(),
                                    open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume
                                }))
                            } : null}
                            decisions={completedDecisions}
                            activeDecision={null}
                            trade={null}
                            trades={completedTrades}
                            simulationOco={ocoState}
                            forceShowAllTrades={true}
                        />
                    </div>
                </div>

                {/* Right Sidebar - Controls */}
                <div className="w-80 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto">
                    <h2 className="text-sm font-bold text-blue-400 uppercase mb-4">Controls</h2>

                    {/* Playback Controls */}
                    <div className="mb-6">
                        <h3 className="text-xs font-bold text-green-400 uppercase mb-2">Playback</h3>

                        <div className="flex gap-2 mb-3">
                            <button
                                onClick={handlePlayPause}
                                disabled={allBarsRef.current.length === 0}
                                className={`flex-1 font-bold py-2 px-3 rounded text-sm ${allBarsRef.current.length === 0
                                    ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                                    : playbackState === 'PLAYING'
                                        ? 'bg-yellow-600 hover:bg-yellow-500 text-white'
                                        : 'bg-green-600 hover:bg-green-500 text-white'
                                    }`}
                            >
                                {playbackState === 'PLAYING' ? '⏸ Pause' : '▶ Play'}
                            </button>
                            <button
                                onClick={handleStop}
                                disabled={playbackState === 'STOPPED'}
                                className="flex-1 bg-red-600 hover:bg-red-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-bold py-2 px-3 rounded text-sm"
                            >
                                ■ Stop
                            </button>
                        </div>

                        <div className="flex gap-2 mb-3">
                            <button
                                onClick={handleRewind}
                                disabled={playbackState === 'STOPPED'}
                                className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white py-1 px-2 rounded text-xs"
                            >
                                ⏪ -100
                            </button>
                            <button
                                onClick={handleFastForward}
                                disabled={playbackState === 'STOPPED'}
                                className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white py-1 px-2 rounded text-xs"
                            >
                                +100 ⏩
                            </button>
                        </div>

                        <div className="mb-3">
                            <label className="text-xs text-slate-400">Speed (ms per bar)</label>
                            <select
                                value={speed}
                                onChange={e => setSpeed(parseInt(e.target.value))}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            >
                                <option value={500}>Slow (500ms)</option>
                                <option value={200}>Normal (200ms)</option>
                                <option value={100}>Fast (100ms)</option>
                                <option value={50}>Very Fast (50ms)</option>
                                <option value={10}>Max (10ms)</option>
                            </select>
                        </div>

                        {/* Seek Bar */}
                        <div className="mb-3">
                            <label className="text-xs text-slate-400">Position: {currentIndex} / {allBarsRef.current.length}</label>
                            <input
                                type="range"
                                min={startIndex}
                                max={Math.max(startIndex, allBarsRef.current.length - 1)}
                                value={currentIndex}
                                onChange={e => handleSeek(parseInt(e.target.value))}
                                disabled={playbackState === 'STOPPED'}
                                className="w-full"
                            />
                        </div>
                    </div>

                    {/* Data Source Specific Settings */}
                    {dataSourceMode === 'YFINANCE' && (
                        <div className="mb-6">
                            <h3 className="text-xs font-bold text-purple-400 uppercase mb-2">YFinance Settings</h3>
                            <div className="mb-3">
                                <label className="text-xs text-slate-400">Ticker</label>
                                <input
                                    type="text"
                                    value={ticker}
                                    onChange={e => setTicker(e.target.value)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                />
                            </div>
                            <div className="mb-3">
                                <label className="text-xs text-slate-400">Days History</label>
                                <select
                                    value={yfinanceDays}
                                    onChange={e => setYfinanceDays(parseInt(e.target.value))}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                >
                                    <option value={1}>1 day</option>
                                    <option value={3}>3 days</option>
                                    <option value={7}>7 days (max)</option>
                                </select>
                            </div>
                        </div>
                    )}

                    {/* Model Selection - Checkbox + Dropdown */}
                    <div className="mb-6">
                        <h3 className="text-xs font-bold text-cyan-400 uppercase mb-2">Trigger Sources</h3>

                        {/* CNN Model Checkbox */}
                        <div className="mb-3">
                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={useCnnModel}
                                    onChange={e => setUseCnnModel(e.target.checked)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-4 h-4"
                                />
                                Use CNN Model
                            </label>
                            {useCnnModel && (
                                <select
                                    value={selectedModel}
                                    onChange={e => setSelectedModel(e.target.value)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                >
                                    {availableModels.map(model => (
                                        <option key={model} value={model}>{model.split('/').pop()}</option>
                                    ))}
                                </select>
                            )}
                        </div>

                        {/* Pattern Scanner Checkbox */}
                        <div className="mb-3">
                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={usePatternScanner}
                                    onChange={e => setUsePatternScanner(e.target.checked)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-4 h-4"
                                />
                                Use Pattern Scanner
                            </label>
                            {usePatternScanner && (
                                <select
                                    value={selectedScanner}
                                    onChange={e => setSelectedScanner(e.target.value)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                >
                                    <option value="ifvg">IFVG</option>
                                    <option value="ema_cross">EMA Cross</option>
                                    <option value="ema_bounce">EMA Bounce</option>
                                </select>
                            )}
                        </div>

                        {useCnnModel && usePatternScanner && (
                            <p className="text-xs text-yellow-400 mt-1">⚠ Both enabled: requires BOTH to trigger (AND)</p>
                        )}
                    </div>

                    {/* Entry Configuration */}
                    <div className="mb-6">
                        <h3 className="text-xs font-bold text-purple-400 uppercase mb-2">Entry Configuration</h3>

                        <div className="mb-3">
                            <label className="text-xs text-slate-400">Entry Type</label>
                            <div className="flex gap-3 mt-1">
                                <label className="flex items-center gap-1 text-xs text-slate-300 cursor-pointer">
                                    <input
                                        type="radio"
                                        name="entryType"
                                        value="market"
                                        checked={entryType === 'market'}
                                        onChange={() => setEntryType('market')}
                                        disabled={playbackState === 'PLAYING'}
                                    />
                                    Market
                                </label>
                                <label className="flex items-center gap-1 text-xs text-slate-300 cursor-pointer">
                                    <input
                                        type="radio"
                                        name="entryType"
                                        value="limit"
                                        checked={entryType === 'limit'}
                                        onChange={() => setEntryType('limit')}
                                        disabled={playbackState === 'PLAYING'}
                                    />
                                    Limit
                                </label>
                            </div>
                        </div>

                        <div className="mb-3">
                            <label className="text-xs text-slate-400">Stop Placement</label>
                            <select
                                value={stopMethod}
                                onChange={e => setStopMethod(e.target.value as any)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            >
                                <option value="atr">ATR Multiple</option>
                                <option value="swing">Behind Swing</option>
                                <option value="fixed_bars">Fixed Bars</option>
                            </select>
                        </div>

                        <div className="mb-3">
                            <label className="text-xs text-slate-400">Take Profit</label>
                            <select
                                value={tpMethod}
                                onChange={e => setTpMethod(e.target.value as any)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            >
                                <option value="atr">ATR Multiple</option>
                                <option value="r_multiple">R-Multiple</option>
                            </select>
                        </div>
                    </div>

                    {/* OCO Settings */}
                    <div className="mb-6">
                        <h3 className="text-xs font-bold text-orange-400 uppercase mb-2">OCO Settings</h3>
                        <div className="mb-3">
                            <label className="text-xs text-slate-400">Threshold</label>
                            <input
                                type="number"
                                step="0.01"
                                min="0.1"
                                max="0.9"
                                value={threshold}
                                onChange={e => setThreshold(parseFloat(e.target.value) || 0.35)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            />
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
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            />
                        </div>
                        <div className="mb-3">
                            <label className="text-xs text-slate-400">Take Profit (ATR ×)</label>
                            <input
                                type="number"
                                step="0.5"
                                min="0.5"
                                max="20"
                                value={tpAtr}
                                onChange={e => setTpAtr(parseFloat(e.target.value) || 4)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            />
                        </div>
                    </div>

                    {/* Status & Stats */}
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-slate-400">Status:</span>
                            <span className="text-white bg-slate-900 px-2 py-1 rounded text-xs">{status}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Mode:</span>
                            <span className="text-blue-400">{dataSourceMode}</span>
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
                        <div className="flex justify-between">
                            <span className="text-slate-400">Win Rate:</span>
                            <span className="text-cyan-400">
                                {(wins + losses) > 0 ? ((wins / (wins + losses)) * 100).toFixed(1) : '0.0'}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
