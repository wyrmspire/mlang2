import React, { useState, useCallback, useRef, useEffect } from 'react';
import { CandleChart } from './CandleChart';
import { VizTrade, VizDecision } from '../types/viz';
import { api } from '../api/client';

interface LiveSessionViewProps {
    onClose: () => void;
    runId?: string;
    lastTradeTimestamp?: string;
    initialMode?: 'SIMULATION' | 'YFINANCE';
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

const SidebarSection: React.FC<{
    title: string;
    children: React.ReactNode;
    defaultOpen?: boolean;
    colorClass?: string;
}> = ({ title, children, defaultOpen = false, colorClass = "text-blue-400" }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="mb-2 border-b border-slate-700 pb-2 last:border-0">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`flex items-center justify-between w-full text-xs font-bold uppercase py-1 ${colorClass} hover:opacity-80`}
            >
                {title}
                <span className="text-slate-500">{isOpen ? '▼' : '▶'}</span>
            </button>
            {isOpen && <div className="mt-2 text-sm">{children}</div>}
        </div>
    );
};

export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
    onClose,
    runId,
    lastTradeTimestamp,
    initialMode = 'SIMULATION'
}) => {
    // Data Source
    const [dataSourceMode, setDataSourceMode] = useState<DataSourceMode>(initialMode);

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
        'models/best_model.pth',
        'models/puller_xgb_4class.json'
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
    const [isLiveStreaming, setIsLiveStreaming] = useState(false);

    // Refs
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const allBarsRef = useRef<BarData[]>([]);
    const ocoRef = useRef<typeof ocoState>(null);
    const completedTradesRef = useRef<VizTrade[]>([]);
    const completedDecisionsRef = useRef<VizDecision[]>([]);
    const eventSourceRef = useRef<EventSource | null>(null);
    const dataSourceModeRef = useRef<DataSourceMode>(initialMode);

    // Load data based on selected mode
    useEffect(() => {
        dataSourceModeRef.current = dataSourceMode;
        if (dataSourceMode === 'SIMULATION') {
            loadSimulationData();
        } else {
            // YFinance: auto-load historical data
            resetState();
            setStatus('Loading YFinance data...');
            fetchYFinanceHistory();
        }
    }, [dataSourceMode, lastTradeTimestamp, runId]);

    const resetState = () => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
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
        setPlaybackState('STOPPED');
        setIsLiveStreaming(false);
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


    // Fetch YFinance history using simple JSON endpoint
    const fetchYFinanceHistory = async (): Promise<boolean> => {
        setStatus(`Fetching ${ticker} data from Yahoo...`);
        try {
            const data = await api.getYFinanceData(ticker, yfinanceDays);

            if (!data.bars || data.bars.length === 0) {
                setStatus(data.message || 'No data returned from YFinance');
                return false;
            }

            const historyBars: BarData[] = data.bars.map((b: any) => ({
                time: new Date(b.time).getTime() / 1000,
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close,
                volume: b.volume || 0
            }));

            console.log('[YFinance] Loaded', historyBars.length, 'bars');
            console.log('[YFinance] First bar:', historyBars[0]);
            console.log('[YFinance] Last bar:', historyBars[historyBars.length - 1]);

            allBarsRef.current = historyBars;
            setStartIndex(0);
            setCurrentIndex(0);
            setStatus(`Ready: ${historyBars.length} bars loaded. Press Play or Go Live.`);
            return true;
        } catch (e: any) {
            setStatus(`YFinance error: ${e.message}`);
            return false;
        }
    };

    // Helper to fetch a single new candle (manual trigger)
    const fetchNextBar = async () => {
        try {
            const data = await api.getYFinanceData(ticker, 1);
            if (data.bars && data.bars.length > 0) {
                const latestBar = data.bars[data.bars.length - 1];
                const latestTime = new Date(latestBar.time).getTime() / 1000;
                const ourLatestTime = allBarsRef.current.length > 0 ? allBarsRef.current[allBarsRef.current.length - 1].time : 0;
                if (latestTime > ourLatestTime) {
                    const newBar: BarData = {
                        time: latestTime,
                        open: latestBar.open,
                        high: latestBar.high,
                        low: latestBar.low,
                        close: latestBar.close,
                        volume: latestBar.volume || 0
                    };
                    console.log('[Manual] New bar fetched:', new Date(latestTime * 1000).toLocaleTimeString());
                    allBarsRef.current.push(newBar);
                    setBars(prev => [...prev, newBar]);
                    setCurrentIndex(prev => prev + 1);
                    processBar(newBar, allBarsRef.current.length - 1);
                    setStatus(`Manual fetch: ${new Date(latestTime * 1000).toLocaleTimeString()}`);
                } else {
                    setStatus('Manual fetch: No newer candle available yet');
                }
            }
        } catch (e) {
            console.error('[Manual] Error fetching next bar:', e);
            setStatus('Error fetching next candle');
        }
    };

    const goLive = async () => {
        // Collect params from UI
        const pctInput = document.getElementById('param_pct') as HTMLInputElement;
        const tfInput = document.getElementById('param_tf') as HTMLSelectElement;

        const entryParams: any = {};
        if (pctInput) entryParams.pct = parseFloat(pctInput.value);
        if (tfInput) entryParams.timeframe = tfInput.value;

        try {
            setStatus('Initializing Backend Session...');
            const config = {
                entry_type: entryType,
                entry_params: entryParams,
                stop_method: stopMethod,
                tp_method: tpMethod,
                stop_atr: stopAtr,
                tp_atr: tpAtr
            };

            const res = await api.startLiveReplay(
                ticker,
                selectedScanner, // Use scanner as strategy
                yfinanceDays,
                speed,
                config
            );

            if (res.session_id) {
                setStatus(`Session Started: ${res.session_id}. Connecting stream...`);
                setupEventSource(res.session_id);
            }
        } catch (e: any) {
            console.error(e);
            setStatus(`Failed to start live session: ${e.message}`);
        }
    };

    const setupEventSource = (sessionId: string) => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
        }

        const url = api.getReplayStreamUrl(sessionId);
        const eventSource = new EventSource(url);
        eventSourceRef.current = eventSource;

        setIsLiveStreaming(true);
        setPlaybackState('PLAYING');

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'BAR') {
                    // Update latest bar
                    const newBar: BarData = {
                        time: new Date(data.timestamp).getTime() / 1000,
                        open: data.data.open,
                        high: data.data.high,
                        low: data.data.low,
                        close: data.data.close,
                        volume: data.data.volume
                    };

                    // Append or update if timestamp matches last
                    setBars(prev => {
                        const last = prev[prev.length - 1];
                        if (last && last.time === newBar.time) {
                            // Update existing (unlikely in this stream but possible)
                            const updated = [...prev];
                            updated[updated.length - 1] = newBar;
                            allBarsRef.current = updated;
                            return updated;
                        } else {
                            // Append
                            const updated = [...prev, newBar];
                            allBarsRef.current = updated;
                            return updated;
                        }
                    });
                    setCurrentIndex(prev => prev + 1); // Auto scroll?
                    setStatus(`Live: ${new Date(newBar.time * 1000).toLocaleTimeString()}`);
                }
                else if (data.type === 'DECISION') {
                    // Backend triggered a pattern
                    // We can visualize this?
                    // For now, key is ORDER_SUBMIT usually follows
                }
                else if (data.type === 'ORDER_SUBMIT') {
                    const ocoData = data.data;
                    const newOco = {
                        entry: ocoData.entry_price,
                        stop: ocoData.stop_price,
                        tp: ocoData.tp_price,
                        startTime: new Date(data.timestamp).getTime() / 1000,
                        direction: ocoData.direction
                    };
                    ocoRef.current = newOco;
                    setOcoState(newOco);
                    setTriggers(prev => prev + 1);
                }
                else if (data.type === 'FILL') {
                    // Trade completed
                    // We construct VizTrade from event data or wait for full update?
                    // Simpler: Backend manages state, we just visualize active OCO closure
                    // OCOEngine (Py) emits outcome.
                    // But here we just clear the OCO box for now, 
                    // ideally we get the trade result from the backend event.

                    // For now, client logic clears OCO if it sees price hit levels? No, backend tells us.
                    ocoRef.current = null;
                    setOcoState(null);

                    // Note: Ideally we parse PnL from FILL event to update stats
                    const pnl = data.data.pnl_dollars || 0;
                    if (pnl > 0) setWins(prev => prev + 1);
                    else setLosses(prev => prev + 1);
                }
                else if (data.type === 'STREAM_END') {
                    eventSource.close();
                    setPlaybackState('STOPPED');
                    setStatus('Session Ends.');
                    setIsLiveStreaming(false);
                }
            } catch (e) {
                console.error('SSE Parse Error', e);
            }
        };

        eventSource.onerror = (e) => {
            console.error('SSE Error', e);
            eventSource.close();
            setIsLiveStreaming(false);
            setPlaybackState('STOPPED');
            setStatus('Connection connection lost.');
        };
    };

    // UI button for manual candle fetch (place near other controls)
    const manualFetchButton = (
        <button
            className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded"
            onClick={fetchNextBar}
            disabled={dataSourceMode !== 'YFINANCE' || isLiveStreaming === false}
        >
            Fetch Next Candle
        </button>
    );


    const handlePlayPause = useCallback(async () => {
        // In live streaming mode, Play/Pause has no effect - use Stop to exit
        if (isLiveStreaming) {
            return;
        }

        if (playbackState === 'PLAYING') {
            // Pause
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            setPlaybackState('PAUSED');
            setStatus('Paused');
        } else {
            // For YFinance mode: fetch history first if not loaded
            if (dataSourceMode === 'YFINANCE' && allBarsRef.current.length === 0) {
                const success = await fetchYFinanceHistory();
                if (!success) return;
            }
            // Play or Resume - same behavior for both Simulation and YFinance
            startPlayback();
        }
    }, [playbackState, currentIndex, startIndex, dataSourceMode, isLiveStreaming]);

    const handleStop = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
        }
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        setPlaybackState('STOPPED');
        setIsLiveStreaming(false);
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

        // IMPORTANT: Check if resuming BEFORE setting state to PLAYING
        const isResuming = playbackState === 'PAUSED';

        // Clear any existing interval first
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        setPlaybackState('PLAYING');

        // If resuming, start from current position. Otherwise reset.
        let idx: number;
        if (isResuming) {
            // Resume from where we left off (currentIndex is already the next bar to show)
            idx = currentIndex + 1;
            setStatus('Resuming...');
        } else {
            // Fresh start
            idx = startIndex;
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
            setStatus('Playing...');
        }

        intervalRef.current = setInterval(() => {
            if (idx >= allBarsRef.current.length) {
                // In YFinance mode: Wait for new bars instead of stopping
                if (dataSourceModeRef.current === 'YFINANCE') {
                    const lastBar = allBarsRef.current[allBarsRef.current.length - 1];
                    const lastTime = lastBar ? new Date(lastBar.time * 1000).toLocaleTimeString() : 'N/A';
                    setStatus(`Live: Waiting for new candle... (Last: ${lastTime})`);
                    return; // Skip this tick, but keep interval alive!
                }

                // In Simulation mode: Stop as usual
                if (intervalRef.current) {
                    clearInterval(intervalRef.current);
                    intervalRef.current = null;
                }
                setPlaybackState('STOPPED');
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
    }, [speed, startIndex, currentIndex, playbackState]);

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

        // Trigger Logic (Entry) - SIMULATION mode only, either CNN model or pattern scanner
        // In YFinance mode, the backend (session_live.py) handles strategy triggering
        const canTrigger = dataSourceModeRef.current === 'SIMULATION' && !ocoRef.current && idx % 5 === 0 && idx >= 60;
        const shouldUseCnn = canTrigger && useCnnModel;
        const shouldUseScanner = canTrigger && usePatternScanner;

        // Pattern Scanner Trigger (simple local implementation)
        if (shouldUseScanner && !shouldUseCnn) {
            const recentBars = allBarsRef.current.slice(Math.max(0, idx - 13), idx + 1);
            const avgRange = recentBars.reduce((sum, b) => sum + (b.high - b.low), 0) / recentBars.length;
            const atr = avgRange || (bar.close * 0.001);

            // Simple pattern detection based on selected scanner
            let triggered = false;
            let direction: 'LONG' | 'SHORT' | null = null;

            if (selectedScanner === 'ema_cross' && recentBars.length >= 9) {
                // Simple EMA cross check
                const closes = recentBars.map(b => b.close);
                const fast = closes.slice(-3).reduce((a, b) => a + b, 0) / 3;
                const slow = closes.slice(-9).reduce((a, b) => a + b, 0) / 9;
                const prevFast = closes.slice(-4, -1).reduce((a, b) => a + b, 0) / 3;
                const prevSlow = closes.slice(-10, -1).reduce((a, b) => a + b, 0) / 9;

                if (prevFast <= prevSlow && fast > slow) {
                    triggered = true;
                    direction = 'LONG';
                } else if (prevFast >= prevSlow && fast < slow) {
                    triggered = true;
                    direction = 'SHORT';
                }
            } else if (selectedScanner === 'ifvg' && recentBars.length >= 3) {
                // Simple IFVG detection (fair value gap)
                const b1 = recentBars[recentBars.length - 3];
                const b2 = recentBars[recentBars.length - 2];
                const b3 = recentBars[recentBars.length - 1];

                // Bullish FVG: gap between bar1 high and bar3 low
                if (b1.high < b3.low && b2.close > b2.open) {
                    triggered = true;
                    direction = 'LONG';
                }
                // Bearish FVG: gap between bar1 low and bar3 high  
                else if (b1.low > b3.high && b2.close < b2.open) {
                    triggered = true;
                    direction = 'SHORT';
                }
            }

            if (triggered && direction) {
                const entry = bar.close;
                const isLong = direction === 'LONG';
                const stop = isLong ? entry - (stopAtr * atr) : entry + (stopAtr * atr);
                const tp = isLong ? entry + (tpAtr * atr) : entry - (tpAtr * atr);

                const newOco = { entry, stop, tp, startTime: bar.time, direction };
                ocoRef.current = newOco;
                setOcoState(newOco);
                setTriggers(prev => prev + 1);
            }
        }

        // CNN Model Trigger (calls backend /infer endpoint)
        if (shouldUseCnn) {
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

    // Restart interval when speed changes during playback
    useEffect(() => {
        if (playbackState === 'PLAYING' && intervalRef.current && !isLiveStreaming) {
            // Clear old interval
            clearInterval(intervalRef.current);
            intervalRef.current = null;

            // Create new interval with updated speed, starting from current position + 1
            let idx = currentIndex + 1;
            intervalRef.current = setInterval(() => {
                if (idx >= allBarsRef.current.length) {
                    if (intervalRef.current) {
                        clearInterval(intervalRef.current);
                        intervalRef.current = null;
                    }
                    setPlaybackState('STOPPED');
                    setStatus('Completed');
                    return;
                }

                const bar = allBarsRef.current[idx];
                setBars(prev => [...prev, bar]);
                setCurrentIndex(idx);
                processBar(bar, idx);
                idx++;
            }, speed);

            setStatus(`Speed changed to ${speed}ms`);
        }
    }, [speed]);

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
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
                <h1 className="text-white font-bold">Live Session Mode</h1>
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

            <div className="flex-1 flex overflow-hidden min-h-0">
                {/* Main Chart Area */}
                <div className="flex-1 flex flex-col min-h-0">
                    <div className="flex-1 min-h-[400px]">
                        {/* Debug: Log first few bars to console */}
                        {bars.length > 0 && console.log('[Chart Input] First 3 bars:', bars.slice(0, 3).map(b => ({
                            time: b.time,
                            timeISO: new Date(b.time * 1000).toISOString(),
                            open: b.open, high: b.high, low: b.low, close: b.close,
                            range: b.high - b.low
                        })))}
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
                        />
                    </div>
                </div>

                {/* Right Sidebar - Controls */}
                <div className="w-80 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto min-h-0 max-h-full">

                    {/* Playback Controls */}
                    <SidebarSection title="Playback" defaultOpen={true} colorClass="text-green-400">
                        <div className="flex gap-2 mb-3">
                            <button
                                onClick={handlePlayPause}
                                disabled={dataSourceMode === 'SIMULATION' && allBarsRef.current.length === 0}
                                className={`flex-1 font-bold py-2 px-3 rounded text-sm ${(dataSourceMode === 'SIMULATION' && allBarsRef.current.length === 0)
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

                        {/* Go Live button - YFinance only */}
                        {dataSourceMode === 'YFINANCE' && (
                            <div className="mb-3">
                                <button
                                    onClick={goLive}
                                    disabled={isLiveStreaming}
                                    className={`w-full font-bold py-2 px-3 rounded text-sm ${isLiveStreaming
                                        ? 'bg-orange-700 text-orange-300 cursor-not-allowed'
                                        : 'bg-orange-600 hover:bg-orange-500 text-white'
                                        }`}
                                >
                                    {isLiveStreaming ? '📡 Live Streaming...' : '⏩ Go Live (Realtime)'}
                                </button>
                            </div>
                        )}

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
                            <label className="text-xs text-slate-400 mb-1 block">Speed (ms per bar)</label>
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

                        <div className="mb-1">
                            <label className="text-xs text-slate-400 mb-1 block">
                                Position: {currentIndex} / {allBarsRef.current.length}
                            </label>
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
                    </SidebarSection>

                    {/* Data Source Specific Settings */}
                    {dataSourceMode === 'YFINANCE' && (
                        <SidebarSection title="YFinance Settings" defaultOpen={true} colorClass="text-purple-400">
                            <div className="mb-1">
                                <label className="text-xs text-slate-400 mb-1 block">Ticker</label>
                                <input
                                    type="text"
                                    value={ticker}
                                    onChange={e => setTicker(e.target.value)}
                                    disabled={playbackState === 'PLAYING' || isLiveStreaming}
                                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                />
                            </div>
                        </SidebarSection>
                    )}

                    {/* Model Selection */}
                    <SidebarSection title="Trigger Sources" colorClass="text-cyan-400">
                        <div className="mb-3">
                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer mb-1">
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

                        <div className="mb-1">
                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer mb-1">
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
                    </SidebarSection>

                    {/* Entry Configuration */}
                    <SidebarSection title="Entry Configuration" colorClass="text-purple-400">
                        <div className="mb-3">
                            <label className="text-xs text-slate-400 mb-1 block">Entry Strategy</label>
                            <select
                                value={entryType}
                                onChange={e => setEntryType(e.target.value as any)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white mb-2"
                            >
                                <option value="market">Market</option>
                                <option value="limit">Limit (Legacy)</option>
                                <option value="retrace_signal">Retrace (Signal Bar)</option>
                                <option value="retrace_timeframe">Retrace (Timeframe)</option>
                                <option value="breakout">Breakout</option>
                            </select>

                            {/* Dynamic Params based on strategy */}
                            {(entryType === 'retrace_signal' || entryType === 'retrace_timeframe') && (
                                <div className="mb-2 pl-2 border-l-2 border-slate-700">
                                    <label className="text-xs text-slate-500 mb-1 block">Retrace % (0.0 - 1.0)</label>
                                    <input
                                        type="number"
                                        step="0.1"
                                        min="0.1"
                                        max="1.0"
                                        defaultValue="0.5"
                                        id="param_pct"
                                        className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                                    />
                                </div>
                            )}

                            {entryType === 'retrace_timeframe' && (
                                <div className="mb-2 pl-2 border-l-2 border-slate-700">
                                    <label className="text-xs text-slate-500 mb-1 block">Timeframe</label>
                                    <select
                                        id="param_tf"
                                        defaultValue="15m"
                                        className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                                    >
                                        <option value="5m">5m</option>
                                        <option value="15m">15m</option>
                                        <option value="1h">1h</option>
                                    </select>
                                </div>
                            )}
                        </div>

                        <div className="flex gap-2 mb-2">
                            <button
                                className={`flex-1 px-3 py-1 rounded text-sm font-medium ${isLiveStreaming ? 'bg-red-600 hover:bg-red-500' : 'bg-green-600 hover:bg-green-500'} disabled:opacity-50`}
                                onClick={isLiveStreaming ? handleStop : goLive}
                                disabled={playbackState === 'PLAYING' && !isLiveStreaming}
                            >
                                {isLiveStreaming ? 'Stop Live' : 'Go Live'}
                            </button>
                            {isLiveStreaming && (
                                <button
                                    className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-sm"
                                    onClick={fetchNextBar}
                                    title="Force check for new data"
                                >
                                    Force Fetch
                                </button>
                            )}
                        </div>
                        {dataSourceMode === 'YFINANCE' && (
                            <p className="text-xs text-yellow-500 mb-2">
                                Note: YFinance data is delayed ~15 mins.
                                <br />Last bar: {allBarsRef.current.length > 0 ? new Date(allBarsRef.current[allBarsRef.current.length - 1].time * 1000).toLocaleTimeString() : 'N/A'}
                            </p>
                        )}

                        <div className="mb-3">
                            <label className="text-xs text-slate-400 mb-1 block">Stop Placement</label>
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

                        <div className="mb-1">
                            <label className="text-xs text-slate-400 mb-1 block">Take Profit</label>
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
                    </SidebarSection>

                    {/* OCO Settings */}
                    <SidebarSection title="OCO Settings" colorClass="text-orange-400">
                        <div className="mb-3">
                            <label className="text-xs text-slate-400 mb-1 block">Threshold</label>
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
                            <label className="text-xs text-slate-400 mb-1 block">Stop Loss (ATR ×)</label>
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
                        <div className="mb-1">
                            <label className="text-xs text-slate-400 mb-1 block">Take Profit (ATR ×)</label>
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
                    </SidebarSection>

                    {/* Status & Stats */}
                    <SidebarSection title="Status & Stats" defaultOpen={true} colorClass="text-white">
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
                    </SidebarSection>
                </div>
            </div>
        </div>
    );
};
