import React, { useState, useCallback, useRef, useEffect } from 'react';
import { CandleChart } from './CandleChart';
import { VizTrade, VizDecision } from '../types/viz';
import { api } from '../api/client';

interface SimulationViewProps {
    onClose: () => void;
    runId?: string;
    lastTradeTimestamp?: string;
}

interface BarData {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
}

interface PendingOrder {
    order_id: string;
    type: string;
    direction: string;
    price: number;
    status: string;
}

interface ActiveOCO {
    name: string;
    entry_price: number;
    stop_price: number;
    tp_price: number;
    status: string;
}

/**
 * Simulation View 2.0 - Interactive Trading Lab
 * 
 * Backend-owned simulation using the new /sim/* API.
 * The backend runs the full market simulation (OMS), frontend displays state.
 * 
 * Architecture:
 * - Backend: Runs DataStream, Strategy, and OMS
 * - Frontend: Controls + Visualization
 * 
 * Features:
 * - Strategy selection and configuration
 * - Play/Pause/Step controls
 * - Live event log showing OMS activity
 * - Chart rendering with pending orders and active OCOs
 * - Mid-stream parameter updates
 */
export const SimulationView: React.FC<SimulationViewProps> = ({
    onClose,
    runId,
    lastTradeTimestamp
}) => {
    // Session state
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [status, setStatus] = useState('Ready - Configure and start simulation');
    
    // Configuration
    const [strategy, setStrategy] = useState('random');
    const [entryType, setEntryType] = useState('MARKET');
    const [stopAtr, setStopAtr] = useState(1.0);
    const [tpMultiple, setTpMultiple] = useState(1.4);
    const [speed, setSpeed] = useState(200); // ms per step
    
    // Visualization state
    const [bars, setBars] = useState<BarData[]>([]);
    const [events, setEvents] = useState<string[]>([]);
    const [pendingOrders, setPendingOrders] = useState<PendingOrder[]>([]);
    const [activeOCOs, setActiveOCOs] = useState<ActiveOCO[]>([]);
    const [completedTrades, setCompletedTrades] = useState<VizTrade[]>([]);
    const [completedDecisions, setCompletedDecisions] = useState<VizDecision[]>([]);
    
    // Stats
    const [progress, setProgress] = useState(0);
    const [totalOCOs, setTotalOCOs] = useState(0);
    const [activePositions, setActivePositions] = useState(0);
    const [closedPositions, setClosedPositions] = useState(0);
    
    // Playback control
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const eventLogRef = useRef<HTMLDivElement>(null);
    
    // Auto-scroll event log
    useEffect(() => {
        if (eventLogRef.current) {
            eventLogRef.current.scrollTop = eventLogRef.current.scrollHeight;
        }
    }, [events]);
    
    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
            if (sessionId) {
                // Stop session on unmount
                api.stopSimulation(sessionId).catch((error) => {
                    console.warn('Failed to cleanup simulation session:', error);
                });
            }
        };
    }, [sessionId]);
    
    // Start simulation
    const startSimulation = useCallback(async () => {
        try {
            setStatus('Starting simulation...');
            
            // Determine start date (2 days before last trade for context, or default)
            let startDate: string | undefined;
            let endDate: string | undefined;
            
            if (lastTradeTimestamp) {
                const lastTrade = new Date(lastTradeTimestamp);
                const start = new Date(lastTrade.getTime() - 2 * 24 * 60 * 60 * 1000);
                const end = new Date(lastTrade.getTime() + 14 * 24 * 60 * 60 * 1000);
                startDate = start.toISOString().split('T')[0];
                endDate = end.toISOString().split('T')[0];
            }
            
            // Start backend session
            const result = await api.startSimulation({
                strategy_name: strategy,
                config: {
                    entry_type: entryType,
                    stop_atr: stopAtr,
                    tp_multiple: tpMultiple,
                    auto_submit_ocos: true,
                    direction: 'LONG'  // Default to LONG, strategy can override
                },
                start_date: startDate,
                end_date: endDate
            });
            
            setSessionId(result.session_id);
            setStatus(`Session ${result.session_id} started - ${result.data_range.total_bars} bars`);
            setIsRunning(true);
            
            // Clear previous state
            setBars([]);
            setEvents([`Session started: ${result.session_id}`]);
            setCompletedTrades([]);
            setCompletedDecisions([]);
            setPendingOrders([]);
            setActiveOCOs([]);
            setProgress(0);
            
            // Start playback loop
            startPlayback(result.session_id);
            
        } catch (error) {
            setStatus(`Error: ${error}`);
            setIsRunning(false);
        }
    }, [strategy, entryType, stopAtr, tpMultiple, lastTradeTimestamp]);
    
    // Playback loop
    const startPlayback = useCallback((sessId: string) => {
        intervalRef.current = setInterval(async () => {
            try {
                // Step forward 1 bar
                const result = await api.stepSimulation(sessId, 1);
                
                if (result.done) {
                    // Simulation complete
                    stopSimulation();
                    setStatus('Simulation completed');
                    return;
                }
                
                // Update bars
                if (result.bars && result.bars.length > 0) {
                    setBars(prev => [...prev, ...result.bars]);
                }
                
                // Update events
                if (result.events && result.events.length > 0) {
                    setEvents(prev => [...prev, ...result.events]);
                }
                
                // Update state
                if (result.state) {
                    const state = result.state;
                    
                    // Update OMS state
                    if (state.oms) {
                        setPendingOrders(state.oms.pending_orders || []);
                        setActiveOCOs(state.oms.active_ocos || []);
                        setActivePositions(state.oms.open_positions?.length || 0);
                        
                        // Convert completed OCOs to VizTrade records AND VizDecision records
                        if (state.oms.completed_ocos && state.oms.completed_ocos.length > 0) {
                            const newTrades: VizTrade[] = [];
                            const newDecisions: VizDecision[] = [];
                            
                            state.oms.completed_ocos.forEach((oco: any) => {
                                // Determine outcome from status
                                let outcome = 'UNKNOWN';
                                let exit_reason = 'UNKNOWN';
                                if (oco.status === 'STOPPED_OUT') {
                                    outcome = 'LOSS';
                                    exit_reason = 'SL';
                                } else if (oco.status === 'TARGET_HIT') {
                                    outcome = 'WIN';
                                    exit_reason = 'TP';
                                } else if (oco.status === 'TIMED_OUT') {
                                    outcome = 'TIMEOUT';
                                    exit_reason = 'TIMEOUT';
                                }
                                
                                const tradeId = oco.name || 'unknown';
                                
                                // Create VizTrade
                                newTrades.push({
                                    trade_id: tradeId,
                                    decision_id: tradeId,
                                    index: 0,
                                    direction: oco.direction || 'LONG',
                                    size: 1,
                                    entry_time: oco.entry_time,
                                    entry_bar: oco.entry_bar || 0,
                                    entry_price: oco.entry_price || 0,
                                    exit_time: oco.exit_time,
                                    exit_bar: oco.exit_bar || 0,
                                    exit_price: oco.exit_price || 0,
                                    exit_reason: exit_reason,
                                    outcome: outcome,
                                    pnl_points: 0,
                                    pnl_dollars: 0,
                                    r_multiple: 0,
                                    bars_held: oco.bars_in_trade || 0,
                                    mae: oco.mae || 0,
                                    mfe: oco.mfe || 0,
                                    fills: []
                                });
                                
                                // Create matching VizDecision with OCO data
                                newDecisions.push({
                                    decision_id: tradeId,
                                    timestamp: oco.entry_time,
                                    bar_idx: oco.entry_bar || 0,
                                    index: 0,
                                    scanner_id: 'simulation',
                                    scanner_context: {},
                                    action: 'PLACE_ORDER',
                                    skip_reason: '',
                                    current_price: oco.entry_price || 0,
                                    atr: 0,
                                    cf_outcome: outcome,
                                    cf_pnl_dollars: 0,
                                    oco: {
                                        entry_price: oco.entry_price || 0,
                                        stop_price: oco.stop_price || 0,
                                        tp_price: oco.tp_price || 0,
                                        entry_type: 'MARKET',
                                        direction: oco.direction || 'LONG',
                                        reference_type: 'PRICE',
                                        reference_value: 0,
                                        atr_at_creation: 0,
                                        max_bars: 200,
                                        stop_atr: 1.0,
                                        tp_multiple: 1.4
                                    }
                                });
                            });
                            
                            setCompletedTrades(newTrades);
                            setCompletedDecisions(newDecisions);
                        }
                    }
                    
                    // Update stats
                    if (state.stats) {
                        setTotalOCOs(state.stats.total_ocos || 0);
                        setClosedPositions(state.stats.closed_positions || 0);
                    }
                    
                    // Update progress
                    setProgress(state.progress || 0);
                }
                
            } catch (error) {
                console.error('Step error:', error);
                stopSimulation();
                setStatus(`Error during playback: ${error}`);
            }
        }, speed);
    }, [speed]);
    
    // Stop simulation
    const stopSimulation = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsRunning(false);
    }, []);
    
    // Step one bar manually
    const stepOnce = useCallback(async () => {
        if (!sessionId) return;
        
        try {
            const result = await api.stepSimulation(sessionId, 1);
            
            if (result.done) {
                setStatus('Simulation completed');
                return;
            }
            
            // Update bars
            if (result.bars && result.bars.length > 0) {
                setBars(prev => [...prev, ...result.bars]);
            }
            
            // Update events
            if (result.events && result.events.length > 0) {
                setEvents(prev => [...prev, ...result.events]);
            }
            
            // Update state
            if (result.state) {
                const state = result.state;
                
                if (state.oms) {
                    setPendingOrders(state.oms.pending_orders || []);
                    setActiveOCOs(state.oms.active_ocos || []);
                    setActivePositions(state.oms.open_positions?.length || 0);
                    
                    // Convert completed OCOs to VizTrade records AND VizDecision records
                    if (state.oms.completed_ocos && state.oms.completed_ocos.length > 0) {
                        const newTrades: VizTrade[] = [];
                        const newDecisions: VizDecision[] = [];
                        
                        state.oms.completed_ocos.forEach((oco: any) => {
                            // Determine outcome from status
                            let outcome = 'UNKNOWN';
                            let exit_reason = 'UNKNOWN';
                            if (oco.status === 'STOPPED_OUT') {
                                outcome = 'LOSS';
                                exit_reason = 'SL';
                            } else if (oco.status === 'TARGET_HIT') {
                                outcome = 'WIN';
                                exit_reason = 'TP';
                            } else if (oco.status === 'TIMED_OUT') {
                                outcome = 'TIMEOUT';
                                exit_reason = 'TIMEOUT';
                            }
                            
                            const tradeId = oco.name || 'unknown';
                            
                            // Create VizTrade
                            newTrades.push({
                                trade_id: tradeId,
                                decision_id: tradeId,
                                index: 0,
                                direction: oco.direction || 'LONG',
                                size: 1,
                                entry_time: oco.entry_time,
                                entry_bar: oco.entry_bar || 0,
                                entry_price: oco.entry_price || 0,
                                exit_time: oco.exit_time,
                                exit_bar: oco.exit_bar || 0,
                                exit_price: oco.exit_price || 0,
                                exit_reason: exit_reason,
                                outcome: outcome,
                                pnl_points: 0,
                                pnl_dollars: 0,
                                r_multiple: 0,
                                bars_held: oco.bars_in_trade || 0,
                                mae: oco.mae || 0,
                                mfe: oco.mfe || 0,
                                fills: []
                            });
                            
                            // Create matching VizDecision with OCO data
                            newDecisions.push({
                                decision_id: tradeId,
                                timestamp: oco.entry_time,
                                bar_idx: oco.entry_bar || 0,
                                index: 0,
                                scanner_id: 'simulation',
                                scanner_context: {},
                                action: 'PLACE_ORDER',
                                skip_reason: '',
                                current_price: oco.entry_price || 0,
                                atr: 0,
                                cf_outcome: outcome,
                                cf_pnl_dollars: 0,
                                oco: {
                                    entry_price: oco.entry_price || 0,
                                    stop_price: oco.stop_price || 0,
                                    tp_price: oco.tp_price || 0,
                                    entry_type: 'MARKET',
                                    direction: oco.direction || 'LONG',
                                    reference_type: 'PRICE',
                                    reference_value: 0,
                                    atr_at_creation: 0,
                                    max_bars: 200,
                                    stop_atr: 1.0,
                                    tp_multiple: 1.4
                                }
                            });
                        });
                        
                        setCompletedTrades(newTrades);
                        setCompletedDecisions(newDecisions);
                    }
                }
                
                if (state.stats) {
                    setTotalOCOs(state.stats.total_ocos || 0);
                    setClosedPositions(state.stats.closed_positions || 0);
                }
                
                setProgress(state.progress || 0);
            }
            
        } catch (error) {
            setStatus(`Error: ${error}`);
        }
    }, [sessionId]);
    
    // Update parameters mid-stream
    const updateParams = useCallback(async () => {
        if (!sessionId) return;
        
        try {
            await api.updateSimParams(sessionId, {
                entry_type: entryType,
                stop_atr: stopAtr,
                tp_multiple: tpMultiple
            });
            
            setEvents(prev => [...prev, `Parameters updated: ${entryType}, SL=${stopAtr}ATR, TP=${tpMultiple}x`]);
        } catch (error) {
            setStatus(`Error updating params: ${error}`);
        }
    }, [sessionId, entryType, stopAtr, tpMultiple]);
    
    return (
        <div className="fixed inset-0 bg-slate-900 z-50 flex flex-col">
            {/* Header */}
            <div className="h-14 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-4">
                <h1 className="text-white font-bold">Interactive Simulation Lab</h1>
                <button
                    onClick={onClose}
                    className="text-slate-400 hover:text-white"
                >
                    ✕ Close
                </button>
            </div>

            {/* Main content */}
            <div className="flex-1 flex">
                {/* Chart area */}
                <div className="flex-1 flex flex-col min-h-0">
                    <div className="flex-1 min-h-[400px]">
                        <CandleChart
                            continuousData={bars.length > 0 ? {
                                timeframe: '1m',
                                count: bars.length,
                                bars: bars
                            } : null}
                            decisions={completedDecisions}
                            activeDecision={null}
                            trade={null}
                            trades={completedTrades}
                            simulationOco={activeOCOs.length > 0 ? {
                                entry: activeOCOs[0].entry_price,
                                stop: activeOCOs[0].stop_price,
                                tp: activeOCOs[0].tp_price,
                                startTime: 0  // Not used in rendering
                            } : null}
                            forceShowAllTrades={true}
                        />
                    </div>
                    
                    {/* Event log */}
                    <div className="h-32 bg-slate-950 border-t border-slate-700 p-2 overflow-y-auto font-mono text-xs"
                         ref={eventLogRef}>
                        {events.map((event, i) => (
                            <div key={i} className="text-green-400 mb-1">
                                {event}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Controls sidebar */}
                <div className="w-80 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto">
                    <h2 className="text-sm font-bold text-blue-400 uppercase mb-4">Controls</h2>

                    {/* Strategy */}
                    <div className="mb-4">
                        <label className="text-xs text-slate-400 block mb-1">Strategy</label>
                        <select
                            value={strategy}
                            onChange={e => setStrategy(e.target.value)}
                            disabled={isRunning}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                        >
                            <option value="random">Random (Test)</option>
                            <option value="always_long">Always Long (Test)</option>
                            <option value="ifvg_cnn">IFVG CNN (Future)</option>
                        </select>
                    </div>

                    {/* Entry Type */}
                    <div className="mb-4">
                        <label className="text-xs text-slate-400 block mb-1">Entry Type</label>
                        <select
                            value={entryType}
                            onChange={e => setEntryType(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                        >
                            <option value="MARKET">Market</option>
                            <option value="LIMIT">Limit</option>
                        </select>
                    </div>

                    {/* Stop ATR */}
                    <div className="mb-4">
                        <label className="text-xs text-slate-400 block mb-1">
                            Stop Loss (ATR): {stopAtr.toFixed(1)}
                        </label>
                        <input
                            type="range"
                            min="0.5"
                            max="3.0"
                            step="0.1"
                            value={stopAtr}
                            onChange={e => setStopAtr(parseFloat(e.target.value))}
                            className="w-full"
                        />
                    </div>

                    {/* TP Multiple */}
                    <div className="mb-4">
                        <label className="text-xs text-slate-400 block mb-1">
                            Take Profit (Multiple): {tpMultiple.toFixed(1)}
                        </label>
                        <input
                            type="range"
                            min="1.0"
                            max="3.0"
                            step="0.1"
                            value={tpMultiple}
                            onChange={e => setTpMultiple(parseFloat(e.target.value))}
                            className="w-full"
                        />
                    </div>

                    {/* Update Params Button */}
                    {sessionId && (
                        <button
                            onClick={updateParams}
                            className="w-full bg-yellow-600 hover:bg-yellow-500 text-white font-bold py-2 px-4 rounded mb-4"
                        >
                            Update Parameters
                        </button>
                    )}

                    {/* Speed */}
                    <div className="mb-4">
                        <label className="text-xs text-slate-400 block mb-1">Playback Speed</label>
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
                        </select>
                    </div>

                    <div className="border-t border-slate-700 pt-4 mb-4">
                        {/* Play/Stop/Step */}
                        {!sessionId ? (
                            <button
                                onClick={startSimulation}
                                className="w-full bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-4 rounded mb-2"
                            >
                                ▶ Start Simulation
                            </button>
                        ) : (
                            <>
                                {!isRunning ? (
                                    <button
                                        onClick={() => startPlayback(sessionId)}
                                        className="w-full bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-4 rounded mb-2"
                                    >
                                        ▶ Resume
                                    </button>
                                ) : (
                                    <button
                                        onClick={stopSimulation}
                                        className="w-full bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-4 rounded mb-2"
                                    >
                                        ■ Pause
                                    </button>
                                )}
                                
                                <button
                                    onClick={stepOnce}
                                    disabled={isRunning}
                                    className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
                                >
                                    → Step
                                </button>
                            </>
                        )}
                    </div>

                    {/* Status */}
                    <div className="border-t border-slate-700 pt-4">
                        <h3 className="text-xs font-bold text-blue-400 uppercase mb-2">Status</h3>
                        <div className="space-y-2 text-xs">
                            <div className="text-slate-300 break-words">{status}</div>
                            
                            {sessionId && (
                                <>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Progress:</span>
                                        <span className="text-white">{(progress * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Bars:</span>
                                        <span className="text-white">{bars.length}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Total OCOs:</span>
                                        <span className="text-yellow-400">{totalOCOs}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Active OCOs:</span>
                                        <span className="text-yellow-400">{activeOCOs.length}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Active Positions:</span>
                                        <span className="text-cyan-400">{activePositions}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Closed Positions:</span>
                                        <span className="text-slate-400">{closedPositions}</span>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Active OCOs */}
                    {activeOCOs.length > 0 && (
                        <div className="border-t border-slate-700 pt-4 mt-4">
                            <h3 className="text-xs font-bold text-yellow-400 uppercase mb-2">Active OCOs</h3>
                            {activeOCOs.map(oco => (
                                <div key={oco.name} className="bg-yellow-900/30 border border-yellow-600 rounded p-2 mb-2">
                                    <div className="text-yellow-400 font-bold text-xs">{oco.name}</div>
                                    <div className="text-xs text-slate-300 mt-1">
                                        Entry: ${oco.entry_price.toFixed(2)}<br />
                                        Stop: ${oco.stop_price.toFixed(2)}<br />
                                        TP: ${oco.tp_price.toFixed(2)}<br />
                                        Status: {oco.status}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
