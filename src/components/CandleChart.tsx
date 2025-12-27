import React, { useEffect, useRef, useState, useMemo } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, Time, SeriesMarker } from 'lightweight-charts';
import { VizDecision, VizTrade, ContinuousData, BarData } from '../types/viz';
import { PositionBox, createTradePositionBoxes } from './PositionBox';

interface SimulationOco {
    entry: number;
    stop: number;
    tp: number;
    startTime: number;  // Unix timestamp
}

interface CandleChartProps {
    continuousData: ContinuousData | null;  // Full contract data
    decisions: VizDecision[];               // All decisions for markers
    activeDecision: VizDecision | null;     // Currently selected decision
    trade: VizTrade | null;                 // Active trade for position box
    trades?: VizTrade[];                    // All trades for overlay mode
    simulationOco?: SimulationOco | null;   // OCO state for simulation mode
    // NOTE: forceShowAllTrades and defaultShowAllTrades were REMOVED
    // The "Show All Trades" feature had broken bars_held parsing
}

type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h';


// Aggregation helper for higher timeframes - aligned to time boundaries
const aggregateData = (bars: BarData[], intervalMinutes: number): BarData[] => {
    if (intervalMinutes === 1) return bars;
    if (bars.length === 0) return [];

    const aggregated: BarData[] = [];
    const intervalMs = intervalMinutes * 60 * 1000;

    // Group bars by aligned time bucket
    const buckets = new Map<number, BarData[]>();

    bars.forEach(bar => {
        const barTime = new Date(bar.time).getTime();
        // Align to interval boundary (floor to nearest interval)
        const bucketTime = Math.floor(barTime / intervalMs) * intervalMs;

        if (!buckets.has(bucketTime)) {
            buckets.set(bucketTime, []);
        }
        buckets.get(bucketTime)!.push(bar);
    });

    // Convert buckets to aggregated bars
    const sortedBuckets = Array.from(buckets.entries()).sort((a, b) => a[0] - b[0]);

    for (const [bucketTime, barsInBucket] of sortedBuckets) {
        if (barsInBucket.length === 0) continue;

        const open = barsInBucket[0].open;
        const close = barsInBucket[barsInBucket.length - 1].close;
        let high = -Infinity;
        let low = Infinity;
        let vol = 0;

        barsInBucket.forEach(c => {
            if (c.high > high) high = c.high;
            if (c.low < low) low = c.low;
            vol += c.volume;
        });

        // Use the first bar's time string (it has the correct local part we want)
        aggregated.push({
            time: barsInBucket[0].time,
            open,
            high,
            low,
            close,
            volume: vol
        });
    }
    return aggregated;
};

// Parse ISO time string to Unix timestamp
const parseTime = (timeStr: string): number => {
    if (!timeStr) return 0;

    // If it contains a timezone offset or Z, let the browser parse it
    if (timeStr.match(/([+-]\d{2}:?\d{2}|Z)$/)) {
        const ts = Date.parse(timeStr);
        if (!isNaN(ts)) return Math.floor(ts / 1000);
    }

    // Naive fallback (treats string as UTC components if no timezone info)
    const m = timeStr.match(/(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})/);
    if (!m) return Math.floor(new Date(timeStr).getTime() / 1000);

    return Math.floor(Date.UTC(
        parseInt(m[1]),
        parseInt(m[2]) - 1,
        parseInt(m[3]),
        parseInt(m[4]),
        parseInt(m[5]),
        parseInt(m[6])
    ) / 1000);
};

// Find bar index by timestamp
const findBarIndex = (bars: BarData[], timestamp: string): number => {
    const targetTime = parseTime(timestamp);
    for (let i = 0; i < bars.length; i++) {
        const barTime = parseTime(bars[i].time);
        if (barTime >= targetTime) return i;
    }
    return Math.max(0, bars.length - 1);
};

export const CandleChart: React.FC<CandleChartProps> = ({
    continuousData,
    decisions,
    activeDecision,
    trade,
    trades = [],
    simulationOco,
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    // References for position box primitives
    const activeBoxesRef = useRef<PositionBox[]>([]);
    // NOTE: allTradesBoxesRef was REMOVED - "Show All Trades" feature was broken
    const simOcoBoxesRef = useRef<PositionBox[]>([]);
    const simPriceLinesRef = useRef<any[]>([]);

    const [timeframe, setTimeframe] = useState<Timeframe>('1m');
    const [isLoading, setIsLoading] = useState(true);

    // Process continuous data with current timeframe
    const chartData = useMemo(() => {
        if (!continuousData?.bars?.length) return [];

        const intervalMap: Record<Timeframe, number> = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 };
        const interval = intervalMap[timeframe];
        const aggregated = aggregateData(continuousData.bars, interval);

        return aggregated.map(bar => ({
            time: parseTime(bar.time) as Time,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close
        }));
    }, [continuousData, timeframe]);

    // Get aggregated bars for timestamp lookups
    const aggregatedBars = useMemo(() => {
        if (!continuousData?.bars?.length) return [];
        const intervalMap: Record<Timeframe, number> = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 };
        return aggregateData(continuousData.bars, intervalMap[timeframe]);
    }, [continuousData, timeframe]);

    // Create decision markers
    const decisionMarkers = useMemo((): SeriesMarker<Time>[] => {
        if (!decisions.length || !aggregatedBars.length) return [];

        return decisions
            .filter(d => d.timestamp)
            .map(d => {
                const barIdx = findBarIndex(aggregatedBars, d.timestamp!);
                const bar = aggregatedBars[barIdx];
                if (!bar) return null;

                const isActive = d.decision_id === activeDecision?.decision_id;
                const direction = d.scanner_context?.direction || d.oco?.direction || 'LONG';

                return {
                    time: parseTime(bar.time) as Time,
                    position: direction === 'LONG' ? 'belowBar' : 'aboveBar',
                    color: isActive ? '#f59e0b' : '#6366f1', // amber for active, indigo for others
                    shape: direction === 'LONG' ? 'arrowUp' : 'arrowDown',
                    text: isActive ? 'ACTIVE' : '',
                    size: isActive ? 2 : 1
                } as SeriesMarker<Time>;
            })
            .filter((m): m is SeriesMarker<Time> => m !== null);
    }, [decisions, activeDecision, aggregatedBars]);

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#1e293b' },
                textColor: '#cbd5e1',
            },
            grid: {
                vertLines: { color: '#334155' },
                horzLines: { color: '#334155' },
            },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        const newSeries = chart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderVisible: false,
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
        });

        chartRef.current = chart;
        seriesRef.current = newSeries;

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            // Detach all position boxes
            activeBoxesRef.current.forEach(box => {
                try { seriesRef.current?.detachPrimitive(box); } catch { }
            });
            activeBoxesRef.current = [];
            chart.remove();
        };
    }, []);

    // Update chart data when continuous data or timeframe changes
    useEffect(() => {
        if (!seriesRef.current || !chartData.length) return;

        setIsLoading(true);
        seriesRef.current.setData(chartData);

        // Force resize to pick up container dimensions (fixes simulation view)
        if (chartRef.current && chartContainerRef.current) {
            const width = chartContainerRef.current.clientWidth;
            const height = chartContainerRef.current.clientHeight;
            if (width > 0 && height > 0) {
                chartRef.current.applyOptions({ width, height });
            }
        }

        setIsLoading(false);
    }, [chartData]);

    // Update markers when decisions change
    useEffect(() => {
        if (!seriesRef.current) return;
        seriesRef.current.setMarkers(decisionMarkers);
    }, [decisionMarkers]);

    // ========================================
    // SHOW ALL TRADE BOXES ON LOAD - FIXED 2025-12-25
    // ========================================
    // This useEffect renders position boxes (TP/SL zones) for ALL trades on the chart.
    //
    // CRITICAL DEPENDENCIES:
    // - decisions: Re-run when decision data changes
    // - trades: Re-run when trade data changes (for exit_time/bars_held)
    // - aggregatedBars: Re-run when bar data changes (for timestamp lookups)
    // - continuousData: Re-run when raw data loads
    // - chartData: Re-run when chart data is set (ensures chart is ready)
    // - isLoading: Re-run after loading completes
    // - timeframe: CRITICAL - must re-run when timeframe changes so boxes align correctly
    //
    // PREVIOUS BUG: The old "Show All Trades" toggle had broken bars_held parsing
    // (string vs number types) that caused boxes to extend past actual TP/SL hits.
    // This was fixed by using the same endTime calculation as the active decision path.
    // ========================================
    const allTradesBoxesRef = useRef<PositionBox[]>([]);

    useEffect(() => {
        console.log('[ALL_TRADES] useEffect running, timeframe:', timeframe, 'decisions:', decisions.length, 'trades:', trades.length);

        if (!seriesRef.current || !chartRef.current) {
            console.log('[ALL_TRADES] No chart/series ref');
            return;
        }
        if (!aggregatedBars.length || !continuousData?.bars?.length) {
            console.log('[ALL_TRADES] No bars');
            return;
        }

        // Clear old boxes - MUST do this on every timeframe change
        console.log('[ALL_TRADES] Clearing', allTradesBoxesRef.current.length, 'old boxes');
        allTradesBoxesRef.current.forEach(box => {
            try { seriesRef.current?.detachPrimitive(box); } catch { }
        });
        allTradesBoxesRef.current = [];

        // Get all decisions with OCO data
        const decisionsWithOco = decisions.filter(d => d.oco && d.timestamp);
        console.log('[ALL_TRADES] Decisions with OCO:', decisionsWithOco.length);
        if (!decisionsWithOco.length) return;

        const newBoxes: PositionBox[] = [];

        decisionsWithOco.forEach((decision, idx) => {
            const oco = decision.oco;
            if (!oco?.entry_price || !oco?.stop_price || !oco?.tp_price) return;

            // Find matching trade for this decision (for exit_time/bars_held)
            const matchingTrade = trades.find(t => t.decision_id === decision.decision_id);

            const entryPrice = oco.entry_price;
            const stopPrice = oco.stop_price;
            const tpPrice = oco.tp_price;
            const direction = (decision.scanner_context?.direction || oco.direction || 'LONG') as 'LONG' | 'SHORT';

            // ========================================
            // CRITICAL: Snap times to nearest ACTUAL bars on the chart
            // timeToCoordinate returns null if the exact time doesn't exist
            // On 5m bars, 09:39 doesn't exist - only 09:35, 09:40, etc.
            // So we find the nearest bar and use ITS time
            // ========================================
            const startBarIdx = findBarIndex(aggregatedBars, decision.timestamp!);
            const startBar = aggregatedBars[startBarIdx];
            if (!startBar) return;  // Skip if no matching bar found

            const startTime = parseTime(startBar.time) as Time;

            // End time calculation - snap to nearest bar
            let endTime = startTime;

            if (matchingTrade?.exit_time) {
                // Find the bar at or after exit time
                const endBarIdx = findBarIndex(aggregatedBars, matchingTrade.exit_time);
                const endBar = aggregatedBars[endBarIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            } else if (matchingTrade?.bars_held) {
                // bars_held is in 1-minute bars - need to convert to current timeframe
                const intervalMinutes = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 }[timeframe] || 1;
                const barsOnThisTF = Math.max(1, Math.ceil(matchingTrade.bars_held / intervalMinutes));
                const endBarIdx = Math.min(startBarIdx + barsOnThisTF, aggregatedBars.length - 1);
                const endBar = aggregatedBars[endBarIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            } else if (oco.max_bars) {
                // Fallback: use max_bars converted to current timeframe
                const intervalMinutes = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 }[timeframe] || 1;
                const barsOnThisTF = Math.max(1, Math.ceil(oco.max_bars / intervalMinutes));
                const endBarIdx = Math.min(startBarIdx + barsOnThisTF, aggregatedBars.length - 1);
                const endBar = aggregatedBars[endBarIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            } else {
                // Default: 2 hours (120 minutes) converted to bars
                const intervalMinutes = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 }[timeframe] || 1;
                const barsOnThisTF = Math.max(1, Math.ceil(120 / intervalMinutes));
                const endBarIdx = Math.min(startBarIdx + barsOnThisTF, aggregatedBars.length - 1);
                const endBar = aggregatedBars[endBarIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            }
            const { slBox, tpBox } = createTradePositionBoxes(
                entryPrice,
                stopPrice,
                tpPrice,
                startTime,
                endTime,
                direction,
                decision.decision_id || `all_${idx}`
            );

            seriesRef.current?.attachPrimitive(slBox);
            seriesRef.current?.attachPrimitive(tpBox);
            newBoxes.push(slBox, tpBox);
            console.log('[ALL_TRADES] Attached boxes, total now:', newBoxes.length);
        });

        allTradesBoxesRef.current = newBoxes;
        console.log('[ALL_TRADES] Done! Created', newBoxes.length, 'boxes total');

    }, [decisions, trades, aggregatedBars, continuousData, chartData, isLoading, timeframe]);

    // Handle active decision - scroll to it and show position boxes
    useEffect(() => {
        if (!seriesRef.current || !chartRef.current) return;
        if (!aggregatedBars.length || !continuousData?.bars?.length) return;

        // Remove old active position boxes
        activeBoxesRef.current.forEach(box => {
            try { seriesRef.current?.detachPrimitive(box); } catch { }
        });
        activeBoxesRef.current = [];

        // If no active decision, just clear boxes
        if (!activeDecision?.timestamp) return;

        // Get decision bar
        const decisionIdx = findBarIndex(aggregatedBars, activeDecision.timestamp);
        const decisionBar = aggregatedBars[decisionIdx];
        if (!decisionBar) return;

        // Scroll to decision with context (more bars for better view)
        const fromIdx = Math.max(0, decisionIdx - 50);
        const toIdx = Math.min(aggregatedBars.length - 1, decisionIdx + 50);

        if (aggregatedBars[fromIdx] && aggregatedBars[toIdx]) {
            chartRef.current.timeScale().setVisibleRange({
                from: parseTime(aggregatedBars[fromIdx].time) as Time,
                to: parseTime(aggregatedBars[toIdx].time) as Time
            });
        }

        // Add position boxes if OCO data available
        const oco = activeDecision.oco;
        if (oco) {
            const entryPrice = oco.entry_price;
            const stopPrice = oco.stop_price;
            const tpPrice = oco.tp_price;
            const direction = (activeDecision.scanner_context?.direction || oco.direction || 'LONG') as 'LONG' | 'SHORT';

            // Calculate start time from decision timestamp
            const startTime = parseTime(decisionBar.time) as Time;

            // Calculate end time based on trade data or estimate
            let endTime = startTime;

            if (trade?.exit_time) {
                // Use actual exit timestamp from trade
                endTime = parseTime(trade.exit_time) as Time;
            } else if (trade?.bars_held) {
                // Estimate end time: decision time + bars_held minutes (assuming 1m bars)
                const decisionTimeMs = parseTime(decisionBar.time) * 1000;
                const endTimeMs = decisionTimeMs + (trade.bars_held * 60 * 1000);
                endTime = Math.floor(endTimeMs / 1000) as Time;
            } else if (oco.max_bars) {
                // Fallback: use max_bars from OCO
                const decisionTimeMs = parseTime(decisionBar.time) * 1000;
                const endTimeMs = decisionTimeMs + (oco.max_bars * 60 * 1000);
                endTime = Math.floor(endTimeMs / 1000) as Time;
            } else {
                // Default: 50 bars (50 minutes)
                const decisionTimeMs = parseTime(decisionBar.time) * 1000;
                const endTimeMs = decisionTimeMs + (50 * 60 * 1000);
                endTime = Math.floor(endTimeMs / 1000) as Time;
            }

            // Create position boxes with actual timestamps
            const { slBox, tpBox } = createTradePositionBoxes(
                entryPrice,
                stopPrice,
                tpPrice,
                startTime,
                endTime,
                direction,
                activeDecision.decision_id
            );

            // Attach primitives to series
            seriesRef.current.attachPrimitive(slBox);
            seriesRef.current.attachPrimitive(tpBox);

            activeBoxesRef.current = [slBox, tpBox];
        }

    }, [activeDecision, trade, aggregatedBars, timeframe, continuousData]);

    // Render simulation OCO position boxes
    // Render simulation OCO price lines
    useEffect(() => {
        if (!seriesRef.current) return;

        // Clear existing simulation lines
        simPriceLinesRef.current.forEach(line => {
            try { seriesRef.current?.removePriceLine(line); } catch { }
        });
        simPriceLinesRef.current = [];

        // If no simulation OCO, we're done
        if (!simulationOco) return;

        // Entry Line
        const entryLine = seriesRef.current.createPriceLine({
            price: simulationOco.entry,
            color: '#3b82f6', // Blue
            lineWidth: 2,
            lineStyle: 0, // Solid
            axisLabelVisible: true,
            title: '',
        });

        // TP Line
        const tpLine = seriesRef.current.createPriceLine({
            price: simulationOco.tp,
            color: '#22c55e', // Green
            lineWidth: 2,
            lineStyle: 0, // Solid
            axisLabelVisible: true,
            title: '',
        });

        // SL Line
        const slLine = seriesRef.current.createPriceLine({
            price: simulationOco.stop,
            color: '#ef4444', // Red
            lineWidth: 2,
            lineStyle: 0, // Solid
            axisLabelVisible: true,
            title: '',
        });

        simPriceLinesRef.current = [entryLine, tpLine, slLine];

    }, [simulationOco]);

    return (
        <div className="relative w-full h-full group">
            <div ref={chartContainerRef} className="w-full h-full" />

            {/* Loading indicator */}
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50">
                    <div className="text-slate-400">Loading chart...</div>
                </div>
            )}

            {/* No data message */}
            {!continuousData?.bars?.length && !isLoading && (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-slate-500">
                        <p className="text-lg mb-2">No continuous data loaded</p>
                        <p className="text-sm">Waiting for market data...</p>
                    </div>
                </div>
            )}

            {/* Controls */}
            <div className="absolute top-3 right-3 flex flex-col gap-2 z-10">
                {/* Timeframe Controls */}
                <div className="flex bg-slate-800 rounded-md border border-slate-700 shadow-lg overflow-hidden">
                    {(['1m', '5m', '15m', '1h', '4h'] as Timeframe[]).map((tf, idx, arr) => (
                        <button
                            key={tf}
                            onClick={() => setTimeframe(tf)}
                            className={`px-3 py-1 text-xs font-bold transition-colors ${timeframe === tf
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                                } ${idx !== arr.length - 1 ? 'border-r border-slate-700' : ''}`}
                        >
                            {tf}
                        </button>
                    ))}
                </div>
            </div>

            {/* Decision count overlay */}
            {decisions.length > 0 && (
                <div className="absolute bottom-3 right-3 bg-slate-800/80 px-2 py-1 rounded text-xs text-slate-400 z-10">
                    {decisions.length} decisions
                </div>
            )}
        </div>
    );
};
