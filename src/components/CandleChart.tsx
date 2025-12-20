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
    forceShowAllTrades?: boolean;           // Force showing all trades
}

type Timeframe = '1m' | '5m' | '15m';

// Aggregation helper for higher timeframes
const aggregateData = (bars: BarData[], interval: number): BarData[] => {
    if (interval === 1) return bars;

    const aggregated: BarData[] = [];

    for (let i = 0; i < bars.length; i += interval) {
        const chunk = bars.slice(i, i + interval);
        if (chunk.length === 0) continue;

        const open = chunk[0].open;
        const close = chunk[chunk.length - 1].close;
        let high = -Infinity;
        let low = Infinity;
        let vol = 0;

        chunk.forEach(c => {
            if (c.high > high) high = c.high;
            if (c.low < low) low = c.low;
            vol += c.volume;
        });

        aggregated.push({
            time: chunk[0].time,
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
    return Math.floor(new Date(timeStr).getTime() / 1000);
};

// Find bar index by timestamp
const findBarIndex = (bars: BarData[], timestamp: string): number => {
    const targetTime = parseTime(timestamp);
    for (let i = 0; i < bars.length; i++) {
        const barTime = parseTime(bars[i].time);
        if (barTime >= targetTime) return i;
    }
    return bars.length - 1;
};

export const CandleChart: React.FC<CandleChartProps> = ({
    continuousData,
    decisions,
    activeDecision,
    trade,
    trades = [],
    simulationOco,
    forceShowAllTrades = false
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    // References for position box primitives
    const activeBoxesRef = useRef<PositionBox[]>([]);
    const allTradesBoxesRef = useRef<Map<string, PositionBox[]>>(new Map());
    const simOcoBoxesRef = useRef<PositionBox[]>([]);
    const simPriceLinesRef = useRef<any[]>([]); // Store price line objects

    const [timeframe, setTimeframe] = useState<Timeframe>('1m');
    const [isLoading, setIsLoading] = useState(true);
    const [showAllTrades, setShowAllTrades] = useState(false);

    // Process continuous data with current timeframe
    const chartData = useMemo(() => {
        if (!continuousData?.bars?.length) return [];

        const intervalMap = { '1m': 1, '5m': 5, '15m': 15 };
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
        const intervalMap = { '1m': 1, '5m': 5, '15m': 15 };
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
            allTradesBoxesRef.current.forEach(boxes => {
                boxes.forEach(box => {
                    try { seriesRef.current?.detachPrimitive(box); } catch { }
                });
            });
            allTradesBoxesRef.current.clear();
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

    // Render all trades as position boxes
    useEffect(() => {
        const shouldShowAll = showAllTrades || forceShowAllTrades;
        const interval = timeframe === '5m' ? 5 : timeframe === '15m' ? 15 : 1;

        if (!seriesRef.current) return;

        // Always clear first to handle updates or toggling off
        allTradesBoxesRef.current.forEach(boxes => {
            boxes.forEach(box => {
                try { seriesRef.current?.detachPrimitive(box); } catch { }
            });
        });
        allTradesBoxesRef.current.clear();

        if (!shouldShowAll) return;
        if (!aggregatedBars.length || !continuousData?.bars?.length) return;
        if (!trades.length) return;

        // Create boxes for each trade
        trades.forEach(t => {
            const decision = decisions.find(d => d.decision_id === t.decision_id);
            if (!decision?.oco || !decision.timestamp) return;

            const oco = decision.oco;
            // Snap start time to current timeframe interval
            const rawStartIdx = findBarIndex(continuousData.bars, decision.timestamp);
            const snappedStartIdx = Math.floor(rawStartIdx / interval);
            const snappedStartBar = aggregatedBars[Math.min(snappedStartIdx, aggregatedBars.length - 1)];
            if (!snappedStartBar) return;
            const startTime = parseTime(snappedStartBar.time) as Time;

            // Calculate end time
            let endTime = startTime;
            if (t.exit_time) {
                const rawExitIdx = findBarIndex(continuousData.bars, t.exit_time);
                const snappedExitIdx = Math.floor(rawExitIdx / interval);
                const snappedExitBar = aggregatedBars[Math.min(snappedExitIdx, aggregatedBars.length - 1)];
                if (snappedExitBar) {
                    endTime = parseTime(snappedExitBar.time) as Time;
                }
            } else if (t.bars_held) {
                // Estimate based on bars held (adjusted for interval)
                const barsHeldAdjusted = Math.ceil(t.bars_held / interval);
                const endIdx = Math.min(snappedStartIdx + barsHeldAdjusted, aggregatedBars.length - 1);
                const endBar = aggregatedBars[endIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            }

            const direction = (decision.scanner_context?.direction || oco.direction || 'LONG') as 'LONG' | 'SHORT';

            const { slBox, tpBox } = createTradePositionBoxes(
                oco.entry_price,
                oco.stop_price,
                oco.tp_price,
                startTime,
                endTime,
                direction,
                t.trade_id || t.decision_id
            );

            // Attach boxes
            seriesRef.current?.attachPrimitive(slBox);
            seriesRef.current?.attachPrimitive(tpBox);

            allTradesBoxesRef.current.set(t.trade_id || t.decision_id, [slBox, tpBox]);
        });

    }, [showAllTrades, forceShowAllTrades, trades, decisions, aggregatedBars, continuousData, timeframe]);

    // Handle active decision - scroll to it and show position boxes
    useEffect(() => {
        if (!seriesRef.current || !chartRef.current) return;
        if (!aggregatedBars.length || !continuousData?.bars?.length) return;

        // Remove old active position boxes (unless showing all trades)
        if (!showAllTrades) {
            activeBoxesRef.current.forEach(box => {
                try { seriesRef.current?.detachPrimitive(box); } catch { }
            });
            activeBoxesRef.current = [];
        }

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

            // Only show active trade boxes if not showing all trades
            if (!showAllTrades) {
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
        }

    }, [activeDecision, trade, aggregatedBars, timeframe, continuousData, showAllTrades]);

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
                    {(['1m', '5m', '15m'] as Timeframe[]).map((tf) => (
                        <button
                            key={tf}
                            onClick={() => setTimeframe(tf)}
                            className={`px-3 py-1 text-xs font-bold transition-colors ${timeframe === tf
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                                } ${tf !== '15m' ? 'border-r border-slate-700' : ''}`}
                        >
                            {tf}
                        </button>
                    ))}
                </div>

                {/* Show All Trades Toggle */}
                {trades.length > 0 && (
                    <button
                        onClick={() => setShowAllTrades(!showAllTrades)}
                        className={`px-3 py-1.5 text-xs font-semibold rounded-md border transition-colors ${showAllTrades
                            ? 'bg-indigo-600 text-white border-indigo-500'
                            : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700 hover:text-slate-200'
                            }`}
                    >
                        {showAllTrades ? `All Trades (${trades.length})` : 'Show All Trades'}
                    </button>
                )}
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
