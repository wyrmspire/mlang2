import React, { useEffect, useRef, useState, useMemo } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, Time, SeriesMarker } from 'lightweight-charts';
import { VizDecision, VizTrade, ContinuousData, BarData } from '../types/viz';
import { PositionBox, createTradePositionBoxes } from './PositionBox';

interface CandleChartProps {
    continuousData: ContinuousData | null;  // Full contract data
    decisions: VizDecision[];               // All decisions for markers
    activeDecision: VizDecision | null;     // Currently selected decision
    trade: VizTrade | null;                 // Active trade for position box
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
    trade
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    // References for position box primitives
    const positionBoxesRef = useRef<PositionBox[]>([]);

    const [timeframe, setTimeframe] = useState<Timeframe>('1m');
    const [isLoading, setIsLoading] = useState(true);

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
            positionBoxesRef.current.forEach(box => {
                try { seriesRef.current?.detachPrimitive(box); } catch { }
            });
            positionBoxesRef.current = [];
            chart.remove();
        };
    }, []);

    // Update chart data when continuous data or timeframe changes
    useEffect(() => {
        if (!seriesRef.current || !chartData.length) return;

        setIsLoading(true);
        seriesRef.current.setData(chartData);
        setIsLoading(false);
    }, [chartData]);

    // Update markers when decisions change
    useEffect(() => {
        if (!seriesRef.current) return;
        seriesRef.current.setMarkers(decisionMarkers);
    }, [decisionMarkers]);

    // Handle active decision - scroll to it and show position boxes
    useEffect(() => {
        if (!seriesRef.current || !chartRef.current) return;
        if (!aggregatedBars.length || !continuousData?.bars?.length) return;

        // Remove old position boxes
        positionBoxesRef.current.forEach(box => {
            try { seriesRef.current?.detachPrimitive(box); } catch { }
        });
        positionBoxesRef.current = [];

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
            const { slBox, tpBox, entryBox } = createTradePositionBoxes(
                entryPrice,
                stopPrice,
                tpPrice,
                startTime,
                endTime,
                direction
            );

            // Attach primitives to series
            seriesRef.current.attachPrimitive(slBox);
            seriesRef.current.attachPrimitive(tpBox);
            seriesRef.current.attachPrimitive(entryBox);

            positionBoxesRef.current = [slBox, tpBox, entryBox];
        }

    }, [activeDecision, trade, aggregatedBars, timeframe, continuousData]);

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

            {/* Timeframe Controls */}
            <div className="absolute top-3 right-3 flex bg-slate-800 rounded-md border border-slate-700 shadow-lg overflow-hidden z-10">
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

            {/* Decision count overlay */}
            {decisions.length > 0 && (
                <div className="absolute bottom-3 right-3 bg-slate-800/80 px-2 py-1 rounded text-xs text-slate-400 z-10">
                    {decisions.length} decisions
                </div>
            )}
        </div>
    );
};
