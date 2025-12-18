import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { VizDecision, VizTrade } from '../types/viz';

interface CandleChartProps {
    decision: VizDecision | null;
    trade: VizTrade | null;
}

type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h';

// Aggregation helper: turns 1m candles into Xm candles
// data: [open, high, low, close, volume]
const aggregateData = (data: number[][], interval: number): number[][] => {
    if (interval === 1) return data;

    const aggregated: number[][] = [];

    // Iterate backwards to anchor on the most recent data point (decision time)
    // This ensures the last bar on the chart aligns with the decision regardless of total count
    for (let i = data.length - 1; i >= 0; i -= interval) {
        const chunk: number[][] = [];
        // Collect up to 'interval' candles for this period
        for (let j = 0; j < interval; j++) {
            const idx = i - j;
            if (idx >= 0) {
                // We unshift to keep them chronological [oldest ... newest] within the chunk
                chunk.unshift(data[idx]);
            }
        }

        if (chunk.length === 0) continue;

        const open = chunk[0][0]; // Open of the first candle in chunk
        const close = chunk[chunk.length - 1][3]; // Close of the last candle in chunk
        let high = -Infinity;
        let low = Infinity;
        let vol = 0;

        chunk.forEach(c => {
            if (c[1] > high) high = c[1];
            if (c[2] < low) low = c[2];
            vol += c[4];
        });

        // Unshift to result to maintain [oldest ... newest] order for the final array
        aggregated.unshift([open, high, low, close, vol]);
    }
    return aggregated;
};

export const CandleChart: React.FC<CandleChartProps> = ({ decision, trade }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    // References for OCO zones
    const entryZoneRef = useRef<any>(null);
    const stopZoneRef = useRef<any>(null);
    const tpZoneRef = useRef<any>(null);

    const [timeframe, setTimeframe] = useState<Timeframe>('1m');

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
            chart.remove();
        };
    }, []);

    // Update Data when decision or timeframe changes
    useEffect(() => {
        if (!seriesRef.current || !decision) return;

        // Use RAW OHLCV data (not normalized CNN input) for chart display
        // Fall back to x_price_1m for backward compatibility with old data
        const rawData = decision.window?.raw_ohlcv_1m || decision.window?.x_price_1m || (decision as any).x_price_1m;

        if (!rawData || rawData.length === 0) return;

        // Determine aggregation interval
        const intervalMap = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 };
        const interval = intervalMap[timeframe];

        // Process data (already includes history + future in raw_ohlcv_1m)
        const processedData = aggregateData(rawData, interval);

        // Calculate Timestamps
        // raw_ohlcv_1m has 60 bars before decision + 20 after
        // Decision is at index 60 (0-indexed), so bar 60 is decision time
        const decisionBarIndex = Math.min(60, rawData.length - 1);
        const baseTime = new Date(decision.timestamp || Date.now()).getTime() / 1000;

        // Map all bars with correct timestamps
        const chartData = processedData.map((d, i) => {
            // Calculate offset from decision bar after aggregation
            const aggregatedDecisionIdx = Math.floor(decisionBarIndex / interval);
            const offset = aggregatedDecisionIdx - i;
            return {
                time: (baseTime - (offset * interval * 60)) as Time,
                open: d[0],
                high: d[1],
                low: d[2],
                close: d[3],
            };
        });

        seriesRef.current.setData(chartData);

        // Remove old zones
        if (entryZoneRef.current) { chartRef.current?.removeSeries(entryZoneRef.current); entryZoneRef.current = null; }
        if (stopZoneRef.current) { chartRef.current?.removeSeries(stopZoneRef.current); stopZoneRef.current = null; }
        if (tpZoneRef.current) { chartRef.current?.removeSeries(tpZoneRef.current); tpZoneRef.current = null; }

        // Add OCO Zones if present (bounded rectangles, not infinite lines)
        if (decision.oco && chartRef.current) {
            const oco = decision.oco;
            
            // Calculate zone boundaries based on decision time and max_bars
            const aggregatedDecisionIdx = Math.floor(60 / intervalMap[timeframe]);
            const decisionTime = chartData[aggregatedDecisionIdx]?.time;
            
            // Zone extends from decision time to decision time + max_bars (or end of data)
            const maxBarsAggregated = Math.ceil((oco.max_bars || 200) / intervalMap[timeframe]);
            const endIdx = Math.min(aggregatedDecisionIdx + maxBarsAggregated, chartData.length - 1);
            const endTime = chartData[endIdx]?.time;
            
            if (decisionTime && endTime) {
                // Entry zone (small band around entry price)
                const entryBand = oco.atr_at_creation * 0.1; // 0.1 ATR band
                entryZoneRef.current = chartRef.current.addLineSeries({
                    color: 'rgba(59, 130, 246, 0.15)', // blue with transparency
                    lineWidth: 0,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });
                entryZoneRef.current.setData([
                    { time: decisionTime, value: oco.entry_price },
                    { time: endTime, value: oco.entry_price },
                ]);
                
                // Stop zone (area from entry to stop)
                stopZoneRef.current = chartRef.current.addAreaSeries({
                    topColor: 'rgba(239, 68, 68, 0.2)', // red with transparency
                    bottomColor: 'rgba(239, 68, 68, 0.05)',
                    lineColor: 'rgba(239, 68, 68, 0.6)',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });
                stopZoneRef.current.setData([
                    { time: decisionTime, value: oco.stop_price },
                    { time: endTime, value: oco.stop_price },
                ]);
                
                // TP zone (area from entry to TP)
                tpZoneRef.current = chartRef.current.addAreaSeries({
                    topColor: 'rgba(34, 197, 94, 0.2)', // green with transparency
                    bottomColor: 'rgba(34, 197, 94, 0.05)',
                    lineColor: 'rgba(34, 197, 94, 0.6)',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });
                tpZoneRef.current.setData([
                    { time: decisionTime, value: oco.tp_price },
                    { time: endTime, value: oco.tp_price },
                ]);
            }
        }

        // Add Decision Time Marker
        // The decision bar is at decisionBarIndex in original data
        // After aggregation, figure out which chartData index that corresponds to
        const aggregatedDecisionIdx = Math.floor(60 / intervalMap[timeframe]);
        const markerBar = chartData[aggregatedDecisionIdx];

        if (markerBar) {
            seriesRef.current.setMarkers([
                {
                    time: markerBar.time,
                    position: 'aboveBar',
                    color: '#f59e0b', // amber
                    shape: 'arrowDown',
                    text: 'Signal',
                }
            ]);
        }

        if (chartRef.current) {
            chartRef.current.timeScale().fitContent();
        }

    }, [decision, timeframe]);

    return (
        <div className="relative w-full h-full group">
            <div ref={chartContainerRef} className="w-full h-full" />

            {/* Timeframe Controls */}
            <div className="absolute top-3 right-3 flex bg-slate-800 rounded-md border border-slate-700 shadow-lg overflow-hidden z-10">
                {(['1m', '5m', '15m', '1h', '4h'] as Timeframe[]).map((tf) => (
                    <button
                        key={tf}
                        onClick={() => setTimeframe(tf)}
                        className={`px-3 py-1 text-xs font-bold transition-colors ${timeframe === tf
                            ? 'bg-blue-600 text-white'
                            : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                            } ${tf !== '4h' ? 'border-r border-slate-700' : ''}`}
                    >
                        {tf}
                    </button>
                ))}
            </div>
        </div>
    );
};
