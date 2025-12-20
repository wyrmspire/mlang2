import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';

interface SimChartProps {
    /** Called when simulation needs a new bar */
    onRequestBar?: () => void;
    /** Called when simulation ends */
    onSimulationEnd?: () => void;
}

interface BarData {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
}

interface OCOZone {
    entryPrice: number;
    stopPrice: number;
    tpPrice: number;
    entryTime: number;
    active: boolean;
}

/**
 * Simulation Chart - Bar-by-bar forward simulation
 * 
 * Unlike CandleChart which shows a fixed window, this chart:
 * - Starts empty or with minimal history
 * - Adds bars one at a time
 * - Shows OCO zones when model triggers
 * - Animates OCO resolution
 */
export const SimulationChart: React.FC<SimChartProps> = ({
    onRequestBar,
    onSimulationEnd
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

    const [bars, setBars] = useState<BarData[]>([]);
    const [ocoZone, setOcoZone] = useState<OCOZone | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [currentBar, setCurrentBar] = useState<number>(0);

    // Initialize chart
    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#0f172a' },
                textColor: '#94a3b8',
            },
            grid: {
                vertLines: { color: '#1e293b' },
                horzLines: { color: '#1e293b' },
            },
            width: containerRef.current.clientWidth,
            height: 400,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderUpColor: '#22c55e',
            borderDownColor: '#ef4444',
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
        });

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;

        // Resize handler
        const handleResize = () => {
            if (containerRef.current) {
                chart.resize(containerRef.current.clientWidth, 400);
            }
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Update chart when bars change
    useEffect(() => {
        if (candleSeriesRef.current && bars.length > 0) {
            candleSeriesRef.current.setData(bars.map(b => ({
                time: b.time as any,
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close,
            })));

            // Auto-scroll to latest bar
            chartRef.current?.timeScale().scrollToRealTime();
        }
    }, [bars]);

    // Add a new bar
    const addBar = useCallback((bar: BarData) => {
        setBars(prev => [...prev, bar]);
        setCurrentBar(prev => prev + 1);
    }, []);

    // Show OCO zone
    const showOCO = useCallback((entry: number, stop: number, tp: number) => {
        setOcoZone({
            entryPrice: entry,
            stopPrice: stop,
            tpPrice: tp,
            entryTime: bars.length > 0 ? bars[bars.length - 1].time : Date.now() / 1000,
            active: true,
        });

        // Draw horizontal lines for OCO
        if (candleSeriesRef.current) {
            // Note: In full implementation, would use priceLine API
        }
    }, [bars]);

    // Clear OCO zone
    const clearOCO = useCallback(() => {
        setOcoZone(null);
    }, []);

    // Reset simulation
    const reset = useCallback(() => {
        setBars([]);
        setOcoZone(null);
        setCurrentBar(0);
        setIsRunning(false);
        if (candleSeriesRef.current) {
            candleSeriesRef.current.setData([]);
        }
    }, []);

    return (
        <div className="relative">
            <div ref={containerRef} className="w-full h-[400px]" />

            {/* Overlay info */}
            <div className="absolute top-2 left-2 bg-slate-900/80 px-3 py-2 rounded text-xs">
                <div className="text-slate-400">Bars: <span className="text-white">{bars.length}</span></div>
                {ocoZone && (
                    <div className="text-yellow-400 mt-1">
                        OCO Active: Entry ${ocoZone.entryPrice.toFixed(2)}
                    </div>
                )}
            </div>

            {/* Empty state */}
            {bars.length === 0 && (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-slate-500 text-sm">
                        Waiting for simulation to start...
                    </div>
                </div>
            )}
        </div>
    );
};

// Export utility to control the simulation externally
export interface SimulationController {
    addBar: (bar: BarData) => void;
    showOCO: (entry: number, stop: number, tp: number) => void;
    clearOCO: () => void;
    reset: () => void;
}
