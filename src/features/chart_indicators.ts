/**
 * Chart Indicators - Reusable calculation module
 * 
 * This module provides indicator calculations that can be used by:
 * - Chart rendering (via useIndicators hook)
 * - Strategy triggers
 * - Backend analysis
 * 
 * All calculations are pure functions for easy testing and reuse.
 */

export interface OHLCV {
    time: number | string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
}

export interface IndicatorPoint {
    time: number | string;
    value: number;
}

export interface BandPoint {
    time: number | string;
    upper: number;
    lower: number;
    middle?: number;
}

// =============================================================================
// EMA (Exponential Moving Average)
// =============================================================================

/**
 * Calculate EMA for a series of candles
 */
export function calculateEMA(candles: OHLCV[], period: number): IndicatorPoint[] {
    if (candles.length < period) return [];

    const multiplier = 2 / (period + 1);
    const result: IndicatorPoint[] = [];

    // Initial SMA for first EMA value
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += candles[i].close;
    }
    let ema = sum / period;

    result.push({ time: candles[period - 1].time, value: ema });

    // Calculate EMA for remaining candles
    for (let i = period; i < candles.length; i++) {
        ema = (candles[i].close - ema) * multiplier + ema;
        result.push({ time: candles[i].time, value: ema });
    }

    return result;
}

// =============================================================================
// VWAP (Volume Weighted Average Price)
// =============================================================================

/**
 * Calculate VWAP - resets at each session start (9:30 ET)
 */
export function calculateVWAP(candles: OHLCV[]): IndicatorPoint[] {
    const result: IndicatorPoint[] = [];

    let cumulativeTPV = 0;  // Cumulative (TP * Volume)
    let cumulativeVolume = 0;
    let lastSessionDate = '';

    for (const candle of candles) {
        // Get session date for reset detection
        const candleDate = typeof candle.time === 'number'
            ? new Date(candle.time * 1000).toDateString()
            : new Date(candle.time).toDateString();

        // Reset at new session
        if (candleDate !== lastSessionDate) {
            cumulativeTPV = 0;
            cumulativeVolume = 0;
            lastSessionDate = candleDate;
        }

        const typicalPrice = (candle.high + candle.low + candle.close) / 3;
        const volume = candle.volume || 1;

        cumulativeTPV += typicalPrice * volume;
        cumulativeVolume += volume;

        const vwap = cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : typicalPrice;
        result.push({ time: candle.time, value: vwap });
    }

    return result;
}

// =============================================================================
// ATR (Average True Range) + Bands
// =============================================================================

/**
 * Calculate ATR
 */
export function calculateATR(candles: OHLCV[], period: number = 14): IndicatorPoint[] {
    if (candles.length < 2) return [];

    const trueRanges: number[] = [];

    // Calculate True Range for each candle
    for (let i = 1; i < candles.length; i++) {
        const high = candles[i].high;
        const low = candles[i].low;
        const prevClose = candles[i - 1].close;

        const tr = Math.max(
            high - low,
            Math.abs(high - prevClose),
            Math.abs(low - prevClose)
        );
        trueRanges.push(tr);
    }

    if (trueRanges.length < period) return [];

    const result: IndicatorPoint[] = [];

    // Initial ATR (SMA of first n TRs)
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += trueRanges[i];
    }
    let atr = sum / period;

    result.push({ time: candles[period].time, value: atr });

    // Smoothed ATR for remaining
    for (let i = period; i < trueRanges.length; i++) {
        atr = (atr * (period - 1) + trueRanges[i]) / period;
        result.push({ time: candles[i + 1].time, value: atr });
    }

    return result;
}

/**
 * Calculate ATR Bands (price ± ATR multiple)
 */
export function calculateATRBands(candles: OHLCV[], period: number = 14, multiple: number = 2): BandPoint[] {
    const atr = calculateATR(candles, period);
    const result: BandPoint[] = [];

    // Map ATR to bands using close price as center
    for (const point of atr) {
        const candle = candles.find(c => c.time === point.time);
        if (candle) {
            result.push({
                time: point.time,
                upper: candle.close + point.value * multiple,
                lower: candle.close - point.value * multiple,
                middle: candle.close
            });
        }
    }

    return result;
}

// =============================================================================
// Bollinger Bands
// =============================================================================

/**
 * Calculate Bollinger Bands (SMA ± std dev)
 */
export function calculateBollingerBands(candles: OHLCV[], period: number = 20, stdDev: number = 2): BandPoint[] {
    if (candles.length < period) return [];

    const result: BandPoint[] = [];

    for (let i = period - 1; i < candles.length; i++) {
        // Calculate SMA
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sum += candles[j].close;
        }
        const sma = sum / period;

        // Calculate Standard Deviation
        let sqSum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sqSum += Math.pow(candles[j].close - sma, 2);
        }
        const std = Math.sqrt(sqSum / period);

        result.push({
            time: candles[i].time,
            upper: sma + stdDev * std,
            lower: sma - stdDev * std,
            middle: sma
        });
    }

    return result;
}

// =============================================================================
// Donchian Channels
// =============================================================================

/**
 * Calculate Donchian Channels (highest high, lowest low over period)
 */
export function calculateDonchianChannels(candles: OHLCV[], period: number = 20): BandPoint[] {
    if (candles.length < period) return [];

    const result: BandPoint[] = [];

    for (let i = period - 1; i < candles.length; i++) {
        let highestHigh = -Infinity;
        let lowestLow = Infinity;

        for (let j = i - period + 1; j <= i; j++) {
            highestHigh = Math.max(highestHigh, candles[j].high);
            lowestLow = Math.min(lowestLow, candles[j].low);
        }

        result.push({
            time: candles[i].time,
            upper: highestHigh,
            lower: lowestLow,
            middle: (highestHigh + lowestLow) / 2
        });
    }

    return result;
}

// =============================================================================
// SMA (Simple Moving Average)
// =============================================================================

/**
 * Calculate SMA for a series of candles
 */
export function calculateSMA(candles: OHLCV[], period: number): IndicatorPoint[] {
    if (candles.length < period) return [];

    const result: IndicatorPoint[] = [];

    for (let i = period - 1; i < candles.length; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sum += candles[j].close;
        }
        result.push({ time: candles[i].time, value: sum / period });
    }

    return result;
}

// =============================================================================
// ADR (Average Daily Range)
// =============================================================================

export interface AdrZones {
    time: number | string;
    resTop: number;      // Resistance zone top
    resBottom: number;   // Resistance zone bottom  
    supTop: number;      // Support zone top (same as resBottom for middle)
    supBottom: number;   // Support zone bottom
    sessionOpen: number; // Session open price (midpoint)
}

/**
 * Calculate ADR zones - outputs ADR levels for EVERY candle so lines render horizontally.
 * 
 * ADR zones are based on average daily range projected from session open.
 * - Resistance zone: sessionOpen + halfRange (14-period) to sessionOpen + halfRange * 0.5 (7-period)
 * - Support zone: sessionOpen - halfRange * 0.5 to sessionOpen - halfRange
 * 
 * @param candles - OHLCV data (should be at your display timeframe)
 * @param period - Lookback period for ADR calculation (default 14)
 */
export function calculateADR(candles: OHLCV[], period: number = 14): AdrZones[] {
    if (candles.length < period + 1) return [];

    // Step 1: Group candles by day and calculate daily high/low/open
    const dailyData: Map<string, { high: number; low: number; open: number; candles: OHLCV[] }> = new Map();

    for (const candle of candles) {
        const date = typeof candle.time === 'number'
            ? new Date(candle.time * 1000).toDateString()
            : new Date(candle.time).toDateString();

        const existing = dailyData.get(date);
        if (existing) {
            existing.high = Math.max(existing.high, candle.high);
            existing.low = Math.min(existing.low, candle.low);
            existing.candles.push(candle);
        } else {
            dailyData.set(date, {
                high: candle.high,
                low: candle.low,
                open: candle.open,
                candles: [candle]
            });
        }
    }

    const days = Array.from(dailyData.values());
    if (days.length < period + 1) return [];

    // Step 2: Calculate ADR for each candle
    const result: AdrZones[] = [];
    let dayIndex = 0;

    for (const [dateStr, dayData] of dailyData) {
        dayIndex++;

        // Need at least 'period' previous days to calculate ADR
        if (dayIndex <= period) continue;

        // Calculate average range from previous 'period' days
        let sumRange = 0;
        let count = 0;
        const prevDays = Array.from(dailyData.values()).slice(dayIndex - period - 1, dayIndex - 1);

        for (const prevDay of prevDays) {
            sumRange += prevDay.high - prevDay.low;
            count++;
        }

        if (count < period) continue;

        const avgRange = sumRange / period;
        const halfRange = avgRange / 2;
        const sessionOpen = dayData.open;

        // Output ADR levels for EVERY candle in this day (creates horizontal lines)
        for (const candle of dayData.candles) {
            result.push({
                time: candle.time,
                resTop: sessionOpen + halfRange,          // Red zone top (14-period)
                resBottom: sessionOpen + halfRange * 0.5, // Red zone bottom (7-period approx)
                supTop: sessionOpen - halfRange * 0.5,    // Green zone top
                supBottom: sessionOpen - halfRange,       // Green zone bottom (14-period)
                sessionOpen,
            });
        }
    }

    return result;
}

// =============================================================================
// Custom Indicator Type
// =============================================================================

export interface CustomIndicator {
    id: string;
    type: 'ema' | 'sma';
    period: number;
    color: string;
}

// =============================================================================
// Indicator Settings Type
// =============================================================================

export interface IndicatorSettings {
    ema9: boolean;
    ema21: boolean;
    ema200: boolean;
    vwap: boolean;
    atrBands: boolean;
    bollingerBands: boolean;
    donchianChannels: boolean;
    adr: boolean;
    customIndicators?: CustomIndicator[];
}

export const DEFAULT_INDICATOR_SETTINGS: IndicatorSettings = {
    ema9: false,
    ema21: false,
    ema200: false,
    vwap: false,
    atrBands: false,
    bollingerBands: false,
    donchianChannels: false,
    adr: false,
    customIndicators: [],
};
