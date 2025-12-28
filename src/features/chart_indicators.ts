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
}

export const DEFAULT_INDICATOR_SETTINGS: IndicatorSettings = {
    ema9: false,
    ema21: false,
    ema200: false,
    vwap: false,
    atrBands: false,
    bollingerBands: false,
    donchianChannels: false,
};
