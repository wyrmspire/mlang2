/**
 * useIndicators Hook
 * 
 * Calculates all enabled indicators from candle data.
 * Memoized for performance - only recalculates when data or settings change.
 */

import { useMemo } from 'react';
import type { OHLCV, IndicatorSettings, IndicatorPoint, BandPoint, AdrZones } from '../features/chart_indicators';
import {
    calculateEMA,
    calculateSMA,
    calculateVWAP,
    calculateATRBands,
    calculateBollingerBands,
    calculateDonchianChannels,
    calculateADR,
} from '../features/chart_indicators';

export interface IndicatorData {
    ema9: IndicatorPoint[];
    ema21: IndicatorPoint[];
    ema200: IndicatorPoint[];
    vwap: IndicatorPoint[];
    atrBands: BandPoint[];
    bollingerBands: BandPoint[];
    donchianChannels: BandPoint[];
    adr: AdrZones[];
    customIndicators: Map<string, IndicatorPoint[]>;  // id -> data
}

/**
 * Hook to calculate indicators based on settings.
 * Only calculates indicators that are enabled.
 */
export function useIndicators(candles: OHLCV[], settings: IndicatorSettings): IndicatorData {
    return useMemo(() => {
        const data: IndicatorData = {
            ema9: [],
            ema21: [],
            ema200: [],
            vwap: [],
            atrBands: [],
            bollingerBands: [],
            donchianChannels: [],
            adr: [],
            customIndicators: new Map(),
        };

        if (!candles || candles.length < 3) {
            return data;
        }

        // EMAs
        if (settings.ema9) {
            data.ema9 = calculateEMA(candles, 9);
        }
        if (settings.ema21) {
            data.ema21 = calculateEMA(candles, 21);
        }
        if (settings.ema200) {
            data.ema200 = calculateEMA(candles, 200);
        }

        // VWAP
        if (settings.vwap) {
            data.vwap = calculateVWAP(candles);
        }

        // Bands
        if (settings.atrBands) {
            data.atrBands = calculateATRBands(candles, 14, 2);
        }
        if (settings.bollingerBands) {
            data.bollingerBands = calculateBollingerBands(candles, 20, 2);
        }
        if (settings.donchianChannels) {
            data.donchianChannels = calculateDonchianChannels(candles, 20);
        }

        // ADR
        if (settings.adr) {
            data.adr = calculateADR(candles, 14);
        }

        // Custom Indicators
        if (settings.customIndicators) {
            for (const custom of settings.customIndicators) {
                if (custom.type === 'ema') {
                    data.customIndicators.set(custom.id, calculateEMA(candles, custom.period));
                } else if (custom.type === 'sma') {
                    data.customIndicators.set(custom.id, calculateSMA(candles, custom.period));
                }
            }
        }

        return data;
    }, [candles, settings]);
}
