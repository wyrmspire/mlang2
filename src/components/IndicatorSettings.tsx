/**
 * Indicator Settings Panel - Vertical Layout
 * 
 * Toggle menu for enabling/disabling chart indicators.
 * Positioned top-left of chart, vertical layout.
 */

import React, { useState } from 'react';
import type { IndicatorSettings, CustomIndicator } from '../features/chart_indicators';

interface IndicatorSettingsProps {
    settings: IndicatorSettings;
    onChange: (settings: IndicatorSettings) => void;
}

interface IndicatorToggle {
    key: keyof Omit<IndicatorSettings, 'customIndicators'>;
    label: string;
    color: string;
}

const PRESET_INDICATORS: IndicatorToggle[] = [
    // EMAs
    { key: 'ema9', label: 'EMA 9', color: '#fbbf24' },   // yellow
    { key: 'ema21', label: 'EMA 21', color: '#3b82f6' }, // blue
    { key: 'ema200', label: 'EMA 200', color: '#ffffff' }, // white

    // VWAP
    { key: 'vwap', label: 'VWAP', color: '#ec4899' }, // pink

    // Bands
    { key: 'atrBands', label: 'ATR Bands', color: '#6b7280' },  // gray
    { key: 'bollingerBands', label: 'Bollinger', color: '#a855f7' }, // purple
    { key: 'donchianChannels', label: 'Donchian', color: '#22d3ee' }, // cyan

    // ADR
    { key: 'adr', label: 'ADR', color: '#f97316' }, // orange
];

export const IndicatorSettingsPanel: React.FC<IndicatorSettingsProps> = ({ settings, onChange }) => {
    const [showCustomForm, setShowCustomForm] = useState(false);
    const [customType, setCustomType] = useState<'ema' | 'sma'>('ema');
    const [customPeriod, setCustomPeriod] = useState<number>(50);
    const [customColor, setCustomColor] = useState<string>('#10b981');

    const toggle = (key: keyof Omit<IndicatorSettings, 'customIndicators'>) => {
        onChange({ ...settings, [key]: !settings[key] });
    };

    const addCustomIndicator = () => {
        const newIndicator: CustomIndicator = {
            id: `${customType}_${customPeriod}_${Date.now()}`,
            type: customType,
            period: customPeriod,
            color: customColor,
        };

        const customIndicators = [...(settings.customIndicators || []), newIndicator];
        onChange({ ...settings, customIndicators });
        setShowCustomForm(false);
    };

    const removeCustomIndicator = (id: string) => {
        const customIndicators = (settings.customIndicators || []).filter(i => i.id !== id);
        onChange({ ...settings, customIndicators });
    };

    return (
        <div className="flex flex-col gap-1 bg-slate-800/90 rounded-lg p-2 min-w-[120px] backdrop-blur-sm">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-700 pb-1 mb-1">
                <span className="text-xs text-slate-400 font-medium">Indicators</span>
                <button
                    onClick={() => setShowCustomForm(!showCustomForm)}
                    className="w-5 h-5 rounded bg-slate-700 hover:bg-slate-600 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
                    title="Add custom indicator"
                >
                    <span className="text-sm">+</span>
                </button>
            </div>

            {/* Custom Indicator Form */}
            {showCustomForm && (
                <div className="bg-slate-700/50 rounded p-2 mb-1 space-y-2">
                    <div className="flex gap-1">
                        <button
                            onClick={() => setCustomType('ema')}
                            className={`flex-1 px-2 py-1 text-[10px] rounded ${customType === 'ema' ? 'bg-blue-600' : 'bg-slate-600'}`}
                        >
                            EMA
                        </button>
                        <button
                            onClick={() => setCustomType('sma')}
                            className={`flex-1 px-2 py-1 text-[10px] rounded ${customType === 'sma' ? 'bg-blue-600' : 'bg-slate-600'}`}
                        >
                            SMA
                        </button>
                    </div>
                    <input
                        type="number"
                        value={customPeriod}
                        onChange={(e) => setCustomPeriod(Number(e.target.value))}
                        className="w-full px-2 py-1 text-xs bg-slate-800 border border-slate-600 rounded"
                        placeholder="Period"
                        min={1}
                        max={500}
                    />
                    <div className="flex items-center gap-2">
                        <input
                            type="color"
                            value={customColor}
                            onChange={(e) => setCustomColor(e.target.value)}
                            className="w-6 h-6 rounded cursor-pointer"
                        />
                        <button
                            onClick={addCustomIndicator}
                            className="flex-1 px-2 py-1 text-[10px] bg-green-600 hover:bg-green-500 rounded"
                        >
                            Add
                        </button>
                    </div>
                </div>
            )}

            {/* Preset Indicators */}
            {PRESET_INDICATORS.map(ind => (
                <button
                    key={ind.key}
                    onClick={() => toggle(ind.key)}
                    className={`
            flex items-center gap-2 px-2 py-1.5 text-xs rounded transition-all w-full text-left
            ${settings[ind.key]
                            ? 'bg-slate-600/80 text-white'
                            : 'text-slate-500 hover:text-slate-300 hover:bg-slate-700/50'
                        }
          `}
                >
                    <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: ind.color, opacity: settings[ind.key] ? 1 : 0.4 }}
                    />
                    <span className="truncate">{ind.label}</span>
                </button>
            ))}

            {/* Custom Indicators */}
            {(settings.customIndicators || []).map(ind => (
                <div
                    key={ind.id}
                    className="flex items-center gap-2 px-2 py-1.5 text-xs bg-slate-600/80 rounded"
                >
                    <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: ind.color }}
                    />
                    <span className="flex-1 truncate">{ind.type.toUpperCase()} {ind.period}</span>
                    <button
                        onClick={() => removeCustomIndicator(ind.id)}
                        className="text-slate-400 hover:text-red-400 text-[10px]"
                    >
                        ✕
                    </button>
                </div>
            ))}
        </div>
    );
};

export const INDICATOR_COLORS = {
    ema9: '#fbbf24',
    ema21: '#3b82f6',
    ema200: '#ffffff',
    vwap: '#ec4899',
    atrBands: '#6b7280',
    bollingerBands: '#a855f7',
    donchianChannels: '#22d3ee',
    adr: '#f97316',
};
