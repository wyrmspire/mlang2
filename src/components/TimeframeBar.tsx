import React from 'react';

export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h';

interface TimeframeBarProps {
    selected: Timeframe;
    onSelect: (tf: Timeframe) => void;
}

const TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '1h', '4h'];

export const TimeframeBar: React.FC<TimeframeBarProps> = ({ selected, onSelect }) => {
    return (
        <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-1">
            {TIMEFRAMES.map(tf => (
                <button
                    key={tf}
                    onClick={() => onSelect(tf)}
                    className={`px-3 py-1.5 text-xs font-medium rounded transition-all ${selected === tf
                            ? 'bg-blue-600 text-white shadow-lg'
                            : 'text-slate-400 hover:text-white hover:bg-slate-700'
                        }`}
                >
                    {tf}
                </button>
            ))}
        </div>
    );
};
