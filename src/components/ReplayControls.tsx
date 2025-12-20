import React, { useState, useEffect, useRef, useCallback } from 'react';

interface ReplayControlsProps {
    maxIndex: number;
    currentIndex: number;
    onIndexChange: (index: number) => void;
    onReplayStart: () => void;
    onReplayEnd: () => void;
}

/**
 * Replay Controls - Auto-step through existing decisions
 * 
 * This doesn't load new data - it animates through the bars/decisions
 * already loaded in the chart.
 */
export const ReplayControls: React.FC<ReplayControlsProps> = ({
    maxIndex,
    currentIndex,
    onIndexChange,
    onReplayStart,
    onReplayEnd
}) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState<number>(500); // ms per step
    const [playbackIndex, setPlaybackIndex] = useState<number>(0);

    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    // Start playback
    const startReplay = useCallback(() => {
        if (maxIndex <= 0) return;

        setIsPlaying(true);
        setPlaybackIndex(0);
        onIndexChange(0);
        onReplayStart();

        // Start stepping through
        intervalRef.current = setInterval(() => {
            setPlaybackIndex(prev => {
                const next = prev + 1;
                if (next > maxIndex) {
                    // Reached end
                    stopReplay();
                    return prev;
                }
                onIndexChange(next);
                return next;
            });
        }, speed);
    }, [maxIndex, speed, onIndexChange, onReplayStart]);

    // Stop playback
    const stopReplay = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsPlaying(false);
        onReplayEnd();
    }, [onReplayEnd]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, []);

    // Update speed while playing
    useEffect(() => {
        if (isPlaying && intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = setInterval(() => {
                setPlaybackIndex(prev => {
                    const next = prev + 1;
                    if (next > maxIndex) {
                        stopReplay();
                        return prev;
                    }
                    onIndexChange(next);
                    return next;
                });
            }, speed);
        }
    }, [speed, isPlaying, maxIndex, onIndexChange, stopReplay]);

    const progress = maxIndex > 0 ? (playbackIndex / maxIndex) * 100 : 0;

    return (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
            <h3 className="text-sm font-bold text-blue-400 uppercase mb-3">Replay Mode</h3>

            {/* Speed Control */}
            <div className="mb-3">
                <label className="text-xs text-slate-400">Speed</label>
                <select
                    value={speed}
                    onChange={e => setSpeed(parseInt(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                >
                    <option value={2000}>0.5x (2s per decision)</option>
                    <option value={1000}>1x (1s per decision)</option>
                    <option value={500}>2x (0.5s per decision)</option>
                    <option value={200}>5x (0.2s per decision)</option>
                    <option value={100}>10x (0.1s per decision)</option>
                </select>
            </div>

            {/* Progress Bar */}
            <div className="mb-3">
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Progress</span>
                    <span>{playbackIndex} / {maxIndex}</span>
                </div>
                <div className="w-full bg-slate-900 rounded-full h-2">
                    <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-200"
                        style={{ width: `${progress}%` }}
                    />
                </div>
            </div>

            {/* Controls */}
            <div className="flex gap-2 mb-3">
                {!isPlaying ? (
                    <button
                        onClick={startReplay}
                        disabled={maxIndex <= 0}
                        className={`flex-1 font-bold py-2 px-4 rounded text-sm ${maxIndex > 0
                                ? 'bg-green-600 hover:bg-green-500 text-white'
                                : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                            }`}
                    >
                        ▶ Play
                    </button>
                ) : (
                    <button
                        onClick={stopReplay}
                        className="flex-1 bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-4 rounded text-sm"
                    >
                        ■ Stop
                    </button>
                )}
            </div>

            {/* Status */}
            <div className="text-xs space-y-1">
                <div className="flex justify-between">
                    <span className="text-slate-400">Status:</span>
                    <span className={isPlaying ? 'text-green-400' : 'text-slate-300'}>
                        {maxIndex <= 0 ? 'No data loaded' : isPlaying ? 'Playing...' : 'Ready'}
                    </span>
                </div>
                {maxIndex <= 0 && (
                    <p className="text-slate-500 text-xs mt-2">
                        Select a run with decisions to enable replay.
                    </p>
                )}
            </div>
        </div>
    );
};
