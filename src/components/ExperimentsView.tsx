
import React, { useState, useEffect } from 'react';
import { api } from '../api/client';

interface Experiment {
    run_id: string;
    created_at: string;
    strategy: string;
    config: any;
    total_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_pnl: number;
    avg_pnl_per_trade: number;
}

interface ExperimentsViewProps {
    onOpenRun: (runId: string) => void;
}

export const ExperimentsView: React.FC<ExperimentsViewProps> = ({ onOpenRun }) => {
    const [experiments, setExperiments] = useState<Experiment[]>([]);
    const [loading, setLoading] = useState(true);
    const [sort, setSort] = useState("created_at");

    const loadExperiments = () => {
        setLoading(true);
        api.getExperiments(sort)
            .then(data => {
                setExperiments(data);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });
    };

    useEffect(() => {
        loadExperiments();
    }, [sort]);

    const handleDelete = async (runId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm(`Delete run ${runId}?`)) return;

        try {
            await api.deleteExperiment(runId);
            setExperiments(prev => prev.filter(ex => ex.run_id !== runId));
        } catch (err) {
            alert("Failed to delete run");
        }
    };

    return (
        <div className="h-full flex flex-col bg-slate-900 text-slate-200 p-8 overflow-hidden">
            <div className="flex justify-between items-center mb-6 shrink-0">
                <h1 className="text-2xl font-bold">Experiments & Backtests</h1>
                <div className="flex gap-2">
                    <select
                        value={sort}
                        onChange={(e) => setSort(e.target.value)}
                        className="bg-slate-800 border border-slate-700 rounded px-3 py-1 text-sm outline-none"
                    >
                        <option value="created_at">Date (Newest)</option>
                        <option value="total_pnl">PnL (High to Low)</option>
                        <option value="win_rate">Win Rate (High to Low)</option>
                    </select>
                    <button
                        onClick={loadExperiments}
                        className="bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-sm"
                    >
                        Refresh
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-auto border border-slate-700 rounded-lg bg-slate-800/50">
                <table className="w-full text-left text-sm">
                    <thead className="bg-slate-800 sticky top-0">
                        <tr>
                            <th className="p-4 font-semibold text-slate-400">Run ID</th>
                            <th className="p-4 font-semibold text-slate-400">Date</th>
                            <th className="p-4 font-semibold text-slate-400">Strategy</th>
                            <th className="p-4 font-semibold text-slate-400 text-right">Trades</th>
                            <th className="p-4 font-semibold text-slate-400 text-right">Win Rate</th>
                            <th className="p-4 font-semibold text-slate-400 text-right">PnL ($)</th>
                            <th className="p-4 font-semibold text-slate-400 text-right">Avg Trade</th>
                            <th className="p-4 font-semibold text-slate-400 text-center">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700">
                        {loading ? (
                            <tr><td colSpan={8} className="p-8 text-center text-slate-500">Loading...</td></tr>
                        ) : experiments.length === 0 ? (
                            <tr><td colSpan={8} className="p-8 text-center text-slate-500">No experiments found.</td></tr>
                        ) : (
                            experiments.map(ex => (
                                <tr
                                    key={ex.run_id}
                                    onClick={() => onOpenRun(ex.run_id)}
                                    className="hover:bg-slate-700/50 cursor-pointer transition-colors"
                                >
                                    <td className="p-4 font-mono text-xs text-blue-400">{ex.run_id}</td>
                                    <td className="p-4 text-slate-400 whitespace-nowrap">
                                        {new Date(ex.created_at).toLocaleString()}
                                    </td>
                                    <td className="p-4">
                                        <span className="bg-slate-800 px-2 py-1 rounded text-xs border border-slate-700">
                                            {ex.strategy}
                                        </span>
                                    </td>
                                    <td className="p-4 text-right font-mono">{ex.total_trades}</td>
                                    <td className={`p-4 text-right font-mono font-bold ${ex.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}`}>
                                        {(ex.win_rate * 100).toFixed(1)}%
                                    </td>
                                    <td className={`p-4 text-right font-mono font-bold ${ex.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        ${ex.total_pnl.toFixed(2)}
                                    </td>
                                    <td className="p-4 text-right font-mono text-slate-400">
                                        ${ex.avg_pnl_per_trade.toFixed(2)}
                                    </td>
                                    <td className="p-4 text-center">
                                        <button
                                            onClick={(e) => handleDelete(ex.run_id, e)}
                                            className="text-slate-500 hover:text-red-400 p-1"
                                            title="Delete"
                                        >
                                            🗑️
                                        </button>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
