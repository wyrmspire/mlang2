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
    has_viz: boolean;
}

interface ExperimentsViewProps {
    onLoadRun: (runId: string) => void;
}

export const ExperimentsView: React.FC<ExperimentsViewProps> = ({ onLoadRun }) => {
    const [experiments, setExperiments] = useState<Experiment[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [sortBy, setSortBy] = useState<string>('created_at');
    const [sortDesc, setSortDesc] = useState<boolean>(true);
    const [processing, setProcessing] = useState<string | null>(null);

    const fetchExperiments = async () => {
        setLoading(true);
        try {
            const data = await api.getExperiments({
                sort_by: sortBy,
                sort_desc: sortDesc,
                limit: 100
            });
            setExperiments(data.items || []);
        } catch (error) {
            console.error("Failed to fetch experiments", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchExperiments();
    }, [sortBy, sortDesc]);

    const handleDelete = async (runId: string) => {
        if (!window.confirm(`Are you sure you want to delete run ${runId}?`)) return;
        try {
            await api.deleteExperiment(runId);
            fetchExperiments();
        } catch (error) {
            console.error("Failed to delete run", error);
            alert("Failed to delete run");
        }
    };

    const handleVisualize = async (runId: string) => {
        setProcessing(runId);
        try {
            await api.visualizeExperiment(runId);
            // Reload experiments to update 'has_viz' status
            await fetchExperiments();
        } catch (error) {
            console.error("Failed to generate viz", error);
            alert("Failed to generate visualization. Check logs.");
        } finally {
            setProcessing(null);
        }
    };

    const formatPnL = (val: number) => {
        return (
            <span className={val >= 0 ? "text-green-500" : "text-red-500"}>
                ${val.toFixed(2)}
            </span>
        );
    };

    return (
        <div className="flex flex-col h-full bg-gray-900 text-white p-6 overflow-hidden">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold">Experiments & Backtests</h1>
                <button
                    onClick={fetchExperiments}
                    className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition"
                >
                    Refresh
                </button>
            </div>

            <div className="flex-1 overflow-auto bg-gray-800 rounded-lg shadow-lg">
                <table className="w-full text-left border-collapse">
                    <thead className="bg-gray-700 sticky top-0 z-10">
                        <tr>
                            <th className="p-3 font-semibold cursor-pointer hover:bg-gray-600" onClick={() => { setSortBy('created_at'); setSortDesc(!sortDesc); }}>
                                Date {sortBy === 'created_at' && (sortDesc ? '↓' : '↑')}
                            </th>
                            <th className="p-3 font-semibold">Strategy</th>
                            <th className="p-3 font-semibold text-right cursor-pointer hover:bg-gray-600" onClick={() => { setSortBy('total_trades'); setSortDesc(!sortDesc); }}>
                                Trades {sortBy === 'total_trades' && (sortDesc ? '↓' : '↑')}
                            </th>
                            <th className="p-3 font-semibold text-right cursor-pointer hover:bg-gray-600" onClick={() => { setSortBy('win_rate'); setSortDesc(!sortDesc); }}>
                                Win Rate {sortBy === 'win_rate' && (sortDesc ? '↓' : '↑')}
                            </th>
                            <th className="p-3 font-semibold text-right cursor-pointer hover:bg-gray-600" onClick={() => { setSortBy('total_pnl'); setSortDesc(!sortDesc); }}>
                                Total PnL {sortBy === 'total_pnl' && (sortDesc ? '↓' : '↑')}
                            </th>
                            <th className="p-3 font-semibold text-center">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading ? (
                            <tr><td colSpan={6} className="p-8 text-center text-gray-400">Loading...</td></tr>
                        ) : experiments.length === 0 ? (
                            <tr><td colSpan={6} className="p-8 text-center text-gray-400">No experiments found.</td></tr>
                        ) : (
                            experiments.map((exp) => (
                                <tr key={exp.run_id} className="border-b border-gray-700 hover:bg-gray-750 transition-colors">
                                    <td className="p-3 text-gray-300">
                                        <div className="font-mono text-sm">{new Date(exp.created_at).toLocaleString()}</div>
                                        <div className="text-xs text-gray-500 mt-1">{exp.run_id}</div>
                                    </td>
                                    <td className="p-3 font-medium text-blue-300">
                                        {exp.strategy}
                                        {exp.config && exp.config.entry_trigger && (
                                            <div className="text-xs text-gray-400 mt-1 font-mono">
                                                {exp.config.entry_trigger.type} ({JSON.stringify(exp.config.entry_trigger).slice(0, 30)}...)
                                            </div>
                                        )}
                                    </td>
                                    <td className="p-3 text-right font-mono">{exp.total_trades}</td>
                                    <td className="p-3 text-right font-mono">
                                        {(exp.win_rate * 100).toFixed(1)}%
                                    </td>
                                    <td className="p-3 text-right font-mono font-bold">
                                        {formatPnL(exp.total_pnl)}
                                    </td>
                                    <td className="p-3">
                                        <div className="flex justify-center gap-2">
                                            {exp.has_viz ? (
                                                <button
                                                    onClick={() => onLoadRun(exp.run_id)}
                                                    className="px-3 py-1 bg-green-600 text-xs rounded hover:bg-green-700"
                                                >
                                                    Load Viz
                                                </button>
                                            ) : (
                                                <button
                                                    onClick={() => handleVisualize(exp.run_id)}
                                                    disabled={processing === exp.run_id}
                                                    className={`px-3 py-1 bg-gray-600 text-xs rounded hover:bg-gray-500 ${processing === exp.run_id ? 'opacity-50 cursor-wait' : ''}`}
                                                >
                                                    {processing === exp.run_id ? 'Generating...' : 'Re-run for Viz'}
                                                </button>
                                            )}

                                            <button
                                                onClick={() => handleDelete(exp.run_id)}
                                                className="px-3 py-1 bg-red-900/50 text-red-300 text-xs rounded hover:bg-red-900"
                                            >
                                                Delete
                                            </button>
                                        </div>
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

export default ExperimentsView;
