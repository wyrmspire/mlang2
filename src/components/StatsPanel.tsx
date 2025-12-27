import React, { useMemo } from 'react';
import { VizDecision } from '../types/viz';

interface StatsPanelProps {
    decisions: VizDecision[];
    startingBalance?: number;
}

interface ScanStats {
    totalTrades: number;
    wins: number;
    losses: number;
    winRate: number;
    totalPnL: number;
    avgPnL: number;
    maxDrawdown: number;
    endBalance: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;
}

export const StatsPanel: React.FC<StatsPanelProps> = ({
    decisions,
    startingBalance = 50000
}) => {
    const stats = useMemo<ScanStats>(() => {
        if (!decisions || decisions.length === 0) {
            return {
                totalTrades: 0,
                wins: 0,
                losses: 0,
                winRate: 0,
                totalPnL: 0,
                avgPnL: 0,
                maxDrawdown: 0,
                endBalance: startingBalance,
                profitFactor: 0,
                avgWin: 0,
                avgLoss: 0,
            };
        }

        let balance = startingBalance;
        let peakBalance = startingBalance;
        let maxDrawdown = 0;
        let wins = 0;
        let losses = 0;
        let totalWinAmount = 0;
        let totalLossAmount = 0;

        decisions.forEach(d => {
            // Get PnL from oco_results (multiple formats) or cf_pnl_dollars
            const ocoResults = d.oco_results || {};
            let pnl = 0;

            // Format 1: Direct pnl_dollars on oco_results (from IFVG debug scanner)
            if (typeof ocoResults.pnl_dollars === 'number') {
                pnl = ocoResults.pnl_dollars;
            }
            // Format 2: Nested OCO results (from OR scanner)
            else if (typeof Object.values(ocoResults)[0] === 'object') {
                const bestOco = Object.values(ocoResults)[0] as { pnl_dollars?: number } | undefined;
                pnl = bestOco?.pnl_dollars ?? 0;
            }
            // Fallback: cf_pnl_dollars
            else {
                pnl = d.cf_pnl_dollars ?? 0;
            }

            balance += pnl;

            if (pnl > 0) {
                wins++;
                totalWinAmount += pnl;
            } else if (pnl < 0) {
                losses++;
                totalLossAmount += Math.abs(pnl);
            }

            // Track peak and drawdown
            if (balance > peakBalance) {
                peakBalance = balance;
            }
            const currentDrawdown = peakBalance - balance;
            if (currentDrawdown > maxDrawdown) {
                maxDrawdown = currentDrawdown;
            }
        });

        const totalTrades = wins + losses;
        const totalPnL = balance - startingBalance;

        return {
            totalTrades,
            wins,
            losses,
            winRate: totalTrades > 0 ? (wins / totalTrades) * 100 : 0,
            totalPnL,
            avgPnL: totalTrades > 0 ? totalPnL / totalTrades : 0,
            maxDrawdown,
            endBalance: balance,
            profitFactor: totalLossAmount > 0 ? totalWinAmount / totalLossAmount : totalWinAmount > 0 ? Infinity : 0,
            avgWin: wins > 0 ? totalWinAmount / wins : 0,
            avgLoss: losses > 0 ? totalLossAmount / losses : 0,
        };
    }, [decisions, startingBalance]);

    const formatCurrency = (val: number) => {
        const sign = val >= 0 ? '+' : '';
        return `${sign}$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    };

    const StatBox = ({ label, value, color = 'text-white', subValue, trend }: {
        label: string;
        value: string | number;
        color?: string;
        subValue?: string;
        trend?: 'up' | 'down' | 'neutral';
    }) => (
        <div className="bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-lg p-3 flex flex-col hover:bg-slate-800/60 transition-colors shadow-sm group">
            <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1">{label}</span>
            <span className={`text-base font-bold font-mono ${color} group-hover:scale-105 transition-transform origin-left`}>{value}</span>
            {subValue && (
                <div className="flex items-center mt-1">
                    <span className="text-[10px] text-slate-500 font-mono">{subValue}</span>
                </div>
            )}
        </div>
    );

    if (decisions.length === 0) {
        return (
            <div className="p-4 bg-slate-900 border-b border-slate-800">
                <div className="text-sm text-slate-500 text-center italic">No scan data loaded to analyze.</div>
            </div>
        );
    }

    return (
        <div className="px-4 py-3 bg-slate-900 border-b border-slate-800 shadow-md z-10">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                {/* Starting Balance */}
                <StatBox
                    label="Initial Capital"
                    value={`$${startingBalance.toLocaleString()}`}
                    color="text-slate-300"
                />

                {/* End Balance */}
                <StatBox
                    label="Current Balance"
                    value={`$${stats.endBalance.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    color={stats.endBalance >= startingBalance ? 'text-green-400' : 'text-red-400'}
                    trend={stats.endBalance >= startingBalance ? 'up' : 'down'}
                />

                {/* Total P&L */}
                <StatBox
                    label="Net P&L"
                    value={formatCurrency(stats.totalPnL)}
                    color={stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
                    subValue={`${stats.totalTrades} Trades`}
                />

                {/* Win Rate */}
                <StatBox
                    label="Win Rate"
                    value={`${stats.winRate.toFixed(1)}%`}
                    color={stats.winRate >= 50 ? 'text-emerald-400' : 'text-amber-400'}
                    subValue={`${stats.wins}W - ${stats.losses}L`}
                />

                {/* Max Drawdown */}
                <StatBox
                    label="Max Drawdown"
                    value={`-$${stats.maxDrawdown.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    color="text-rose-400"
                    subValue={`${((stats.maxDrawdown / startingBalance) * 100).toFixed(1)}%`}
                />

                {/* Profit Factor */}
                <StatBox
                    label="Profit Factor"
                    value={stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2)}
                    color={stats.profitFactor >= 1.5 ? 'text-purple-400' : stats.profitFactor >= 1 ? 'text-blue-400' : 'text-slate-400'}
                    subValue={`Avg: ${formatCurrency(stats.avgPnL)}`}
                />
            </div>
        </div>
    );
};
