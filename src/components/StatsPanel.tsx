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

    const StatBox = ({ label, value, color = 'text-white', subValue }: {
        label: string;
        value: string | number;
        color?: string;
        subValue?: string;
    }) => (
        <div className="bg-slate-900/50 rounded px-3 py-2 flex flex-col">
            <span className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</span>
            <span className={`text-sm font-bold ${color}`}>{value}</span>
            {subValue && <span className="text-[10px] text-slate-400">{subValue}</span>}
        </div>
    );

    if (decisions.length === 0) {
        return (
            <div className="p-3 bg-slate-800 border-b border-slate-700">
                <div className="text-xs text-slate-500 text-center">No scan data loaded</div>
            </div>
        );
    }

    return (
        <div className="p-3 bg-slate-800 border-b border-slate-700">
            <div className="grid grid-cols-6 gap-2">
                {/* Starting Balance */}
                <StatBox
                    label="Start"
                    value={`$${startingBalance.toLocaleString()}`}
                    color="text-slate-300"
                />

                {/* End Balance */}
                <StatBox
                    label="End Balance"
                    value={`$${stats.endBalance.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    color={stats.endBalance >= startingBalance ? 'text-green-400' : 'text-red-400'}
                />

                {/* Total P&L */}
                <StatBox
                    label="Total P&L"
                    value={formatCurrency(stats.totalPnL)}
                    color={stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
                    subValue={`${stats.totalTrades} trades`}
                />

                {/* Win Rate */}
                <StatBox
                    label="Win Rate"
                    value={`${stats.winRate.toFixed(1)}%`}
                    color={stats.winRate >= 50 ? 'text-green-400' : 'text-amber-400'}
                    subValue={`${stats.wins}W / ${stats.losses}L`}
                />

                {/* Max Drawdown */}
                <StatBox
                    label="Max Drawdown"
                    value={`-$${stats.maxDrawdown.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    color="text-red-400"
                    subValue={`${((stats.maxDrawdown / startingBalance) * 100).toFixed(1)}%`}
                />

                {/* Profit Factor */}
                <StatBox
                    label="Profit Factor"
                    value={stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2)}
                    color={stats.profitFactor >= 1 ? 'text-green-400' : 'text-red-400'}
                    subValue={`Avg: ${formatCurrency(stats.avgPnL)}`}
                />
            </div>
        </div>
    );
};
