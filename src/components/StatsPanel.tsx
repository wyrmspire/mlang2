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

    const formatCurrency = (val: number, decimals = 2) => {
        const sign = val >= 0 ? '+' : '-';
        return `${sign}$${Math.abs(val).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
    };

    const StatBox = ({ label, value, subValue, trend, trendColor }: {
        label: string;
        value: string | number;
        subValue?: string;
        trend?: string;
        trendColor?: string;
    }) => (
        <div className="bg-slate-900/40 border border-slate-800/60 rounded-lg p-3 flex flex-col justify-between hover:bg-slate-800/40 transition-colors group">
            <div className="flex justify-between items-start mb-1">
                <span className="text-[10px] uppercase tracking-wider font-bold text-slate-500 group-hover:text-slate-400 transition-colors">{label}</span>
                {trend && (
                   <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${trendColor === 'green' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                       {trend}
                   </span>
                )}
            </div>
            <div className="flex items-baseline gap-2">
                 <span className="text-lg font-mono font-medium text-slate-200">{value}</span>
            </div>
            {subValue && (
                <div className="text-[10px] text-slate-600 font-medium mt-1 truncate">{subValue}</div>
            )}
        </div>
    );

    if (decisions.length === 0) {
        return (
             <div className="bg-slate-950 border-b border-slate-800/60 py-8">
                <div className="flex flex-col items-center justify-center gap-2 text-slate-600">
                    <svg className="w-8 h-8 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <span className="text-xs font-medium uppercase tracking-widest opacity-50">No Data Available</span>
                </div>
            </div>
        );
    }

    const pnlColor = stats.totalPnL >= 0 ? 'text-emerald-400' : 'text-rose-400';
    const winRateColor = stats.winRate >= 50 ? 'text-emerald-400' : 'text-amber-400';

    return (
        <div className="bg-slate-950 border-b border-slate-800/60 shadow-sm z-10">
            <div className="max-w-[1920px] mx-auto px-4 py-3">
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">

                    {/* Net P&L */}
                    <div className="bg-slate-900/40 border border-slate-800/60 rounded-lg p-3 relative overflow-hidden group">
                        <div className={`absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity ${stats.totalPnL >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                             <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L11 8.586 15.586 4H12z" clipRule="evenodd" /></svg>
                        </div>
                        <div className="text-[10px] uppercase tracking-wider font-bold text-slate-500 mb-1">Net P&L</div>
                        <div className={`text-xl font-mono font-bold ${pnlColor}`}>
                            {formatCurrency(stats.totalPnL)}
                        </div>
                         <div className="text-[10px] text-slate-500 mt-1 flex items-center gap-1">
                             <span className={stats.endBalance >= startingBalance ? 'text-emerald-500' : 'text-rose-500'}>
                                 {((stats.totalPnL / startingBalance) * 100).toFixed(2)}%
                             </span>
                             <span>return</span>
                         </div>
                    </div>

                    {/* Win Rate */}
                    <StatBox
                        label="Win Rate"
                        value={`${stats.winRate.toFixed(1)}%`}
                        trend={`${stats.wins}W - ${stats.losses}L`}
                        trendColor={stats.winRate >= 50 ? 'green' : 'red'}
                        subValue={`Avg Win: ${formatCurrency(stats.avgWin, 0)}`}
                    />

                    {/* Profit Factor */}
                    <StatBox
                        label="Profit Factor"
                        value={stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2)}
                        subValue={`PF > 1.5 is ideal`}
                    />

                     {/* Max Drawdown */}
                     <StatBox
                        label="Max Drawdown"
                        value={`-${formatCurrency(stats.maxDrawdown, 0).replace('+', '').replace('-', '')}`}
                        trend={`${((stats.maxDrawdown / startingBalance) * 100).toFixed(1)}%`}
                        trendColor="red"
                        subValue="Peak to Valley"
                    />

                    {/* Trades */}
                    <StatBox
                        label="Total Trades"
                        value={stats.totalTrades}
                        subValue={`Avg PnL: ${formatCurrency(stats.avgPnL, 2)}`}
                    />

                    {/* Balance */}
                    <div className="bg-slate-900/40 border border-slate-800/60 rounded-lg p-3 flex flex-col justify-between">
                         <div className="text-[10px] uppercase tracking-wider font-bold text-slate-500 mb-1">Account Balance</div>
                         <div className="text-lg font-mono text-slate-300">
                             ${stats.endBalance.toLocaleString()}
                         </div>
                         <div className="w-full bg-slate-800 h-1 mt-2 rounded-full overflow-hidden">
                             <div
                                className={`h-full ${stats.endBalance >= startingBalance ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                style={{ width: '100%' }} // Just a visual indicator line
                             ></div>
                         </div>
                    </div>

                </div>
            </div>
        </div>
    );
};
