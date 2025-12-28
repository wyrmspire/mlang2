import React from 'react';
import { VizDecision, VizTrade } from '../types/viz';

interface DetailsPanelProps {
  decision: VizDecision | null;
  trade: VizTrade | null;
}

export const DetailsPanel: React.FC<DetailsPanelProps> = ({ decision, trade }) => {
  if (!decision) return (
    <div className="flex flex-col items-center justify-center p-8 text-slate-600 gap-3 min-h-[200px]">
        <svg className="w-8 h-8 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span className="text-[10px] uppercase tracking-widest font-medium opacity-50">No Selection</span>
    </div>
  );

  return (
    <div className="h-full flex flex-col bg-slate-900/30">
      <div className="bg-slate-950/50 border-b border-slate-800/60 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${decision.action === 'ENTER' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)] animate-pulse-subtle' : 'bg-slate-600'}`}></div>
            <h3 className="font-bold text-xs text-slate-200 tracking-wide">
            {decision.action}
            </h3>
            <span className="font-mono text-slate-600 text-[9px] bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800/80">ID: {decision.decision_id}</span>
        </div>

        {trade && (
          <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${
              trade.outcome === 'WIN'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/20'
            }`}>
            {trade.outcome}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4 flex flex-col gap-6 custom-scrollbar">

        {/* Stats Section */}
        <div className="space-y-5">
          <section>
            <h4 className="text-[9px] font-bold text-slate-500 uppercase mb-3 tracking-widest flex items-center gap-2">
                Specs
                <span className="h-px flex-1 bg-slate-800"></span>
            </h4>
            <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-[11px]">
              <span className="text-slate-500">Scanner</span>
              <span className="text-right font-mono font-medium text-blue-400 truncate">{decision.scanner_id || 'unknown'}</span>

              <span className="text-slate-500">Price</span>
              <span className="text-right font-mono text-slate-300">{decision.current_price?.toFixed?.(2) ?? decision.current_price ?? '-'}</span>

              <span className="text-slate-500">ATR</span>
              <span className="text-right font-mono text-slate-300">{decision.atr?.toFixed?.(2) ?? decision.atr ?? '-'}</span>

              {decision.skip_reason && (
                <>
                    <span className="text-slate-500">Status</span>
                    <span className="text-right font-mono text-amber-500 font-bold text-[10px] leading-tight truncate" title={decision.skip_reason}>{decision.skip_reason}</span>
                </>
              )}
            </div>
          </section>

          {trade && (
            <section>
              <h4 className="text-[9px] font-bold text-slate-500 uppercase mb-3 tracking-widest flex items-center gap-2">
                Performance
                <span className="h-px flex-1 bg-slate-800"></span>
              </h4>
              <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-[11px]">
                <span className="text-slate-500">PnL</span>
                <span className={`text-right font-mono font-bold ${trade.pnl_dollars >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {trade.pnl_dollars >= 0 ? '+' : '-'}${Math.abs(trade.pnl_dollars).toFixed(2)}
                </span>

                <span className="text-slate-500">R-Multiple</span>
                <span className={`text-right font-mono font-bold ${trade.r_multiple && trade.r_multiple > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {trade.r_multiple?.toFixed?.(2) ?? '-'}R
                </span>

                <span className="text-slate-500">Entry / Exit</span>
                <span className="text-right font-mono text-slate-400 text-[10px]">{trade.entry_price?.toFixed?.(2)} → {trade.exit_price?.toFixed?.(2)}</span>
              </div>
            </section>
          )}
        </div>

        {/* JSON Context */}
        <div className="flex-1 flex flex-col min-h-0">
             <div className="text-[9px] text-slate-500 uppercase tracking-widest font-bold mb-2 flex items-center justify-between">
                 <span>Context Data</span>
                 <span className="text-[9px] font-mono text-slate-600 bg-slate-900 px-1 rounded">JSON</span>
             </div>
             <div className="bg-slate-950/80 rounded-lg p-3 overflow-auto text-[10px] font-mono border border-slate-800/60 shadow-inner flex-1 custom-scrollbar">
                {decision.scanner_context && (
                    <div className="mb-4">
                        <div className="text-blue-500/70 mb-1 font-bold">// Scanner Context</div>
                        <pre className="text-slate-400 whitespace-pre-wrap leading-relaxed">
                            {JSON.stringify(decision.scanner_context, null, 2)}
                        </pre>
                    </div>
                )}

                {decision.oco && (
                    <div>
                        <div className="text-emerald-500/70 mb-1 font-bold">// OCO Params</div>
                        <pre className="text-slate-400 whitespace-pre-wrap leading-relaxed">
                            {JSON.stringify(decision.oco, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        </div>

      </div>
    </div>
  );
};
