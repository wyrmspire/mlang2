import React from 'react';
import { VizDecision, VizTrade } from '../types/viz';

interface DetailsPanelProps {
  decision: VizDecision | null;
  trade: VizTrade | null;
}

export const DetailsPanel: React.FC<DetailsPanelProps> = ({ decision, trade }) => {
  if (!decision) return <div className="p-8 text-slate-500 text-center italic text-sm">No data point selected</div>;

  return (
    <div className="h-full flex flex-col bg-slate-900/30">
      <div className="bg-slate-950/50 border-b border-slate-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${decision.action === 'ENTER' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-slate-500'}`}></span>
            <h3 className="font-bold text-sm text-slate-200">
            {decision.action}
            </h3>
            <span className="font-mono text-slate-600 text-[10px] bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">#{decision.decision_id}</span>
        </div>

        {trade && (
          <span className={`text-[10px] font-bold px-2 py-1 rounded border ${
              trade.outcome === 'WIN'
                ? 'bg-green-500/10 text-green-400 border-green-500/20 shadow-[0_0_10px_-3px_rgba(34,197,94,0.4)]'
                : 'bg-red-500/10 text-red-400 border-red-500/20 shadow-[0_0_10px_-3px_rgba(239,68,68,0.4)]'
            }`}>
            {trade.outcome} <span className="ml-1 opacity-75">(${trade.pnl_dollars})</span>
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4 flex flex-col gap-5 custom-scrollbar">

        {/* Stats Section */}
        <div className="space-y-4">
          <section>
            <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-2 tracking-widest flex items-center gap-2">
                <span className="h-px w-3 bg-slate-700"></span>
                Decision Specs
                <span className="h-px flex-1 bg-slate-700"></span>
            </h4>
            <div className="grid grid-cols-2 gap-y-2 text-xs">
              <span className="text-slate-500">Scanner</span>
              <span className="text-right font-mono font-semibold text-blue-400">{decision.scanner_id || 'unknown'}</span>

              <span className="text-slate-500">Price</span>
              <span className="text-right font-mono text-slate-300">{decision.current_price?.toFixed?.(2) ?? decision.current_price ?? '-'}</span>

              <span className="text-slate-500">ATR</span>
              <span className="text-right font-mono text-slate-300">{decision.atr?.toFixed?.(2) ?? decision.atr ?? '-'}</span>

              {decision.skip_reason && (
                <>
                    <span className="text-slate-500">Skip Reason</span>
                    <span className="text-right font-mono text-amber-500 font-bold text-[10px] leading-tight">{decision.skip_reason}</span>
                </>
              )}
            </div>
          </section>

          {trade && (
            <section>
              <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-2 tracking-widest flex items-center gap-2">
                <span className="h-px w-3 bg-slate-700"></span>
                Trade Performance
                <span className="h-px flex-1 bg-slate-700"></span>
              </h4>
              <div className="grid grid-cols-2 gap-y-2 text-xs">
                <span className="text-slate-500">Entry</span>
                <span className="text-right font-mono text-slate-300">{trade.entry_price?.toFixed?.(2) ?? trade.entry_price ?? '-'}</span>

                <span className="text-slate-500">Exit</span>
                <span className="text-right font-mono text-slate-300">{trade.exit_price?.toFixed?.(2) ?? trade.exit_price ?? '-'}</span>

                <span className="text-slate-500">R-Multiple</span>
                <span className={`text-right font-mono font-bold ${trade.r_multiple && trade.r_multiple > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {trade.r_multiple?.toFixed?.(2) ?? '-'}R
                </span>

                <span className="text-slate-500">MAE / MFE</span>
                <span className="text-right font-mono text-slate-400">{trade.mae?.toFixed?.(2) ?? '-'} / {trade.mfe?.toFixed?.(2) ?? '-'}</span>
              </div>
            </section>
          )}
        </div>

        {/* JSON Context */}
        <div className="space-y-2">
             <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Context Data</div>
             <div className="bg-slate-950 rounded-md p-3 overflow-auto text-[10px] font-mono border border-slate-800 shadow-inner max-h-60 custom-scrollbar">
                {decision.scanner_context && (
                    <div className="mb-3">
                        <div className="text-blue-500/70 mb-1 font-bold">// Scanner Context</div>
                        <pre className="text-slate-300 whitespace-pre-wrap leading-relaxed">
                            {JSON.stringify(decision.scanner_context, null, 2)}
                        </pre>
                    </div>
                )}

                {decision.oco && (
                    <div>
                        <div className="text-green-500/70 mb-1 font-bold">// OCO Params</div>
                        <pre className="text-slate-300 whitespace-pre-wrap leading-relaxed">
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
