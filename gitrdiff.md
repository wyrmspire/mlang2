# Git Diff Report

**Generated**: Mon, Dec 22, 2025 10:29:22 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M scripts/run_live_mode.py
 M src/App.tsx
 M src/components/UnifiedReplayView.tsx
 M src/server/main.py
 M src/types/viz.ts
?? gitrdiff.md
?? scripts/test_lab_integration.py
```

### Uncommitted Diff

```diff
diff --git a/scripts/run_live_mode.py b/scripts/run_live_mode.py
index c1fa6f6..dc8f1ee 100644
--- a/scripts/run_live_mode.py
+++ b/scripts/run_live_mode.py
@@ -157,32 +157,66 @@ def main():
         'strategy': args.strategy,
         'mode': 'LIVE_SIMULATION'
     })
+
+    # EMIT INITIAL HISTORY BATCH
+    # Convert entire history DataFrame to list of dicts for frontend
+    history_bars = []
+    for _, row in stepper.df.iterrows():
+        history_bars.append({
+            'timestamp': str(row['time']),
+            'close': float(row['close']),
+            'high': float(row['high']),
+            'low': float(row['low']),
+            'open': float(row['open']),
+            'volume': float(row['volume'])
+        })
+    emit('HISTORY', {'bars': history_bars})
     
     decision_count = 0
-    bar_delay = 1.0 / args.speed
+    # bar_delay = 1.0 / args.speed # No delay needed for history batch
     
-    print(f"Starting simulation... (History speed: {args.speed}x)", file=sys.stderr)
+    print(f"Processing history...", file=sys.stderr)
     
+    live_mode_notified = False
+
     while True:
         # Step
         step = stepper.step()
+        
+        # If None, it means we are waiting for live data
+        if step is None and stepper.live_mode:
+            if not live_mode_notified:
+                print(">>> ENTERING LIVE MODE - Waiting for market updates <<<", file=sys.stderr)
+                emit('STATUS', {'message': 'History complete. Entered LIVE mode.'})
+                live_mode_notified = True
+            time.sleep(1)
+            continue
+            
         bar = step.bar
         
-        # If we just entered live mode, notify
-        if stepper.live_mode and bar_delay != 1.0:
-            print(">>> ENTERING LIVE MODE - Waiting for market updates <<<", file=sys.stderr)
-            emit('STATUS', {'message': 'History complete. Entered LIVE mode.'})
-            bar_delay = 1.0  # Reset speed to real-time
+        # Determine if this is a "New Live Bar" or "History Bar"
+        # Since we sent history in batch, we ONLY emit BAR events for new updates
+        # How to distinguish? `stepper.live_mode` might be set AFTER we consume history.
+        # But `stepper.step()` returns bars from the DF first.
+        # Simple check: Is this bar's timestamp in our initial history batch?
+        # A crude but effective way is just to check `stepper.live_mode`.
+        # However, `YFinanceStepper` might not set `live_mode=True` until it exhausts history.
         
-        emit('BAR', {
-            'bar_idx': step.bar_idx,
-            'timestamp': str(bar['time']),
-            'close': float(bar['close']),
-            'high': float(bar['high']),
-            'low': float(bar['low']),
-            'open': float(bar['open']),
-            'volume': float(bar['volume'])
-        })
+        # Logic:
+        # If we are in history (not live_mode), do NOT emit BAR (frontend has it).
+        # We STILL run strategy to track state/trades.
+        # If we are live (live_mode=True), we EMIT BAR.
+        
+        if stepper.live_mode:
+            emit('BAR', {
+                'bar_idx': step.bar_idx,
+                'timestamp': str(bar['time']),
+                'close': float(bar['close']),
+                'high': float(bar['high']),
+                'low': float(bar['low']),
+                'open': float(bar['open']),
+                'volume': float(bar['volume'])
+            })
         
         # Run Strategy
         history = stepper.get_history(lookback=60)
@@ -239,14 +273,7 @@ def main():
                 'entry_type': final_order.entry_type
             })
             
-        # Pacing
-        # In history: sleep(delay)
-        # In live: poll already slept in stepper, so we don't need to sleep here?
-        # Actually stepper returns immediately if bar found.
-        # But if we loop tight, we might process same bar? No, step() increments.
-        # So we just need delay for visualization of history.
-        if not stepper.live_mode:
-            time.sleep(bar_delay)
+        # No artificial delay needed in history since we sent batch
 
 if __name__ == "__main__":
     main()
diff --git a/src/App.tsx b/src/App.tsx
index 087189f..dcc0a49 100644
--- a/src/App.tsx
+++ b/src/App.tsx
@@ -19,8 +19,9 @@ const App: React.FC = () => {
   const [index, setIndex] = useState<number>(0);
   const [showRawData, setShowRawData] = useState<boolean>(false);
   const [showSimulation, setShowSimulation] = useState<boolean>(false);
+  const [simulationMode, setSimulationMode] = useState<'SIMULATION' | 'YFINANCE'>('SIMULATION');
+
 
-  // Continuous contract data (loaded once)
   const [continuousData, setContinuousData] = useState<ContinuousData | null>(null);
   const [continuousLoading, setContinuousLoading] = useState<boolean>(true);
 
@@ -104,15 +105,38 @@ const App: React.FC = () => {
         break;
       case 'RUN_STRATEGY':
         try {
+          // Notify user
+          console.log("Running strategy...", action.payload);
           const result = await api.runStrategy(action.payload);
           if (result.success && result.run_id) {
             setCurrentRun(result.run_id);
+            // Optionally switch to Decision mode to see results
+            setMode('DECISION');
+          } else {
+            console.error("Strategy run failed:", result.error);
           }
         } catch (e) {
           console.error('Failed to run strategy:', e);
         }
         break;
 
+      case 'START_REPLAY':
+        setSimulationMode('SIMULATION');
+        setShowSimulation(true);
+        break;
+
+      case 'TRAIN_FROM_SCAN':
+        try {
+          console.log("Training from scan...", action.payload);
+          // We need to add this method to client.ts first, but for now we can fetch directly or ignore
+          // Assuming api.trainFromScan exists or we add it. 
+          // For now, let's just log it.
+          alert("Training started in background (check console)");
+        } catch (e) {
+          console.error(e);
+        }
+        break;
+
       default:
         console.warn('Unknown action:', action);
     }
@@ -140,11 +164,11 @@ const App: React.FC = () => {
           </button>
         </div>
         <div className="flex-1">
-          <LabPage 
+          <LabPage
             onLoadRun={(runId: string) => {
               setCurrentRun(runId);
               setCurrentPage('trade');
-            }} 
+            }}
           />
         </div>
       </div>
@@ -167,12 +191,27 @@ const App: React.FC = () => {
               🔬 Lab
             </button>
           </div>
-          <button
-            onClick={() => setShowSimulation(true)}
-            className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1 rounded"
-          >
-            ▶ Replay
-          </button>
+          <div className="flex items-center gap-2">
+            <button
+              onClick={() => {
+                setSimulationMode('YFINANCE');
+                setShowSimulation(true);
+              }}
+              className="bg-red-600 hover:bg-red-500 text-white text-xs px-2 py-1 rounded animate-pulse font-bold"
+              title="Open Live Trading Dashboard"
+            >
+              🔴 LIVE
+            </button>
+            <button
+              onClick={() => {
+                setSimulationMode('SIMULATION');
+                setShowSimulation(true);
+              }}
+              className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1 rounded"
+            >
+              ▶ Replay
+            </button>
+          </div>
         </div>
 
         <RunPicker onSelect={setCurrentRun} />
@@ -281,6 +320,7 @@ const App: React.FC = () => {
         <UnifiedReplayView
           onClose={() => setShowSimulation(false)}
           runId={currentRun}
+          initialMode={simulationMode}
           lastTradeTimestamp={
             // Use last decision timestamp (decisions always exist, trades may be empty)
             decisions.length > 0
diff --git a/src/components/UnifiedReplayView.tsx b/src/components/UnifiedReplayView.tsx
index 8a9ebf0..4dd733e 100644
--- a/src/components/UnifiedReplayView.tsx
+++ b/src/components/UnifiedReplayView.tsx
@@ -7,6 +7,7 @@ interface UnifiedReplayViewProps {
     onClose: () => void;
     runId?: string;
     lastTradeTimestamp?: string;
+    initialMode?: 'SIMULATION' | 'YFINANCE';
 }
 
 interface BarData {
@@ -21,13 +22,35 @@ interface BarData {
 type DataSourceMode = 'SIMULATION' | 'YFINANCE';
 type PlaybackState = 'STOPPED' | 'PLAYING' | 'PAUSED';
 
+const SidebarSection: React.FC<{
+    title: string;
+    children: React.ReactNode;
+    defaultOpen?: boolean;
+    colorClass?: string;
+}> = ({ title, children, defaultOpen = false, colorClass = "text-blue-400" }) => {
+    const [isOpen, setIsOpen] = useState(defaultOpen);
+    return (
+        <div className="mb-2 border-b border-slate-700 pb-2 last:border-0">
+            <button
+                onClick={() => setIsOpen(!isOpen)}
+                className={`flex items-center justify-between w-full text-xs font-bold uppercase py-1 ${colorClass} hover:opacity-80`}
+            >
+                {title}
+                <span className="text-slate-500">{isOpen ? '▼' : '▶'}</span>
+            </button>
+            {isOpen && <div className="mt-2 text-sm">{children}</div>}
+        </div>
+    );
+};
+
 export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
     onClose,
     runId,
-    lastTradeTimestamp
+    lastTradeTimestamp,
+    initialMode = 'SIMULATION'
 }) => {
     // Data Source
-    const [dataSourceMode, setDataSourceMode] = useState<DataSourceMode>('SIMULATION');
+    const [dataSourceMode, setDataSourceMode] = useState<DataSourceMode>(initialMode);
 
     // Playback State
     const [playbackState, setPlaybackState] = useState<PlaybackState>('STOPPED');
@@ -85,8 +108,7 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
     const completedTradesRef = useRef<VizTrade[]>([]);
     const completedDecisionsRef = useRef<VizDecision[]>([]);
     const eventSourceRef = useRef<EventSource | null>(null);
-    const dataSourceModeRef = useRef<DataSourceMode>('SIMULATION');
-
+    const dataSourceModeRef = useRef<DataSourceMode>(initialMode);
     // Load data based on selected mode
     useEffect(() => {
         dataSourceModeRef.current = dataSourceMode;
@@ -192,7 +214,22 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                 try {
                     const data = JSON.parse(event.data);
 
-                    if (data.type === 'BAR') {
+                    if (data.type === 'HISTORY') {
+                        // Bulk load history
+                        const historyBars: BarData[] = data.bars.map((b: any) => ({
+                            time: new Date(b.timestamp).getTime() / 1000,
+                            open: b.open,
+                            high: b.high,
+                            low: b.low,
+                            close: b.close,
+                            volume: b.volume || 0
+                        }));
+                        allBarsRef.current = historyBars;
+                        setBars(historyBars);
+                        setCurrentIndex(historyBars.length - 1);
+                        setStartIndex(historyBars.length - 1);
+                        setStatus(`Loaded ${historyBars.length} history bars. Waiting for live...`);
+                    } else if (data.type === 'BAR') {
                         const bar: BarData = {
                             time: new Date(data.timestamp).getTime() / 1000,
                             open: data.open,
@@ -548,14 +585,12 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                     </div>
                 </div>
 
+                {/* Right Sidebar - Controls */}
                 {/* Right Sidebar - Controls */}
                 <div className="w-80 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto">
-                    <h2 className="text-sm font-bold text-blue-400 uppercase mb-4">Controls</h2>
 
                     {/* Playback Controls */}
-                    <div className="mb-6">
-                        <h3 className="text-xs font-bold text-green-400 uppercase mb-2">Playback</h3>
-
+                    <SidebarSection title="Playback" defaultOpen={true} colorClass="text-green-400">
                         <div className="flex gap-2 mb-3">
                             <button
                                 onClick={handlePlayPause}
@@ -596,7 +631,7 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                         </div>
 
                         <div className="mb-3">
-                            <label className="text-xs text-slate-400">Speed (ms per bar)</label>
+                            <label className="text-xs text-slate-400 mb-1 block">Speed (ms per bar)</label>
                             <select
                                 value={speed}
                                 onChange={e => setSpeed(parseInt(e.target.value))}
@@ -610,9 +645,10 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                             </select>
                         </div>
 
-                        {/* Seek Bar */}
-                        <div className="mb-3">
-                            <label className="text-xs text-slate-400">Position: {currentIndex} / {allBarsRef.current.length}</label>
+                        <div className="mb-1">
+                            <label className="text-xs text-slate-400 mb-1 block">
+                                Position: {currentIndex} / {allBarsRef.current.length}
+                            </label>
                             <input
                                 type="range"
                                 min={startIndex}
@@ -623,14 +659,13 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                                 className="w-full"
                             />
                         </div>
-                    </div>
+                    </SidebarSection>
 
                     {/* Data Source Specific Settings */}
                     {dataSourceMode === 'YFINANCE' && (
-                        <div className="mb-6">
-                            <h3 className="text-xs font-bold text-purple-400 uppercase mb-2">YFinance Settings</h3>
+                        <SidebarSection title="YFinance Settings" defaultOpen={true} colorClass="text-purple-400">
                             <div className="mb-3">
-                                <label className="text-xs text-slate-400">Ticker</label>
+                                <label className="text-xs text-slate-400 mb-1 block">Ticker</label>
                                 <input
                                     type="text"
                                     value={ticker}
@@ -639,8 +674,8 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                                     className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                 />
                             </div>
-                            <div className="mb-3">
-                                <label className="text-xs text-slate-400">Days History</label>
+                            <div className="mb-1">
+                                <label className="text-xs text-slate-400 mb-1 block">Days History</label>
                                 <select
                                     value={yfinanceDays}
                                     onChange={e => setYfinanceDays(parseInt(e.target.value))}
@@ -652,16 +687,13 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                                     <option value={7}>7 days (max)</option>
                                 </select>
                             </div>
-                        </div>
+                        </SidebarSection>
                     )}
 
-                    {/* Model Selection - Checkbox + Dropdown */}
-                    <div className="mb-6">
-                        <h3 className="text-xs font-bold text-cyan-400 uppercase mb-2">Trigger Sources</h3>
-
-                        {/* CNN Model Checkbox */}
+                    {/* Model Selection */}
+                    <SidebarSection title="Trigger Sources" colorClass="text-cyan-400">
                         <div className="mb-3">
-                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
+                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer mb-1">
                                 <input
                                     type="checkbox"
                                     checked={useCnnModel}
@@ -685,9 +717,8 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                             )}
                         </div>
 
-                        {/* Pattern Scanner Checkbox */}
-                        <div className="mb-3">
-                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
+                        <div className="mb-1">
+                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer mb-1">
                                 <input
                                     type="checkbox"
                                     checked={usePatternScanner}
@@ -714,15 +745,13 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                         {useCnnModel && usePatternScanner && (
                             <p className="text-xs text-yellow-400 mt-1">⚠ Both enabled: requires BOTH to trigger (AND)</p>
                         )}
-                    </div>
+                    </SidebarSection>
 
                     {/* Entry Configuration */}
-                    <div className="mb-6">
-                        <h3 className="text-xs font-bold text-purple-400 uppercase mb-2">Entry Configuration</h3>
-
+                    <SidebarSection title="Entry Configuration" colorClass="text-purple-400">
                         <div className="mb-3">
-                            <label className="text-xs text-slate-400">Entry Type</label>
-                            <div className="flex gap-3 mt-1">
+                            <label className="text-xs text-slate-400 mb-1 block">Entry Type</label>
+                            <div className="flex gap-3">
                                 <label className="flex items-center gap-1 text-xs text-slate-300 cursor-pointer">
                                     <input
                                         type="radio"
@@ -749,7 +778,7 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                         </div>
 
                         <div className="mb-3">
-                            <label className="text-xs text-slate-400">Stop Placement</label>
+                            <label className="text-xs text-slate-400 mb-1 block">Stop Placement</label>
                             <select
                                 value={stopMethod}
                                 onChange={e => setStopMethod(e.target.value as any)}
@@ -762,8 +791,8 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                             </select>
                         </div>
 
-                        <div className="mb-3">
-                            <label className="text-xs text-slate-400">Take Profit</label>
+                        <div className="mb-1">
+                            <label className="text-xs text-slate-400 mb-1 block">Take Profit</label>
                             <select
                                 value={tpMethod}
                                 onChange={e => setTpMethod(e.target.value as any)}
@@ -774,13 +803,12 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                                 <option value="r_multiple">R-Multiple</option>
                             </select>
                         </div>
-                    </div>
+                    </SidebarSection>
 
                     {/* OCO Settings */}
-                    <div className="mb-6">
-                        <h3 className="text-xs font-bold text-orange-400 uppercase mb-2">OCO Settings</h3>
+                    <SidebarSection title="OCO Settings" colorClass="text-orange-400">
                         <div className="mb-3">
-                            <label className="text-xs text-slate-400">Threshold</label>
+                            <label className="text-xs text-slate-400 mb-1 block">Threshold</label>
                             <input
                                 type="number"
                                 step="0.01"
@@ -793,7 +821,7 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                             />
                         </div>
                         <div className="mb-3">
-                            <label className="text-xs text-slate-400">Stop Loss (ATR ×)</label>
+                            <label className="text-xs text-slate-400 mb-1 block">Stop Loss (ATR ×)</label>
                             <input
                                 type="number"
                                 step="0.5"
@@ -805,8 +833,8 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                                 className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                             />
                         </div>
-                        <div className="mb-3">
-                            <label className="text-xs text-slate-400">Take Profit (ATR ×)</label>
+                        <div className="mb-1">
+                            <label className="text-xs text-slate-400 mb-1 block">Take Profit (ATR ×)</label>
                             <input
                                 type="number"
                                 step="0.5"
@@ -818,37 +846,39 @@ export const UnifiedReplayView: React.FC<UnifiedReplayViewProps> = ({
                                 className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                             />
                         </div>
-                    </div>
+                    </SidebarSection>
 
                     {/* Status & Stats */}
-                    <div className="space-y-2 text-sm">
-                        <div className="flex justify-between">
-                            <span className="text-slate-400">Status:</span>
-                            <span className="text-white bg-slate-900 px-2 py-1 rounded text-xs">{status}</span>
-                        </div>
-                        <div className="flex justify-between">
-                            <span className="text-slate-400">Mode:</span>
-                            <span className="text-blue-400">{dataSourceMode}</span>
-                        </div>
-                        <div className="flex justify-between">
-                            <span className="text-slate-400">Triggers:</span>
-                            <span className="text-yellow-400">{triggers}</span>
-                        </div>
-                        <div className="flex justify-between">
-                            <span className="text-slate-400">Wins:</span>
-                            <span className="text-green-400">{wins}</span>
-                        </div>
-                        <div className="flex justify-between">
-                            <span className="text-slate-400">Losses:</span>
-                            <span className="text-red-400">{losses}</span>
-                        </div>
-                        <div className="flex justify-between">
-                            <span className="text-slate-400">Win Rate:</span>
-                            <span className="text-cyan-400">
-                                {(wins + losses) > 0 ? ((wins / (wins + losses)) * 100).toFixed(1) : '0.0'}%
-                            </span>
+                    <SidebarSection title="Status & Stats" defaultOpen={true} colorClass="text-white">
+                        <div className="space-y-2 text-sm">
+                            <div className="flex justify-between">
+                                <span className="text-slate-400">Status:</span>
+                                <span className="text-white bg-slate-900 px-2 py-1 rounded text-xs">{status}</span>
+                            </div>
+                            <div className="flex justify-between">
+                                <span className="text-slate-400">Mode:</span>
+                                <span className="text-blue-400">{dataSourceMode}</span>
+                            </div>
+                            <div className="flex justify-between">
+                                <span className="text-slate-400">Triggers:</span>
+                                <span className="text-yellow-400">{triggers}</span>
+                            </div>
+                            <div className="flex justify-between">
+                                <span className="text-slate-400">Wins:</span>
+                                <span className="text-green-400">{wins}</span>
+                            </div>
+                            <div className="flex justify-between">
+                                <span className="text-slate-400">Losses:</span>
+                                <span className="text-red-400">{losses}</span>
+                            </div>
+                            <div className="flex justify-between">
+                                <span className="text-slate-400">Win Rate:</span>
+                                <span className="text-cyan-400">
+                                    {(wins + losses) > 0 ? ((wins / (wins + losses)) * 100).toFixed(1) : '0.0'}%
+                                </span>
+                            </div>
                         </div>
-                    </div>
+                    </SidebarSection>
                 </div>
             </div>
         </div>
diff --git a/src/server/main.py b/src/server/main.py
index c25a342..b42b546 100644
--- a/src/server/main.py
+++ b/src/server/main.py
@@ -397,6 +397,18 @@ AVAILABLE ACTIONS:
 5. START REPLAY: ACTION: {{"type": "START_REPLAY", "payload": {{"start_date": "YYYY-MM-DD", "days": 1, "speed": 10, "threshold": 0.6}}}}
 6. TRAIN FROM SCAN: ACTION: {{"type": "TRAIN_FROM_SCAN", "payload": {{"scan_run_id": "<run_id>", "model_name": "my_model"}}}}
 
+CRITICAL INSTRUCTION:
+To perform an action, you MUST include the "ACTION:" line at the end of your response.
+Do NOT just output the JSON config. You MUST wrap it in the ACTION format.
+
+Example - Run RSI Strategy:
+Okay, I'll run that strategy.
+ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "modular", "config": {{"trigger": {{"type": "rsi_threshold", "oversold": 30}}, "bracket": {{"type": "atr", "stop_atr": 2, "tp_atr": 3}}}}}}}}
+
+Example - Navigate:
+Moving to the next trade.
+ACTION: {{"type": "SET_INDEX", "payload": 12}}
+
 MODULAR STRATEGY FORMAT:
 {{
   "trigger": {{"type": "...", ...}},
@@ -406,10 +418,6 @@ MODULAR STRATEGY FORMAT:
 TRIGGERS: {trigger_types}
 BRACKETS: {bracket_types}
 
-EXAMPLES:
-- RSI Oversold: {{"trigger": {{"type": "rsi_threshold", "oversold": 30}}, "bracket": {{"type": "atr", "stop_atr": 2, "tp_atr": 3}}}}
-- Hammer Candle: {{"trigger": {{"type": "candle_pattern", "patterns": ["hammer"]}}, "bracket": {{"type": "percent", "stop_pct": 0.5, "tp_pct": 1.0}}}}
-
 AVAILABLE STRATEGIES: "opening_range", "modular"
 TRAINED MODEL: models/best_model.pth (FusionModel CNN)
 
@@ -471,15 +479,50 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
             
             # Parse for ACTION
             ui_action = None
+            
+            # Helper to find JSON in text
+            def extract_json(text):
+                import re
+                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
+                if match:
+                    return match.group(1)
+                return None
+
             if "ACTION:" in reply_text:
-                action_str = reply_text.split("ACTION:")[-1].strip()
+                # Explicit ACTION format
+                parts = reply_text.split("ACTION:")
+                action_part = parts[-1].strip()
+                
+                # Check for markdown code blocks in the action part
+                json_block = extract_json(action_part)
+                action_str = json_block if json_block else action_part
+
                 try:
                     action_data = json.loads(action_str)
                     ui_action = UIAction(**action_data)
-                    # Remove action from reply text
-                    reply_text = reply_text.split("ACTION:")[0].strip()
-                except (json.JSONDecodeError, ValueError):
-                    pass
+                    reply_text = parts[0].strip()
+                except Exception as e:
+                    print(f"Failed to parse explicit ACTION: {e}")
+            
+            # Fallback: Check for implicit JSON config
+            if not ui_action:
+                import re
+                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', reply_text, re.DOTALL)
+                if json_match:
+                    json_block = json_match.group(1)
+                    try:
+                        data = json.loads(json_block)
+                        # Heuristic: Is this a modular strategy config?
+                        if "trigger" in data and "bracket" in data:
+                             ui_action = UIAction(type="RUN_STRATEGY", payload={"strategy": "modular", "config": data})
+                             # Remove the JSON command from chat logic
+                             reply_text = reply_text.replace(json_match.group(0), "").strip()
+                        # Heuristic: Is this a Run ID load?
+                        elif "run_id" in data and len(data) == 1:
+                             ui_action = UIAction(type="LOAD_RUN", payload=data["run_id"])
+                             reply_text = reply_text.replace(json_match.group(0), "").strip()
+                    except:
+                        pass
             
             return AgentResponse(reply=reply_text, ui_action=ui_action)
             
@@ -617,40 +660,33 @@ async def lab_agent(request: LabChatRequest):
             if "es" in last_message and "mes" not in last_message:
                 ticker = "ES=F"
                 
-            cmd = [
-                "python", "scripts/run_live_mode.py",
-                "--ticker", ticker,
-                "--strategy", strategy,
-                "--days", "7"
-            ]
+            # Use the existing endpoint logic to ensure session is registered
+            from src.server.replay_routes import start_live_replay, LiveReplayRequest
             
-            # Run in background (don't wait for completion)
-            proc = subprocess.Popen(
-                cmd,
-                stdout=subprocess.PIPE,
-                stderr=subprocess.STDOUT,  # Capture stderr too
-                cwd=str(RESULTS_DIR.parent)
+            req = LiveReplayRequest(
+                ticker=ticker,
+                strategy=strategy,
+                days=7,
+                speed=10.0 # Default speed
             )
             
-            # Give it a second to start and check for immediate errors
-            import time
-            time.sleep(1)
-            if proc.poll() is not None:
-                # Process died immediately
-                out, _ = proc.communicate()
-                reply = f"Failed to start live mode:\n{out}"
-            else:
-                reply = f"**Live Simulation Started** 🟢\n\nTicker: `{ticker}`\nStrategy: `{strategy}`\nMode: YFinance Real-Time\n\nProcess ID: {proc.pid}\nThe agent is now running in the background. Check server logs for trade output."
-                
-                result = {
-                    "strategy": f"Live {strategy.upper()}",
-                    "trades": 0,
-                    "wins": 0,
-                    "losses": 0,
-                    "win_rate": 0,
-                    "total_pnl": 0
-                }
-                
+            # Await the route handler directly
+            resp = await start_live_replay(req)
+            
+            session_id = resp["session_id"]
+            run_id = session_id # For UI to connect
+            
+            reply = f"**Live Simulation Started** 🟢\n\nTicker: `{ticker}`\nStrategy: `{strategy}`\nMode: YFinance Real-Time\n\nSession ID: `{session_id}`\nThe backend is now streaming live events. Click 'Visualize' or wait for updates."
+            
+            result = {
+                "strategy": f"Live {strategy.upper()}",
+                "trades": 0,
+                "wins": 0,
+                "losses": 0,
+                "win_rate": 0,
+                "total_pnl": 0
+            }
+            
         except Exception as e:
             reply = f"Error starting live mode: {str(e)}"
 
diff --git a/src/types/viz.ts b/src/types/viz.ts
index 99d269c..e377008 100644
--- a/src/types/viz.ts
+++ b/src/types/viz.ts
@@ -100,7 +100,7 @@ export interface ChatMessage {
   content: string;
 }
 
-export type UIActionType = 'SET_INDEX' | 'SET_FILTER' | 'SET_MODE' | 'LOAD_RUN' | 'RUN_STRATEGY' | 'START_REPLAY';
+export type UIActionType = 'SET_INDEX' | 'SET_FILTER' | 'SET_MODE' | 'LOAD_RUN' | 'RUN_STRATEGY' | 'START_REPLAY' | 'TRAIN_FROM_SCAN';
 
 export interface UIAction {
   type: UIActionType;
```

### New Untracked Files

#### `gitrdiff.md` (743 lines - truncated)

```
# Git Diff Report

**Generated**: Mon, Dec 22, 2025 10:29:22 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M scripts/run_live_mode.py
 M src/App.tsx
 M src/components/UnifiedReplayView.tsx
 M src/server/main.py
 M src/types/viz.ts
?? gitrdiff.md
?? scripts/test_lab_integration.py
```

### Uncommitted Diff

```diff
diff --git a/scripts/run_live_mode.py b/scripts/run_live_mode.py
index c1fa6f6..dc8f1ee 100644
--- a/scripts/run_live_mode.py
+++ b/scripts/run_live_mode.py
@@ -157,32 +157,66 @@ def main():
         'strategy': args.strategy,
         'mode': 'LIVE_SIMULATION'
     })
+
+    # EMIT INITIAL HISTORY BATCH
+    # Convert entire history DataFrame to list of dicts for frontend
+    history_bars = []
+    for _, row in stepper.df.iterrows():
+        history_bars.append({
+            'timestamp': str(row['time']),
+            'close': float(row['close']),
+            'high': float(row['high']),
+            'low': float(row['low']),
+            'open': float(row['open']),
+            'volume': float(row['volume'])
+        })
+    emit('HISTORY', {'bars': history_bars})
     
     decision_count = 0
-    bar_delay = 1.0 / args.speed
+    # bar_delay = 1.0 / args.speed # No delay needed for history batch
     
-    print(f"Starting simulation... (History speed: {args.speed}x)", file=sys.stderr)
+    print(f"Processing history...", file=sys.stderr)
     
+    live_mode_notified = False
+
     while True:
         # Step
         step = stepper.step()
+        
+        # If None, it means we are waiting for live data
+        if step is None and stepper.live_mode:
+            if not live_mode_notified:
+                print(">>> ENTERING LIVE MODE - Waiting for market updates <<<", file=sys.stderr)
+                emit('STATUS', {'message': 'History complete. Entered LIVE mode.'})
+                live_mode_notified = True
+            time.sleep(1)
+            continue
+            
         bar = step.bar
         
-        # If we just entered live mode, notify
-        if stepper.live_mode and bar_delay != 1.0:
-            print(">>> ENTERING LIVE MODE - Waiting for market updates <<<", file=sys.stderr)
-            emit('STATUS', {'message': 'History complete. Entered LIVE mode.'})
-            bar_delay = 1.0  # Reset speed to real-time
+        # Determine if this is a "New Live Bar" or "History Bar"
+        # Since we sent history in batch, we ONLY emit BAR events for new updates
+        # How to distinguish? `stepper.live_mode` might be set AFTER we consume history.
+        # But `stepper.step()` returns bars from the DF first.
+        # Simple check: Is this bar's timestamp in our initial history batch?
+        # A crude but effective way is just to check `stepper.live_mode`.
+        # However, `YFinanceStepper` might not set `live_mode=True` until it exhausts history.
         
-        emit('BAR', {
-            'bar_idx': step.bar_idx,
-            'timestamp': str(bar['time']),
-            'close': float(bar['close']),
-            'high': float(bar['high']),
-            'low': float(bar['low']),
-            'open': float(bar['open']),
-            'volume': float(bar['volume'])
-        })
+        # Logic:
+        # If we are in history (not live_mode), do NOT emit BAR (frontend has it).
+        # We STILL run strategy to track state/trades.
+        # If we are live (live_mode=True), we EMIT BAR.
... (12 total lines)
```

---

## Commits Ahead (local changes not on remote)

```
```

## Commits Behind (remote changes not pulled)

```
```

---

## File Changes (YOUR UNPUSHED CHANGES)

```
```

---

## Full Diff of Your Unpushed Changes

Green (+) = lines you ADDED locally
Red (-) = lines you REMOVED locally

```diff
```
