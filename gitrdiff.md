# Git Diff Report

**Generated**: Tue, Dec 23, 2025 11:53:33 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M src/App.tsx
 M src/api/client.ts
 M src/components/LiveSessionView.tsx
 M src/server/main.py
?? gitrdiff.md
?? goLive_simple.txt
?? src/components/LiveSessionView.tsx.backup
```

### Uncommitted Diff

```diff
diff --git a/src/App.tsx b/src/App.tsx
index 00fd33f..2c22f17 100644
--- a/src/App.tsx
+++ b/src/App.tsx
@@ -191,27 +191,15 @@ const App: React.FC = () => {
               🔬 Lab
             </button>
           </div>
-          <div className="flex items-center gap-2">
-            <button
-              onClick={() => {
-                setSimulationMode('YFINANCE');
-                setShowSimulation(true);
-              }}
-              className="bg-red-600 hover:bg-red-500 text-white text-xs px-2 py-1 rounded animate-pulse font-bold"
-              title="Open Live Trading Dashboard"
-            >
-              🔴 LIVE
-            </button>
-            <button
-              onClick={() => {
-                setSimulationMode('SIMULATION');
-                setShowSimulation(true);
-              }}
-              className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1 rounded"
-            >
-              ▶ Replay
-            </button>
-          </div>
+          <button
+            onClick={() => {
+              setSimulationMode('SIMULATION');
+              setShowSimulation(true);
+            }}
+            className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1 rounded"
+          >
+            ▶ Replay
+          </button>
         </div>
 
         <RunPicker onSelect={setCurrentRun} />
diff --git a/src/api/client.ts b/src/api/client.ts
index 6bc60d2..6243354 100644
--- a/src/api/client.ts
+++ b/src/api/client.ts
@@ -3,12 +3,13 @@ import { VizDecision, VizTrade, RunManifest, AgentResponse, ChatMessage, Continu
 // API base URL - auto-detect port (8000 or 8001)
 let API_BASE = import.meta.env.VITE_API_URL || '';
 
-// Flag to track if backend is available
+// Flag to track if backend is available - only cache success, always retry on failure
 let backendAvailable: boolean | null = null;
 
 // Check backend availability, auto-detecting port if needed
 async function checkBackend(): Promise<boolean> {
-    if (backendAvailable !== null) return backendAvailable;
+    // Only cache success - if previously failed, try again
+    if (backendAvailable === true) return true;
 
     // If no explicit URL, try both ports
     if (!API_BASE) {
diff --git a/src/components/LiveSessionView.tsx b/src/components/LiveSessionView.tsx
index 3e9db72..0345a0f 100644
--- a/src/components/LiveSessionView.tsx
+++ b/src/components/LiveSessionView.tsx
@@ -111,15 +111,16 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
     const eventSourceRef = useRef<EventSource | null>(null);
     const dataSourceModeRef = useRef<DataSourceMode>(initialMode);
 
-    // Load data based on selected mode - but NOT for YFinance (user must press Play/Live)
+    // Load data based on selected mode
     useEffect(() => {
         dataSourceModeRef.current = dataSourceMode;
         if (dataSourceMode === 'SIMULATION') {
             loadSimulationData();
         } else {
-            // YFinance: reset state, wait for user to press Play or Go Live
+            // YFinance: auto-load historical data
             resetState();
-            setStatus('Ready - Press Play to replay history or Go Live for realtime');
+            setStatus('Loading YFinance data...');
+            fetchYFinanceHistory();
         }
     }, [dataSourceMode, lastTradeTimestamp, runId]);
 
@@ -214,6 +215,12 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
 
                 es.onmessage = (event) => {
                     try {
+                        // Skip non-JSON lines (debug output from backend)
+                        if (!event.data.startsWith('{')) {
+                            console.log('[SSE debug]', event.data);
+                            return;
+                        }
+
                         const data = JSON.parse(event.data);
 
                         if (data.type === 'HISTORY') {
@@ -225,6 +232,11 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
                                 close: b.close,
                                 volume: b.volume || 0
                             }));
+
+                            // Debug: log first 3 bars to console
+                            console.log('[YFinance] First 3 bars:', historyBars.slice(0, 3));
+                            console.log('[YFinance] Bars 50-53:', historyBars.slice(50, 53));
+
                             allBarsRef.current = historyBars;
                             setStartIndex(0);
                             setCurrentIndex(0);
@@ -257,117 +269,132 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
         }
     };
 
-    // Go Live: Fast-forward to latest candle and start realtime streaming
+    // Ref for polling interval separate from playback interval
+    const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
+
     const goLive = async () => {
-        setStatus('Connecting to live feed...');
-        try {
-            // Close any existing connection
-            if (eventSourceRef.current) {
-                eventSourceRef.current.close();
-                eventSourceRef.current = null;
+        if (allBarsRef.current.length === 0) {
+            setStatus('No bars loaded. Switch to YFinance mode first.');
+            return;
+        }
+
+        // Close any existing intervals
+        if (intervalRef.current) {
+            clearInterval(intervalRef.current);
+            intervalRef.current = null;
+        }
+        if (pollIntervalRef.current) {
+            clearInterval(pollIntervalRef.current);
+            pollIntervalRef.current = null;
+        }
+
+        // Show ALL bars immediately
+        setBars([...allBarsRef.current]);
+        setCurrentIndex(allBarsRef.current.length - 1);
+        setIsLiveStreaming(true);
+        setPlaybackState('PLAYING');
+
+        const lastBar = allBarsRef.current[allBarsRef.current.length - 1];
+        const lastTime = new Date(lastBar.time * 1000).toLocaleTimeString();
+        setStatus(`Live: ${allBarsRef.current.length} bars. Last: ${lastTime}. Running CNN...`);
+
+        // Run CNN on all bars in background (fast, 10ms per bar)
+        let processIdx = 60;
+        const processInterval = setInterval(() => {
+            if (processIdx >= allBarsRef.current.length) {
+                clearInterval(processInterval);
+                setStatus(`Live: Waiting for new candle... (Last: ${lastTime})`);
+                return;
             }
-            if (intervalRef.current) {
-                clearInterval(intervalRef.current);
-                intervalRef.current = null;
+            processBar(allBarsRef.current[processIdx], processIdx);
+            processIdx++;
+        }, 10);
+
+        // Start a "waiting" playback loop - it will just wait at the end
+        let idx = allBarsRef.current.length;
+        intervalRef.current = setInterval(() => {
+            if (idx >= allBarsRef.current.length) {
+                // Waiting for new bars - the poll will push them
+                return;
             }
+            // New bar arrived! Display it
+            const bar = allBarsRef.current[idx];
+            setBars(prev => [...prev, bar]);
+            setCurrentIndex(idx);
+            processBar(bar, idx);
 
-            const session = await api.startLiveReplay(ticker, selectedScanner, yfinanceDays, 10.0, {
-                entry_type: entryType,
-                stop_method: stopMethod,
-                tp_method: tpMethod,
-                stop_atr: stopAtr,
-                tp_atr: tpAtr
-            });
-            setStatus(`Live session started: ${session.session_id}`);
+            const newLastTime = new Date(bar.time * 1000).toLocaleTimeString();
+            setStatus(`Live: New bar! Total: ${allBarsRef.current.length}. Last: ${newLastTime}`);
+            idx++;
+        }, 200); // Check every 200ms for new bars
 
-            // Connect to SSE stream for continuous updates
-            const es = new EventSource(`http://localhost:8000/replay/stream/${session.session_id}`);
-            eventSourceRef.current = es;
-            setIsLiveStreaming(true);
-            setPlaybackState('PLAYING');
+        // Poll for new bars every 60 seconds
+        console.log('[Go Live] Starting 60-second poll');
+        pollIntervalRef.current = setInterval(async () => {
+            console.log('[Poll]', new Date().toLocaleTimeString());
+            try {
+                const session = await api.startLiveReplay(ticker, selectedScanner, 1, 10.0, {
+                    entry_type: entryType,
+                    stop_method: stopMethod,
+                    tp_method: tpMethod,
+                    stop_atr: stopAtr,
+                    tp_atr: tpAtr
+                });
 
-            es.onmessage = (event) => {
-                try {
-                    const data = JSON.parse(event.data);
-
-                    if (data.type === 'HISTORY') {
-                        // Bulk load history - show all at once
-                        const historyBars: BarData[] = data.bars.map((b: any) => ({
-                            time: new Date(b.timestamp).getTime() / 1000,
-                            open: b.open,
-                            high: b.high,
-                            low: b.low,
-                            close: b.close,
-                            volume: b.volume || 0
-                        }));
-                        allBarsRef.current = historyBars;
-                        setBars(historyBars);
-                        setCurrentIndex(historyBars.length - 1);
-                        setStartIndex(0);
-                        setStatus(`Live: ${historyBars.length} bars loaded. Waiting for new bars...`);
-                    } else if (data.type === 'BAR') {
-                        const bar: BarData = {
-                            time: new Date(data.timestamp).getTime() / 1000,
-                            open: data.open,
-                            high: data.high,
-                            low: data.low,
-                            close: data.close,
-                            volume: data.volume || 0
-                        };
-                        allBarsRef.current.push(bar);
-                        setBars([...allBarsRef.current]);
-                        setCurrentIndex(allBarsRef.current.length - 1);
-
-                        // Process OCO exits and run scanner
-                        processBar(bar, allBarsRef.current.length - 1);
-                    } else if (data.type === 'OCO_OPEN' || (data.type === 'DECISION' && data.triggered)) {
-                        // Backend triggered a trade entry
-                        const newOco = {
-                            entry: data.entry_price || data.price,
-                            stop: data.stop_price,
-                            tp: data.tp_price,
-                            startTime: new Date(data.timestamp || Date.now()).getTime() / 1000,
-                            direction: data.direction as 'LONG' | 'SHORT'
-                        };
-                        ocoRef.current = newOco;
-                        setOcoState(newOco);
-                        setTriggers(prev => prev + 1);
-                    } else if (data.type === 'STATUS') {
-                        setStatus(data.message || 'Live streaming...');
-                    } else if (data.type === 'STREAM_END') {
-                        setStatus(`Stream ended (code: ${data.exit_code})`);
-                        es.close();
-                        eventSourceRef.current = null;
-                        setIsLiveStreaming(false);
-                        setPlaybackState('STOPPED');
-                    } else if (data.type === 'ERROR') {
-                        setStatus(`Stream error: ${data.message}`);
+                const es = new EventSource(`http://localhost:8000/replay/stream/${session.session_id}`);
+
+                es.onmessage = (event) => {
+                    if (!event.data.startsWith('{')) return;
+                    try {
+                        const data = JSON.parse(event.data);
+                        if (data.type === 'HISTORY' && data.bars && data.bars.length > 0) {
+                            const latestTime = allBarsRef.current[allBarsRef.current.length - 1]?.time || 0;
+                            const newBars: BarData[] = data.bars
+                                .map((b: any) => ({
+                                    time: new Date(b.timestamp).getTime() / 1000,
+                                    open: b.open,
+                                    high: b.high,
+                                    low: b.low,
+                                    close: b.close,
+                                    volume: b.volume || 0
+                                }))
+                                .filter((b: BarData) => b.time > latestTime);
+
+                            if (newBars.length > 0) {
+                                console.log('[Poll] Found', newBars.length, 'new bars - pushing to ref');
+                                // Just push to ref - the playback loop will pick them up
+                                newBars.forEach((bar: BarData) => {
+                                    allBarsRef.current.push(bar);
+                                });
+                            } else {
+                                console.log('[Poll] No new bars');
+                            }
+                            es.close();
+                            api.stopReplay(session.session_id).catch(() => { });
+                        }
+                    } catch { }
+                };
+
+                es.onerror = () => es.close();
+                setTimeout(() => {
+                    if (es.readyState !== EventSource.CLOSED) {
                         es.close();
-                        eventSourceRef.current = null;
-                        setIsLiveStreaming(false);
-                        setPlaybackState('STOPPED');
+                        api.stopReplay(session.session_id).catch(() => { });
                     }
-                } catch (parseErr) {
-                    console.error('SSE parse error:', parseErr, event.data);
-                }
-            };
-
-            es.onerror = (err) => {
-                console.error('SSE connection error:', err);
-                setStatus('Live stream error - check console');
-                es.close();
-                eventSourceRef.current = null;
-                setIsLiveStreaming(false);
-                setPlaybackState('STOPPED');
-            };
-        } catch (e: any) {
-            setStatus(`Live error: ${e.message}`);
-            setIsLiveStreaming(false);
-        }
+                }, 10000);
+            } catch (e) {
+                console.error('[Poll] Error:', e);
+            }
+        }, 60000);
     };
 
 
     const handlePlayPause = useCallback(async () => {
+        // In live streaming mode, Play/Pause has no effect - use Stop to exit
+        if (isLiveStreaming) {
+            return;
+        }
+
         if (playbackState === 'PLAYING') {
             // Pause
             if (intervalRef.current) {
@@ -382,16 +409,20 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
                 const success = await fetchYFinanceHistory();
                 if (!success) return;
             }
-            // Play or Resume
+            // Play or Resume - same behavior for both Simulation and YFinance
             startPlayback();
         }
-    }, [playbackState, currentIndex, startIndex, dataSourceMode]);
+    }, [playbackState, currentIndex, startIndex, dataSourceMode, isLiveStreaming]);
 
     const handleStop = useCallback(() => {
         if (intervalRef.current) {
             clearInterval(intervalRef.current);
             intervalRef.current = null;
         }
+        if (pollIntervalRef.current) {
+            clearInterval(pollIntervalRef.current);
+            pollIntervalRef.current = null;
+        }
         if (eventSourceRef.current) {
             eventSourceRef.current.close();
             eventSourceRef.current = null;
@@ -450,10 +481,26 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
             return;
         }
 
+        // IMPORTANT: Check if resuming BEFORE setting state to PLAYING
+        const isResuming = playbackState === 'PAUSED';
+
+        // Clear any existing interval first
+        if (intervalRef.current) {
+            clearInterval(intervalRef.current);
+            intervalRef.current = null;
+        }
+
         setPlaybackState('PLAYING');
 
-        let idx = playbackState === 'PAUSED' ? currentIndex : startIndex;
-        if (playbackState !== 'PAUSED') {
+        // If resuming, start from current position. Otherwise reset.
+        let idx: number;
+        if (isResuming) {
+            // Resume from where we left off (currentIndex is already the next bar to show)
+            idx = currentIndex + 1;
+            setStatus('Resuming...');
+        } else {
+            // Fresh start
+            idx = startIndex;
             setCurrentIndex(startIndex);
             setBars([]);
             setOcoState(null);
@@ -465,13 +512,25 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
             setCompletedDecisions([]);
             completedTradesRef.current = [];
             completedDecisionsRef.current = [];
+            setStatus('Playing...');
         }
 
-        setStatus('Playing...');
-
         intervalRef.current = setInterval(() => {
             if (idx >= allBarsRef.current.length) {
-                handleStop();
+                // In YFinance mode: Wait for new bars instead of stopping
+                if (dataSourceModeRef.current === 'YFINANCE') {
+                    const lastBar = allBarsRef.current[allBarsRef.current.length - 1];
+                    const lastTime = lastBar ? new Date(lastBar.time * 1000).toLocaleTimeString() : 'N/A';
+                    setStatus(`Live: Waiting for new candle... (Last: ${lastTime})`);
+                    return; // Skip this tick, but keep interval alive!
+                }
+
+                // In Simulation mode: Stop as usual
+                if (intervalRef.current) {
+                    clearInterval(intervalRef.current);
+                    intervalRef.current = null;
+                }
+                setPlaybackState('STOPPED');
                 setStatus('Completed');
                 return;
             }
@@ -485,7 +544,7 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
 
             idx++;
         }, speed);
-    }, [speed, startIndex, currentIndex, playbackState, handleStop]);
+    }, [speed, startIndex, currentIndex, playbackState]);
 
     const processBar = (bar: BarData, idx: number) => {
         // OCO Exit Logic
@@ -566,9 +625,70 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
             }
         }
 
-        // Model Trigger Logic (Entry) - Only for SIMULATION mode with CNN enabled
+        // Trigger Logic (Entry) - SIMULATION mode only, either CNN model or pattern scanner
         // In YFinance mode, the backend (session_live.py) handles strategy triggering
-        if (dataSourceModeRef.current === 'SIMULATION' && useCnnModel && !ocoRef.current && idx % 5 === 0 && idx >= 60) {
+        const canTrigger = dataSourceModeRef.current === 'SIMULATION' && !ocoRef.current && idx % 5 === 0 && idx >= 60;
+        const shouldUseCnn = canTrigger && useCnnModel;
+        const shouldUseScanner = canTrigger && usePatternScanner;
+
+        // Pattern Scanner Trigger (simple local implementation)
+        if (shouldUseScanner && !shouldUseCnn) {
+            const recentBars = allBarsRef.current.slice(Math.max(0, idx - 13), idx + 1);
+            const avgRange = recentBars.reduce((sum, b) => sum + (b.high - b.low), 0) / recentBars.length;
+            const atr = avgRange || (bar.close * 0.001);
+
+            // Simple pattern detection based on selected scanner
+            let triggered = false;
+            let direction: 'LONG' | 'SHORT' | null = null;
+
+            if (selectedScanner === 'ema_cross' && recentBars.length >= 9) {
+                // Simple EMA cross check
+                const closes = recentBars.map(b => b.close);
+                const fast = closes.slice(-3).reduce((a, b) => a + b, 0) / 3;
+                const slow = closes.slice(-9).reduce((a, b) => a + b, 0) / 9;
+                const prevFast = closes.slice(-4, -1).reduce((a, b) => a + b, 0) / 3;
+                const prevSlow = closes.slice(-10, -1).reduce((a, b) => a + b, 0) / 9;
+
+                if (prevFast <= prevSlow && fast > slow) {
+                    triggered = true;
+                    direction = 'LONG';
+                } else if (prevFast >= prevSlow && fast < slow) {
+                    triggered = true;
+                    direction = 'SHORT';
+                }
+            } else if (selectedScanner === 'ifvg' && recentBars.length >= 3) {
+                // Simple IFVG detection (fair value gap)
+                const b1 = recentBars[recentBars.length - 3];
+                const b2 = recentBars[recentBars.length - 2];
+                const b3 = recentBars[recentBars.length - 1];
+
+                // Bullish FVG: gap between bar1 high and bar3 low
+                if (b1.high < b3.low && b2.close > b2.open) {
+                    triggered = true;
+                    direction = 'LONG';
+                }
+                // Bearish FVG: gap between bar1 low and bar3 high  
+                else if (b1.low > b3.high && b2.close < b2.open) {
+                    triggered = true;
+                    direction = 'SHORT';
+                }
+            }
+
+            if (triggered && direction) {
+                const entry = bar.close;
+                const isLong = direction === 'LONG';
+                const stop = isLong ? entry - (stopAtr * atr) : entry + (stopAtr * atr);
+                const tp = isLong ? entry + (tpAtr * atr) : entry - (tpAtr * atr);
+
+                const newOco = { entry, stop, tp, startTime: bar.time, direction };
+                ocoRef.current = newOco;
+                setOcoState(newOco);
+                setTriggers(prev => prev + 1);
+            }
+        }
+
+        // CNN Model Trigger (calls backend /infer endpoint)
+        if (shouldUseCnn) {
             const windowBars = allBarsRef.current.slice(Math.max(0, idx - 29), idx + 1);
             const recentBars = allBarsRef.current.slice(Math.max(0, idx - 13), idx + 1);
             const avgRange = recentBars.reduce((sum, b) => sum + (b.high - b.low), 0) / recentBars.length;
@@ -622,9 +742,41 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
         }
     };
 
+    // Restart interval when speed changes during playback
+    useEffect(() => {
+        if (playbackState === 'PLAYING' && intervalRef.current && !isLiveStreaming) {
+            // Clear old interval
+            clearInterval(intervalRef.current);
+            intervalRef.current = null;
+
+            // Create new interval with updated speed, starting from current position + 1
+            let idx = currentIndex + 1;
+            intervalRef.current = setInterval(() => {
+                if (idx >= allBarsRef.current.length) {
+                    if (intervalRef.current) {
+                        clearInterval(intervalRef.current);
+                        intervalRef.current = null;
+                    }
+                    setPlaybackState('STOPPED');
+                    setStatus('Completed');
+                    return;
+                }
+
+                const bar = allBarsRef.current[idx];
+                setBars(prev => [...prev, bar]);
+                setCurrentIndex(idx);
+                processBar(bar, idx);
+                idx++;
+            }, speed);
+
+            setStatus(`Speed changed to ${speed}ms`);
+        }
+    }, [speed]);
+
     useEffect(() => {
         return () => {
             if (intervalRef.current) clearInterval(intervalRef.current);
+            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
             if (eventSourceRef.current) {
                 eventSourceRef.current.close();
                 eventSourceRef.current = null;
@@ -654,10 +806,17 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
                 </div>
             </div>
 
-            <div className="flex-1 flex">
+            <div className="flex-1 flex overflow-hidden min-h-0">
                 {/* Main Chart Area */}
                 <div className="flex-1 flex flex-col min-h-0">
                     <div className="flex-1 min-h-[400px]">
+                        {/* Debug: Log first few bars to console */}
+                        {bars.length > 0 && console.log('[Chart Input] First 3 bars:', bars.slice(0, 3).map(b => ({
+                            time: b.time,
+                            timeISO: new Date(b.time * 1000).toISOString(),
+                            open: b.open, high: b.high, low: b.low, close: b.close,
+                            range: b.high - b.low
+                        })))}
                         <CandleChart
                             continuousData={bars.length > 0 ? {
                                 timeframe: '1m',
@@ -678,8 +837,7 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
                 </div>
 
                 {/* Right Sidebar - Controls */}
-                {/* Right Sidebar - Controls */}
-                <div className="w-80 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto">
+                <div className="w-80 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto min-h-0 max-h-full">
 
                     {/* Playback Controls */}
                     <SidebarSection title="Playback" defaultOpen={true} colorClass="text-green-400">
diff --git a/src/server/main.py b/src/server/main.py
index b42b546..d050341 100644
--- a/src/server/main.py
+++ b/src/server/main.py
@@ -375,11 +375,13 @@ def build_agent_prompt(context: ChatContext, decisions: List[Dict], trades: List
     current_json = json.dumps(current, indent=2) if current else "None selected"
     
     # Discovery info for modular system
-    trigger_types = ["time", "candle_pattern", "ema_cross", "rsi_threshold"]
+    trigger_types = ["time", "candle_pattern", "ema_cross", "ema_bounce", "rsi_threshold", "ifvg", "orb"]
     bracket_types = ["atr", "percent", "fixed"]
 
-    return f"""You are a trade analysis assistant for the MLang2 trading research platform.
-You can BOTH analyze existing data AND run new strategies to create data.
+    return f"""You are a STRATEGY SCAN agent for the MLang2 trading research platform.
+
+YOUR PURPOSE: Create and run strategy scans on specified time windows so the user can visually 
+analyze if trades make sense. You DO NOT run replays or live trading - that is a separate mode.
 
 CURRENT CONTEXT:
 - Run ID: {context.runId}
@@ -390,41 +392,34 @@ CURRENT {item_type.upper()} DATA:
 {current_json}
 
 AVAILABLE ACTIONS:
-1. Navigate: ACTION: {{"type": "SET_INDEX", "payload": <number>}}
-2. Switch mode: ACTION: {{"type": "SET_MODE", "payload": "DECISION" or "TRADE"}}
-3. Load run: ACTION: {{"type": "LOAD_RUN", "payload": "<run_id>"}}
-4. RUN STRATEGY: ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "modular", "config": <config_dict>}}}}
-5. START REPLAY: ACTION: {{"type": "START_REPLAY", "payload": {{"start_date": "YYYY-MM-DD", "days": 1, "speed": 10, "threshold": 0.6}}}}
-6. TRAIN FROM SCAN: ACTION: {{"type": "TRAIN_FROM_SCAN", "payload": {{"scan_run_id": "<run_id>", "model_name": "my_model"}}}}
-
-CRITICAL INSTRUCTION:
-To perform an action, you MUST include the "ACTION:" line at the end of your response.
-Do NOT just output the JSON config. You MUST wrap it in the ACTION format.
-
-Example - Run RSI Strategy:
-Okay, I'll run that strategy.
-ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "modular", "config": {{"trigger": {{"type": "rsi_threshold", "oversold": 30}}, "bracket": {{"type": "atr", "stop_atr": 2, "tp_atr": 3}}}}}}}}
-
-Example - Navigate:
-Moving to the next trade.
-ACTION: {{"type": "SET_INDEX", "payload": 12}}
-
-MODULAR STRATEGY FORMAT:
+1. Navigate decisions/trades: ACTION: {{"type": "SET_INDEX", "payload": <number>}}
+2. Switch view mode: ACTION: {{"type": "SET_MODE", "payload": "DECISION" or "TRADE"}}
+3. Load existing run: ACTION: {{"type": "LOAD_RUN", "payload": "<run_id>"}}
+4. RUN STRATEGY SCAN: ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "modular", "start_date": "YYYY-MM-DD", "weeks": N, "config": <config_dict>}}}}
+5. TRAIN MODEL FROM SCAN: ACTION: {{"type": "TRAIN_FROM_SCAN", "payload": {{"scan_run_id": "<run_id>", "model_name": "my_model"}}}}
+
+CRITICAL: To execute ANY action, you MUST include "ACTION:" at the end of your response.
+Do NOT just describe the config - you MUST wrap it in: ACTION: {{"type": "...", "payload": ...}}
+
+MODULAR STRATEGY CONFIG:
 {{
-  "trigger": {{"type": "...", ...}},
-  "bracket": {{"type": "...", ...}}
+  "trigger": {{"type": "<trigger_type>", ...trigger_params}},
+  "bracket": {{"type": "<bracket_type>", "stop_atr": N, "tp_atr": M}}
 }}
 
-TRIGGERS: {trigger_types}
-BRACKETS: {bracket_types}
+TRIGGER TYPES: {trigger_types}
+BRACKET TYPES: {bracket_types}
+
+DATA RANGE: Historical data covers March 18 - September 17, 2025.
+
+EXAMPLE - User asks "Run an EMA cross scan for last 2 weeks of May":
+I'll run an EMA cross strategy scan for May 19-31, 2025.
+ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "modular", "start_date": "2025-05-19", "weeks": 2, "config": {{"trigger": {{"type": "ema_cross", "fast": 9, "slow": 21}}, "bracket": {{"type": "atr", "stop_atr": 2, "tp_atr": 3}}}}}}}}
 
-AVAILABLE STRATEGIES: "opening_range", "modular"
-TRAINED MODEL: models/best_model.pth (FusionModel CNN)
+EXAMPLE - User asks "Show me the next trade":
+ACTION: {{"type": "SET_INDEX", "payload": {context.currentIndex + 1}}}
 
-When user asks to run/create/generate data, use RUN_STRATEGY with "modular" and a config.
-When user asks to replay/visualize/watch model triggers, use START_REPLAY.
-Include action at END in format: ACTION: {{"type": "...", "payload": ...}}
-Be concise."""
+Be concise. Focus on creating scans the user can visually evaluate."""
 
 
 @app.post("/agent/chat")
```

### New Untracked Files

#### `gitrdiff.md` (693 lines - truncated)

```
# Git Diff Report

**Generated**: Tue, Dec 23, 2025 11:53:33 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M src/App.tsx
 M src/api/client.ts
 M src/components/LiveSessionView.tsx
 M src/server/main.py
?? gitrdiff.md
?? goLive_simple.txt
?? src/components/LiveSessionView.tsx.backup
```

### Uncommitted Diff

```diff
diff --git a/src/App.tsx b/src/App.tsx
index 00fd33f..2c22f17 100644
--- a/src/App.tsx
+++ b/src/App.tsx
@@ -191,27 +191,15 @@ const App: React.FC = () => {
               🔬 Lab
             </button>
           </div>
-          <div className="flex items-center gap-2">
-            <button
-              onClick={() => {
-                setSimulationMode('YFINANCE');
-                setShowSimulation(true);
-              }}
-              className="bg-red-600 hover:bg-red-500 text-white text-xs px-2 py-1 rounded animate-pulse font-bold"
-              title="Open Live Trading Dashboard"
-            >
-              🔴 LIVE
-            </button>
-            <button
-              onClick={() => {
-                setSimulationMode('SIMULATION');
-                setShowSimulation(true);
-              }}
-              className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1 rounded"
-            >
-              ▶ Replay
-            </button>
-          </div>
+          <button
+            onClick={() => {
+              setSimulationMode('SIMULATION');
+              setShowSimulation(true);
+            }}
+            className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1 rounded"
+          >
+            ▶ Replay
+          </button>
         </div>
 
         <RunPicker onSelect={setCurrentRun} />
diff --git a/src/api/client.ts b/src/api/client.ts
index 6bc60d2..6243354 100644
--- a/src/api/client.ts
+++ b/src/api/client.ts
@@ -3,12 +3,13 @@ import { VizDecision, VizTrade, RunManifest, AgentResponse, ChatMessage, Continu
 // API base URL - auto-detect port (8000 or 8001)
 let API_BASE = import.meta.env.VITE_API_URL || '';
 
-// Flag to track if backend is available
+// Flag to track if backend is available - only cache success, always retry on failure
 let backendAvailable: boolean | null = null;
 
 // Check backend availability, auto-detecting port if needed
 async function checkBackend(): Promise<boolean> {
-    if (backendAvailable !== null) return backendAvailable;
+    // Only cache success - if previously failed, try again
+    if (backendAvailable === true) return true;
 
     // If no explicit URL, try both ports
     if (!API_BASE) {
diff --git a/src/components/LiveSessionView.tsx b/src/components/LiveSessionView.tsx
index 3e9db72..0345a0f 100644
--- a/src/components/LiveSessionView.tsx
+++ b/src/components/LiveSessionView.tsx
@@ -111,15 +111,16 @@ export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
     const eventSourceRef = useRef<EventSource | null>(null);
     const dataSourceModeRef = useRef<DataSourceMode>(initialMode);
 
-    // Load data based on selected mode - but NOT for YFinance (user must press Play/Live)
+    // Load data based on selected mode
     useEffect(() => {
         dataSourceModeRef.current = dataSourceMode;
... (12 total lines)
```

#### `goLive_simple.txt`

```
// Simple working goLive - connects to backend SSE stream and displays bars as they come
const goLive = async () => {
    setStatus('Connecting to live feed...');
    try {
        // Close any existing connection
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        const session = await api.startLiveReplay(ticker, selectedScanner, yfinanceDays, 10.0, {
            entry_type: entryType,
            stop_method: stopMethod,
            tp_method: tpMethod,
            stop_atr: stopAtr,
            tp_atr: tpAtr
        });
        setStatus(`Live session started: ${session.session_id}`);

        // Connect to SSE stream for continuous updates
        const es = new EventSource(`http://localhost:8000/replay/stream/${session.session_id}`);
        eventSourceRef.current = es;
        setIsLiveStreaming(true);
        setPlaybackState('PLAYING');

        es.onmessage = (event) => {
            try {
                // Skip non-JSON lines
                if (!event.data.startsWith('{')) {
                    console.log('[SSE debug]', event.data);
                    return;
                }

                const data = JSON.parse(event.data);

                if (data.type === 'HISTORY') {
                    // Bulk load history - show all at once
                    const historyBars: BarData[] = data.bars.map((b: any) => ({
                        time: new Date(b.timestamp).getTime() / 1000,
                        open: b.open,
                        high: b.high,
                        low: b.low,
                        close: b.close,
                        volume: b.volume || 0
                    }));
                    allBarsRef.current = historyBars;
                    setBars(historyBars);
                    setCurrentIndex(historyBars.length - 1);
                    setStartIndex(0);
                    setStatus(`Live: ${historyBars.length} bars loaded. Waiting for new bars...`);
                } else if (data.type === 'BAR') {
                    const bar: BarData = {
                        time: new Date(data.timestamp).getTime() / 1000,
                        open: data.open,
                        high: data.high,
                        low: data.low,
                        close: data.close,
                        volume: data.volume || 0
                    };
                    allBarsRef.current.push(bar);
                    setBars([...allBarsRef.current]);
                    setCurrentIndex(allBarsRef.current.length - 1);

                    // Process OCO exits and run scanner
                    processBar(bar, allBarsRef.current.length - 1);
                } else if (data.type === 'OCO_OPEN' || (data.type === 'DECISION' && data.triggered)) {
                    // Backend triggered a trade entry
                    const newOco = {
                        entry: data.entry_price || data.price,
                        stop: data.stop_price,
                        tp: data.tp_price,
                        startTime: new Date(data.timestamp || Date.now()).getTime() / 1000,
                        direction: data.direction as 'LONG' | 'SHORT'
                    };
                    ocoRef.current = newOco;
                    setOcoState(newOco);
                    setTriggers(prev => prev + 1);
                } else if (data.type === 'STATUS') {
                    setStatus(data.message || 'Live streaming...');
                } else if (data.type === 'STREAM_END') {
                    setStatus(`Stream ended (code: ${data.exit_code})`);
                    es.close();
                    eventSourceRef.current = null;
                    setIsLiveStreaming(false);
                    setPlaybackState('STOPPED');
                } else if (data.type === 'ERROR') {
                    setStatus(`Stream error: ${data.message}`);
                    es.close();
                    eventSourceRef.current = null;
                    setIsLiveStreaming(false);
                    setPlaybackState('STOPPED');
                }
            } catch (parseErr) {
                console.error('SSE parse error:', parseErr, event.data);
            }
        };

        es.onerror = (err) => {
            console.error('SSE connection error:', err);
            setStatus('Live stream error - check console');
            es.close();
            eventSourceRef.current = null;
            setIsLiveStreaming(false);
            setPlaybackState('STOPPED');
        };
    } catch (e: any) {
        setStatus(`Live error: ${e.message}`);
        setIsLiveStreaming(false);
    }
};
```

#### `src/components/LiveSessionView.tsx.backup` (1181 lines - truncated)

```
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { CandleChart } from './CandleChart';
import { VizTrade, VizDecision } from '../types/viz';
import { api } from '../api/client';

interface LiveSessionViewProps {
    onClose: () => void;
    runId?: string;
    lastTradeTimestamp?: string;
    initialMode?: 'SIMULATION' | 'YFINANCE';
}

interface BarData {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

type DataSourceMode = 'SIMULATION' | 'YFINANCE';
type PlaybackState = 'STOPPED' | 'PLAYING' | 'PAUSED';

const SidebarSection: React.FC<{
    title: string;
    children: React.ReactNode;
    defaultOpen?: boolean;
    colorClass?: string;
}> = ({ title, children, defaultOpen = false, colorClass = "text-blue-400" }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="mb-2 border-b border-slate-700 pb-2 last:border-0">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`flex items-center justify-between w-full text-xs font-bold uppercase py-1 ${colorClass} hover:opacity-80`}
            >
                {title}
                <span className="text-slate-500">{isOpen ? '▼' : '▶'}</span>
            </button>
            {isOpen && <div className="mt-2 text-sm">{children}</div>}
        </div>
    );
};

export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
    onClose,
    runId,
    lastTradeTimestamp,
    initialMode = 'SIMULATION'
}) => {
    // Data Source
    const [dataSourceMode, setDataSourceMode] = useState<DataSourceMode>(initialMode);

    // Playback State
    const [playbackState, setPlaybackState] = useState<PlaybackState>('STOPPED');
    const [speed, setSpeed] = useState(200); // ms per bar
    const [bars, setBars] = useState<BarData[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [startIndex, setStartIndex] = useState(0);
    const [status, setStatus] = useState('Ready');

    // Model/Scanner Selection - Now with enable checkboxes (OFF by default)
    const [useCnnModel, setUseCnnModel] = useState(false);       // OFF by default
    const [usePatternScanner, setUsePatternScanner] = useState(false); // OFF by default
    const [selectedModel, setSelectedModel] = useState('models/ifvg_4class_cnn.pth');
    const [selectedScanner, setSelectedScanner] = useState('ifvg');
    const [availableModels, setAvailableModels] = useState<string[]>([
        'models/ifvg_4class_cnn.pth',
        'models/ifvg_cnn.pth',
        'models/best_model.pth'
    ]);

    // Entry Configuration (sent to backend)
    const [entryType, setEntryType] = useState<'market' | 'limit'>('market');
    const [stopMethod, setStopMethod] = useState<'atr' | 'swing' | 'fixed_bars'>('atr');
    const [tpMethod, setTpMethod] = useState<'atr' | 'r_multiple'>('atr');

    // OCO State
    const [ocoState, setOcoState] = useState<{
        entry: number;
        stop: number;
        tp: number;
        startTime: number;
        direction: 'LONG' | 'SHORT';
    } | null>(null);

    // Trade Settings
    const [threshold, setThreshold] = useState(0.35);
    const [stopAtr, setStopAtr] = useState(2.0);
    const [tpAtr, setTpAtr] = useState(4.0);

    // Trade Tracking
    const [triggers, setTriggers] = useState(0);
    const [wins, setWins] = useState(0);
    const [losses, setLosses] = useState(0);
    const [completedTrades, setCompletedTrades] = useState<VizTrade[]>([]);
    const [completedDecisions, setCompletedDecisions] = useState<VizDecision[]>([]);

    // YFinance specific
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
