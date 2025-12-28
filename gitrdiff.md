# Git Diff Report

**Generated**: Sun, Dec 28, 2025 12:15:29 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M src/App.tsx
 M src/api/client.ts
 M src/components/ChatAgent.tsx
 M src/server/fast_viz_routes.py
 M src/server/main.py
?? gitrdiff.md
?? scanners.md
```

### Uncommitted Diff

```diff
diff --git a/src/App.tsx b/src/App.tsx
index 394ad73..9f66ef8 100644
--- a/src/App.tsx
+++ b/src/App.tsx
@@ -456,6 +456,7 @@ const App: React.FC = () => {
             runId={currentRun || 'none'}
             currentIndex={index}
             currentMode={mode}
+            fastVizMode={fastVizEnabled}
             onAction={handleAgentAction}
           />
         </div>
diff --git a/src/api/client.ts b/src/api/client.ts
index 1d7761b..1ee11e7 100644
--- a/src/api/client.ts
+++ b/src/api/client.ts
@@ -126,7 +126,7 @@ export const api = {
 
     postAgent: async (
         messages: ChatMessage[],
-        context: { runId: string, currentIndex: number, currentMode: 'DECISION' | 'TRADE' }
+        context: { runId: string, currentIndex: number, currentMode: 'DECISION' | 'TRADE', fastVizMode?: boolean }
     ): Promise<AgentResponse> => {
         const hasBackend = await checkBackend();
         if (!hasBackend) {
diff --git a/src/components/ChatAgent.tsx b/src/components/ChatAgent.tsx
index 1fde1a7..bcd6866 100644
--- a/src/components/ChatAgent.tsx
+++ b/src/components/ChatAgent.tsx
@@ -7,10 +7,11 @@ interface ChatAgentProps {
   runId: string;
   currentIndex: number;
   currentMode: 'DECISION' | 'TRADE';
+  fastVizMode?: boolean;
   onAction: (action: UIAction) => void;
 }
 
-export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, currentMode, onAction }) => {
+export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, currentMode, fastVizMode = false, onAction }) => {
   const [messages, setMessages] = useState<ChatMessage[]>([
     { role: 'assistant', content: 'Hello! I am the **Trade Viz Agent**. How can I help with your analysis today?' }
   ]);
@@ -35,7 +36,7 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
     setLoading(true);
 
     try {
-      const response = await api.postAgent([...messages, userMsg], { runId, currentIndex, currentMode });
+      const response = await api.postAgent([...messages, userMsg], { runId, currentIndex, currentMode, fastVizMode });
 
       setMessages(prev => [...prev, { role: 'assistant', content: response.reply }]);
 
@@ -56,14 +57,14 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
       {/* Header */}
       <div className="px-4 py-3 bg-slate-950 border-b border-slate-800 flex items-center justify-between shrink-0">
         <div className="flex items-center gap-2">
-           <div className="relative">
-             <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
-             <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-500 animate-ping opacity-20"></div>
-           </div>
-           <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest">Agent Terminal</h3>
+          <div className="relative">
+            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
+            <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-500 animate-ping opacity-20"></div>
+          </div>
+          <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest">Agent Terminal</h3>
         </div>
         <div className="text-[10px] text-slate-600 font-mono">
-           {runId === 'none' ? 'DISCONNECTED' : 'ONLINE'}
+          {runId === 'none' ? 'DISCONNECTED' : 'ONLINE'}
         </div>
       </div>
 
@@ -72,18 +73,17 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
         {messages.map((m, i) => (
           <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
             {m.role === 'assistant' && (
-                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-xs text-white font-bold shrink-0 mr-3 shadow-lg mt-1">
-                    AI
-                </div>
+              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-xs text-white font-bold shrink-0 mr-3 shadow-lg mt-1">
+                AI
+              </div>
             )}
 
             <div className={`max-w-[85%] relative group-message ${m.role === 'user' ? 'items-end flex flex-col' : ''}`}>
-               {m.role === 'user' && (
-                  <div className="text-[10px] text-slate-500 mb-1 mr-1 uppercase tracking-wider font-bold">You</div>
-               )}
+              {m.role === 'user' && (
+                <div className="text-[10px] text-slate-500 mb-1 mr-1 uppercase tracking-wider font-bold">You</div>
+              )}
 
-               <div className={`px-5 py-3.5 text-sm shadow-md transition-all ${
-                  m.role === 'user'
+              <div className={`px-5 py-3.5 text-sm shadow-md transition-all ${m.role === 'user'
                   ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm'
                   : 'bg-slate-900 text-slate-300 border border-slate-800 rounded-2xl rounded-tl-sm'
                 }`}>
@@ -105,23 +105,23 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
             </div>
 
             {m.role === 'user' && (
-                <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-xs text-slate-400 font-bold shrink-0 ml-3 shadow-lg mt-1 border border-slate-700">
-                    U
-                </div>
+              <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-xs text-slate-400 font-bold shrink-0 ml-3 shadow-lg mt-1 border border-slate-700">
+                U
+              </div>
             )}
           </div>
         ))}
 
         {loading && (
           <div className="flex justify-start animate-fade-in">
-              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-xs text-white font-bold shrink-0 mr-3 shadow-lg mt-1">
-                 AI
-              </div>
-              <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm flex items-center gap-1.5">
-                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"></span>
-                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-75"></span>
-                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-150"></span>
-              </div>
+            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-xs text-white font-bold shrink-0 mr-3 shadow-lg mt-1">
+              AI
+            </div>
+            <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm flex items-center gap-1.5">
+              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"></span>
+              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-75"></span>
+              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-150"></span>
+            </div>
           </div>
         )}
       </div>
@@ -130,17 +130,17 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
       <div className="p-4 bg-slate-950 border-t border-slate-800 shrink-0">
         <form onSubmit={handleSubmit} className="relative flex items-center gap-2 max-w-4xl mx-auto w-full">
           <div className="relative flex-1">
-              <input
-                ref={inputRef}
-                value={input}
-                onChange={e => setInput(e.target.value)}
-                placeholder={runId === 'none' ? "Select a run to start chatting..." : "Ask for analysis, valid setups, or strategy insights..."}
-                disabled={runId === 'none' || loading}
-                className="w-full bg-slate-900 border border-slate-800 text-slate-200 placeholder-slate-600 rounded-xl px-4 py-3.5 pl-5 pr-12 text-sm focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all shadow-inner disabled:opacity-50 disabled:cursor-not-allowed"
-              />
-              <div className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-slate-600 font-mono hidden md:block border border-slate-800 px-1.5 py-0.5 rounded">
-                  ↵ Enter
-              </div>
+            <input
+              ref={inputRef}
+              value={input}
+              onChange={e => setInput(e.target.value)}
+              placeholder={runId === 'none' ? "Select a run to start chatting..." : "Ask for analysis, valid setups, or strategy insights..."}
+              disabled={runId === 'none' || loading}
+              className="w-full bg-slate-900 border border-slate-800 text-slate-200 placeholder-slate-600 rounded-xl px-4 py-3.5 pl-5 pr-12 text-sm focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all shadow-inner disabled:opacity-50 disabled:cursor-not-allowed"
+            />
+            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-slate-600 font-mono hidden md:block border border-slate-800 px-1.5 py-0.5 rounded">
+              ↵ Enter
+            </div>
           </div>
           <button
             type="submit"
@@ -148,12 +148,12 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
             className="bg-blue-600 hover:bg-blue-500 text-white rounded-xl p-3.5 shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95 flex items-center justify-center aspect-square"
           >
             <svg className="w-5 h-5 translate-x-0.5 -translate-y-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
-                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
+              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
             </svg>
           </button>
         </form>
         <div className="text-center mt-2">
-            <p className="text-[10px] text-slate-600">AI can make mistakes. Verify important trading decisions.</p>
+          <p className="text-[10px] text-slate-600">AI can make mistakes. Verify important trading decisions.</p>
         </div>
       </div>
     </div>
diff --git a/src/server/fast_viz_routes.py b/src/server/fast_viz_routes.py
index 4a9ee86..72d3eab 100644
--- a/src/server/fast_viz_routes.py
+++ b/src/server/fast_viz_routes.py
@@ -209,10 +209,21 @@ async def save_fast_viz_run(run_id: str):
             "--end-date", result.end_date
         ]
         
-        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
+        print(f"[FAST_VIZ] Promoting {run_id} to full simulation: {' '.join(cmd)}")
+        proc_result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
         
-        # Remove from ephemeral cache
+        # Check if the subprocess succeeded
+        if proc_result.returncode != 0:
+            error_msg = proc_result.stderr or proc_result.stdout or "Unknown error"
+            print(f"[FAST_VIZ] Promotion failed: {error_msg}")
+            raise HTTPException(
+                status_code=500, 
+                detail=f"Full simulation failed: {error_msg[:500]}"
+            )
+        
+        # Only remove from ephemeral cache AFTER successful promotion
         del _fast_viz_runs[run_id]
+        print(f"[FAST_VIZ] Successfully promoted {run_id} -> {new_run_id}")
         
         return {
             "success": True,
@@ -221,7 +232,12 @@ async def save_fast_viz_run(run_id: str):
             "message": "Promoted to full simulation with viz artifacts"
         }
         
+    except subprocess.TimeoutExpired:
+        raise HTTPException(status_code=500, detail="Full simulation timed out after 5 minutes")
+    except HTTPException:
+        raise  # Re-raise HTTP exceptions as-is
     except Exception as e:
+        print(f"[FAST_VIZ] Exception during promotion: {e}")
         raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")
     finally:
         try:
diff --git a/src/server/main.py b/src/server/main.py
index d89a2c0..266c652 100644
--- a/src/server/main.py
+++ b/src/server/main.py
@@ -107,6 +107,7 @@ class ChatContext(BaseModel):
     runId: str
     currentIndex: int
     currentMode: str  # 'DECISION' or 'TRADE'
+    fastVizMode: bool = False  # When True, agent emits RUN_FAST_VIZ instead of RUN_STRATEGY
 
 
 class ChatRequest(BaseModel):
@@ -922,17 +923,42 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
                                     "tp_atr": fn_args.get("tp_atr", 3.0)
                                 }
                             }
-                            ui_action = UIAction(
-                                type="RUN_STRATEGY",
-                                payload={
-                                    "strategy": fn_args.get("strategy", "modular"),
-                                    "start_date": fn_args.get("start_date", "2025-03-18"),
-                                    "weeks": fn_args.get("weeks", 1),
-                                    "run_name": fn_args.get("run_name"),
-                                    "config": config
-                                }
-                            )
-                            reply_text = f"Running {fn_args.get('trigger_type', 'modular')} strategy scan from {fn_args.get('start_date')} for {fn_args.get('weeks')} week(s)..."
+                            
+                            # Check if Fast Viz mode is enabled
+                            fast_viz_enabled = request.context.fastVizMode if hasattr(request.context, 'fastVizMode') else False
+                            
+                            if fast_viz_enabled:
+                                # Emit RUN_FAST_VIZ for instant ideation
+                                from datetime import timedelta
+                                import pandas as pd
+                                start_date = fn_args.get('start_date', '2025-05-01')
+                                weeks = fn_args.get('weeks', 2)
+                                start_dt = pd.to_datetime(start_date)
+                                end_dt = start_dt + timedelta(weeks=weeks)
+                                
+                                ui_action = UIAction(
+                                    type="RUN_FAST_VIZ",
+                                    payload={
+                                        "config": config,
+                                        "start_date": start_date,
+                                        "end_date": end_dt.strftime("%Y-%m-%d"),
+                                        "run_name": fn_args.get('run_name')
+                                    }
+                                )
+                                reply_text = f"⚡ Fast Viz: {fn_args.get('trigger_type', 'modular')} strategy from {start_date} ({weeks} week(s)). Results are approximate - click 💾 to verify with full simulation."
+                            else:
+                                # Normal full simulation
+                                ui_action = UIAction(
+                                    type="RUN_STRATEGY",
+                                    payload={
+                                        "strategy": fn_args.get("strategy", "modular"),
+                                        "start_date": fn_args.get("start_date", "2025-03-18"),
+                                        "weeks": fn_args.get("weeks", 1),
+                                        "run_name": fn_args.get("run_name"),
+                                        "config": config
+                                    }
+                                )
+                                reply_text = f"Running {fn_args.get('trigger_type', 'modular')} strategy scan from {fn_args.get('start_date')} for {fn_args.get('weeks')} week(s)..."
                         
                         elif fn_name == "set_index":
                             ui_action = UIAction(type="SET_INDEX", payload=fn_args.get("index", 0))
```

### New Untracked Files

#### `gitrdiff.md`

```
```

#### `scanners.md`

```
# Scanners Architecture - Dev Notes

> This document captures insights on how scanners work and what's needed to make "verified scans" automatically available in Replay Mode.

## Current State

**With the code as it stands, "turning a trade into a scan" will NOT automatically make that scan show up as an available scanner in Replay Mode.**

### Why it doesn't auto-appear (today)

#### 1. Replay Mode's "scanner list" is not driven by past scan artifacts

`RunManifest` explicitly separates **SCAN** vs **REPLAY** runs:
- `create_for_scan(...)` stores `scanners=[ScannerConfig(...)]`
- `create_for_replay(...)` stores `models=[ModelConfig(...)]` and **does not include scanners**

Even if a scan run produced a good scanner config, Replay isn't reading that manifest field because Replay manifests don't carry scanners.

#### 2. New scanners created by `scripts/create_strategy.py` are just files — not auto-registered

- `create_strategy.py` scaffolds a new scanner class into `src/policy/library/*.py`
- But `src/policy/scanner_registry_init.py` only registers a couple built-ins (`always`, `interval`, `modular`) and doesn't import/discover the library folder

**Meaning:** Unless something imports that new module (or you add discovery), the Replay UI won't "see" it as an option.

---

## What it would take (clean path)

Two missing bridges: **(A) discovery/registration** and **(B) replay consumption**.

### A) Make scanners "discoverable" automatically

| Option | Approach |
|--------|----------|
| **A1 (simple/robust)** | Plugin discovery on startup. Scan `src/policy/library/` and import every module so scanner classes exist. Add a "register decorator" pattern for library scanners. |
| **A2 (cleaner long-term)** | Treat "verified scans" as data, not code. Store a "scanner recipe" (e.g., `trigger_config`) and load dynamically via `ScannerRegistry.create("modular", trigger_config=...)`. Avoids generating python files for every idea. |

### B) Let Replay Mode run a scanner (not just a model)

Right now Replay's manifest factory is model-only, but Replay UI expects a "scanner selection" concept. The missing wiring:

1. Add `scanner_id` + `scanner_params` to Replay "start" request (backend + frontend).
2. In replay engine startup, instantiate a scanner from the registry (or from a recipe).
3. Optionally: allow "Load Scanner From Run" where UI reads a prior SCAN run's manifest and offers **"Use this scanner in Replay"**.

---

## Minimal "Works Fast" Implementation

1. **Verified scan writes a `scanner_recipe.json`** into the run artifact folder.

2. **Replay "scanner dropdown" gets a second tab:**
   - **Built-ins** (always/interval/modular)
   - **Verified scans** (pulled from `results/viz/*/manifest.json` where `run_mode == SCAN` and it has `scanners[]`)

3. **Replay start uses:**
   - `modular` scanner + the saved recipe (fast, no import games), OR
   - a registry-created scanner if you implement discovery

**Result:** Ideation can "backcheat" and be instant, then **Verify** produces:
- A persistent artifact (scan run + recipe)
- A replay-usable scanner (immediately selectable)

---

## Files to Wire

| Component | File | Change Needed |
|-----------|------|---------------|
| Replay Start Request | `src/server/replay_routes.py` | Add `scanner_id`, `scanner_params` fields |
| Replay Engine | `scripts/session_replay.py` | Instantiate scanner from registry |
| Scanner Registry | `src/policy/scanner_registry_init.py` | Add discovery for library folder |
| Frontend Dropdown | `src/components/LiveSessionView.tsx` | Add "Verified Scans" tab to scanner picker |
| Run Manifest | `src/viz/manifest.py` | Store `scanner_recipe` in verified runs |
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
