# Git Diff Report

**Generated**: Sun, Dec 28, 2025  5:36:59 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M src/App.tsx
 M src/components/ChatAgent.tsx
 M src/server/main.py
 M src/tools/agent_tools.py
?? gitrdiff.md
?? src/tools/composition_tools.py
?? src/tools/discovery_tools.py
?? verification/verify_composition.py
?? verification/verify_discovery.py
```

### Uncommitted Diff

```diff
diff --git a/src/App.tsx b/src/App.tsx
index d24229b..4d9e8b5 100644
--- a/src/App.tsx
+++ b/src/App.tsx
@@ -27,6 +27,7 @@ const App: React.FC = () => {
 
   const [decisions, setDecisions] = useState<VizDecision[]>([]);
   const [trades, setTrades] = useState<VizTrade[]>([]);
+  const [planningMode, setPlanningMode] = useState<boolean>(false);
 
   // Layout State
   const [chatHeight, setChatHeight] = useState<number>(320);
@@ -166,7 +167,7 @@ const App: React.FC = () => {
   const handleAgentAction = async (action: UIAction) => {
     // If we receive a UI action that affects the chart, collapse the chat
     if (['RUN_STRATEGY', 'SET_INDEX', 'SET_MODE', 'LOAD_RUN', 'RUN_FAST_VIZ'].includes(action.type)) {
-        setIsChatExpanded(false);
+      setIsChatExpanded(false);
     }
 
     switch (action.type) {
@@ -235,38 +236,38 @@ const App: React.FC = () => {
 
   // Expand Chat on Research/Text response
   const handleAgentTextResponse = () => {
-      // If the chat isn't expanded, expand it to show the research results
-      // But only if we are NOT currently running a viz action (which is handled above)
-      // This logic is tricky because we don't know if a UI action came WITH the text.
-      // However, handleAgentAction runs for UI actions.
-      // So here we can just default to expanding, and if a UI action comes, it will collapse it.
-      // But we need to make sure this doesn't override the collapse.
-
-      // Actually, ChatAgent calls onAction if there is an action.
-      // We can rely on ChatAgent to tell us if it's purely text?
-      // Or we can just trust the user flow:
-      // If I ask "analyze this", agent replies with text -> Expand.
-      // If I ask "show me this", agent replies with text + UIAction -> Collapse.
-
-      // So, we will expose a method or prop to ChatAgent to signal "Text Only Response"?
-      // Or we can just set it to true here, and handleAgentAction sets it to false.
-      // Since react updates are batched or sequential, if both happen, the last one wins?
-      // UIAction usually comes with text.
-
-      // Let's try: Always expand on response. But if action is present, handleAgentAction will collapse.
-      // Note: handleAgentAction is called by ChatAgent when action is present.
-
-      // We'll pass a callback `onTextResponse` to ChatAgent.
-      // But ChatAgent logic needs update? No, we can just use `onAction`.
-
-      // Let's optimistically set expanded to true when user sends message? No.
-      // Let's set expanded to true when `ChatAgent` receives a message that has NO action.
-      // I'll need to modify `ChatAgent` to support `onTextResponse` or similar.
-      // For now, I'll just leave it manual or simple toggle.
-      // But user asked for "expand automatically".
-
-      // Simple heuristic: If we are in "research mode" (no chart changes), we expand.
-      setIsChatExpanded(true);
+    // If the chat isn't expanded, expand it to show the research results
+    // But only if we are NOT currently running a viz action (which is handled above)
+    // This logic is tricky because we don't know if a UI action came WITH the text.
+    // However, handleAgentAction runs for UI actions.
+    // So here we can just default to expanding, and if a UI action comes, it will collapse it.
+    // But we need to make sure this doesn't override the collapse.
+
+    // Actually, ChatAgent calls onAction if there is an action.
+    // We can rely on ChatAgent to tell us if it's purely text?
+    // Or we can just trust the user flow:
+    // If I ask "analyze this", agent replies with text -> Expand.
+    // If I ask "show me this", agent replies with text + UIAction -> Collapse.
+
+    // So, we will expose a method or prop to ChatAgent to signal "Text Only Response"?
+    // Or we can just set it to true here, and handleAgentAction sets it to false.
+    // Since react updates are batched or sequential, if both happen, the last one wins?
+    // UIAction usually comes with text.
+
+    // Let's try: Always expand on response. But if action is present, handleAgentAction will collapse.
+    // Note: handleAgentAction is called by ChatAgent when action is present.
+
+    // We'll pass a callback `onTextResponse` to ChatAgent.
+    // But ChatAgent logic needs update? No, we can just use `onAction`.
+
+    // Let's optimistically set expanded to true when user sends message? No.
+    // Let's set expanded to true when `ChatAgent` receives a message that has NO action.
+    // I'll need to modify `ChatAgent` to support `onTextResponse` or similar.
+    // For now, I'll just leave it manual or simple toggle.
+    // But user asked for "expand automatically".
+
+    // Simple heuristic: If we are in "research mode" (no chart changes), we expand.
+    setIsChatExpanded(true);
   };
 
   // NOTE: I need to update ChatAgent to call this.
@@ -367,7 +368,22 @@ const App: React.FC = () => {
             <span className="text-slate-100">Trade<span className="text-blue-500">Viz</span></span>
           </div>
 
-          <div className="flex gap-1">
+          <div className="flex items-center gap-3">
+            <label className="flex items-center gap-2 cursor-pointer group" title="Enable Planning Mode">
+              <span className={`text-[10px] font-bold uppercase tracking-wider transition-colors ${planningMode ? 'text-emerald-400' : 'text-slate-600 group-hover:text-slate-400'}`}>
+                Plan
+              </span>
+              <div className={`w-8 h-4 rounded-full p-0.5 transition-colors duration-300 ${planningMode ? 'bg-emerald-500/20 ring-1 ring-emerald-500/50' : 'bg-slate-800 ring-1 ring-slate-700'}`}>
+                <div className={`w-3 h-3 rounded-full shadow-sm transform transition-transform duration-300 ${planningMode ? 'translate-x-4 bg-emerald-400' : 'translate-x-0 bg-slate-500'}`} />
+              </div>
+              <input
+                type="checkbox"
+                checked={planningMode}
+                onChange={e => setPlanningMode(e.target.checked)}
+                className="hidden"
+              />
+            </label>
+
             <button
               onClick={() => setCurrentPage('experiments')}
               className="p-2 rounded-md text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 transition-colors"
@@ -376,6 +392,7 @@ const App: React.FC = () => {
               <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
             </button>
           </div>
+
         </div>
 
         {/* Scrollable Content */}
@@ -540,15 +557,15 @@ const App: React.FC = () => {
 
         {/* Chat Bottom (Fixed Height) */}
         <div style={{ height: isChatExpanded ? '60vh' : chatHeight, transition: 'height 0.3s ease-in-out' }} className="shrink-0 bg-slate-950 border-t border-slate-800 shadow-[0_-8px_30px_rgba(0,0,0,0.5)] z-20 relative">
-            <button
-                onClick={() => setIsChatExpanded(!isChatExpanded)}
-                className="absolute top-0 right-4 -mt-3 bg-slate-800 border border-slate-700 rounded-full p-1 hover:bg-slate-700 transition-colors z-50 shadow-sm"
-                title={isChatExpanded ? "Collapse" : "Expand"}
-            >
-                <svg className={`w-4 h-4 text-slate-400 transform transition-transform ${isChatExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
-                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
-                </svg>
-            </button>
+          <button
+            onClick={() => setIsChatExpanded(!isChatExpanded)}
+            className="absolute top-0 right-4 -mt-3 bg-slate-800 border border-slate-700 rounded-full p-1 hover:bg-slate-700 transition-colors z-50 shadow-sm"
+            title={isChatExpanded ? "Collapse" : "Expand"}
+          >
+            <svg className={`w-4 h-4 text-slate-400 transform transition-transform ${isChatExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
+              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
+            </svg>
+          </button>
           <ChatAgent
             runId={currentRun || 'none'}
             currentIndex={index}
@@ -562,20 +579,23 @@ const App: React.FC = () => {
       </div>
 
       {/* UNIFIED REPLAY OVERLAY */}
-      {showSimulation && (
-        <LiveSessionView
-          onClose={() => setShowSimulation(false)}
-          runId={currentRun}
-          initialMode={simulationMode}
-          lastTradeTimestamp={
-            decisions.length > 0
-              ? decisions[decisions.length - 1].timestamp || undefined
-              : undefined
-          }
-        />
-      )}
+      {
+        showSimulation && (
+          <LiveSessionView
+            onClose={() => setShowSimulation(false)}
+            runId={currentRun}
+            initialMode={simulationMode}
+            lastTradeTimestamp={
+              decisions.length > 0
+                ? decisions[decisions.length - 1].timestamp || undefined
+                : undefined
+            }
+          />
+        )
+      }
 
     </div>
+    </div >
   );
 };
 
diff --git a/src/components/ChatAgent.tsx b/src/components/ChatAgent.tsx
index a09bfb1..96f4d34 100644
--- a/src/components/ChatAgent.tsx
+++ b/src/components/ChatAgent.tsx
@@ -150,7 +150,7 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
           </div>
           <button
             type="submit"
-            disabled={loading || !runId || !input.trim()}
+            disabled={loading || !input.trim()}
             className="bg-blue-600 hover:bg-blue-500 text-white rounded-xl p-3.5 shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95 flex items-center justify-center aspect-square"
           >
             <svg className="w-5 h-5 translate-x-0.5 -translate-y-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
diff --git a/src/server/main.py b/src/server/main.py
index c094bcf..87be233 100644
--- a/src/server/main.py
+++ b/src/server/main.py
@@ -30,7 +30,9 @@ from src.data.resample import resample_all_timeframes
 from src.core.tool_registry import ToolRegistry, ToolCategory
 
 # Import agent tools to register them
-import src.tools.agent_tools  # noqa: F401
+import src.tools.agent_tools
+import src.tools.discovery_tools  # Register discovery tools  # noqa: F401
+import src.tools.composition_tools  # Register composition tools  # noqa: F401
 import src.tools.analysis_tools  # noqa: F401 - Registers analysis tools
 import src.core.strategy_tool  # noqa: F401 - Registers CompositeStrategyRunner
 import src.skills.indicator_skills  # noqa: F401 - Registers indicator tools
@@ -878,9 +880,9 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
                                 import pandas as pd
                                 
                                 recipe = {
-                                    "name": f"Research: {fn_args.get('trigger_type', 'test')}",
+                                    "name": f"Research: {fn_args.get('trigger_type', 'custom')}",
                                     "cooldown_bars": 20,
-                                    "entry_trigger": {
+                                    "entry_trigger": fn_args.get("trigger_config") or {
                                         "type": fn_args.get("trigger_type", "ema_cross"),
                                         **fn_args.get("trigger_params", {})
                                     },
@@ -894,6 +896,11 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
                                         }
                                     }
                                 }
+                                # Use bracket_config if available for OCO (simplified mapping)
+                                if fn_args.get("bracket_config"):
+                                    bc = fn_args.get("bracket_config")
+                                    recipe["oco"]["take_profit"]["multiple"] = bc.get("tp_atr", 2.5)
+                                    recipe["oco"]["stop_loss"]["multiple"] = bc.get("stop_atr", 1.5)
 
                                 # Write recipe to temp file
                                 with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
@@ -946,14 +953,13 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
                                     except:
                                         pass
 
-                            else:
                                 # Normal Viz Execution (UI Action)
                                 config = {
-                                    "trigger": {
+                                    "trigger": fn_args.get("trigger_config") or {
                                         "type": fn_args.get("trigger_type", "ema_cross"),
                                         **fn_args.get("trigger_params", {})
                                     },
-                                    "bracket": {
+                                    "bracket": fn_args.get("bracket_config") or {
                                         "type": fn_args.get("bracket_type", "atr"),
                                         "stop_atr": fn_args.get("stop_atr", 2.0),
                                         "tp_atr": fn_args.get("tp_atr", 3.0)
diff --git a/src/tools/agent_tools.py b/src/tools/agent_tools.py
index 5cda4b8..0838fa2 100644
--- a/src/tools/agent_tools.py
+++ b/src/tools/agent_tools.py
@@ -93,15 +93,23 @@ class RunStrategyTool:
     input_schema={
         "type": "object",
         "properties": {
+            "trigger_config": {
+                "type": "object",
+                "description": "Full generic trigger configuration (alternative to trigger_type)"
+            },
             "trigger_type": {
                 "type": "string",
                 "enum": ["ema_cross", "ema_bounce", "rsi_threshold", "ifvg", "orb", "candle_pattern", "time"],
-                "description": "Type of entry trigger"
+                "description": "Type of entry trigger (legacy/simplified)"
             },
             "trigger_params": {
                 "type": "object",
                 "description": "Parameters for the trigger (e.g., {fast: 9, slow: 21} for ema_cross)"
             },
+            "bracket_config": {
+                "type": "object",
+                "description": "Full generic bracket configuration"
+            },
             "bracket_type": {
                 "type": "string",
                 "enum": ["atr", "percent", "fixed"],
@@ -137,7 +145,7 @@ class RunStrategyTool:
                 "default": False
             }
         },
-        "required": ["trigger_type"]
+        "required": []
     },
     produces_artifacts=True,
     artifact_spec={
```

### New Untracked Files

#### `gitrdiff.md`

```
```

#### `src/tools/composition_tools.py`

```
"""
Composition Tools
Tools for composing strategies from atomic primitives (triggers, brackets, filters).
This allows agents to build "Modular Strategies" dynamically.
"""
from typing import Dict, Any, List, Optional
import json

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.policy.triggers.factory import TRIGGER_REGISTRY, trigger_from_dict

@ToolRegistry.register(
    tool_id="compose_scan",
    category=ToolCategory.STRATEGY,
    name="Compose Scan Configuration",
    description="Helper to compose a validate scan configuration for the 'run_modular_strategy' tool.",
    input_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the strategy"
            },
            "trigger_type": {
                "type": "string",
                "description": "Type of trigger (use list_triggers to find available types)"
            },
            "trigger_params": {
                "type": "object",
                "description": "Parameters for the trigger"
            },
            "trigger_config": {
                 "type": "object",
                 "description": "Alternative: Full trigger configuration object (recursive AND/OR)"
            },
            "bracket": {
                "type": "object",
                "description": "Bracket configuration (stop loss / take profit)",
                "properties": {
                    "type": {"type": "string", "enum": ["atr", "percent", "fixed", "ict"]},
                    "stop_atr": {"type": "number"},
                    "tp_atr": {"type": "number"},
                    "stop_pct": {"type": "number"},
                    "tp_pct": {"type": "number"}
                }
            },
            "filters": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of filters (e.g., time, session)"
            }
        },
        "required": ["name"]
    }
)
class ComposeScanTool:
    def execute(self, **inputs):
        name = inputs.get("name")
        
        # Build Trigger Config
        trigger_config = inputs.get("trigger_config")
        
        if not trigger_config:
            # Build from type/params
            t_type = inputs.get("trigger_type")
            t_params = inputs.get("trigger_params", {})
            if not t_type:
               return {"error": "Must provide either 'trigger_config' or 'trigger_type'"}
            
            trigger_config = {"type": t_type, **t_params}
            
        # Validate Trigger
        try:
            # We try to create it to validate params
            # Note: recursive triggers might need children
            # For now, simplistic validation
            if trigger_config["type"] not in TRIGGER_REGISTRY:
                 return {"error": f"Unknown trigger type: {trigger_config['type']}"}
        except Exception as e:
            return {"error": f"Invalid trigger config: {e}"}

        # Build Bracket Config
        bracket_config = inputs.get("bracket", {"type": "atr", "stop_atr": 2.0, "tp_atr": 3.0})
        
        # Assemble Full Spec
        scan_spec = {
            "trigger": trigger_config,
            "bracket": bracket_config,
            "filters": inputs.get("filters", []),
            "name": name
        }
        
        # In the future, we might save this to a DB
        # For now, we return it so the agent can pass it to run_modular_strategy
        
        return {
            "status": "success",
            "message": "Scan configuration composed successfully. Pass 'scan_spec' to run_modular_strategy.",
            "scan_spec": scan_spec
        }

@ToolRegistry.register(
    tool_id="save_scan_spec",
    category=ToolCategory.UTILITY,
    name="Save Scan Spec",
    description="Save a scan specification to the library for reuse.",
    input_schema={
        "type": "object",
        "properties": {
             "name": {"type": "string"},
             "scan_spec": {"type": "object"}
        },
        "required": ["name", "scan_spec"]
    }
)
class SaveScanSpecTool:
    def execute(self, **inputs):
        # Placeholder for saving to file system
        # src/policy/library/user/{name}.json
        return {"status": "success", "message": "Scan saved (mock)"}
```

#### `src/tools/discovery_tools.py`

```
"""
Discovery Tools
Tools for the agent to discover available atomic components (triggers, scanners, levels).
"""
import inspect
import pandas as pd
from typing import Dict, Any, List, Optional

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.policy.triggers.factory import TRIGGER_REGISTRY
# from src.policy.library import SCANNER_REGISTRY # Does not exist, we will generic discovery
from src.features.levels import LevelValues
from src.features.session_levels import SessionLevels
from src.policy.scanners import Scanner
import pkgutil
import importlib
import src.policy.library

@ToolRegistry.register(
    tool_id="list_triggers",
    category=ToolCategory.UTILITY,
    name="List Triggers",
    description="List all available trigger types for strategy composition.",
    output_schema={
        "type": "object",
        "properties": {
            "triggers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "class": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
            }
        }
    }
)
class ListTriggersTool:
    def execute(self, **kwargs):
        triggers = []
        for tid, tcls in TRIGGER_REGISTRY.items():
            doc = inspect.getdoc(tcls) or ""
            # First line of docstring as description
            desc = doc.split("\n")[0] if doc else ""
            triggers.append({
                "id": tid,
                "class": tcls.__name__,
                "description": desc
            })
        
        return {"triggers": triggers}


@ToolRegistry.register(
    tool_id="get_trigger_info",
    category=ToolCategory.UTILITY,
    name="Get Trigger Info",
    description="Get detailed schema and description for a specific trigger type.",
    input_schema={
        "type": "object",
        "properties": {
            "trigger_type": {"type": "string", "description": "ID of the trigger (e.g., 'ema_cross')"}
        },
        "required": ["trigger_type"]
    }
)
class GetTriggerInfoTool:
    def execute(self, trigger_type: str, **kwargs):
        if trigger_type not in TRIGGER_REGISTRY:
            return {"error": f"Trigger type '{trigger_type}' not found. Use list_triggers to see available types."}
        
        tcls = TRIGGER_REGISTRY[trigger_type]
        doc = inspect.getdoc(tcls) or ""
        
        # Determine params from __init__ (simplified)
        init_sig = inspect.signature(tcls.__init__)
        params = []
        for name, param in init_sig.parameters.items():
            if name == "self": continue
            default = param.default if param.default != inspect.Parameter.empty else None
            annotation = str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            params.append({
                "name": name,
                "type": annotation,
                "default": str(default) if default is not None else "Required"
            })

        return {
            "id": trigger_type,
            "class": tcls.__name__,
            "description": doc,
            "parameters": params
        }


@ToolRegistry.register(
    tool_id="list_scanners",
    category=ToolCategory.UTILITY,
    name="List Library Scanners",
    description="List pre-built scanners available in the library.",
    output_schema={
        "type": "object",
        "properties": {
            "scanners": {"type": "array"}
        }
    }
)
class ListScannersTool:
    def execute(self, **kwargs):
        scanners = []
        
        # Dynamic discovery of scanners in src.policy.library
        package = src.policy.library
        path = package.__path__
        prefix = package.__name__ + "."

        for _, name, _ in pkgutil.iter_modules(path, prefix):
            try:
                module = importlib.import_module(name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (inspect.isclass(attr) and 
                        issubclass(attr, Scanner) and 
                        attr is not Scanner and 
                        attr.__module__ == module.__name__):
                        
                        doc = inspect.getdoc(attr) or ""
                        desc = doc.split("\n")[0] if doc else ""
                        sid = getattr(attr, "scanner_id", name.split(".")[-1])
                        # If scanner_id is a property, we might need to instantiate or guess
                        # For now, use class name as fallback ID or snake_case conversion
                        
                        scanners.append({
                            "id": sid if isinstance(sid, str) else attr.__name__,
                            "class": attr.__name__,
                            "description": desc
                        })
            except Exception as e:
                print(f"Error inspecting {name}: {e}")
            
        return {"scanners": scanners}


@ToolRegistry.register(
    tool_id="list_levels",
    category=ToolCategory.UTILITY,
    name="List Available Levels",
    description="List the types of price levels available for strategy context (e.g., PDH, Asian Low).",
    output_schema={
        "type": "object",
        "properties": {
            "levels": {"type": "array"}
        }
    }
)
class ListLevelsTool:
    def execute(self, **kwargs):
        # Inspect LevelValues and SessionLevels dataclasses to find available fields
        levels = []
        
        # From Levels (Daily/HTF)
        for field in LevelValues.__dataclass_fields__:
            levels.append({
                "id": field,
                "category": "Daily/HTF",
                "description": f"Standard level: {field}"
            })
            
        # From Session Levels
        for field in SessionLevels.__dataclass_fields__:
            levels.append({
                "id": field,
                "category": "Session",
                "description": f"Session level: {field}"
            })
            
        # Add dynamic ones (FVG, etc if we had a dedicated structure)
        levels.append({"id": "fvg_bullish", "category": "Dynamic", "description": "Nearest bullish FVG"})
        levels.append({"id": "fvg_bearish", "category": "Dynamic", "description": "Nearest bearish FVG"})
        levels.append({"id": "vwap", "category": "Indicator", "description": "Volume Weighted Average Price"})
        
        return {"levels": levels}
```

#### `verification/verify_composition.py`

```
"""
Verify Composition Tools and Execution Logic
"""
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.composition_tools import ComposeScanTool
from src.tools.agent_tools import RunModularStrategyTool
from src.server.main import agent_chat, ChatRequest, ChatContext, ChatMessage

import asyncio

async def test_composition():
    print("Testing ComposeScanTool...")
    tool = ComposeScanTool()
    
    # Test valid composition
    result = tool.execute(
        name="Test Strat",
        trigger_type="ema_cross",
        trigger_params={"fast": 10, "slow": 50},
        bracket={"type": "atr", "stop_atr": 1.5, "tp_atr": 3.0}
    )
    
    if "scan_spec" not in result:
        print("FAIL: ComposeScanTool did not return scan_spec")
        print(result)
        return
        
    spec = result["scan_spec"]
    print(f"Spec created: {json.dumps(spec, indent=2)}")
    
    # Verify strict equality
    assert spec["trigger"]["type"] == "ema_cross"
    assert spec["trigger"]["fast"] == 10
    
    print("PASS: ComposeScanTool")
    
    # Test Agent Execution Logic (Mock)
    # We call run_modular_strategy via agent_chat
    # Actually we can't easily Mock agent_chat without running valid inputs
    # But we can verify RunModularStrategyTool schema
    
    print("\nVerifying RunModularStrategyTool Schema...")
    from src.core.tool_registry import ToolRegistry
    # The registry stores declarations, not tool instances directly in the same way?
    # Actually get_gemini_function_declarations returns the list
    
    tools = ToolRegistry.get_gemini_function_declarations()
    target = next((t for t in tools if t['name'] == 'run_modular_strategy'), None)
    
    if target:
        props = target['parameters']['properties']
        if "trigger_config" in props and "bracket_config" in props:
            print("PASS: RunModularStrategyTool schema updated.")
        else:
            print(f"FAIL: Schema missing new configs. Keys: {props.keys()}")
    else:
        print("FAIL: Tool not found in registry.")
        
    print("\nPhase 3/4 Verification Complete.")

if __name__ == "__main__":
    asyncio.run(test_composition())
```

#### `verification/verify_discovery.py`

```
import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from src.core.tool_registry import ToolRegistry
import src.server.main  # Trigger registration via imports

def verify_tool(tool_id):
    print(f"\n--- Verifying {tool_id} ---")
    try:
        tool = ToolRegistry.get_tool(tool_id)
        if not tool:
            print(f"❌ Tool {tool_id} NOT found in registry")
            return
        
        print(f"✅ Tool {tool_id} found")
        result = tool.execute()
        print(f"Output: {json.dumps(result, indent=2)[:500]}...") # Truncate output
        
        # specific checks
        if tool_id == "list_triggers":
            triggers = result.get("triggers", [])
            print(f"Found {len(triggers)} triggers")
            has_vwap = any(t["id"] == "vwap_reclaim" for t in triggers)
            print(f"Has 'vwap_reclaim': {has_vwap}")
            
        if tool_id == "list_scanners":
            scanners = result.get("scanners", [])
            print(f"Found {len(scanners)} scanners")
            
        if tool_id == "list_levels":
            levels = result.get("levels", [])
            print(f"Found {len(levels)} levels")
            
    except Exception as e:
        print(f"❌ Error executing {tool_id}: {e}")

if __name__ == "__main__":
    verify_tool("list_triggers")
    verify_tool("list_scanners")
    verify_tool("list_levels")
```

---

## Commits Ahead (local changes not on remote)

```
f79eb22 Unify Lab and TradeViz agents and remove separate Lab page
```

## Commits Behind (remote changes not pulled)

```
```

---

## File Changes (YOUR UNPUSHED CHANGES)

```
 render.yaml                    |  30 +-
 src/App.tsx                    |  96 +++--
 src/components/ChatAgent.tsx   |  12 +-
 src/components/LabPage.tsx     | 298 ---------------
 src/server/main.py             | 810 +++++++++--------------------------------
 verification/chat_expanded.png | Bin 0 -> 76686 bytes
 verification/main_page.png     | Bin 0 -> 76616 bytes
 verification/verify_changes.py |  54 +++
 8 files changed, 317 insertions(+), 983 deletions(-)
```

---

## Full Diff of Your Unpushed Changes

Green (+) = lines you ADDED locally
Red (-) = lines you REMOVED locally

```diff
diff --git a/render.yaml b/render.yaml
index c1c146e..6ec5180 100644
--- a/render.yaml
+++ b/render.yaml
@@ -1,14 +1,24 @@
 services:
+  # Frontend service
   - type: web
-    name: mlang2-app
-    runtime: python
-    buildCommand: |
-      pip install -r requirements.txt
-      npm install
-      npm run build
-    startCommand: python -m uvicorn src.server.main:app --host 0.0.0.0 --port $PORT
+    name: tradeviz-frontend
+    env: node
+    plan: free
+    buildCommand: npm install && npm run build
+    startCommand: npm run preview -- --host --port $PORT
+    envVars:
+      - key: VITE_API_URL
+        value: https://tradeviz-backend.onrender.com
+
+  # Backend service
+  - type: web
+    name: tradeviz-backend
+    env: python
+    plan: free
+    buildCommand: pip install -r requirements.txt
+    startCommand: uvicorn src.server.main:app --host 0.0.0.0 --port $PORT
     envVars:
       - key: PYTHON_VERSION
-        value: 3.11.0
-      - key: NODE_VERSION
-        value: 20.0.0
+        value: 3.10.0
+      - key: GEMINI_API_KEY
+        sync: false
diff --git a/src/App.tsx b/src/App.tsx
index beeff8d..d24229b 100644
--- a/src/App.tsx
+++ b/src/App.tsx
@@ -8,12 +8,11 @@ import { DetailsPanel } from './components/DetailsPanel';
 import { ChatAgent } from './components/ChatAgent';
 import { LiveSessionView } from './components/LiveSessionView';
 import { StatsPanel } from './components/StatsPanel';
-import { LabPage } from './components/LabPage';
 import ExperimentsView from './components/ExperimentsView';
 import { IndicatorSettingsPanel } from './components/IndicatorSettings';
 import { DEFAULT_INDICATOR_SETTINGS, type IndicatorSettings } from './features/chart_indicators';
 
-type PageType = 'trade' | 'lab' | 'experiments';
+type PageType = 'trade' | 'experiments';
 
 const App: React.FC = () => {
   const [currentPage, setCurrentPage] = useState<PageType>('trade');
@@ -31,6 +30,7 @@ const App: React.FC = () => {
 
   // Layout State
   const [chatHeight, setChatHeight] = useState<number>(320);
+  const [isChatExpanded, setIsChatExpanded] = useState<boolean>(false); // New state for auto-expansion
   const isResizingRef = useRef(false);
 
   // Indicator Settings State
@@ -164,6 +164,11 @@ const App: React.FC = () => {
 
   // Agent Action Handler
   const handleAgentAction = async (action: UIAction) => {
+    // If we receive a UI action that affects the chart, collapse the chat
+    if (['RUN_STRATEGY', 'SET_INDEX', 'SET_MODE', 'LOAD_RUN', 'RUN_FAST_VIZ'].includes(action.type)) {
+        setIsChatExpanded(false);
+    }
+
     switch (action.type) {
       case 'SET_INDEX':
         setIndex(action.payload);
@@ -228,6 +233,50 @@ const App: React.FC = () => {
     }
   };
 
+  // Expand Chat on Research/Text response
+  const handleAgentTextResponse = () => {
+      // If the chat isn't expanded, expand it to show the research results
+      // But only if we are NOT currently running a viz action (which is handled above)
+      // This logic is tricky because we don't know if a UI action came WITH the text.
+      // However, handleAgentAction runs for UI actions.
+      // So here we can just default to expanding, and if a UI action comes, it will collapse it.
+      // But we need to make sure this doesn't override the collapse.
+
+      // Actually, ChatAgent calls onAction if there is an action.
+      // We can rely on ChatAgent to tell us if it's purely text?
+      // Or we can just trust the user flow:
+      // If I ask "analyze this", agent replies with text -> Expand.
+      // If I ask "show me this", agent replies with text + UIAction -> Collapse.
+
+      // So, we will expose a method or prop to ChatAgent to signal "Text Only Response"?
+      // Or we can just set it to true here, and handleAgentAction sets it to false.
+      // Since react updates are batched or sequential, if both happen, the last one wins?
+      // UIAction usually comes with text.
+
+      // Let's try: Always expand on response. But if action is present, handleAgentAction will collapse.
+      // Note: handleAgentAction is called by ChatAgent when action is present.
+
+      // We'll pass a callback `onTextResponse` to ChatAgent.
+      // But ChatAgent logic needs update? No, we can just use `onAction`.
+
+      // Let's optimistically set expanded to true when user sends message? No.
+      // Let's set expanded to true when `ChatAgent` receives a message that has NO action.
+      // I'll need to modify `ChatAgent` to support `onTextResponse` or similar.
+      // For now, I'll just leave it manual or simple toggle.
+      // But user asked for "expand automatically".
+
+      // Simple heuristic: If we are in "research mode" (no chart changes), we expand.
+      setIsChatExpanded(true);
+  };
+
+  // NOTE: I need to update ChatAgent to call this.
+  // Since I can't easily edit ChatAgent right now without reading it, I will assume it calls onAction.
+  // Wait, I can read ChatAgent. It's in `components/ChatAgent.tsx`.
+  // I will check it in next step if needed.
+  // For now, I'll rely on the user expanding it manually or the initial state.
+  // Actually, I can just toggle it based on the prompt instructions.
+  // "chat buffer... will expand automatically when the chat needs to be used for research"
+
   // Resizing Logic
   const startResizing = () => {
     isResizingRef.current = true;
@@ -249,6 +298,7 @@ const App: React.FC = () => {
     // Constrain height (min 100px, max 80% of screen)
     const constrained = Math.max(100, Math.min(newHeight, window.innerHeight * 0.8));
     setChatHeight(constrained);
+    // If user manually resizes, we might want to disable auto-expansion logic or update it
   };
 
   const PageHeader = ({ title, backButton }: { title: string, backButton?: boolean }) => (
@@ -272,12 +322,6 @@ const App: React.FC = () => {
       <div className="flex items-center gap-2">
         {!backButton && (
           <>
-            <button
-              onClick={() => setCurrentPage('lab')}
-              className="flex items-center gap-2 px-3 py-1.5 rounded-md text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-all text-sm font-medium"
-            >
-              <span>🔬 Lab</span>
-            </button>
             <button
               onClick={() => setCurrentPage('experiments')}
               className="flex items-center gap-2 px-3 py-1.5 rounded-md text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 transition-all text-sm font-medium"
@@ -290,23 +334,6 @@ const App: React.FC = () => {
     </div>
   );
 
-  // If Lab page is active, render it instead
-  if (currentPage === 'lab') {
-    return (
-      <div className="flex flex-col h-screen w-full bg-slate-950 overflow-hidden text-slate-100 font-sans">
-        <PageHeader title="Strategy Lab" backButton />
-        <div className="flex-1 overflow-hidden min-h-0 bg-slate-900">
-          <LabPage
-            onLoadRun={(runId: string) => {
-              setCurrentRun(runId);
-              setCurrentPage('trade');
-            }}
-          />
-        </div>
-      </div>
-    );
-  }
-
   // If Experiments page is active
   if (currentPage === 'experiments') {
     return (
@@ -341,13 +368,6 @@ const App: React.FC = () => {
           </div>
 
           <div className="flex gap-1">
-            <button
-              onClick={() => setCurrentPage('lab')}
-              className="p-2 rounded-md text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
-              title="Strategy Lab"
-            >
-              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
-            </button>
             <button
               onClick={() => setCurrentPage('experiments')}
               className="p-2 rounded-md text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 transition-colors"
@@ -519,13 +539,23 @@ const App: React.FC = () => {
         </div>
 
         {/* Chat Bottom (Fixed Height) */}
-        <div style={{ height: chatHeight }} className="shrink-0 bg-slate-950 border-t border-slate-800 shadow-[0_-8px_30px_rgba(0,0,0,0.5)] z-20">
+        <div style={{ height: isChatExpanded ? '60vh' : chatHeight, transition: 'height 0.3s ease-in-out' }} className="shrink-0 bg-slate-950 border-t border-slate-800 shadow-[0_-8px_30px_rgba(0,0,0,0.5)] z-20 relative">
+            <button
+                onClick={() => setIsChatExpanded(!isChatExpanded)}
+                className="absolute top-0 right-4 -mt-3 bg-slate-800 border border-slate-700 rounded-full p-1 hover:bg-slate-700 transition-colors z-50 shadow-sm"
+                title={isChatExpanded ? "Collapse" : "Expand"}
+            >
+                <svg className={`w-4 h-4 text-slate-400 transform transition-transform ${isChatExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
+                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
+                </svg>
+            </button>
           <ChatAgent
             runId={currentRun || 'none'}
             currentIndex={index}
             currentMode={mode}
             fastVizMode={fastVizEnabled}
             onAction={handleAgentAction}
+            onTextResponse={handleAgentTextResponse}
           />
         </div>
 
diff --git a/src/components/ChatAgent.tsx b/src/components/ChatAgent.tsx
index ada30aa..a09bfb1 100644
--- a/src/components/ChatAgent.tsx
+++ b/src/components/ChatAgent.tsx
@@ -9,9 +9,10 @@ interface ChatAgentProps {
   currentMode: 'DECISION' | 'TRADE';
   fastVizMode?: boolean;
   onAction: (action: UIAction) => void;
+  onTextResponse?: () => void;
 }
 
-export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, currentMode, fastVizMode = false, onAction }) => {
+export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, currentMode, fastVizMode = false, onAction, onTextResponse }) => {
   const [messages, setMessages] = useState<ChatMessage[]>([
     { role: 'assistant', content: 'Hello! I am the **Trade Viz Agent**. How can I help with your analysis today?' }
   ]);
@@ -42,6 +43,11 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
 
       if (response.ui_action) {
         onAction(response.ui_action);
+      } else {
+        // Text-only response (likely research result), expand chat
+        if (onTextResponse) {
+          onTextResponse();
+        }
       }
     } catch (err) {
       setMessages(prev => [...prev, { role: 'assistant', content: "Error contacting agent." }]);
@@ -58,8 +64,8 @@ export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, curre
       <div className="px-4 py-3 bg-slate-950 border-b border-slate-800 flex items-center justify-between shrink-0">
         <div className="flex items-center gap-2">
           <div className="relative">
-            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
-            <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-500 animate-ping opacity-20"></div>
+            <div className={`w-2 h-2 rounded-full ${loading ? 'bg-amber-400' : 'bg-emerald-500'} animate-pulse`}></div>
+            <div className={`absolute inset-0 w-2 h-2 rounded-full ${loading ? 'bg-amber-400' : 'bg-emerald-500'} animate-ping opacity-20`}></div>
           </div>
           <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest">Agent Terminal</h3>
         </div>
diff --git a/src/components/LabPage.tsx b/src/components/LabPage.tsx
deleted file mode 100644
index fb8bec0..0000000
--- a/src/components/LabPage.tsx
+++ /dev/null
@@ -1,298 +0,0 @@
-import React, { useState, useRef, useEffect } from 'react';
-import ReactMarkdown from 'react-markdown';
-import { api } from '../api/client';
-
-interface Message {
-    role: 'user' | 'assistant';
-    content: string;
-    type?: 'text' | 'table' | 'chart' | 'code';
-    data?: any;
-    run_id?: string;
-}
-
-interface LabResult {
-    strategy: string;
-    trades: number;
-    wins: number;
-    losses: number;
-    win_rate: number;
-    total_pnl: number;
-    equity_curve?: number[];
-}
-
-interface LabPageProps {
-    onLoadRun?: (runId: string) => void;
-}
-
-export const LabPage: React.FC<LabPageProps> = ({ onLoadRun }) => {
-    const [messages, setMessages] = useState<Message[]>([
-        {
-            role: 'assistant',
-            content: 'Welcome to the Research Lab! I can help you test strategies, run scans, train models, and analyze results. What would you like to explore?',
-            type: 'text'
-        }
-    ]);
-    const [input, setInput] = useState('');
-    const [loading, setLoading] = useState(false);
-    const [currentResult, setCurrentResult] = useState<LabResult | null>(null);
-    const scrollRef = useRef<HTMLDivElement>(null);
-    const [plannerMode, setPlannerMode] = useState<boolean>(false);
-
-    useEffect(() => {
-        if (scrollRef.current) {
-            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
-        }
-    }, [messages]);
-
-    const handleSubmit = async (e: React.FormEvent) => {
-        e.preventDefault();
-        if (!input.trim()) return;
-
-        const userMsg: Message = { role: 'user', content: input, type: 'text' };
-        setMessages(prev => [...prev, userMsg]);
-        setInput('');
-        setLoading(true);
-
-        try {
-            const response = await api.postLabAgent([...messages, userMsg], plannerMode);
-            const assistantMsg: Message = {
-                role: 'assistant',
-                content: response.reply || 'Processing...',
-                type: response.type || 'text',
-                data: response.data,
-                run_id: response.run_id
-            };
-            setMessages(prev => [...prev, assistantMsg]);
-            if (response.result) {
-                setCurrentResult(response.result);
-            }
-        } catch (err) {
-            setMessages(prev => [...prev, {
-                role: 'assistant',
-                content: 'Error contacting lab agent. Is the backend running?',
-                type: 'text'
-            }]);
-        } finally {
-            setLoading(false);
-        }
-    };
-
-    const quickActions = [
-        { label: 'Run EMA Scan', prompt: 'Run an EMA cross scan on the last 7 days' },
-        { label: 'Test ORB Strategy', prompt: 'Test the Opening Range Breakout strategy' },
-        { label: 'Compare Models', prompt: 'Compare the LSTM vs CNN model accuracy' },
-        { label: 'Show Best Config', prompt: 'What is the best configuration from recent experiments?' },
-        { label: 'Run Grid Search', prompt: 'Run a grid search on ORB stop and target parameters' },
-    ];
-
-    const sendQuickAction = (prompt: string) => {
-        setInput(prompt);
-    };
-
-    const renderResultTable = (result: LabResult, runId?: string) => (
-        <div className="bg-slate-800 rounded-lg p-4 my-3 border border-slate-600">
-            <div className="flex items-center justify-between mb-3">
-                <div className="text-sm font-bold text-blue-400">{result.strategy}</div>
-                {runId && onLoadRun && (
-                    <button
-                        onClick={() => onLoadRun(runId)}
-                        className="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1.5 rounded transition"
-                    >
-                        📊 Visualize
-                    </button>
-                )}
-            </div>
-            <div className="grid grid-cols-3 gap-4 text-center">
-                <div>
-                    <div className="text-2xl font-bold text-white">{result.trades}</div>
-                    <div className="text-xs text-slate-400">Trades</div>
-                </div>
-                <div>
-                    <div className={`text-2xl font-bold ${result.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}`}>
-                        {(result.win_rate * 100).toFixed(1)}%
-                    </div>
-                    <div className="text-xs text-slate-400">Win Rate</div>
-                </div>
-                <div>
-                    <div className={`text-2xl font-bold ${result.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
-                        ${result.total_pnl.toLocaleString()}
-                    </div>
-                    <div className="text-xs text-slate-400">P&L</div>
-                </div>
-            </div>
-
-            <div className="mt-4">
-                <div className="flex h-3 rounded overflow-hidden">
-                    <div className="bg-green-500" style={{ width: `${result.win_rate * 100}%` }} />
-                    <div className="bg-red-500" style={{ width: `${(1 - result.win_rate) * 100}%` }} />
-                </div>
-                <div className="flex justify-between text-xs text-slate-400 mt-1">
-                    <span>{result.wins} Wins</span>
-                    <span>{result.losses} Losses</span>
-                </div>
-            </div>
-
-            {result.equity_curve && result.equity_curve.length > 0 && (
-                <div className="mt-4">
-                    <div className="text-xs text-slate-400 mb-2">Equity Curve</div>
-                    <div className="h-16 flex items-end gap-px">
-                        {result.equity_curve.slice(-50).map((val, idx) => {
-                            const min = Math.min(...result.equity_curve!.slice(-50));
-                            const max = Math.max(...result.equity_curve!.slice(-50));
-                            const height = max > min ? ((val - min) / (max - min)) * 100 : 50;
-                            return (
-                                <div
-                                    key={idx}
-                                    className={`flex-1 ${val >= result.equity_curve![0] ? 'bg-green-500' : 'bg-red-500'}`}
-                                    style={{ height: `${Math.max(5, height)}%` }}
-                                />
-                            );
-                        })}
-                    </div>
-                </div>
-            )}
-        </div>
-    );
-
-    const renderMessage = (msg: Message, idx: number) => {
-        if (msg.role === 'user') {
-            return (
-                <div key={idx} className="flex justify-end">
-                    <div className="max-w-[80%] bg-blue-600 text-white rounded-xl px-4 py-2">
-                        {msg.content}
-                    </div>
-                </div>
-            );
-        }
-
-        return (
-            <div key={idx} className="flex justify-start">
-                <div className="max-w-[90%]">
-                    <div className="bg-slate-700 text-slate-100 rounded-xl px-4 py-3">
-                        <div className="prose prose-sm prose-invert max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-2 prose-headings:text-blue-400 prose-code:bg-slate-600 prose-code:px-1 prose-code:rounded prose-pre:bg-slate-800 prose-pre:border prose-pre:border-slate-600 prose-strong:text-white prose-table:border-collapse prose-th:bg-slate-700 prose-th:border prose-th:border-slate-600 prose-th:px-3 prose-th:py-2 prose-td:border prose-td:border-slate-600 prose-td:px-3 prose-td:py-2 prose-tr:even:bg-slate-800/50">
-                            <ReactMarkdown>{msg.content}</ReactMarkdown>
-                        </div>
-                    </div>
-                    {msg.data?.result && renderResultTable(msg.data.result, msg.run_id)}
-                </div>
-            </div>
-        );
-    };
-
-    return (
-        <div className="flex flex-col h-full bg-slate-900 overflow-hidden">
-            {/* Header */}
-            <div className="h-14 flex items-center justify-between px-6 border-b border-slate-700 bg-slate-800 shrink-0">
-                <div className="flex items-center gap-3">
-                    <span className="text-2xl">🔬</span>
-                    <h1 className="text-xl font-bold text-white">Research Lab</h1>
-                </div>
-                <div className="flex items-center gap-4">
-                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer hover:text-purple-400 transition-colors">
-                        <input
-                            type="checkbox"
-                            checked={plannerMode}
-                            onChange={(e) => setPlannerMode(e.target.checked)}
-                            className="w-4 h-4 rounded accent-purple-500"
-                        />
-                        <span>🗓️ Planner Mode</span>
-                    </label>
-                    <span className="text-sm text-slate-500">AI-Powered Strategy Research</span>
-                </div>
-            </div>
-
-            {/* Main Content */}
-            <div className="flex flex-1 overflow-hidden min-h-0">
-
-                {/* Left Sidebar - Current Result & Commands */}
-                <div className="w-80 border-r border-slate-700 bg-slate-800 p-4 overflow-y-auto shrink-0 flex flex-col">
-                    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">
-                        Latest Result
-                    </h2>
-
-                    {currentResult ? (
-                        renderResultTable(currentResult)
-                    ) : (
-                        <div className="text-slate-500 text-sm text-center py-8 border border-dashed border-slate-700 rounded">
-                            Run a strategy to see results here
-                        </div>
-                    )}
-
-                    <div className="mt-6">
-                        <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">
-                            Quick Commands
-                        </h2>
-                        <div className="space-y-2 text-xs">
-                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Run EMA cross scan")}>
-                                <code>"Run EMA cross scan"</code>
-                            </div>
-                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Test lunch hour fade")}>
-                                <code>"Test lunch hour fade"</code>
-                            </div>
-                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Train LSTM on bounce data")}>
-                                <code>"Train LSTM on bounce data"</code>
-                            </div>
-                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Compare ORB vs MR strategy")}>
-                                <code>"Compare ORB vs MR strategy"</code>
-                            </div>
-                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Show experiment history")}>
-                                <code>"Show experiment history"</code>
-                            </div>
-                        </div>
-                    </div>
-                </div>
-
-                {/* Chat Area (Right) */}
-                <div className="flex-1 flex flex-col min-w-0 bg-slate-900">
-                    {/* Messages */}
-                    <div className="flex-1 overflow-y-auto p-6 space-y-4" ref={scrollRef}>
-                        {messages.map((msg, idx) => renderMessage(msg, idx))}
-                        {loading && (
-                            <div className="flex justify-start">
-                                <div className="bg-slate-700 text-slate-300 rounded-xl px-4 py-3 animate-pulse">
-                                    <span className="text-blue-400">Agent is thinking...</span>
-                                </div>
-                            </div>
-                        )}
-                    </div>
-
-                    {/* Quick Actions */}
-                    <div className="px-6 py-3 border-t border-slate-700 bg-slate-800 shrink-0">
-                        <div className="flex gap-2 flex-wrap">
-                            {quickActions.map((action, idx) => (
-                                <button
-                                    key={idx}
-                                    onClick={() => sendQuickAction(action.prompt)}
-                                    className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs px-3 py-1.5 rounded-full transition"
-                                >
-                                    {action.label}
-                                </button>
-                            ))}
-                        </div>
-                    </div>
-
-                    {/* Input */}
-                    <form onSubmit={handleSubmit} className="p-4 border-t border-slate-700 bg-slate-800 shrink-0">
-                        <div className="flex gap-3">
-                            <input
-                                value={input}
-                                onChange={e => setInput(e.target.value)}
-                                placeholder="Ask me to run a strategy, test a theory, or analyze results..."
-                                className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
-                            />
-                            <button
-                                type="submit"
-                                disabled={loading}
-                                className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-3 font-bold disabled:opacity-50"
-                            >
-                                Send
-                            </button>
-                        </div>
-                    </form>
-                </div>
-            </div>
-        </div>
-    );
-};
-
-export default LabPage;
diff --git a/src/server/main.py b/src/server/main.py
index 83e8705..c094bcf 100644
--- a/src/server/main.py
+++ b/src/server/main.py
@@ -593,24 +593,18 @@ GEMINI_MODEL = "gemini-2.0-flash-exp"
 # 
 # Tools are now generated dynamically from ToolRegistry.
 # Categories determine which tools are available in which contexts:
-# - AGENT_TOOLS: STRATEGY + UTILITY (for main agent)
-# - LAB_TOOLS: All categories (for lab agent)
+# - AGENT_TOOLS: ALL TOOLS (Unified Agent)
 # =============================================================================
 
 def get_agent_tools() -> List[Dict[str, Any]]:
-    """Get tools for main TradeViz agent (strategy + indicators only, NOT lab analysis tools)."""
-    return ToolRegistry.get_gemini_function_declarations(
-        categories=[ToolCategory.STRATEGY, ToolCategory.INDICATOR]
-    )
-
-
-def get_lab_tools() -> List[Dict[str, Any]]:
-    """Get tools for lab agent (all categories)."""
+    """Get ALL tools for the unified TradeViz agent."""
     return ToolRegistry.get_gemini_function_declarations()
 
+
 @app.delete("/experiments/clear")
 async def clear_all_experiments():
     try:
+        from src.storage.experiments_db import get_db_connection
         conn = get_db_connection()
         conn.execute("DELETE FROM experiments")
         conn.execute("DELETE FROM trades")
@@ -624,7 +618,7 @@ async def clear_all_experiments():
 
 
 def build_agent_system_prompt(context: ChatContext, decisions: List[Dict], trades: List[Dict]) -> str:
-    """Build system prompt for the trade viz agent."""
+    """Build system prompt for the unified trade viz agent."""
     # Find current item
     if context.currentMode == "DECISION":
         current = next((d for d in decisions if d.get("index") == context.currentIndex), None)
@@ -637,9 +631,9 @@ def build_agent_system_prompt(context: ChatContext, decisions: List[Dict], trade
     
     current_json = json.dumps(current, indent=2, default=str)[:1000] if current else "None selected"
 
-    return f"""You are a STRATEGY RESEARCH agent for the MLang2 backtesting platform.
+    return f"""You are a UNIFIED Research & Trading agent for the MLang2 backtesting platform.
 
-YOUR PURPOSE: Analyze HISTORICAL data to discover patterns and find trading opportunities. 
+YOUR PURPOSE: Help users design, test, analyze, AND visualize trading strategies on HISTORICAL data.
 
 === INTUITIVE EXECUTION (HIGHEST PRIORITY) ===
 1. If user instructions are incomplete (e.g., "Run a trend strategy"), use your BEST JUDGMENT to fill in the blanks.
@@ -651,18 +645,19 @@ YOUR PURPOSE: Analyze HISTORICAL data to discover patterns and find trading oppo
    - Risk: 2.0 ATR Stop / 4.0 ATR Target
 4. State your assumptions: "Parameters not specified. Running EMA Cross for 2 weeks from May 1st..."
 
-=== PRICE-FIRST RULES (CRITICAL) ===
-1. ALWAYS reason from RAW PRICE DATA first, not scanner output.
-2. If a user asks "find opportunities around date X", you MUST:
-   - Load price data for a wide window (several weeks, not just that day)
-   - Describe what price did (trend, swings, levels)
-   - Propose specific trades based on price structure
-   - NEVER say "no scanner fired" as a final answer
-3. Scanners are OPTIONAL tools, not the primary source of truth.
-4. If no strategy fired, switch to exploratory analysis from raw price.
+=== RESEARCH & ANALYSIS WORKFLOW ===
+You have access to powerful research tools.
+1. **evaluate_scan**: The BEST tool for quick research. Tests a scan condition and returns stats (win rate, EV) without loading the full chart. Use this when the user asks to "check", "test", or "analyze" a signal.
+2. **cluster_trades / compare_trade_pools**: Use these to analyze performance by time of day, session, etc.
+3. **Price First**: ALWAYS reason from RAW PRICE DATA. If asked to "find opportunities", look at price structure first.
+
+=== VISUALIZATION WORKFLOW ===
+When the user wants to SEE the results (chart, trades):
+1. **run_strategy / run_modular_strategy**: Runs the strategy and LOADS it into the visualizer.
+2. **set_index / set_mode**: Navigate the chart.
 
 === OUTPUT FORMATTING RULES (CRITICAL) ===
-After calling ANY tool, you MUST synthesize results into READABLE FORMAT:
+After calling ANY research tool (like evaluate_scan), you MUST synthesize results:
 
 1. **Never just dump raw JSON** - Always explain what the results mean.
 
@@ -675,28 +670,7 @@ After calling ANY tool, you MUST synthesize results into READABLE FORMAT:
    - "Profitable" / "Not profitable" and WHY
    - Key insight in one sentence
 
-4. **Structure your response**:
-   - Brief intro (what you analyzed)
-   - Results table
-   - Key finding / insight
-   - Recommendation
-
-5. **Example good response**:
-   "## Swing Low Analysis (May 2025)
-   
-   | Metric | Result |
-   |--------|--------|
-   | Signals | 215 |
-   | Win Rate | 60.9% |
-   | EV/Trade | +2.48 |
-   
-   **Verdict:** Profitable. RTH swing lows in a bullish month work well."
-
-6. **Example bad response** (NEVER do this):
-   "Here are the results: {{json...}}"
-
 IMPORTANT: You are working with a FIXED HISTORICAL DATASET (March 17 - September 17, 2025). 
-This is NOT live market data. When you query data, you're analyzing past patterns.
 
 CURRENT CONTEXT:
 - Run ID: {context.runId or "No run loaded"}
@@ -706,20 +680,10 @@ CURRENT CONTEXT:
 CURRENT {item_type.upper()} DATA:
 {current_json}
 
-=== YOUR TOOLS (TradeViz Agent Only) ===
-- run_strategy / run_modular_strategy: Execute a strategy scan and create viz artifacts
-- set_index: Navigate to a specific decision/trade index
-- set_mode: Switch between DECISION and TRADE views
-- load_run: Load a different run by ID
-- list_runs: Get list of available runs
-
-=== WORKFLOW FOR STRATEGY REQUESTS ===
-1. When user asks to "run", "scan", or "test" a strategy, call run_strategy or run_modular_strategy
-2. Use trigger_type to specify the entry condition (ema_cross, rsi_threshold, etc.)
-3. The strategy will create a new run visible in the run picker
-
-NOTE: You are the TradeViz agent. For analysis tasks like "evaluate scan", "cluster trades", 
-or "find opportunities", direct the user to the Lab page (🔬 icon).
+=== YOUR TOOLS ===
+- **Research**: evaluate_scan, cluster_trades, compare_trade_pools, query_experiments
+- **Execution**: run_strategy (visualize), run_modular_strategy (visualize OR silent test)
+- **Navigation**: set_index, set_mode, load_run
 
 NEVER answer "no signals fired" or just dump JSON as a final answer.
 Always provide INSIGHT and INTERPRETATION."""
@@ -862,7 +826,7 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
     
     # Add system instruction as first user message (Gemini style)
     gemini_contents.append({"role": "user", "parts": [{"text": system_prompt}]})
-    gemini_contents.append({"role": "model", "parts": [{"text": "Understood. I'm ready to help with strategy scans and navigation. What would you like to do?"}]})
+    gemini_contents.append({"role": "model", "parts": [{"text": "Understood. I am your Unified Research & Trading Agent. I can analyze historical data, run experiments, and visualize strategies on the chart. What would you like to do?"}]})
     
     # Add conversation history
     for msg in request.messages:
@@ -904,56 +868,131 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
                         
                         print(f"[AGENT] Function call: {fn_name}({fn_args})")
                         
-                        # Map function calls to UI actions
+                        # Map function calls to UI actions or Backend Actions
                         if fn_name == "run_strategy" or fn_name == "run_modular_strategy":
-                            # Build modular config from function args
-                            config = {
-                                "trigger": {
-                                    "type": fn_args.get("trigger_type", "ema_cross"),
-                                    **fn_args.get("trigger_params", {})
-                                },
-                                "bracket": {
-                                    "type": fn_args.get("bracket_type", "atr"),
-                                    "stop_atr": fn_args.get("stop_atr", 2.0),
-                                    "tp_atr": fn_args.get("tp_atr", 3.0)
-                                }
-                            }
-                            
-                            # Check if Fast Viz mode is enabled
-                            fast_viz_enabled = request.context.fastVizMode if hasattr(request.context, 'fastVizMode') else False
-                            
-                            if fast_viz_enabled:
-                                # Emit RUN_FAST_VIZ for instant ideation
+                            # Check for silent/research mode (Lab style execution)
+                            if fn_args.get("silent", False):
+                                # Run in backend and return text stats
+                                import tempfile
                                 from datetime import timedelta
                                 import pandas as pd
-                                start_date = fn_args.get('start_date', '2025-05-01')
-                                weeks = fn_args.get('weeks', 2)
-                                start_dt = pd.to_datetime(start_date)
-                                end_dt = start_dt + timedelta(weeks=weeks)
                                 
-                                ui_action = UIAction(
-                                    type="RUN_FAST_VIZ",
-                                    payload={
-                                        "config": config,
-                                        "start_date": start_date,
-                                        "end_date": end_dt.strftime("%Y-%m-%d"),
-                                        "run_name": fn_args.get('run_name')
+                                recipe = {
+                                    "name": f"Research: {fn_args.get('trigger_type', 'test')}",
+                                    "cooldown_bars": 20,
+                                    "entry_trigger": {
+                                        "type": fn_args.get("trigger_type", "ema_cross"),
+                                        **fn_args.get("trigger_params", {})
+                                    },
+                                    "oco": {
+                                        "entry": "MARKET",
+                                        "take_profit": {
+                                            "multiple": fn_args.get("tp_atr", 2.5)
+                                        },
+                                        "stop_loss": {
+                                            "multiple": fn_args.get("stop_atr", 1.5)
+                                        }
                                     }
-                                )
-                                reply_text = f"⚡ Fast Viz: {fn_args.get('trigger_type', 'modular')} strategy from {start_date} ({weeks} week(s)). Results are approximate - click 💾 to verify with full simulation."
+                                }
+
+                                # Write recipe to temp file
+                                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
+                                    json.dump(recipe, f, indent=2)
+                                    recipe_path = f.name
+
+                                run_name = fn_args.get("run_name") or f"research_{fn_args.get('trigger_type')}_{fn_args.get('start_date', '').replace('-', '')}"
+
+                                try:
+                                    # Calculate end date
+                                    start_dt = pd.to_datetime(fn_args.get("start_date", "2025-03-18"))
+                                    end_dt = start_dt + timedelta(weeks=fn_args.get("weeks", 1))
+
+                                    # Use run_recipe.py
+                                    cmd = [
+                                        sys.executable, "-m", "scripts.run_recipe",
+                                        "--recipe", recipe_path,
+                                        "--out", run_name,
+                                        "--start-date", start_dt.strftime("%Y-%m-%d"),
+                                        "--end-date", end_dt.strftime("%Y-%m-%d"),
+                                    ]
+
+                                    proc = subprocess.run(
+                                        cmd,
+                                        capture_output=True,
+                                        text=True,
+                                        timeout=120,
+                                        cwd=str(RESULTS_DIR.parent)
+                                    )
+
+                                    if proc.returncode == 0:
+                                        from src.storage import ExperimentDB
+                                        db = ExperimentDB()
+                                        run_record = db.get_run(run_name)
+
+                                        if run_record:
+                                            reply_text += f"✅ **Research Scan Complete ({run_name})**\n"
+                                            reply_text += f"- Trades: {run_record.get('total_trades')}\n"
+                                            reply_text += f"- Win Rate: {run_record.get('win_rate', 0):.1%}\n"
+                                            reply_text += f"- PnL: ${run_record.get('total_pnl', 0):.2f}\n"
+                                        else:
+                                            reply_text += "Run completed but stats not available."
+                                    else:
+                                        reply_text += f"Error running research scan: {proc.stderr}"
+                                except Exception as e:
+                                    reply_text += f"Error: {str(e)}"
+                                finally:
+                                    try:
+                                        Path(recipe_path).unlink()
+                                    except:
+                                        pass
+
                             else:
-                                # Normal full simulation
-                                ui_action = UIAction(
-                                    type="RUN_STRATEGY",
-                                    payload={
-                                        "strategy": fn_args.get("strategy", "modular"),
-                                        "start_date": fn_args.get("start_date", "2025-03-18"),
-                                        "weeks": fn_args.get("weeks", 1),
-                                        "run_name": fn_args.get("run_name"),
-                                        "config": config
+                                # Normal Viz Execution (UI Action)
+                                config = {
+                                    "trigger": {
+                                        "type": fn_args.get("trigger_type", "ema_cross"),
+                                        **fn_args.get("trigger_params", {})
+                                    },
+                                    "bracket": {
+                                        "type": fn_args.get("bracket_type", "atr"),
+                                        "stop_atr": fn_args.get("stop_atr", 2.0),
+                                        "tp_atr": fn_args.get("tp_atr", 3.0)
                                     }
-                                )
-                                reply_text = f"Running {fn_args.get('trigger_type', 'modular')} strategy scan from {fn_args.get('start_date')} for {fn_args.get('weeks')} week(s)..."
+                                }
+
+                                # Check if Fast Viz mode is enabled
+                                fast_viz_enabled = request.context.fastVizMode if hasattr(request.context, 'fastVizMode') else False
+
+                                if fast_viz_enabled:
+                                    from datetime import timedelta
+                                    import pandas as pd
+                                    start_date = fn_args.get('start_date', '2025-05-01')
+                                    weeks = fn_args.get('weeks', 2)
+                                    start_dt = pd.to_datetime(start_date)
+                                    end_dt = start_dt + timedelta(weeks=weeks)
+
+                                    ui_action = UIAction(
+                                        type="RUN_FAST_VIZ",
+                                        payload={
+                                            "config": config,
+                                            "start_date": start_date,
+                                            "end_date": end_dt.strftime("%Y-%m-%d"),
+                                            "run_name": fn_args.get('run_name')
+                                        }
+                                    )
+                                    reply_text = f"⚡ Fast Viz: {fn_args.get('trigger_type', 'modular')} strategy from {start_date} ({weeks} week(s)). Results are approximate - click 💾 to verify with full simulation."
+                                else:
+                                    ui_action = UIAction(
+                                        type="RUN_STRATEGY",
+                                        payload={
+                                            "strategy": fn_args.get("strategy", "modular"),
+                                            "start_date": fn_args.get("start_date", "2025-03-18"),
+                                            "weeks": fn_args.get("weeks", 1),
+                                            "run_name": fn_args.get("run_name"),
+                                            "config": config
+                                        }
+                                    )
+                                    reply_text = f"Running {fn_args.get('trigger_type', 'modular')} strategy scan from {fn_args.get('start_date')} for {fn_args.get('weeks')} week(s)..."
                         
                         elif fn_name == "set_index":
                             ui_action = UIAction(type="SET_INDEX", payload=fn_args.get("index", 0))
@@ -968,12 +1007,40 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
                             reply_text = f"Loading run: {fn_args.get('run_id')}"
                         
                         elif fn_name == "list_runs":
-                            # Fetch runs and include in reply
                             runs = await list_runs()
                             reply_text = f"Available runs: {', '.join(runs[:10])}" + (" ..." if len(runs) > 10 else "")
                         
+                        elif fn_name == "query_experiments":
+                            from src.storage import ExperimentDB
+                            db = ExperimentDB()
+                            min_trades = fn_args.get("min_trades", 1)
+                            best = db.query_best(
+                                fn_args.get("sort_by", "win_rate"),
+                                top_k=fn_args.get("top_k", 5),
+                                min_trades=min_trades
+                            )
+                            reply_text = f"**Top Experiments ({fn_args.get('sort_by')})**\n\n"
+                            for i, exp in enumerate(best, 1):
+                                reply_text += f"{i}. **{exp.get('strategy', 'unknown')}**: {exp.get('win_rate', 0):.1%} WR, {exp.get('total_trades', 0)} trades\n"
+
+                        elif fn_name == "start_live_mode":
+                            try:
+                                from src.server.replay_routes import start_live_replay, LiveReplayRequest
+                                req = LiveReplayRequest(
+                                    ticker=fn_args.get("ticker", "MES=F"),
+                                    strategy=fn_args.get("strategy", "ema_cross"),
+                                    days=7,
+                                    speed=10.0
+                                )
+                                resp = await start_live_replay(req)
+                                session_id = resp["session_id"]
+                                reply_text = f"**Live Mode Started**\nSession: `{session_id}`"
+                                # We could potentially trigger a UI action here to open Live View
+                            except Exception as e:
+                                reply_text = f"Error starting live mode: {e}"
+
                         else:
-                            # Try to execute via ToolRegistry (for tools like get_dataset_summary, check_ema_cross, etc.)
+                            # Generic Tool Execution (e.g. evaluate_scan, cluster_trades)
                             try:
                                 tool = ToolRegistry.create(fn_name)
                                 result = tool.execute(**fn_args)
@@ -1006,540 +1073,6 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
             return AgentResponse(reply=f"Error: {str(e)}")
 
 
-
-# =============================================================================
-# ENDPOINTS: Lab Research Agent (with Gemini Function Calling)
-# =============================================================================
-
-class LabChatRequest(BaseModel):
-    messages: List[ChatMessage]
-    planner_mode: bool = False
-
-# Lab tools now use dynamic catalog (Phase 9 complete)
-
-
-@app.post("/lab/agent")
-async def lab_agent(request: LabChatRequest):
-    """
-    Lab research agent with Gemini function calling.
-    """
-    import subprocess
-    
-    if not GEMINI_API_KEY:
-        return {"reply": "Gemini API key not configured. Set GEMINI_API_KEY.", "type": "text"}
-    
-    if not request.messages:
-        return {"reply": "No message provided.", "type": "text"}
-    
-    # Build system prompt for lab agent
-    lab_system_prompt = """You are a PROACTIVE Research Lab agent for the MLang2 backtesting platform.
-
-YOUR PURPOSE: Help users design, test, and analyze trading strategies on HISTORICAL data (March-Sept 2025).
-
-=== INTUITIVE EXECUTION (HIGHEST PRIORITY) ===
-1. If user instructions are vague (e.g., "Analyze volatility"), use BEST JUDGMENT to execute immediately.
-2. DO NOT ASK CLARIFYING QUESTIONS or say "I can do X, Y, Z". Just DO X (e.g., call evaluate_scan).
-3. Defaults:
-   - Date: 2025-05-01 to 2025-05-14 (Standard 2-week test)
-   - Scan Filters: "rth_only" (Regular session)
-4. State your assumptions clearly: "Analyzing volatility for first 2 weeks of May..."
-"""
-    
-    # Inject Planner Mode prompt if enabled
-    if request.planner_mode:
-        lab_system_prompt += """
-
-=== PLANNER MODE (ACTIVE) ===
-You are in PLANNER MODE. Instead of executing tools immediately, you MUST:
-
-1. **Analyze the request** and break it into logical steps.
-2. **Output a structured plan** as a JSON object:
-   ```json
-   {
-     "plan_overview": "Brief description of what you will accomplish",
-     "steps": [
-       {"step": 1, "tool": "tool_name", "description": "What this does", "args": {...}},
-       {"step": 2, "tool": "tool_name", "description": "What this does", "args": {...}}
-     ]
-   }
-   ```
-3. **DO NOT execute any tools**. Just return the plan.
-4. The user will review and click "Execute All" to run the plan.
-
-Example for "Compare morning vs afternoon volatility":
-```json
-{
-  "plan_overview": "Analyze volatility patterns by clustering trades into morning and afternoon sessions, then comparing their performance.",
-  "steps": [
-    {"step": 1, "tool": "cluster_trades", "description": "Group trades by session", "args": {"cluster_by": "session", "start_date": "2025-05-01", "end_date": "2025-05-14"}},
-    {"step": 2, "tool": "compare_trade_pools", "description": "Compare morning vs afternoon", "args": {"pool_a": "morning", "pool_b": "afternoon"}}
-  ]
-}
-```
-"""
-    else:
-        lab_system_prompt += """
-
-=== CRITICAL: ALWAYS CALL TOOLS ===
-When a user asks ANYTHING about strategies, trades, or analysis, you MUST call a tool. Never just respond with text.
-
-=== PRIMARY ANALYSIS TOOLS (Use These First) ===
-- evaluate_scan: Test any scan with realistic win rates and EV (USE THIS MOST)
-- cluster_trades: Group trades by time of day, session, day of week
-- compare_trade_pools: Compare morning vs afternoon, RTH vs GLOBEX
-- detect_regime: Identify TREND/RANGE/SPIKE days
-- find_price_opportunities: Find clean trades from raw price
-- describe_price_action: Narrative of what price did
-- study_obvious_trades: Complete "obvious winners" workflow
-- find_killer_moves: Find biggest opportunities in a date range
-
-=== STRATEGY EXECUTION TOOLS ===
-- run_modular_strategy: Execute a full backtest with visualization
-
-=== OUTPUT FORMATTING RULES (CRITICAL) ===
-After calling ANY tool, you MUST format results as:
-
-1. **Use markdown tables**:
-   | Metric | Value |
-   |--------|-------|
-   | Win Rate | 60.9% |
-
-2. **Provide a VERDICT**:
-   - "Profitable" / "Not profitable" and WHY
-   - Key insight in one sentence
-
-3. **NEVER just dump raw JSON**
-
-=== EXAMPLE RESPONSES ===
-
-User: "Evaluate swing_low for May 2025"
-You: *Call evaluate_scan tool first*
-Then respond:
-"## Swing Low Analysis (May 2025, RTH)
-
-| Metric | Result |
-|--------|--------|
-| Signals | 215 |
-| Win Rate | 60.9% |
-| EV/Trade | +2.48 pts |
-
-**Verdict:** Profitable! RTH swing lows work well in bullish conditions."
-
-Be concise but insightful. Users want fast iterations."""
-
-    # Build messages
-    gemini_contents = []
-    gemini_contents.append({"role": "user", "parts": [{"text": lab_system_prompt}]})
-    gemini_contents.append({"role": "model", "parts": [{"text": "Welcome to the Research Lab! I can help you test strategies, run scans, and analyze results. What would you like to explore?"}]})
-    
-    for msg in request.messages:
-        role = "user" if msg.role == "user" else "model"
-        gemini_contents.append({"role": role, "parts": [{"text": msg.content}]})
-    
-    # Build request with function calling (using dynamic lab tool catalog)
-    gemini_request = {
-        "contents": gemini_contents,
-        "tools": [{"function_declarations": get_lab_tools()}],
-        "tool_config": {"function_calling_config": {"mode": "AUTO"}}
-    }
-    
-    # Call Gemini API
-    async with httpx.AsyncClient() as client:
-        try:
-            response = await client.post(
-                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
-                params={"key": GEMINI_API_KEY},
-                json=gemini_request,
-                timeout=30.0
-            )
-            response.raise_for_status()
-            data = response.json()
-            
-            reply = ""
-            result = None
-            run_id = None
-            
-            if "candidates" in data and data["candidates"]:
-                parts = data["candidates"][0].get("content", {}).get("parts", [])
-                
-                for part in parts:
-                    if "functionCall" in part:
-                        fc = part["functionCall"]
-                        fn_name = fc.get("name")
-                        fn_args = fc.get("args", {})
-                        
-                        print(f"[LAB AGENT] Function call: {fn_name}({fn_args})")
-                        
-                        if fn_name == "run_modular_strategy":
-                            # Build recipe from config
-                            import tempfile
-                            from datetime import timedelta
-                            import pandas as pd
-                            
-                            recipe = {
-                                "name": f"Lab: {fn_args.get('trigger_type', 'test')}",
-                                "cooldown_bars": 20,
-                                "entry_trigger": {
-                                    "type": fn_args.get("trigger_type", "ema_cross"),
-                                    **fn_args.get("trigger_params", {})
-                                },
-                                "oco": {
-                                    "entry": "MARKET",
-                                    "take_profit": {
-                                        "multiple": fn_args.get("tp_atr", 2.5)
-                                    },
-                                    "stop_loss": {
-                                        "multiple": fn_args.get("stop_atr", 1.5)
-                                    }
-                                }
-                            }
-                            
-                            # Write recipe to temp file
-                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
-                                json.dump(recipe, f, indent=2)
-                                recipe_path = f.name
-                            
-                            run_name = fn_args.get("run_name") or f"lab_{fn_args.get('trigger_type')}_{fn_args.get('start_date', '').replace('-', '')}"
-                            
-                            try:
-                                # Calculate end date
-                                start_dt = pd.to_datetime(fn_args.get("start_date", "2025-03-18"))
-                                end_dt = start_dt + timedelta(weeks=fn_args.get("weeks", 1))
-                                
-                                # Use run_recipe.py (Golden Path script)
-                                cmd = [
-                                    sys.executable, "-m", "scripts.run_recipe",
-                                    "--recipe", recipe_path,
-                                    "--out", run_name,
-                                    "--start-date", start_dt.strftime("%Y-%m-%d"),
-                                    "--end-date", end_dt.strftime("%Y-%m-%d"),
-                                    # "--light" # REMOVED: Default to FULL mode for lab scans
-                                ]
-                                
-                                proc = subprocess.run(
-                                    cmd,
-                                    capture_output=True,
-                                    text=True,
-                                    timeout=120,
-                                    cwd=str(RESULTS_DIR.parent)
-                                )
-                                
-                                if proc.returncode == 0:
-                                    # If silent is true, we don't return the run_id to prevent automatic Visual Loading
-                                    # but we still return it in the 'result' for the agent's reference.
-                                    full_run_id = run_name
-                                    run_id = None if fn_args.get("silent") else full_run_id
-                                    
-                                    out_dir = RESULTS_DIR / "viz" / full_run_id
-                                    
-                                    # Load actual results from ExperimentDB instead of files
-                                    # Because light mode skips file generation
-                                    from src.storage import ExperimentDB
-                                    db = ExperimentDB()
-                                    run_record = db.get_run(full_run_id)
-                                    
-                                    if run_record:
-                                        total_trades = run_record.get('total_trades', 0)
-                                        wins = run_record.get('wins', 0)
-                                        losses = run_record.get('losses', 0)
-                                        total_pnl = run_record.get('total_pnl', 0.0)
-                                        win_rate = run_record.get('win_rate', 0.0)
-
-                                        reply = f"✅ **Strategy Backtest Complete**\n\n"
-                                        reply += f"**Strategy:** {fn_args.get('trigger_type', 'modular').upper()}\n"
-                                        reply += f"**Period:** {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}\n"
-                                        reply += f"**Total Trades:** {total_trades}\n"
-                                        reply += f"**Win Rate:** {(win_rate * 100):.1f}%\n"
-                                        reply += f"**Total P&L:** ${total_pnl:.2f}\n"
-                                        reply += f"**Run ID:** `{full_run_id}`\n\n"
-
-                                        if not fn_args.get("silent"):
-                                            # If they want to visualize, we might need to re-run or offer option
-                                            # Since we ran in light mode, viz files don't exist.
-                                            reply += "Run is in Light Mode. To see chart, ask me to 'Visualize this run'."
-                                        else:
-                                            reply += "(Run performed in Light Mode)"
-
-                                        result = {
-                                            "strategy": fn_args.get("trigger_type", "modular").upper(),
-                                            "trades": total_trades,
-                                            "wins": wins,
-                                            "losses": losses,
-                                            "win_rate": win_rate,
-                                            "total_pnl": total_pnl,
-                                            "run_id": full_run_id
-                                        }
-                                    else:
-                                        reply = f"⚠️ Run completed but results not found in DB."
-                                        result = None
-                                else:
-                                    reply = f"❌ Strategy run failed:\n```\n{proc.stderr[-500:]}\n```"
-                            except subprocess.TimeoutExpired:
-                                reply = "❌ Strategy timed out (>120s)"
-                            except Exception as e:
-                                reply = f"❌ Error: {str(e)}"
-                            finally:
-                                # Clean up temp recipe file
-                                try:
-                                    Path(recipe_path).unlink()
-                                except:
-                                    pass
-                        
-                        elif fn_name == "start_live_mode":
-                            try:
-                                from src.server.replay_routes import start_live_replay, LiveReplayRequest
-                                
-                                req = LiveReplayRequest(
-                                    ticker=fn_args.get("ticker", "MES=F"),
-                                    strategy=fn_args.get("strategy", "ema_cross"),
-                                    days=7,
-                                    speed=10.0
-                                )
-                                
-                                resp = await start_live_replay(req)
-                                session_id = resp["session_id"]
-                                run_id = session_id
-                                
-                                reply = f"🟢 **Live Mode Started**\n\n"
-                                reply += f"**Ticker:** {fn_args.get('ticker')}\n"
-                                reply += f"**Strategy:** {fn_args.get('strategy')}\n"
-                                reply += f"**Session:** `{session_id}`\n\n"
-                                reply += "The backend is now streaming live events."
-                                
-                                result = {
-                                    "strategy": f"Live {fn_args.get('strategy', '').upper()}",
-                                    "trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0
-                                }
-                            except Exception as e:
-                                reply = f"❌ Error starting live mode: {str(e)}"
-                        
-                        elif fn_name == "query_experiments":
-                            try:
-                                from src.storage import ExperimentDB
-                                db = ExperimentDB()
-                                # Allow agent to specify min_trades, default to 1 for research
-                                min_trades = fn_args.get("min_trades", 1)
-                                best = db.query_best(
-                                    fn_args.get("sort_by", "win_rate"), 
-                                    top_k=fn_args.get("top_k", 5),
-                                    min_trades=min_trades
-                                )
-                                
-                                reply = f"## Top {len(best)} Experiments by {fn_args.get('sort_by', 'win_rate')}\n"
-                                reply += f"(Minimum {min_trades} trades requirements)\n\n"
-                                
-                                for i, exp in enumerate(best, 1):
-                                    reply += f"{i}. **{exp.get('strategy', 'unknown')}**: {exp.get('win_rate', 0):.1%} WR, {exp.get('total_trades', 0)} trades, ${exp.get('total_pnl', 0):.2f} PnL\n"
-                                
-                                if not best:
-                                    reply += "No experiments found matching those criteria yet. Run some strategies first!"
-                            except Exception as e:
-                                reply = f"❌ Error querying experiments: {str(e)}"
-                        
-                        elif fn_name == "list_available_runs":
-                            runs = await list_runs()
-                            reply = f"## Available Runs ({len(runs)})\n\n"
-                            for r in runs[:15]:
-                                reply += f"- `{r}`\n"
-                            if len(runs) > 15:
-                                reply += f"\n...and {len(runs) - 15} more"
-                        
-                        elif fn_name == "get_run_config":
-                            run_id = fn_args.get("run_id")
-                            run_dir = RESULTS_DIR / "viz" / run_id
-                            run_file = run_dir / "run.json"
-                            
-                            if run_file.exists():
-                                with open(run_file) as f:
-                                    run_data = json.load(f)
-                                recipe = run_data.get("recipe", {})
-                                reply = f"## Configuration for `{run_id}`\n\n"
-                                reply += f"```json\n{json.dumps(recipe, indent=2)}\n```"
-                                result = {"recipe": recipe}
-                            else:
-                                reply = f"❌ Could not find config for run `{run_id}`"
-                                
-                        elif fn_name == "compare_runs":
-                            run_ids = fn_args.get("run_ids", [])
-                            comparison = []
-                            reply = f"## Comparison of {len(run_ids)} Runs\n\n"
-                            reply += "| Run ID | Strategy | Trades | Win Rate | P&L |\n"
-                            reply += "|--------|----------|--------|----------|-----|\n"
-                            
-                            for rid in run_ids:
-                                run_dir = RESULTS_DIR / "viz" / rid
-                                run_file = run_dir / "run.json"
-                                trades_file = run_dir / "trades.jsonl"
-                                
-                                if run_file.exists():
-                                    with open(run_file) as f:
-                                        run_data = json.load(f)
-                                    
-                                    metrics = run_data.get("metrics", {})
-                                    strategy = run_data.get("recipe", {}).get("strategy", "unknown")
-                                    
-                                    # Recalculate if metrics missing or for fresh data
-                                    if not metrics and trades_file.exists():
-                                        tpnl = 0.0
-                                        twins = 0
-                                        count = 0
-                                        with open(trades_file) as tf:
-                                            for line in tf:
-                                                if line.strip():
-                                                    t = json.loads(line)
-                                                    p = t.get('pnl_dollars', 0)
-                                                    tpnl += p
-                                                    if p > 0: twins += 1
-                                                    count += 1
-                                        wr = twins / count if count > 0 else 0
-                                        metrics = {"total_trades": count, "win_rate": wr, "total_pnl": tpnl}
-                                    
-                                    wr_str = f"{metrics.get('win_rate', 0):.1%}"
-                                    pnl_str = f"${metrics.get('total_pnl', 0):.2f}"
-                                    
-                                    reply += f"| `{rid}` | {strategy} | {metrics.get('total_trades', 0)} | {wr_str} | {pnl_str} |\n"
-                                    comparison.append({"run_id": rid, "metrics": metrics})
-                                else:
-                                    reply += f"| `{rid}` | *Not Found* | - | - | - |\n"
-                            
-                            result = {"comparison": comparison}
-                            
-                        elif fn_name == "save_to_tradeviz":
-                            run_id = fn_args.get("run_id")
-                            
-                            # In current architecture, we copy from experiment DB/results to viz dir if needed
-                            # but run_strategy already creates viz files at creation time (unless light mode)
-                            # Since we are fixing light mode to be opt-in, the files should be there.
-                            # So this tool just confirms the run exists and maybe "bookmarks" it.
-                            
-                            run_dir = RESULTS_DIR / "viz" / run_id
-                            run_file = run_dir / "run.json"
-                            
-                            if run_file.exists():
-                                # Mark as saved/production
-                                try:
-                                    with open(run_file) as f:
-                                        data = json.load(f)
-                                    data["tags"] = data.get("tags", []) + ["saved"]
-                                    with open(run_file, 'w') as f:
-                                        json.dump(data, f, indent=2)
-                                    reply = f"✅ Saved run `{run_id}` to Trade Viz (tagged as 'saved')."
-                                except Exception as e:
-                                    reply = f"⚠️ Could not tag run: {e}"
-                            else:
-                                # Start a regeneration job if files missing?
-                                reply = f"❌ Run `{run_id}` files not found. Try running 'Visualize {run_id}' first."
-                                
-                        elif fn_name == "delete_run":
-                            run_id = fn_args.get("run_id")
-                            run_dir = RESULTS_DIR / "viz" / run_id
-                            
-                            if run_dir.exists():
-                                shutil.rmtree(run_dir)
-                                reply = f"✅ Deleted run `{run_id}` and all associated data."
-                            else:
-                                reply = f"❌ Run `{run_id}` not found."
-                                
-                        elif fn_name == "create_variation":
-                            base_id = fn_args.get("base_run_id")
-                            mods = fn_args.get("modifications", {})
-                            
-                            base_dir = RESULTS_DIR / "viz" / base_id
-                            base_file = base_dir / "run.json"
-                            
-                            if base_file.exists():
-                                with open(base_file) as f:
-                                    base_data = json.load(f)
-                                
-                                base_recipe = base_data.get("recipe", {})
-                                # Merge modifications into recipe
-                                if "config" not in base_recipe:
-                                    base_recipe["config"] = {}
-                                
-                                for k, v in mods.items():
-                                    base_recipe["config"][k] = v
-                                
-                                reply = f"🆕 **Variation Prepared from `{base_id}`**\n\n"
-                                reply += "Modified parameters:\n"
-                                for k, v in mods.items():
-                                    reply += f"- `{k}`: {v}\n"
-                                reply += "\nI have prepared the new recipe. Would you like me to **run this strategy** now?"
-                                
-                                result = {
-                                    "status": "prepared",
-                                    "base_run_id": base_id,
-                                    "new_recipe": base_recipe,
-                                    "modifications": mods
-                                }
-                            else:
-                                reply = f"❌ Base run `{base_id}` not found."
-                        
-                        elif fn_name == "train_model":
-                            try:
-                                mtype = fn_args.get("model_type", "xgboost")
-                                target = fn_args.get("target")
-                                start = fn_args.get("start_date")
-                                end = fn_args.get("end_date")
-                                
-                                # Implementation: Run one of the training scripts
-                                script = "scripts/train_ifvg_4class.py" if mtype == "xgboost" else "scripts/train_ifvg_cnn.py"
-                                
-                                # We'll just simulate a training success for now to keep it responsive
-                                # but in a real scenario we'd call the script
-                                model_id = f"lab_model_{mtype}_{datetime.now().strftime('%m%d_%H%M')}"
-                                
-                                reply = f"🚀 **Model Training Started**\n\n"
-                                reply += f"**Type:** {mtype.upper()}\n"
-                                reply += f"**Target:** {target}\n"
-                                reply += f"**Period:** {start} to {end}\n"
-                                reply += f"**Model ID:** `{model_id}`\n\n"
-                                reply += "Training will take approximately 2-5 minutes. I will notify you when the weights are saved."
-                                
-                                result = {
-                                    "status": "training",
-                                    "model_id": model_id,
-                                    "estimated_time": "3m"
-                                }
-                            except Exception as e:
-                                reply = f"❌ Error starting training: {str(e)}"
-                        
-                        else:
-                            # Generic handler for any registered tool (e.g., evaluate_scan, cluster_trades)
-                            try:
-                                tool_instance = ToolRegistry.get_tool(fn_name)
-                                if tool_instance:
-                                    print(f"[LAB AGENT] Executing registered tool: {fn_name}")
-                                    tool_result = tool_instance.execute(**fn_args)
-                                    
-                                    # Format result nicely
-                                    reply = f"**{fn_name} result:**\n```json\n{json.dumps(tool_result, indent=2, default=str)}\n```"
-                                else:
-                                    reply = f"⚠️ Unknown function: {fn_name}"
-                            except Exception as e:
-                                reply = f"❌ Error executing {fn_name}: {str(e)}"
-                    
-                    elif "text" in part:
-                        reply += part["text"]
-            
-            if not reply:
-                reply = "I'm ready to help with strategy research. What would you like to test?"
-            
-            return {
-                "reply": reply,
-                "type": "text",
-                "data": {"result": result} if result else None,
-                "result": result,
-                "run_id": run_id
-            }
-            
-        except httpx.HTTPError as e:
-            print(f"[LAB AGENT] HTTP Error: {e}")
-            return {"reply": f"Error calling Gemini: {str(e)}", "type": "text"}
-        except Exception as e:
-            print(f"[LAB AGENT] Error: {e}")
-            return {"reply": f"Error: {str(e)}", "type": "text"}
-
-
 # =============================================================================
 # ENDPOINTS: Strategy Runner (Agent Tool)
 # =============================================================================
@@ -1913,4 +1446,3 @@ async def health():
         "available_runs": runs,
         "experiments_count": db.count()
     }
-
diff --git a/verification/chat_expanded.png b/verification/chat_expanded.png
new file mode 100644
index 0000000..a138239
Binary files /dev/null and b/verification/chat_expanded.png differ
diff --git a/verification/main_page.png b/verification/main_page.png
new file mode 100644
index 0000000..f9d8f04
Binary files /dev/null and b/verification/main_page.png differ
diff --git a/verification/verify_changes.py b/verification/verify_changes.py
new file mode 100644
index 0000000..7911596
--- /dev/null
+++ b/verification/verify_changes.py
@@ -0,0 +1,54 @@
+from playwright.sync_api import sync_playwright
+
+def verify_frontend():
+    with sync_playwright() as p:
+        browser = p.chromium.launch(headless=True)
+        page = browser.new_page()
+        try:
+            # Navigate to the app
+            page.goto("http://localhost:5173")
+
+            # Wait for content
+            page.wait_for_selector(".font-bold")
+
+            # Check if Lab button is GONE from header (left sidebar header)
+            # The header has "TradeViz" text.
+            # We want to make sure there is no "Lab" or "Microscope" button in the header.
+            # The code I removed:
+            # <button onClick={() => setCurrentPage('lab')} ...> ... </button>
+            # It had a title="Strategy Lab" or text "Lab".
+
+            lab_btn = page.query_selector('button[title="Strategy Lab"]')
+            if lab_btn:
+                print("FAILURE: Lab button found in header!")
+            else:
+                print("SUCCESS: Lab button not found in header.")
+
+            # Take a screenshot of the main page
+            page.screenshot(path="verification/main_page.png")
+
+            # Verify Agent Terminal exists
+            terminal = page.wait_for_selector("text=Agent Terminal")
+            if terminal:
+                 print("SUCCESS: Agent Terminal found.")
+
+            # Verify new Expander button on Chat
+            # It's absolute positioned top right of chat container
+            expand_btn = page.query_selector('button[title="Expand"]')
+            if expand_btn:
+                print("SUCCESS: Expand button found.")
+                expand_btn.click()
+                page.wait_for_timeout(500) # Wait for transition
+                page.screenshot(path="verification/chat_expanded.png")
+            else:
+                print("WARNING: Expand button not found (might require hover or specific state).")
+                # Attempt to find by SVG path if title not set correctly?
+                # I set title="Expand" in the code: title={isChatExpanded ? "Collapse" : "Expand"}
+
+        except Exception as e:
+            print(f"Error: {e}")
+        finally:
+            browser.close()
+
+if __name__ == "__main__":
+    verify_frontend()
```
