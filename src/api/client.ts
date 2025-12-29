import { VizDecision, VizTrade, RunManifest, AgentResponse, ChatMessage, ContinuousData } from '../types/viz';

// API base URL - auto-detect port (8000 or 8001)
let API_BASE = import.meta.env.VITE_API_URL || '';

// Flag to track if backend is available - only cache success, always retry on failure
let backendAvailable: boolean | null = null;

// Check backend availability, auto-detecting port if needed
async function checkBackend(): Promise<boolean> {
    // Only cache success - if previously failed, try again
    if (backendAvailable === true) return true;

    // If no explicit URL, try both ports
    if (!API_BASE) {
        for (const port of [8000, 8001]) {
            try {
                const response = await fetch(`http://localhost:${port}/health`, {
                    method: 'GET',
                    signal: AbortSignal.timeout(2000) // 2s timeout
                });
                if (response.ok) {
                    API_BASE = `http://localhost:${port}`;
                    console.log(`Backend detected on port ${port}`);
                    backendAvailable = true;
                    return true;
                }
            } catch {
                // Try next port
            }
        }
        backendAvailable = false;
        return false;
    }

    try {
        const response = await fetch(`${API_BASE}/health`, { method: 'GET' });
        backendAvailable = response.ok;
    } catch {
        backendAvailable = false;
    }
    return backendAvailable;
}

// ============================================================================
// API CLIENT
// ============================================================================
export const api = {
    // Fetch continuous contract data for base chart
    getContinuousContract: async (
        start?: string,
        end?: string,
        timeframe: string = '1m'
    ): Promise<ContinuousData> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const params = new URLSearchParams();
        if (start) params.set('start', start);
        if (end) params.set('end', end);
        params.set('timeframe', timeframe);
        const response = await fetch(`${API_BASE}/market/continuous?${params}`);
        if (!response.ok) throw new Error('Failed to fetch continuous data');
        return response.json();
    },

    getRuns: async (): Promise<string[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs`);
        if (!response.ok) throw new Error('Failed to fetch runs');
        return response.json();
    },

    clearAllRuns: async (): Promise<void> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/experiments/clear`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to clear runs');
    },

    getRun: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs/${runId}`);
        if (!response.ok) throw new Error(`Failed to fetch run: ${runId}`);
        return response.json();
    },

    getManifest: async (runId: string): Promise<RunManifest> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs/${runId}/manifest`);
        if (!response.ok) throw new Error(`Failed to fetch manifest: ${runId}`);
        return response.json();
    },

    getDecisions: async (runId: string): Promise<VizDecision[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs/${runId}/decisions`);
        if (!response.ok) throw new Error(`Failed to fetch decisions: ${runId}`);
        return response.json();
    },

    getTrades: async (runId: string): Promise<VizTrade[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs/${runId}/trades`);
        if (!response.ok) throw new Error(`Failed to fetch trades: ${runId}`);
        return response.json();
    },

    postAgent: async (
        messages: ChatMessage[],
        context: { runId: string, currentIndex: number, currentMode: 'DECISION' | 'TRADE', fastVizMode?: boolean, planningMode?: boolean }
    ): Promise<AgentResponse> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            return { reply: 'Backend not connected. Start with: ./start.sh' };
        }
        try {
            const response = await fetch(`${API_BASE}/agent/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages, context }),
            });
            if (!response.ok) return { reply: 'Error contacting agent server.' };
            return response.json();
        } catch {
            return { reply: 'Error contacting agent server.' };
        }
    },

    postLabAgent: async (messages: ChatMessage[], plannerMode: boolean = false): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            return { reply: 'Backend not connected. Start with: ./start.sh' };
        }
        try {
            const response = await fetch(`${API_BASE}/lab/agent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages, planner_mode: plannerMode }),
            });
            if (!response.ok) return { reply: 'Error contacting lab agent.' };
            return response.json();
        } catch {
            return { reply: 'Error contacting lab agent.' };
        }
    },

    runStrategy: async (payload: any): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/agent/run-strategy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return response.json();
    },

    startReplay: async (modelPath: string, startDate?: string, days: number = 1, speed: number = 10.0, threshold: number = 0.6, strategy: string = "ifvg_4class"): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');

        const response = await fetch(`${API_BASE}/replay/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: modelPath,
                start_date: startDate,
                days: days,
                speed: speed,
                threshold: threshold,
                strategy: strategy
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to start replay');
        }
        return response.json();
    },

    startLiveReplay: async (
        ticker: string = "MES=F",
        strategy: string = "ema_cross",
        days: number = 7,
        speed: number = 10.0,
        entryConfig?: {
            entry_type?: string;     // Now string to support all strategies
            entry_params?: any;      // Dynamic params for strategies
            stop_method?: 'atr' | 'swing' | 'fixed_bars';
            stop_config?: any;       // Future-proofing
            tp_method?: 'atr' | 'r_multiple';
            stop_atr?: number;
            tp_atr?: number;
            tp_r?: number;
        }
    ): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');

        const body = {
            ticker,
            strategy,
            days,
            speed,
            // Entry scan config
            entry_type: entryConfig?.entry_type || 'market',
            entry_params: entryConfig?.entry_params || {},
            stop_method: entryConfig?.stop_method || 'atr',
            tp_method: entryConfig?.tp_method || 'atr',
            stop_atr: entryConfig?.stop_atr || 1.0,
            tp_atr: entryConfig?.tp_atr || 2.0,
            tp_r: entryConfig?.tp_r || 2.0
        };

        const response = await fetch(`${API_BASE}/replay/start/live`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to start live replay');
        }
        return response.json();
    },

    stopReplay: async (sessionId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) return;

        await fetch(`${API_BASE}/replay/sessions/${sessionId}`, {
            method: 'DELETE'
        });
    },

    getReplayStreamUrl: (sessionId: string) => {
        // API_BASE is set by checkBackend hopefully? 
        // We might need to ensure checkBackend is called, but startReplay calls it.
        return `${API_BASE}/replay/stream/${sessionId}`;
    },

    getYFinanceData: async (ticker: string, days: number): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/market/yfinance?ticker=${ticker}&days=${days}`);
        if (!response.ok) {
            throw new Error(`YFinance fetch failed: ${response.status}`);
        }
        return response.json();
    },

    getExperiments: async (params: any = {}): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const query = new URLSearchParams(params).toString();
        const response = await fetch(`${API_BASE}/experiments?${query}`);
        if (!response.ok) throw new Error('Failed to fetch experiments');
        return response.json();
    },

    deleteExperiment: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/experiments/${runId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to delete experiment');
        return response.json();
    },

    visualizeExperiment: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/experiments/${runId}/visualize`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to visualize experiment');
        return response.json();
    },

    // Fast Viz API
    runFastViz: async (config: any, startDate: string, endDate: string, runName?: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/fast-viz/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config, start_date: startDate, end_date: endDate, run_name: runName })
        });
        if (!response.ok) throw new Error('Failed to run Fast Viz');
        return response.json();
    },

    listFastVizRuns: async (): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/fast-viz/list`);
        if (!response.ok) throw new Error('Failed to list Fast Viz runs');
        return response.json();
    },

    getFastVizRun: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/fast-viz/${runId}`);
        if (!response.ok) throw new Error('Failed to get Fast Viz run');
        return response.json();
    },

    deleteFastVizRun: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/fast-viz/${runId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to delete Fast Viz run');
        return response.json();
    },

    saveFastVizRun: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/fast-viz/save/${runId}`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to save Fast Viz run');
        return response.json();
    },

    addToFastViz: async (triggerType: string, triggerParams: any, startDate: string, endDate: string, stopAtr: number = 2.0, tpAtr: number = 3.0): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/fast-viz/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trigger_type: triggerType,
                trigger_params: triggerParams,
                start_date: startDate,
                end_date: endDate,
                stop_atr: stopAtr,
                tp_atr: tpAtr
            })
        });
        if (!response.ok) throw new Error('Failed to add to Fast Viz');
        return response.json();
    }
};