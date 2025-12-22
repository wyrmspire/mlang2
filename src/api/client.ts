import { VizDecision, VizTrade, RunManifest, AgentResponse, ChatMessage, ContinuousData } from '../types/viz';

// API base URL - auto-detect port (8000 or 8001)
let API_BASE = import.meta.env.VITE_API_URL || '';

// Flag to track if backend is available
let backendAvailable: boolean | null = null;

// Check backend availability, auto-detecting port if needed
async function checkBackend(): Promise<boolean> {
    if (backendAvailable !== null) return backendAvailable;

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
        context: { runId: string, currentIndex: number, currentMode: 'DECISION' | 'TRADE' }
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

    postLabAgent: async (messages: ChatMessage[]): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            return { reply: 'Backend not connected. Start with: ./start.sh' };
        }
        try {
            const response = await fetch(`${API_BASE}/lab/agent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages }),
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
    }
};