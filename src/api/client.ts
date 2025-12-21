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

    // Simulation endpoints
    startSimulation: async (payload: {
        strategy_name: string,
        config?: any,
        start_date?: string,
        end_date?: string,
        start_idx?: number,
        end_idx?: number
    }): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/sim/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error('Failed to start simulation');
        return response.json();
    },

    stepSimulation: async (sessionId: string, nBars: number = 1): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/sim/${sessionId}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_bars: nBars })
        });
        if (!response.ok) throw new Error('Failed to step simulation');
        return response.json();
    },

    updateSimParams: async (sessionId: string, params: any): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/sim/${sessionId}/update_params`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ params })
        });
        if (!response.ok) throw new Error('Failed to update params');
        return response.json();
    },

    getSimState: async (sessionId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/sim/${sessionId}/state`);
        if (!response.ok) throw new Error('Failed to get sim state');
        return response.json();
    },

    stopSimulation: async (sessionId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/sim/${sessionId}/stop`, {
            method: 'POST'
        });
        if (!response.ok) throw new Error('Failed to stop simulation');
        return response.json();
    },

    listSimSessions: async (): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/sim/sessions`);
        if (!response.ok) throw new Error('Failed to list sessions');
        return response.json();
    }
};