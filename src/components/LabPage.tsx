import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api/client';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    type?: 'text' | 'table' | 'chart' | 'code';
    data?: any;
    run_id?: string;
}

interface LabResult {
    strategy: string;
    trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_pnl: number;
    equity_curve?: number[];
}

interface LabPageProps {
    onLoadRun?: (runId: string) => void;
}

export const LabPage: React.FC<LabPageProps> = ({ onLoadRun }) => {
    const [messages, setMessages] = useState<Message[]>([
        {
            role: 'assistant',
            content: 'Welcome to the Research Lab! I can help you test strategies, run scans, train models, and analyze results. What would you like to explore?',
            type: 'text'
        }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [currentResult, setCurrentResult] = useState<LabResult | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const [plannerMode, setPlannerMode] = useState<boolean>(false);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg: Message = { role: 'user', content: input, type: 'text' };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const response = await api.postLabAgent([...messages, userMsg], plannerMode);
            const assistantMsg: Message = {
                role: 'assistant',
                content: response.reply || 'Processing...',
                type: response.type || 'text',
                data: response.data,
                run_id: response.run_id
            };
            setMessages(prev => [...prev, assistantMsg]);
            if (response.result) {
                setCurrentResult(response.result);
            }
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Error contacting lab agent. Is the backend running?',
                type: 'text'
            }]);
        } finally {
            setLoading(false);
        }
    };

    const quickActions = [
        { label: 'Run EMA Scan', prompt: 'Run an EMA cross scan on the last 7 days' },
        { label: 'Test ORB Strategy', prompt: 'Test the Opening Range Breakout strategy' },
        { label: 'Compare Models', prompt: 'Compare the LSTM vs CNN model accuracy' },
        { label: 'Show Best Config', prompt: 'What is the best configuration from recent experiments?' },
        { label: 'Run Grid Search', prompt: 'Run a grid search on ORB stop and target parameters' },
    ];

    const sendQuickAction = (prompt: string) => {
        setInput(prompt);
    };

    const renderResultTable = (result: LabResult, runId?: string) => (
        <div className="bg-slate-800 rounded-lg p-4 my-3 border border-slate-600">
            <div className="flex items-center justify-between mb-3">
                <div className="text-sm font-bold text-blue-400">{result.strategy}</div>
                {runId && onLoadRun && (
                    <button
                        onClick={() => onLoadRun(runId)}
                        className="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1.5 rounded transition"
                    >
                        📊 Visualize
                    </button>
                )}
            </div>
            <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                    <div className="text-2xl font-bold text-white">{result.trades}</div>
                    <div className="text-xs text-slate-400">Trades</div>
                </div>
                <div>
                    <div className={`text-2xl font-bold ${result.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}`}>
                        {(result.win_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-slate-400">Win Rate</div>
                </div>
                <div>
                    <div className={`text-2xl font-bold ${result.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${result.total_pnl.toLocaleString()}
                    </div>
                    <div className="text-xs text-slate-400">P&L</div>
                </div>
            </div>

            <div className="mt-4">
                <div className="flex h-3 rounded overflow-hidden">
                    <div className="bg-green-500" style={{ width: `${result.win_rate * 100}%` }} />
                    <div className="bg-red-500" style={{ width: `${(1 - result.win_rate) * 100}%` }} />
                </div>
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>{result.wins} Wins</span>
                    <span>{result.losses} Losses</span>
                </div>
            </div>

            {result.equity_curve && result.equity_curve.length > 0 && (
                <div className="mt-4">
                    <div className="text-xs text-slate-400 mb-2">Equity Curve</div>
                    <div className="h-16 flex items-end gap-px">
                        {result.equity_curve.slice(-50).map((val, idx) => {
                            const min = Math.min(...result.equity_curve!.slice(-50));
                            const max = Math.max(...result.equity_curve!.slice(-50));
                            const height = max > min ? ((val - min) / (max - min)) * 100 : 50;
                            return (
                                <div
                                    key={idx}
                                    className={`flex-1 ${val >= result.equity_curve![0] ? 'bg-green-500' : 'bg-red-500'}`}
                                    style={{ height: `${Math.max(5, height)}%` }}
                                />
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );

    const renderMessage = (msg: Message, idx: number) => {
        if (msg.role === 'user') {
            return (
                <div key={idx} className="flex justify-end">
                    <div className="max-w-[80%] bg-blue-600 text-white rounded-xl px-4 py-2">
                        {msg.content}
                    </div>
                </div>
            );
        }

        return (
            <div key={idx} className="flex justify-start">
                <div className="max-w-[90%]">
                    <div className="bg-slate-700 text-slate-100 rounded-xl px-4 py-3">
                        <div className="prose prose-sm prose-invert max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-2 prose-headings:text-blue-400 prose-code:bg-slate-600 prose-code:px-1 prose-code:rounded prose-pre:bg-slate-800 prose-pre:border prose-pre:border-slate-600 prose-strong:text-white prose-table:border-collapse prose-th:bg-slate-700 prose-th:border prose-th:border-slate-600 prose-th:px-3 prose-th:py-2 prose-td:border prose-td:border-slate-600 prose-td:px-3 prose-td:py-2 prose-tr:even:bg-slate-800/50">
                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                    </div>
                    {msg.data?.result && renderResultTable(msg.data.result, msg.run_id)}
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 overflow-hidden">
            {/* Header */}
            <div className="h-14 flex items-center justify-between px-6 border-b border-slate-700 bg-slate-800 shrink-0">
                <div className="flex items-center gap-3">
                    <span className="text-2xl">🔬</span>
                    <h1 className="text-xl font-bold text-white">Research Lab</h1>
                </div>
                <div className="flex items-center gap-4">
                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer hover:text-purple-400 transition-colors">
                        <input
                            type="checkbox"
                            checked={plannerMode}
                            onChange={(e) => setPlannerMode(e.target.checked)}
                            className="w-4 h-4 rounded accent-purple-500"
                        />
                        <span>🗓️ Planner Mode</span>
                    </label>
                    <span className="text-sm text-slate-500">AI-Powered Strategy Research</span>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex flex-1 overflow-hidden min-h-0">

                {/* Left Sidebar - Current Result & Commands */}
                <div className="w-80 border-r border-slate-700 bg-slate-800 p-4 overflow-y-auto shrink-0 flex flex-col">
                    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">
                        Latest Result
                    </h2>

                    {currentResult ? (
                        renderResultTable(currentResult)
                    ) : (
                        <div className="text-slate-500 text-sm text-center py-8 border border-dashed border-slate-700 rounded">
                            Run a strategy to see results here
                        </div>
                    )}

                    <div className="mt-6">
                        <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">
                            Quick Commands
                        </h2>
                        <div className="space-y-2 text-xs">
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Run EMA cross scan")}>
                                <code>"Run EMA cross scan"</code>
                            </div>
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Test lunch hour fade")}>
                                <code>"Test lunch hour fade"</code>
                            </div>
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Train LSTM on bounce data")}>
                                <code>"Train LSTM on bounce data"</code>
                            </div>
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Compare ORB vs MR strategy")}>
                                <code>"Compare ORB vs MR strategy"</code>
                            </div>
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Show experiment history")}>
                                <code>"Show experiment history"</code>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Chat Area (Right) */}
                <div className="flex-1 flex flex-col min-w-0 bg-slate-900">
                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-6 space-y-4" ref={scrollRef}>
                        {messages.map((msg, idx) => renderMessage(msg, idx))}
                        {loading && (
                            <div className="flex justify-start">
                                <div className="bg-slate-700 text-slate-300 rounded-xl px-4 py-3 animate-pulse">
                                    <span className="text-blue-400">Agent is thinking...</span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Quick Actions */}
                    <div className="px-6 py-3 border-t border-slate-700 bg-slate-800 shrink-0">
                        <div className="flex gap-2 flex-wrap">
                            {quickActions.map((action, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => sendQuickAction(action.prompt)}
                                    className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs px-3 py-1.5 rounded-full transition"
                                >
                                    {action.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Input */}
                    <form onSubmit={handleSubmit} className="p-4 border-t border-slate-700 bg-slate-800 shrink-0">
                        <div className="flex gap-3">
                            <input
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                placeholder="Ask me to run a strategy, test a theory, or analyze results..."
                                className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                            />
                            <button
                                type="submit"
                                disabled={loading}
                                className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-3 font-bold disabled:opacity-50"
                            >
                                Send
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default LabPage;
