import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api/client';
import { ChatMessage, UIAction } from '../types/viz';

interface ChatAgentProps {
  runId: string;
  currentIndex: number;
  currentMode: 'DECISION' | 'TRADE';
  onAction: (action: UIAction) => void;
}

export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, currentMode, onAction }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hello! I am the **Trade Viz Agent**. How can I help with your analysis today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !runId) return;

    const userMsg: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await api.postAgent([...messages, userMsg], { runId, currentIndex, currentMode });

      setMessages(prev => [...prev, { role: 'assistant', content: response.reply }]);

      if (response.ui_action) {
        onAction(response.ui_action);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Error contacting agent." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 font-sans">
      <div className="px-4 py-3 bg-slate-900/50 backdrop-blur-sm border-b border-slate-800 flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest">Agent Terminal</h3>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar bg-slate-950/50" ref={scrollRef}>
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-md ${m.role === 'user'
              ? 'bg-blue-600 text-white rounded-br-none'
              : 'bg-slate-800 text-slate-200 border border-slate-700/50 rounded-bl-none'
              }`}>
              {m.role === 'assistant' ? (
                <div className="prose prose-sm prose-invert max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-2 prose-code:bg-slate-700 prose-code:px-1 prose-code:rounded prose-pre:bg-slate-900 prose-pre:border prose-pre:border-slate-700">
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-slate-500 ml-4 animate-pulse">
            <span className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce"></span>
            <span className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce delay-75"></span>
            <span className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce delay-150"></span>
            Agent is thinking...
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="p-3 bg-slate-900 border-t border-slate-800 flex gap-3 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)]">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about market structure, setup validation, or strategy..."
          className="flex-1 bg-slate-800/50 border border-slate-700 rounded-full px-5 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all shadow-inner"
        />
        <button
          type="submit"
          disabled={loading || !runId}
          className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-full px-6 py-2.5 text-sm font-bold shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:translate-y-[-1px] active:translate-y-[1px]"
        >
          Send
        </button>
      </form>
    </div>
  );
};
