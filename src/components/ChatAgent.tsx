import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api/client';
import { ChatMessage, UIAction } from '../types/viz';

interface ChatAgentProps {
  runId: string;
  currentIndex: number;
  currentMode: 'DECISION' | 'TRADE';
  fastVizMode?: boolean;
  onAction: (action: UIAction) => void;
  onTextResponse?: () => void;
}

export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, currentMode, fastVizMode = false, onAction, onTextResponse }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hello! I am the **Trade Viz Agent**. How can I help with your analysis today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;  // Allow chat without run selected

    const userMsg: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await api.postAgent([...messages, userMsg], { runId, currentIndex, currentMode, fastVizMode });

      setMessages(prev => [...prev, { role: 'assistant', content: response.reply }]);

      if (response.ui_action) {
        onAction(response.ui_action);
      } else {
        // Text-only response (likely research result), expand chat
        if (onTextResponse) {
          onTextResponse();
        }
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Error contacting agent." }]);
    } finally {
      setLoading(false);
      // Focus back on input after response
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 font-sans relative group">
      {/* Header */}
      <div className="px-4 py-3 bg-slate-950 border-b border-slate-800 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="relative">
            <div className={`w-2 h-2 rounded-full ${loading ? 'bg-amber-400' : 'bg-emerald-500'} animate-pulse`}></div>
            <div className={`absolute inset-0 w-2 h-2 rounded-full ${loading ? 'bg-amber-400' : 'bg-emerald-500'} animate-ping opacity-20`}></div>
          </div>
          <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest">Agent Terminal</h3>
        </div>
        <div className="text-[10px] text-slate-600 font-mono">
          {runId === 'none' ? 'DISCONNECTED' : 'ONLINE'}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar bg-slate-950" ref={scrollRef}>
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
            {m.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-xs text-white font-bold shrink-0 mr-3 shadow-lg mt-1">
                AI
              </div>
            )}

            <div className={`max-w-[85%] relative group-message ${m.role === 'user' ? 'items-end flex flex-col' : ''}`}>
              {m.role === 'user' && (
                <div className="text-[10px] text-slate-500 mb-1 mr-1 uppercase tracking-wider font-bold">You</div>
              )}

              <div className={`px-5 py-3.5 text-sm shadow-md transition-all ${m.role === 'user'
                ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm'
                : 'bg-slate-900 text-slate-300 border border-slate-800 rounded-2xl rounded-tl-sm'
                }`}>
                {m.role === 'assistant' ? (
                  <div className="prose prose-sm prose-invert max-w-none
                    prose-p:my-1 prose-p:leading-relaxed
                    prose-ul:my-2 prose-ul:pl-4
                    prose-li:my-0.5
                    prose-headings:text-slate-200 prose-headings:font-bold prose-headings:my-2
                    prose-code:bg-slate-950 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-blue-300 prose-code:font-mono prose-code:text-xs
                    prose-pre:bg-slate-950 prose-pre:border prose-pre:border-slate-800 prose-pre:p-3 prose-pre:rounded-lg
                    prose-strong:text-white prose-strong:font-bold">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">{m.content}</div>
                )}
              </div>
            </div>

            {m.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-xs text-slate-400 font-bold shrink-0 ml-3 shadow-lg mt-1 border border-slate-700">
                U
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex justify-start animate-fade-in">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-xs text-white font-bold shrink-0 mr-3 shadow-lg mt-1">
              AI
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"></span>
              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-75"></span>
              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-150"></span>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-slate-950 border-t border-slate-800 shrink-0">
        <form onSubmit={handleSubmit} className="relative flex items-center gap-2 max-w-4xl mx-auto w-full">
          <div className="relative flex-1">
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder={runId === 'none' ? "Select a run to start chatting..." : "Ask for analysis, valid setups, or strategy insights..."}
              disabled={runId === 'none' || loading}
              className="w-full bg-slate-900 border border-slate-800 text-slate-200 placeholder-slate-600 rounded-xl px-4 py-3.5 pl-5 pr-12 text-sm focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all shadow-inner disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-slate-600 font-mono hidden md:block border border-slate-800 px-1.5 py-0.5 rounded">
              ↵ Enter
            </div>
          </div>
          <button
            type="submit"
            disabled={loading || !runId || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 text-white rounded-xl p-3.5 shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95 flex items-center justify-center aspect-square"
          >
            <svg className="w-5 h-5 translate-x-0.5 -translate-y-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </form>
        <div className="text-center mt-2">
          <p className="text-[10px] text-slate-600">AI can make mistakes. Verify important trading decisions.</p>
        </div>
      </div>
    </div>
  );
};
