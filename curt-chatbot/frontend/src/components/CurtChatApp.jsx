import React, { useState, useRef, useEffect } from 'react';

export default function CurtChatApp() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    
    const query = input.trim();
    if (!query || isLoading) return;

    setInput('');
    setIsLoading(true);

    const updatedHistory = [...messages, { role: 'user', content: query }];
    setMessages(updatedHistory);

    try {
      const apiHistory = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query,
          history: apiHistory
        }),
      });

      if (!response.ok) throw new Error('Network response failure');
      const data = await response.json();

      let cleanAnswer = data.answer;
      
      cleanAnswer = cleanAnswer.replace(/\*System Note:.*?\*/gi, '');
      
      if (cleanAnswer.includes('**Sources:**')) {
        cleanAnswer = cleanAnswer.split(/\*\*Sources:\*\*/i)[0];
      } else if (cleanAnswer.includes('Sources:')) {
        cleanAnswer = cleanAnswer.split(/Sources:/i)[0];
      }
      
      cleanAnswer = cleanAnswer.trim();

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: cleanAnswer
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: "Connection Error: Could not reach the AI core."
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`flex h-screen w-screen font-sans antialiased justify-center items-center transition-colors duration-300 ${
      darkMode ? 'bg-[#0b0f19] text-slate-100' : 'bg-slate-100 text-slate-800'
    }`}>
      
      <div className={`flex flex-col h-full w-full max-w-3xl shadow-2xl relative transition-colors duration-300 ${
        darkMode ? 'bg-[#111827]/40' : 'bg-white border-x border-slate-200'
      }`}>
        
        <header className={`flex items-center justify-between p-4 border-b backdrop-blur-md transition-colors duration-300 ${
          darkMode ? 'border-slate-800/60 bg-[#111827]/80' : 'border-slate-200 bg-white/90'
        }`}>
          <div className="flex items-center space-x-3">
            <div className="w-2.5 h-2.5 rounded-full bg-red-600 animate-pulse" />
            <h1 className={`text-sm font-bold tracking-wider uppercase ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>
              CURT Assistant
            </h1>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-1.5 rounded-lg border text-xs font-medium transition cursor-pointer ${
                darkMode 
                  ? 'border-slate-700 bg-slate-800 text-amber-400 hover:bg-slate-700' 
                  : 'border-slate-300 bg-slate-50 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {darkMode ? "☀️ Light Mode" : "🌙 Dark Mode"}
            </button>

            <button 
              onClick={() => setMessages([])}
              className={`text-[11px] font-medium transition ${darkMode ? 'text-slate-400 hover:text-red-400' : 'text-slate-500 hover:text-red-600'}`}
            >
              Clear Chat
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center max-w-sm mx-auto space-y-4">
              <span className="text-4xl">🏎️</span>
              <h2 className={`text-lg font-semibold ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>
                How can I help you today?
              </h2>
              <p className={`text-xs leading-relaxed ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                Ask about Cairo University Racing Team vehicle builds, manufacturing timelines, or subsystem structures.
              </p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-md ${
                msg.role === 'user' 
                  ? 'bg-red-600 text-white rounded-tr-none font-medium' 
                  : darkMode 
                    ? 'bg-[#1e293b]/60 border border-slate-800/40 text-slate-200 rounded-tl-none' 
                    : 'bg-slate-50 border border-slate-200 text-slate-800 rounded-tl-none'
              }`}>
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className={`rounded-2xl rounded-tl-none px-4 py-3 border ${
                darkMode ? 'bg-[#1e293b]/40 border-slate-800/40' : 'bg-slate-50 border-slate-200'
              }`}>
                <div className="flex space-x-1 items-center h-3">
                  <div className="h-1.5 w-1.5 bg-red-600 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                  <div className="h-1.5 w-1.5 bg-red-600 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                  <div className="h-1.5 w-1.5 bg-red-600 rounded-full animate-bounce"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className={`p-4 border-t backdrop-blur-md transition-colors duration-300 ${
          darkMode ? 'bg-[#111827]/60 border-slate-800/50' : 'bg-slate-50/80 border-slate-200'
        }`}>
          <form 
            onSubmit={handleSendMessage} 
            className={`flex items-center space-x-2 border rounded-xl px-4 py-2.5 transition-all ${
              darkMode 
                ? 'bg-[#0b0f19] border-slate-800 focus-within:border-red-500/50' 
                : 'bg-white border-slate-300 focus-within:border-red-500 focus-within:ring-1 focus-within:ring-red-500'
            }`}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your question..."
              className={`flex-1 bg-transparent border-none text-sm focus:outline-none focus:ring-0 ${
                darkMode ? 'text-slate-200 placeholder-slate-500' : 'text-slate-800 placeholder-slate-400'
              }`}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className={`p-1.5 transition cursor-pointer ${
                isLoading || !input.trim()
                  ? 'text-slate-300 dark:text-slate-700 cursor-not-allowed'
                  : 'text-slate-500 hover:text-red-600 dark:text-slate-400 dark:hover:text-red-500'
              }`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" className="w-5 h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            </button>
          </form>
        </div>

      </div>
    </div>
  );
}