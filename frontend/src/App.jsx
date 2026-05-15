import { useState, useRef, useEffect } from 'react';
import {
  Send, Menu, Settings, User, Sparkles, Plus,
  TrendingUp, Newspaper, Users, Building2, ChevronRight, X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// ── Markdown renderer ─────────────────────────────────────────────────────────
function formatText(text) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let listBuffer = [];

  const flushList = (key) => {
    if (listBuffer.length > 0) {
      elements.push(
        <ul key={`ul-${key}`} className="lodestone-list">
          {listBuffer.map((item, i) => (
            <li key={i}>{renderInline(item)}</li>
          ))}
        </ul>
      );
      listBuffer = [];
    }
  };

  lines.forEach((line, i) => {
    if (line.startsWith('## ')) {
      flushList(i);
      elements.push(<h2 key={i} className="lodestone-h2">{line.slice(3)}</h2>);
    } else if (line.startsWith('### ')) {
      flushList(i);
      elements.push(<h3 key={i} className="lodestone-h3">{line.slice(4)}</h3>);
    } else if (line.startsWith('* ') || line.startsWith('- ')) {
      listBuffer.push(line.slice(2));
    } else if (line.trim() === '') {
      flushList(i);
      elements.push(<div key={i} className="lodestone-spacer" />);
    } else {
      flushList(i);
      elements.push(<p key={i} className="lodestone-p">{renderInline(line)}</p>);
    }
  });
  flushList('end');
  return elements;
}

function renderInline(text) {
  // Handle **bold** and *italic*
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**'))
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith('*') && part.endsWith('*'))
      return <em key={i}>{part.slice(1, -1)}</em>;
    return part;
  });
}

// ── Quick-start prompts ───────────────────────────────────────────────────────
const STARTERS = [
  { icon: TrendingUp,  label: 'Financials',  prompt: "What are Apple's latest financial results?",      color: '#d1fae5', iconColor: '#059669' },
  { icon: Newspaper,   label: 'Latest News', prompt: 'Give me the latest news on Nvidia',               color: '#dbeafe', iconColor: '#2563eb' },
  { icon: Users,       label: 'Leadership',  prompt: "Who runs Microsoft and what's their background?", color: '#fef3c7', iconColor: '#d97706' },
  { icon: Building2,   label: 'Competitors', prompt: 'Who are the main competitors of Tesla?',          color: '#fce7f3', iconColor: '#db2777' },
];

// ── Recent sessions (static demo) ────────────────────────────────────────────
const RECENTS = [
  { label: 'Apple Financials Check', active: true },
  { label: 'Tesla Revenue 2025',     active: false },
  { label: 'Microsoft Leadership',   active: false },
];

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages]   = useState([{ id: 1, role: 'assistant', isIntro: true }]);
  const [input, setInput]         = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (text) => {
    const msg = (text ?? input).trim();
    if (!msg) return;

    setMessages(prev => [...prev, { id: Date.now(), role: 'user', content: msg }]);
    setInput('');
    setIsLoading(true);

    try {
      const res  = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, session_id: sessionId }),
      });
      const data = await res.json();
      setSessionId(data.session_id);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        followUps: data.follow_ups,
        status: data.status,
      }]);
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: "Connection error — is the LODESTONE backend running?",
        isError: true,
      }]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  return (
    <>
      <style>{STYLES}</style>

      <div className="ls-shell">

        {/* ── Sidebar ── */}
        <AnimatePresence>
          {(sidebarOpen || window.innerWidth >= 768) && (
            <motion.aside
              className="ls-sidebar"
              initial={{ x: -280 }} animate={{ x: 0 }} exit={{ x: -280 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            >
              {/* Logo */}
              <div className="ls-logo-row">
                <div className="ls-logo-icon">
                  <Sparkles size={18} color="#fff" />
                </div>
                <span className="ls-logo-text">LODESTONE</span>
                <button className="ls-close-btn md-hidden" onClick={() => setSidebarOpen(false)}>
                  <X size={16} />
                </button>
              </div>

              {/* New Chat */}
              <button className="ls-new-chat" onClick={() => {
                setMessages([{ id: 1, role: 'assistant', isIntro: true }]);
                setSessionId(null);
                setSidebarOpen(false);
              }}>
                <Plus size={15} strokeWidth={2.5} />
                New Chat
              </button>

              {/* Recents */}
              <p className="ls-section-label">Recent Research</p>
              <div className="ls-recents">
                {RECENTS.map(r => (
                  <button key={r.label} className={`ls-recent-item ${r.active ? 'ls-recent-active' : ''}`}>
                    {r.label}
                  </button>
                ))}
              </div>

              {/* User */}
              <div className="ls-user-row">
                <div className="ls-user-avatar"><User size={16} color="#059669" /></div>
                <div className="ls-user-info">
                  <span className="ls-user-name">User</span>
                  <span className="ls-user-plan">Pro Plan</span>
                </div>
                <Settings size={15} color="#94a3b8" className="ls-settings-icon" />
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* ── Main ── */}
        <main className="ls-main">
          <div className="ls-card">

            {/* Header */}
            <header className="ls-header">
              <div className="ls-header-left">
                <button className="ls-hamburger" onClick={() => setSidebarOpen(o => !o)}>
                  <Menu size={18} />
                </button>
                <div className="ls-header-logo">
                  <div className="ls-logo-icon ls-logo-icon--sm">
                    <Sparkles size={13} color="#fff" />
                  </div>
                  <span className="ls-logo-text">LODESTONE</span>
                </div>
              </div>
              <div className="ls-session-badge">
                SESSION {(sessionId || 'NEW').slice(0,6).toUpperCase()}
              </div>
            </header>

            {/* Messages */}
            <div className="ls-messages">
              <AnimatePresence initial={false}>
                {messages.map(msg => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.28, ease: 'easeOut' }}
                    className={`ls-msg-row ${msg.role === 'user' ? 'ls-msg-row--user' : ''}`}
                  >
                    {/* Avatar */}
                    {msg.role === 'assistant' && (
                      <div className="ls-avatar ls-avatar--bot">
                        <Sparkles size={15} color="#fff" />
                      </div>
                    )}
                    {msg.role === 'user' && (
                      <div className="ls-avatar ls-avatar--user">
                        <User size={15} color="#64748b" />
                      </div>
                    )}

                    {/* Bubble */}
                    <div className={`ls-bubble ${msg.role === 'user' ? 'ls-bubble--user' : 'ls-bubble--bot'} ${msg.isError ? 'ls-bubble--error' : ''}`}>

                      {msg.isIntro ? (
                        <IntroCard onPrompt={p => { setInput(p); handleSend(p); }} />
                      ) : (
                        <div className="ls-content">
                          {formatText(msg.content)}
                        </div>
                      )}

                      {/* Follow-ups */}
                      {msg.followUps?.length > 0 && (
                        <div className="ls-followups">
                          <p className="ls-followups-label">Suggested follow-ups</p>
                          {msg.followUps.map((q, i) => (
                            <button key={i} className="ls-followup-btn" onClick={() => { setInput(q); handleSend(q); }}>
                              <span>{q}</span>
                              <ChevronRight size={14} />
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {/* Typing indicator */}
              {isLoading && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="ls-msg-row">
                  <div className="ls-avatar ls-avatar--bot">
                    <Sparkles size={15} color="#fff" className="ls-pulse" />
                  </div>
                  <div className="ls-bubble ls-bubble--bot ls-typing">
                    <span /><span /><span />
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="ls-input-area">
              <div className="ls-input-wrap">
                <input
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
                  placeholder="Ask me anything ..."
                  disabled={isLoading}
                  className="ls-input"
                />
                <button
                  onClick={() => handleSend()}
                  disabled={!input.trim() || isLoading}
                  className="ls-send-btn"
                >
                  <Send size={16} strokeWidth={2} />
                </button>
              </div>
            </div>

          </div>
        </main>
      </div>
    </>
  );
}

// ── Intro card ────────────────────────────────────────────────────────────────
function IntroCard({ onPrompt }) {
  return (
    <div className="ls-intro">
      <div className="ls-intro-heading">
        <p className="ls-intro-hi">Hi, there!</p>
        <p className="ls-intro-sub">How can I help you?</p>
      </div>
      <div className="ls-starters">
        {STARTERS.map(({ icon: Icon, label, prompt, color, iconColor }) => (
          <button key={label} className="ls-starter" onClick={() => onPrompt(prompt)}>
            <div className="ls-starter-icon" style={{ background: color }}>
              <Icon size={22} color={iconColor} strokeWidth={1.8} />
            </div>
            <span className="ls-starter-label">{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Instrument+Serif:ital@0;1&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: #f4f7f2;
    min-height: 100vh;
    color: #1e293b;
  }

  /* ── Shell ── */
  .ls-shell {
    display: flex;
    height: 100vh;
    overflow: hidden;
    position: relative;
  }

  /* ── Sidebar ── */
  .ls-sidebar {
    width: 256px;
    flex-shrink: 0;
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.6);
    display: flex;
    flex-direction: column;
    padding: 24px 16px;
    gap: 0;
    box-shadow: 4px 0 24px rgba(0,0,0,0.04);
    z-index: 30;
  }

  .ls-logo-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 24px;
    padding: 0 4px;
  }
  .ls-logo-icon {
    width: 34px; height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, #15803d, #34d399);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(52,211,153,0.3);
    flex-shrink: 0;
  }
  .ls-logo-icon--sm { width: 26px; height: 26px; border-radius: 8px; }
  .ls-logo-text {
    font-family: 'DM Sans', sans-serif;
    font-weight: 800;
    font-size: 17px;
    letter-spacing: -0.3px;
    color: #166534;
  }
  .ls-close-btn {
    margin-left: auto;
    background: none; border: none; cursor: pointer;
    color: #94a3b8; padding: 4px;
  }
  .md-hidden { display: none; }

  .ls-new-chat {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px;
    border-radius: 12px;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    font-size: 13.5px; font-weight: 600;
    color: #374151; cursor: pointer;
    transition: background 0.15s, box-shadow 0.15s;
    margin-bottom: 28px;
  }
  .ls-new-chat:hover { background: rgba(255,255,255,0.95); box-shadow: 0 4px 14px rgba(0,0,0,0.08); }

  .ls-section-label {
    font-size: 10.5px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: #94a3b8; padding: 0 6px; margin-bottom: 10px;
  }

  .ls-recents { display: flex; flex-direction: column; gap: 2px; flex: 1; }
  .ls-recent-item {
    padding: 9px 12px; border-radius: 10px;
    font-size: 13px; font-weight: 500; color: #64748b;
    text-align: left; background: none; border: none; cursor: pointer;
    transition: background 0.12s, color 0.12s;
  }
  .ls-recent-item:hover { background: rgba(255,255,255,0.6); color: #374151; }
  .ls-recent-active { background: rgba(255,255,255,0.75) !important; color: #166534 !important; font-weight: 600; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }

  .ls-user-row {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px;
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 14px;
    margin-top: auto;
  }
  .ls-user-avatar {
    width: 34px; height: 34px; border-radius: 10px;
    background: #d1fae5; display: flex; align-items: center; justify-content: center;
    border: 1px solid #a7f3d0; flex-shrink: 0;
  }
  .ls-user-info { display: flex; flex-direction: column; flex: 1; }
  .ls-user-name { font-size: 13px; font-weight: 700; color: #374151; }
  .ls-user-plan { font-size: 11px; color: #059669; font-weight: 600; }
  .ls-settings-icon { cursor: pointer; }

  /* ── Main ── */
  .ls-main {
    flex: 1; display: flex; flex-direction: column;
    padding: 16px; overflow: hidden;
    background: #f4f7f2;
  }

  .ls-card {
    flex: 1; display: flex; flex-direction: column;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(24px);
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.7);
    box-shadow: 0 8px 40px rgba(0,0,0,0.06), 0 1px 0 rgba(255,255,255,0.8) inset;
    overflow: hidden;
  }

  /* ── Header ── */
  .ls-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 28px;
    border-bottom: 1px solid rgba(255,255,255,0.5);
    background: rgba(255,255,255,0.3);
    flex-shrink: 0;
  }
  .ls-header-left { display: flex; align-items: center; gap: 12px; }
  .ls-hamburger {
    background: rgba(255,255,255,0.7); border: 1px solid rgba(255,255,255,0.8);
    border-radius: 10px; width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; color: #64748b;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .ls-header-logo { display: flex; align-items: center; gap: 8px; }
  .ls-session-badge {
    font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
    color: #059669; background: #d1fae5;
    padding: 5px 12px; border-radius: 999px;
    border: 1px solid #a7f3d0;
  }

  /* ── Messages ── */
  .ls-messages {
    flex: 1; overflow-y: auto;
    padding: 28px 24px;
    display: flex; flex-direction: column; gap: 20px;
    scroll-behavior: smooth;
  }
  .ls-messages::-webkit-scrollbar { width: 5px; }
  .ls-messages::-webkit-scrollbar-track { background: transparent; }
  .ls-messages::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.08); border-radius: 10px; }

  .ls-msg-row { display: flex; gap: 12px; align-items: flex-start; max-width: 860px; width: 100%; margin: 0 auto; }
  .ls-msg-row--user { flex-direction: row-reverse; }

  .ls-avatar {
    width: 34px; height: 34px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px;
  }
  .ls-avatar--bot {
    background: linear-gradient(135deg, #16a34a, #34d399);
    box-shadow: 0 4px 12px rgba(52,211,153,0.25);
  }
  .ls-avatar--user {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
  }

  /* ── Bubbles ── */
  .ls-bubble {
    border-radius: 20px; padding: 18px 22px;
    max-width: calc(100% - 46px);
    font-size: 15px; line-height: 1.65;
  }
  .ls-bubble--bot {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(255,255,255,0.85);
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border-top-left-radius: 4px;
    color: #1e293b;
  }
  .ls-bubble--user {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: #fff;
    border-top-right-radius: 4px;
    box-shadow: 0 4px 16px rgba(22,163,74,0.25);
  }
  .ls-bubble--error { background: #fef2f2 !important; border-color: #fecaca !important; color: #dc2626 !important; }

  /* ── Formatted content ── */
  .ls-content { display: flex; flex-direction: column; gap: 4px; }

  .ls-content .lodestone-h2 {
    font-family: 'Instrument Serif', serif;
    font-size: 19px; font-weight: 400;
    color: #166534; margin: 18px 0 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(22,101,52,0.12);
  }
  .ls-content .lodestone-h3 {
    font-size: 14px; font-weight: 700;
    color: #374151; margin: 12px 0 4px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .ls-content .lodestone-p { color: #374151; margin: 0; }
  .ls-content .lodestone-spacer { height: 6px; }
  .ls-content .lodestone-list {
    padding-left: 20px; display: flex; flex-direction: column; gap: 4px;
  }
  .ls-content .lodestone-list li {
    color: #374151; list-style: disc;
    marker-color: #34d399;
  }
  .ls-content .lodestone-list li::marker { color: #16a34a; }
  .ls-content strong { font-weight: 700; color: #166534; }
  .ls-content em { font-style: italic; color: #475569; }

  /* ── Typing indicator ── */
  .ls-typing {
    display: flex; gap: 5px; align-items: center;
    padding: 14px 18px !important;
  }
  .ls-typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #34d399; display: block;
    animation: lsBounce 1.1s ease-in-out infinite;
  }
  .ls-typing span:nth-child(2) { animation-delay: 0.15s; }
  .ls-typing span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes lsBounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
    30% { transform: translateY(-6px); opacity: 1; }
  }
  .ls-pulse { animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* ── Follow-ups ── */
  .ls-followups {
    margin-top: 20px; padding-top: 16px;
    border-top: 1px solid rgba(0,0,0,0.07);
    display: flex; flex-direction: column; gap: 8px;
  }
  .ls-followups-label {
    font-size: 10.5px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: #94a3b8; margin-bottom: 4px;
  }
  .ls-followup-btn {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 14px;
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 12px;
    font-size: 13.5px; font-weight: 500; color: #374151;
    text-align: left; cursor: pointer;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }
  .ls-followup-btn:hover {
    background: rgba(255,255,255,0.9);
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    transform: translateY(-1px);
    color: #166534;
  }
  .ls-followup-btn svg { flex-shrink: 0; color: #059669; opacity: 0; transition: opacity 0.15s; }
  .ls-followup-btn:hover svg { opacity: 1; }

  /* ── Intro card ── */
  .ls-intro { padding: 8px 4px; }
  .ls-intro-heading { margin-bottom: 28px; }
  .ls-intro-hi {
    font-family: 'Instrument Serif', serif;
    font-size: 32px; color: #1e293b; line-height: 1.1;
  }
  .ls-intro-sub {
    font-family: 'Instrument Serif', serif;
    font-size: 28px; color: #94a3b8; line-height: 1.2;
    font-style: italic;
  }

  .ls-starters {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;
  }
  .ls-starter {
    display: flex; flex-direction: column; align-items: flex-start; gap: 12px;
    padding: 18px 18px;
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 18px;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s, background 0.15s;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    text-align: left;
  }
  .ls-starter:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    background: rgba(255,255,255,0.9);
  }
  .ls-starter-icon {
    width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
  }
  .ls-starter-label {
    font-size: 13.5px; font-weight: 700; color: #374151;
  }

  /* ── Input ── */
  .ls-input-area {
    padding: 16px 20px 20px;
    border-top: 1px solid rgba(255,255,255,0.5);
    background: rgba(255,255,255,0.3);
    flex-shrink: 0;
  }
  .ls-input-wrap {
    display: flex; align-items: center; gap: 10px;
    max-width: 800px; margin: 0 auto;
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(255,255,255,0.95);
    border-radius: 999px;
    padding: 6px 8px 6px 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
  }
  .ls-input-wrap:focus-within {
    box-shadow: 0 4px 24px rgba(22,163,74,0.12), 0 0 0 2px rgba(52,211,153,0.2);
  }
  .ls-input {
    flex: 1; background: none; border: none; outline: none;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px; color: #1e293b;
  }
  .ls-input::placeholder { color: #94a3b8; }
  .ls-input:disabled { opacity: 0.5; }

  .ls-send-btn {
    width: 40px; height: 40px; border-radius: 50%;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: #fff; flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(22,163,74,0.3);
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
  }
  .ls-send-btn:hover:not(:disabled) { transform: scale(1.08); box-shadow: 0 4px 16px rgba(22,163,74,0.4); }
  .ls-send-btn:disabled { opacity: 0.35; cursor: not-allowed; }

  @media (max-width: 767px) {
    .ls-sidebar { position: fixed; top: 0; left: 0; height: 100%; }
    .md-hidden { display: flex !important; }
    .ls-main { padding: 10px; }
    .ls-card { border-radius: 20px; }
    .ls-starters { grid-template-columns: repeat(2, 1fr); }
    .ls-messages { padding: 16px 12px; }
    .ls-bubble { padding: 14px 16px; }
  }
`;