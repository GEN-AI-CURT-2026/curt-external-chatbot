import streamlit as st
import time
import sys
import os

# --------------------------------------------------------------------------
# FIX: Add the 'backend' folder to Python's search path
# This allows 'rag_pipeline.py' to find 'prompts.py' without code changes
# --------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "..", "backend")
sys.path.append(backend_dir)

# Import directly (now that backend is in the path)
from rag_pipeline import CURTRagPipeline
from memory_manager import MemoryManager

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CURT AI Assistant",
    page_icon="🏎️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
# INITIALIZE BACKEND (Cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def initialize_pipeline():
    """Initialize RAG pipeline (cached to avoid reloading)."""
    return CURTRagPipeline()

@st.cache_resource
def initialize_memory_manager():
    """Initialize memory manager (cached, singleton per session)."""
    return MemoryManager(window_size=5)

# Load resources
try:
    pipeline = initialize_pipeline()
    memory_mgr = initialize_memory_manager()
except Exception as e:
    st.error(f"❌ Failed to initialize system: {e}")
    st.info("💡 Make sure ChromaDB is built. Run: `python backend/build_chroma.py`")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #E31B23;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Source badge */
    .source-badge {
        background-color: #f0f2f6;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem;
        display: inline-block;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("<h1 class='main-title'>🏎️ CURT AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your guide to Cairo University Racing Team</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR (Optional Information)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **CURT AI Assistant** helps you learn about:
    - Team structure & members
    - Competition history & achievements
    - Current projects
    - How to join CURT
    
    Powered by RAG + LangChain + GPT-3.5
    """)
    
    # Memory stats (for debugging)
    if st.checkbox("Show Memory Stats"):
        stats = memory_mgr.get_stats(st.session_state.messages)
        st.json(stats)
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []

        st.rerun()

# ═══════════════════════════════════════════════════════════════
# DISPLAY CHAT HISTORY
# ═══════════════════════════════════════════════════════════════
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ═══════════════════════════════════════════════════════════════
# WELCOME MESSAGE (First Time)
# ═══════════════════════════════════════════════════════════════
if not st.session_state.conversation_started and len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Welcome to CURT!**
        
        I'm here to answer your questions about the Cairo University Racing Team. 
        
        Try asking me:
        - "What is CURT?"
        - "Tell me about CURT's achievements"
        - "How can I join the team?"
        - "What competitions does CURT participate in?"
        """)

# ═══════════════════════════════════════════════════════════════
# MAIN CHAT INPUT & PROCESSING
# ═══════════════════════════════════════════════════════════════
if prompt := st.chat_input("Ask me anything about CURT..."):
    
    # Mark conversation as started
    st.session_state.conversation_started = True
    
    # 1️⃣ ADD USER MESSAGE
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2️⃣ GET RELEVANT MEMORY (Last 5 message pairs = 10 messages)
    relevant_history = memory_mgr.get_recent_history(st.session_state.messages)
    
    # 3️⃣ GENERATE AI RESPONSE
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Call RAG pipeline with memory context
                response_data = pipeline.run(prompt, chat_history=relevant_history)
                
                ai_answer = response_data.get("answer", "I apologize, but I couldn't generate a response.")
                
                # Simulate streaming effect (character by character looks smoother)
                for char in ai_answer:
                    full_response += char
                    time.sleep(0.01)  # Adjust speed here
                    message_placeholder.markdown(full_response + "▌")
                
                # Final display without cursor
                message_placeholder.markdown(full_response)
                
                # 4️⃣ DISPLAY SOURCES (If Available)
                sources = response_data.get("sources", [])
                if sources and len(sources) > 0:
                    with st.expander("📚 View Sources", expanded=False):
                        st.caption("Information retrieved from:")
                        
                        seen_sources = set()
                        for doc in sources:
                            source = doc.metadata.get('source', 'Unknown')
                            source_name = os.path.basename(source)
                            
                            if source_name not in seen_sources:
                                st.markdown(f"- `{source_name}`")
                                seen_sources.add(source_name)
                
             
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                full_response = "I encountered an error processing your request. Please try again."
                message_placeholder.markdown(full_response)
    
    # 6️⃣ SAVE AI RESPONSE TO SESSION STATE
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.caption("🏎️ Built by CURT Gen AI Team | Powered by LangChain & OpenAI")