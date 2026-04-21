"""
Research Paper Q&A — Streamlit UI
capstone_streamlit.py  |  Capstone Project  |  Agentic AI Course 2026
Run: streamlit run capstone_streamlit.py
"""

import os
import uuid
import streamlit as st
from agent import init_agent, DOCUMENTS

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="wide"
)

# ═════════════════════════════════════════════════════════════════════════════
# EXPENSIVE INIT — runs only once per session (cached)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_agent(groq_api_key: str):
    """Load LLM, embedder, ChromaDB and compile graph — cached across reruns."""
    return init_agent(groq_api_key)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

if "messages"   not in st.session_state:
    st.session_state.messages   = []
if "thread_id"  not in st.session_state:
    st.session_state.thread_id  = str(uuid.uuid4())
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📚 Research Paper Q&A")
    st.caption("Agentic AI Capstone 2026")
    st.divider()

    # API key input
    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )

    if groq_key and not st.session_state.agent_ready:
        with st.spinner("Loading agent…"):
            try:
                st.session_state.app = load_agent(groq_key)
                st.session_state.agent_ready = True
                st.success("Agent ready ✓")
            except Exception as e:
                st.error(f"Failed to load agent: {e}")

    st.divider()
    st.subheader("📖 What I know about")
    for doc in DOCUMENTS:
        st.markdown(f"- {doc['topic']}")

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}…`")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═════════════════════════════════════════════════════════════════════════════

st.title("📚 Research Paper Q&A Assistant")
st.caption(
    "Ask me anything about reading papers, citations, peer review, "
    "publication venues, research metrics, and more."
)

# ── Render existing messages ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────
if not st.session_state.agent_ready:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
else:
    if prompt := st.chat_input("Ask a question about research papers…"):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = st.session_state.app.invoke(
                        {"question": prompt},
                        config=config
                    )
                    answer = result.get("answer", "Sorry, I could not generate a response.")
                    route  = result.get("route", "?")
                    faith  = result.get("faithfulness", None)
                    srcs   = result.get("sources", [])

                    st.markdown(answer)

                    # Debug info in expander
                    with st.expander("🔍 Agent trace", expanded=False):
                        st.markdown(f"**Route:** `{route}`")
                        if faith is not None:
                            st.markdown(f"**Faithfulness score:** `{faith:.2f}`")
                        if srcs:
                            st.markdown("**Sources retrieved:**")
                            for s in srcs:
                                st.markdown(f"  - {s}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    err = f"⚠️ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
