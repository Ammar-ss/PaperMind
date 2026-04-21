# 📚 PaperMind

> **An agentic AI assistant for researchers — answers questions about academic papers, research methodology, citations, peer review, and publication venues. Grounded. Honest. Never hallucinates.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-green?style=flat-square)](https://console.groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple?style=flat-square)](https://www.trychroma.com)

---

## What is PaperMind?

PhD students and academic researchers spend hours searching for reliable answers to recurring questions — how to read and evaluate a paper, what evaluation metrics mean, how citations work, which venues to target, how peer review operates. A simple web search returns contradictory or shallow results.

**PaperMind** is a stateful, self-reflecting agentic AI assistant that answers these questions faithfully from a curated research knowledge base. It remembers the conversation, knows when it doesn't know something, and never makes up information.

Built with **LangGraph**, **ChromaDB**, **Groq (llama-3.3-70b)**, and **Streamlit** as part of the Agentic AI Hands-On Course 2026.

---

## Demo

| Asking about evaluation metrics | Agent admitting it doesn't know | Multi-turn memory |
|---|---|---|
| Routes to KB → retrieves DSC, F1, BLEU | "I do not have that information..." | Remembers name and research field across turns |

---

## Features

- 🧠 **8-node LangGraph StateGraph** — memory → router → retrieve/tool/skip → answer → eval → save
- 📖 **12-document curated knowledge base** — one topic per document, embedded with SentenceTransformers
- 🔁 **Self-reflection eval loop** — every answer is scored for faithfulness (0.0–1.0) before reaching the user; retried automatically if below 0.70
- 💬 **Stateful multi-turn memory** — MemorySaver + thread_id persists user name and context across the full conversation
- 🛠️ **Tool use** — datetime and arithmetic calculator for non-KB queries
- 🚫 **Honest failure mode** — clearly admits when a question is out of scope instead of hallucinating
- ⚡ **Smart routing** — LLM-powered router distinguishes between retrieve / tool / memory-only queries

---

## Knowledge Base Topics

| # | Topic |
|---|---|
| 1 | How to Read a Research Paper (Three-Pass Method) |
| 2 | Understanding the Abstract |
| 3 | Introduction and Problem Statement Structure |
| 4 | Research Methodology Types in CS Papers |
| 5 | Evaluation Metrics (Accuracy, F1, Dice, BLEU, ROUGE) |
| 6 | Literature Review and Related Work |
| 7 | Citation Formats (IEEE, APA, ACM) |
| 8 | Academic Publication Venues and Impact Factor |
| 9 | Research Metrics (H-Index, Citation Count, CiteScore) |
| 10 | The Peer Review Process |
| 11 | IMRaD Paper Structure |
| 12 | Identifying Contributions, Novelty, and Limitations |

---

## Architecture

```
User Question
      ↓
[memory_node]     →  append to history, sliding window (last 6), extract name
      ↓
[router_node]     →  LLM prompt → retrieve / tool / memory_only
      ↓
[retrieval_node / tool_node / skip_node]
      ↓
[answer_node]     →  system prompt + context + history → grounded LLM response
      ↓
[eval_node]       →  faithfulness 0.0–1.0 → retry if < 0.70 (max 2 retries)
      ↓
[save_node]       →  append answer to messages → END
```

**State fields:** `question`, `messages`, `route`, `retrieved`, `sources`, `tool_result`, `answer`, `faithfulness`, `eval_retries`, `user_name`

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph, MemorySaver) |
| LLM | llama-3.3-70b-versatile via Groq API |
| Embedding Model | SentenceTransformer — all-MiniLM-L6-v2 |
| Vector Database | ChromaDB (in-memory) |
| Tool | Python datetime + arithmetic calculator |
| Evaluation | Manual LLM-based faithfulness scoring (RAGAS-style) |
| Deployment | Streamlit (st.cache_resource, st.session_state) |
| Language | Python 3.10+ |

---

## Evaluation Results

### RAGAS Baseline (Part 6)

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | **0.90** | > 0.80 ✅ |
| Answer Relevancy | **0.92** | > 0.85 ✅ |

### Test Results (Part 5 — 11 questions)

| # | Question | Route | Faithfulness | Result |
|---|---|---|---|---|
| 1 | Three-pass method | retrieve | 0.92 | ✅ PASS |
| 2 | What does an abstract contain | retrieve | 0.90 | ✅ PASS |
| 3 | Dice Similarity Coefficient | retrieve | 0.88 | ✅ PASS |
| 4 | What is the h-index? | retrieve | 0.91 | ✅ PASS |
| 5 | Peer review process | retrieve | 0.89 | ✅ PASS |
| 6 | IMRaD structure | retrieve | 0.93 | ✅ PASS |
| 7 | Journal vs Conference in CS | retrieve | 0.87 | ✅ PASS |
| 8 | IEEE citation format | retrieve | 0.90 | ✅ PASS |
| 9 | Today's date | tool | 1.00 | ✅ PASS |
| 10 | Boiling point of mercury *(out-of-scope)* | retrieve | 1.00 | ✅ PASS (admit) |
| 11 | False premise: abstract is last section | retrieve | 0.91 | ✅ PASS (corrected) |

**11 / 11 tests passed.**

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/papermind.git
cd papermind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create key.

### 4. Run the Streamlit app
```bash
streamlit run capstone_streamlit.py
```

Open `http://localhost:8501` in your browser. Paste your Groq API key in the sidebar. Start asking.

---

## Project Structure

```
papermind/
├── agent.py                  # Core agent: KB, State, all 8 nodes, graph assembly
├── capstone_streamlit.py     # Streamlit UI
├── day13_capstone.ipynb      # Full notebook: Parts 1–8 with tests and evaluation
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account → select this repo
4. Set **Main file path** to `capstone_streamlit.py`
5. Under **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
6. Click **Deploy** : you'll get a live public URL

---

## Future Improvements

- **Dynamic PDF ingestion** : upload your own papers, chunk and embed at runtime
- **Full RAGAS suite** : add context_precision and context_recall metrics
- **Citation auto-formatter** : input paper metadata, output IEEE/APA/ACM citation
- **Persistent ChromaDB** : replace in-memory DB with Pinecone or Weaviate for multi-user support
- **Multilingual support** : Hindi and regional languages for broader accessibility

---

## Course Information

Built as the Capstone Project for the **Agentic AI Hands-On Course 2026**  

**Author:** Ammar Bhilwarawala  
**Roll Number:** 2305279  
**Batch:** Agentic AI 2026  

---

## License

MIT License — free to use, modify, and distribute with attribution.
