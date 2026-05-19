# LectureRAG — RAG Pipeline from Scratch

A complete Retrieval-Augmented Generation (RAG) pipeline built from first principles — no pre-built abstractions, every stage implemented and visible. Purpose: deeply understand how production RAG systems work under the hood before using higher-level frameworks.

---

## What it does

Takes a document (PDF or text), ingests it, chunks it, embeds it, stores it in a vector database, then answers questions over it using a locally-running LLM — grounding every response in retrieved context rather than relying on model parametric memory.

---

## Pipeline stages

```
Document input (PDF / text)
        │
        ▼
  1. Ingestion          pypdf — extract raw text from PDF
        │
        ▼
  2. Chunking           LangChain text splitters — recursive character splitting
        │               with configurable chunk size and overlap
        ▼
  3. Embedding          sentence-transformers (Hugging Face)
        │               — local dense vector embeddings, no API calls
        ▼
  4. Vector store       ChromaDB — persistent local vector database
        │               — cosine similarity indexing
        ▼
  5. Retrieval          LangChain retriever — top-k semantic search
        │               against the ChromaDB store
        ▼
  6. Generation         Ollama (local LLM) via LangChain-Ollama
        │               — context-injected prompt, grounded response
        ▼
  Answer with source context
```

---

## Why each choice was made

| Component | Choice | Reason |
|---|---|---|
| Embeddings | `sentence-transformers` | Runs fully locally, no API cost, high quality semantic representations |
| Vector DB | ChromaDB | Simple persistent local store, easy to inspect internals |
| LLM | Ollama | Local inference — understand the full pipeline without API dependency |
| Orchestration | LangChain + LangGraph | Industry standard; LangGraph for explicit step control |
| PDF parsing | pypdf | Lightweight, no external dependencies |

---

## Tech stack

- **Python** — core language
- **PyTorch** — underlying tensor operations for embedding model
- **Hugging Face** (`transformers`, `huggingface_hub`, `sentence-transformers`) — embedding models
- **LangChain** (`langchain-core`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`, `langchain-ollama`, `langchain-text-splitters`) — RAG orchestration
- **LangGraph** (`langgraph`, `langgraph-prebuilt`) — pipeline step graph
- **ChromaDB** — vector store
- **Ollama** — local LLM inference
- **pypdf** — document ingestion
- **scikit-learn** — similarity utilities
- **numpy / scipy** — numerical operations

---

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running

```bash
git clone https://github.com/mahmud60/scratch-rag
cd scratch-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Pull a local model via Ollama (e.g. Llama 3):
```bash
ollama pull llama3
```

Run the pipeline:
```bash
python rag.py
```

---

## What I learned building this

- How chunking strategy (size, overlap, method) directly impacts retrieval quality
- Why embedding model choice matters more than vector DB choice for most use cases
- How context window limits force trade-offs between retrieved chunk count and chunk size
- Where LangGraph's explicit step graph beats a simple LangChain chain — and when it doesn't
- How `sentence-transformers` wraps Hugging Face `transformers` and what happens at the token level

---

## Related project

**[ielts-anywhere-backend](https://github.com/mahmud60/ielts-anywhere-backend)** — production FastAPI backend applying LLM integration (Claude Haiku, Celery, Redis, GCP) at scale.

---

## Author

Mahmudul Hasan — [LinkedIn](https://linkedin.com/in/mahmudhasan60) · [GitHub](https://github.com/mahmud60)

MSc Data Science, FAU Erlangen-Nuremberg
