# harness_agent
# Three-Stage AI Agent

A production-grade agentic pipeline demonstrating all three engineering stages:
**Prompt Engineering → Context Engineering → Harness Engineering**

---

## Project Structure

```
harness-agent/
├── agent.py          # Main pipeline (all three stages annotated)
├── requirements.txt
├── .env.example      # Copy to .env and add your key
└── data/             # Local knowledge base (JSON docs)
    ├── doc1.json     # Harness engineering concepts
    ├── doc2.json     # RAG and context engineering
    └── doc3.json     # Prompt engineering best practices
```

---

## Setup

### 1. Clone / copy the project

```bash
cd harness-agent
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
cp .env.example .env
# Then open .env and replace the placeholder with your real key:
# OPENAI_API_KEY=sk-...
```

---

## Run

```bash
python agent.py
```

You'll see a REPL prompt:

```
============================================================
  Three-Stage AI Agent  |  type 'exit' to quit
============================================================

You: 
```

---

## Example Queries to Test Each Stage

### Tests retrieval (Stage 2)
```
What is harness engineering?
What is the lost-in-the-middle problem?
How should I chunk documents for RAG?
```

### Tests calculator tool (Stage 3 — tool contracts)
```
What is 2 to the power of 10?
Calculate (128 * 3) + 47
```

### Tests multi-turn memory (State store)
```
[Turn 1] What is harness engineering?
[Turn 2] How does that relate to prompt engineering?
[Turn 3] What should I build first?
```

### Tests direct answer (no tool needed)
```
Explain what a state machine is in simple terms
```

---

## What the Logs Tell You

Every run prints stage-by-stage traces:

```
[planner]  Building execution plan...
[planner]  Plan: 1 step(s)
[tools]    Calling 'search_docs' with args: {'query': 'harness engineering'}
[reasoner] Synthesising answer...
[judge]    Verifying answer quality...
[judge]    Quality: ok — Answer is factually grounded and complete.
```

If the judge requests a revision:
```
[judge]    Quality: revise — Answer lacks specific examples.
[router]   Revision needed (attempt 1/2)
[revise]   Applying revision #1: Add concrete examples...
```

---

## Adding Your Own Documents

Drop any `.json` file into `data/` with this shape:

```json
{
  "id": "doc4",
  "title": "Your Document Title",
  "content": "The full text content of the document..."
}
```

Restart the agent — it reloads docs on startup.

---

## Production Upgrade Path

| Component | Current (dev) | Production swap |
|---|---|---|
| Retrieval | Keyword scoring | Vector embeddings + cross-encoder reranking |
| Model | `gpt-4o-mini` | Any `ChatOpenAI`-compatible model |
| State | In-memory dict | Redis / Postgres |
| Observability | `print()` traces | LangSmith / OpenTelemetry |
| Data store | Local JSON files | Vector DB (Pinecone, Weaviate, pgvector) |

---

## Harness Components Checklist

- [x] Execution loop with `MAX_REVISIONS` hard cap
- [x] State store (`State` dict) persists across all nodes
- [x] Typed tool contracts with documented error modes
- [x] Tool output validation before reaching the reasoner
- [x] Judge verification layer before output reaches the user
- [x] Retry decorator on every tool call
- [x] Fallback JSON parsing with safe defaults
- [x] Conversation history passed through every turn
