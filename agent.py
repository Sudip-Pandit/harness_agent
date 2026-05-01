"""
Three-Stage AI Agent — annotated implementation
================================================
Stage 1 — Prompt Engineering  : focused system prompts per node
Stage 2 — Context Engineering : RAG retrieval with provenance
Stage 3 — Harness Engineering : state machine, tool contracts,
                                 verification, retry, revision guard

Run:
    python agent.py
"""

import os, json, time
from typing import Dict, Any, List
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# ── Model setup ────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def call_llm(system: str, user: str) -> str:
    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ])
    return resp.content


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — HARNESS: Reliability primitive
# Every tool call is wrapped. One failure
# doesn't cascade through the pipeline.
# ═══════════════════════════════════════════════════════════════
def with_retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        return f"ERROR after {max_retries} attempts: {e}"
                    time.sleep(delay * (attempt + 1))
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — CONTEXT ENGINEERING: RAG
# Production: replace keyword scoring with
# vector embeddings + cross-encoder reranking.
# ═══════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_docs() -> List[Dict[str, Any]]:
    docs = []
    if not os.path.isdir(DATA_DIR):
        print(f"[WARN] data/ directory not found at {DATA_DIR}. Retrieval will return nothing.")
        return docs
    for f in os.listdir(DATA_DIR):
        if f.endswith(".json"):
            with open(os.path.join(DATA_DIR, f)) as fp:
                docs.append(json.load(fp))
    print(f"[INFO] Loaded {len(docs)} documents from data/")
    return docs

DOCUMENTS = load_docs()

@with_retry(max_retries=2)
def search_docs(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """Keyword-based retrieval. Swap for vector search in production."""
    q = query.lower()
    scored = [
        (sum(1 for t in q.split() if t in (d["title"] + " " + d["content"]).lower()), d)
        for d in DOCUMENTS
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [d for score, d in scored[:top_k] if score > 0]
    for r in results:
        r["retrieved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return results

@with_retry(max_retries=2)
def calculator(expression: str) -> str:
    """Safe math evaluator — no builtins exposed."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"CALCULATOR_ERROR: {e}"

TOOLS = {"search_docs": search_docs, "calculator": calculator}


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — HARNESS: Typed tool contracts
# Prevents the "rate-limit-summarised-as-research" bug.
# ═══════════════════════════════════════════════════════════════
TOOL_CONTRACTS = {
    "search_docs": {
        "description": "Retrieve relevant documents from the local knowledge base.",
        "inputs":      {"query": "str", "top_k": "int (optional, default 2)"},
        "output":      "List[Dict] — keys: id, title, content, retrieved_at",
        "error_modes": ["No matching documents", "Load failure"],
    },
    "calculator": {
        "description": "Evaluate a safe arithmetic expression.",
        "inputs":      {"expression": "str — e.g. '2 ** 10' or '(3 + 5) * 2'"},
        "output":      "str — numeric result or 'CALCULATOR_ERROR: ...'",
        "error_modes": ["Invalid expression", "Division by zero"],
    },
}


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — HARNESS: State store
# Authoritative record of what happened.
# NOT logs. Prevents the duplicate-paper-summary bug.
# ═══════════════════════════════════════════════════════════════
class State(dict):
    query:          str
    history:        List[Dict[str, str]]
    plan:           Dict[str, Any]
    tool_results:   List[Dict[str, Any]]
    draft:          str
    final:          str
    judge:          Dict[str, Any]
    revision_count: int


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — PROMPT ENGINEERING
# One prompt per node, single responsibility, schema-explicit.
# ═══════════════════════════════════════════════════════════════
PLANNER_SYSTEM = """
You are a senior AI planner.
Given the conversation history, the user's current query, and the available tool contracts below,
decide which tools (if any) to call and in what order.

Return ONLY valid JSON — no preamble, no markdown fences:
{
  "steps": [
    {
      "tool": "<tool_name or 'none'>",
      "args": { "<arg_name>": "<value>" },
      "note": "<one line explaining why>"
    }
  ],
  "final_answer_strategy": "<brief description of how to synthesise results>"
}

If no tools are needed, use "tool": "none".
""".strip()

REASONER_SYSTEM = """
You are a senior AI engineer giving precise, well-grounded answers.
You receive: conversation history, the user query, the execution plan, and validated tool results.
Synthesise everything into a clear, accurate Markdown response.
Always prioritise factual accuracy. Reference retrieved documents naturally where relevant.
""".strip()

JUDGE_SYSTEM = """
You are a quality verifier. Review the proposed answer against the user's query.
Assess: factual grounding, completeness, clarity, and safety.

Return ONLY valid JSON — no preamble, no markdown fences:
{
  "quality": "ok" | "revise",
  "reason": "<brief explanation>",
  "revision_instructions": "<specific fix instructions, or empty string if quality is ok>"
}
""".strip()

REVISION_SYSTEM = """
You are revising a previous answer based on explicit judge feedback.
You receive: the original query, the previous draft, revision instructions, and conversation history.
Fix exactly — and only — what the instructions specify. Do not rewrite sections that passed.
""".strip()


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — HARNESS: Pipeline nodes
# ═══════════════════════════════════════════════════════════════
def planner_node(state: State) -> State:
    print("\n[planner] Building execution plan...")
    payload = {
        "history":        state.get("history", []),
        "current_query":  state["query"],
        "tool_contracts": TOOL_CONTRACTS,
    }
    raw = call_llm(PLANNER_SYSTEM, json.dumps(payload, indent=2))
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[planner] WARNING: Could not parse plan JSON, falling back to direct answer.\nRaw: {raw[:200]}")
        plan = {
            "steps": [{"tool": "none", "args": {}, "note": "JSON parse failed — answering directly"}],
            "final_answer_strategy": "Answer directly from model knowledge.",
        }
    print(f"[planner] Plan: {len(plan.get('steps', []))} step(s)")
    state["plan"]           = plan
    state["revision_count"] = 0
    return state


def tool_node(state: State) -> State:
    print("[tools]   Executing tool calls...")
    results = []
    for step in state["plan"].get("steps", []):
        tool_name = step.get("tool", "none")
        args      = step.get("args", {}) or {}
        if tool_name == "none":
            print("[tools]   No tool needed — skipping.")
            continue
        fn = TOOLS.get(tool_name)
        if not fn:
            print(f"[tools]   CONTRACT_ERROR: Unknown tool '{tool_name}'")
            results.append({"tool": tool_name, "output": "CONTRACT_ERROR: Unknown tool", "valid": False})
            continue
        print(f"[tools]   Calling '{tool_name}' with args: {args}")
        output   = fn(**args)
        is_error = isinstance(output, str) and "ERROR" in output.upper()
        if is_error:
            print(f"[tools]   Tool returned error: {output}")
        results.append({"tool": tool_name, "args": args, "output": output, "valid": not is_error})
    state["tool_results"] = results
    return state


def reasoner_node(state: State) -> State:
    """Where all three stages converge."""
    print("[reasoner] Synthesising answer...")
    payload = {
        "history":      state.get("history", []),
        "query":        state["query"],
        "plan":         state["plan"],
        "tool_results": state["tool_results"],
    }
    state["draft"] = call_llm(REASONER_SYSTEM, json.dumps(payload, indent=2))
    return state


def judge_node(state: State) -> State:
    """Verification before output reaches any user."""
    print("[judge]   Verifying answer quality...")
    raw = call_llm(
        JUDGE_SYSTEM,
        json.dumps({"query": state["query"], "answer": state["draft"]}, indent=2),
    )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[judge]   WARNING: Could not parse judge JSON, defaulting to 'ok'.\nRaw: {raw[:200]}")
        result = {"quality": "ok", "reason": "Judge parse failed — accepting draft.", "revision_instructions": ""}
    print(f"[judge]   Quality: {result.get('quality')} — {result.get('reason', '')}")
    state["judge"] = result
    return state


def revision_node(state: State) -> State:
    instructions = state["judge"].get("revision_instructions", "Improve clarity and accuracy.")
    print(f"[revise]  Applying revision #{state.get('revision_count', 0) + 1}: {instructions[:80]}")
    payload = {
        "history":         state.get("history", []),
        "query":           state["query"],
        "previous_answer": state["draft"],
        "instructions":    instructions,
    }
    state["final"]          = call_llm(REVISION_SYSTEM, json.dumps(payload, indent=2))
    state["revision_count"] = state.get("revision_count", 0) + 1
    return state


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — HARNESS: Router
# MAX_REVISIONS is the structural guarantee that replaces
# "please stop revising" in a system prompt.
# ═══════════════════════════════════════════════════════════════
MAX_REVISIONS = 2

def router(state: State) -> str:
    needs  = state.get("judge", {}).get("quality") == "revise"
    capped = state.get("revision_count", 0) >= MAX_REVISIONS
    if needs and not capped:
        print(f"[router]  Revision needed (attempt {state.get('revision_count', 0) + 1}/{MAX_REVISIONS})")
        return "revise"
    if needs and capped:
        print("[router]  Revision needed but MAX_REVISIONS reached — accepting current draft.")
    return END


# ── Graph assembly ─────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("tools",   tool_node)
graph.add_node("reason",  reasoner_node)
graph.add_node("judge",   judge_node)
graph.add_node("revise",  revision_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "tools")
graph.add_edge("tools",   "reason")
graph.add_edge("reason",  "judge")
graph.add_conditional_edges("judge", router, {"revise": "revise", END: END})
graph.add_edge("revise", END)

app = graph.compile()


# ── REPL ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Three-Stage AI Agent  |  type 'exit' to quit")
    print("=" * 60)
    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input or user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = app.invoke({"query": user_input, "history": history})
        answer = result.get("final") or result.get("draft", "No answer generated.")

        print(f"\nAgent:\n{answer}")

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": answer})
