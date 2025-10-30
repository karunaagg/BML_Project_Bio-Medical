# Copyright 2025
# Minimal ADK agent that answers ONLY from a SQuAD-style JSON dataset.
# - No RAG, no training
# - One function tool: lookup_answer(question, threshold)
# - ADK auto-wraps Python functions as tools (per docs)

from __future__ import annotations
import json, os, re, string
from pathlib import Path
from typing import Any, Dict, List
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.lite_llm import LiteLlm

from google.adk.agents import Agent  # per Quickstart
# If you prefer LlmAgent explicitly, you can: from google.adk.agents import LlmAgent

# -------------------- dataset load (module-level, once) --------------------

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s)

def _load_squad(path: str) -> List[Dict[str, Any]]:
    """
    Load SQuAD-style JSON -> flattened list of:
      {id, question, answers: [str], context}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))

    flat: List[Dict[str, Any]] = []
    for art in data.get("data", []):
        for para in art.get("paragraphs", []):
            ctx = _clean(para.get("context", ""))
            for qa in para.get("qas", []):
                q = _clean(qa.get("question", ""))
                answers = []
                for a in qa.get("answers", []):
                    t = _clean(a.get("text", ""))
                    if t:
                        answers.append(t)
                if q and answers:
                    flat.append({
                        "id": qa.get("id", ""),
                        "question": q,
                        "answers": answers,
                        "context": ctx,
                    })
    if not flat:
        raise ValueError("No SQuAD-style items found in JSON.")
    return flat

BIO_JSON = os.getenv("BIO_JSON", "Dataset.json")
_ITEMS: List[Dict[str, Any]] = _load_squad(BIO_JSON)

# -------------------- simple fuzzy matching --------------------

def _similarity(a: str, b: str) -> float:
    # lightweight similarity without extra deps; normalized 0..1
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()

def _best(query: str, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    qn = _norm(query)
    scored = []
    for rec in _ITEMS:
        score = _similarity(qn, _norm(rec["question"]))
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]

def _snippet(ctx: str, answers: List[str], window: int = 80) -> str:
    cl = ctx.lower()
    pos = -1
    sel = ""
    for a in answers:
        al = a.lower()
        pos = cl.find(al)
        if pos != -1:
            sel = a
            break
    if pos == -1:
        return ctx[: max(2*window, 160)].strip()
    start = max(0, pos - window)
    end = min(len(ctx), pos + len(sel) + window)
    return ctx[start:end].strip()

# -------------------- FUNCTION TOOL --------------------
# ADK auto-wraps Python functions as tools when passed in Agent.tools (docs) :contentReference[oaicite:3]{index=3}

def lookup_answer(question: str, threshold: float = 0.80) -> Dict[str, Any]:
    """
    Look up the closest question in the dataset and return its gold answer.
    Args:
      question: User's question.
      threshold: similarity gate in [0,1]. If best match < threshold -> not found.
    Returns:
      dict with:
        - found (bool)
        - answer (str | None)
        - matched_question (str | None)
        - score (float)
        - evidence (str | None)
        - id (str | None)
        - note (str | None)
    """
    q = question or ""
    if not q.strip():
        return {"found": False, "note": "Empty question."}

    top = _best(q, k=3)
    score, rec = top[0]
    if score < float(threshold):
        return {
            "found": False,
            "score": float(score),
            "note": "Low similarity; not in dataset.",
            "closest_question": rec["question"],
        }

    ans = rec["answers"][0]
    evid = _snippet(rec["context"], rec["answers"])
    return {
        "found": True,
        "answer": ans,
        "matched_question": rec["question"],
        "score": float(score),
        "evidence": evid,
        "id": rec.get("id", ""),
    }

# -------------------- ROOT AGENT --------------------
# Quickstart shows creating an Agent and listing tools; ADK reads .env for model/auth :contentReference[oaicite:4]{index=4}

  # NEW: route to OpenAI via LiteLLM

# ... keep the dataset loader and lookup_answer() tool exactly as before ...

root_agent = Agent(
    name="biomed_json_agent",
    # Use OpenAI via LiteLlm; pick any deployed model you have access to
    # Examples: "openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-5"
    model=LiteLlm(model="openai/gpt-4o-mini"),
    description="Answers strictly from a SQuAD-style biomedical JSON dataset.",
    instruction=(
        "You are a biomedical Q→A agent.\n"
        "- Always call the 'lookup_answer' tool first.\n"
        "- If tool returns found=false: reply 'Not found in the dataset.' and if the query is related to biomedical then try to attempt the query but give the footer that this is AI overview might not be correct and for other queries ask the user to give queries related to Biomedical but handle greetings queries\n"
        "- If found=true: reply with: the answer and explain the medical terms .\n"
        "- Then on a new line print: Evidence: <short snippet>.\n"
        "- Do not give medical advice. Do not speculate."
    ),
    tools=[lookup_answer],
)
