import requests
import os
import json

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("RAG_LLM_MODEL", "qwen2.5:3b-instruct")

def judge_answer(answer: str, contexts: list[dict]) -> dict:
    """
    Uses Qwen2.5 as an LLM-as-a-Judge to evaluate groundedness and hallucination risk.
    """

    sources_text = "\n\n".join(
        [f"Source {i+1}:\n{c['text'][:1200]}" for i, c in enumerate(contexts)]
    )

    prompt = f"""
You are a strict banking compliance auditor.

Evaluate the ANSWER strictly against the SOURCES.

Rules:
- Do NOT add new facts.
- If a statement is not supported by the sources, mark it as unsupported.
- Be conservative.

SOURCES:
{sources_text}

ANSWER:
{answer}

Return ONLY valid JSON in this exact schema:
{{
  "groundedness": number between 0 and 1,
  "confidence": number between 0 and 1,
  "hallucination_risk": "low" | "medium" | "high",
  "unsupported_points": [short phrases]
}}
"""

    payload = {
        "model": MODEL,
        "prompt": prompt.strip(),
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 200
        }
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()

    raw = resp.json().get("response", "").strip()

    try:
        return json.loads(raw)
    except Exception:
        # Hard fallback if model breaks JSON
        return {
            "groundedness": 0.0,
            "confidence": 0.0,
            "hallucination_risk": "high",
            "unsupported_points": ["Model failed to produce valid evaluation"]
        }
