from typing import List, Dict, Tuple
from requests.exceptions import ReadTimeout, ConnectionError

from rag_starterkit.api.schemas import Citation, AnswerQuality
from rag_starterkit.rag.prompt import build_rag_prompt
from rag_starterkit.llm.ollama_client import generate_llm_answer
from rag_starterkit.llm.qwen_judge import judge_answer
from rag_starterkit.llm.safety import SAFE_REFUSAL_MESSAGE

def generate_answer(
    query: str,
    contexts: List[Dict]
) -> Tuple[str, List[Citation], AnswerQuality]:

    # Case 1: No retrieved context → safe refusal
    if not contexts:
        return (
            "I do not have sufficient information from the provided documents.",
            [],
            AnswerQuality(
                groundedness=0.0,
                confidence=0.0,
                hallucination_risk="high",
                unsupported_points=[]
            )
        )

    # -----------------------------
    # 1) Generate answer (Qwen2.5)
    # -----------------------------
    prompt = build_rag_prompt(query, contexts)

    try:
        answer = generate_llm_answer(prompt).strip()
    except (ReadTimeout, ConnectionError):
        answer = (
            "The language model did not respond in time. "
            "Please try again or reduce the query scope."
        )

    # -----------------------------
    # 2) Build citations
    # -----------------------------
    citations = [
        Citation(
            source_id=c["id"],
            snippet=(c["text"] or "")[:300]
        )
        for c in contexts
    ]

    # -----------------------------
    # 3) Judge answer (Qwen2.5)
    # -----------------------------
    judge_result = judge_answer(answer, contexts)
    quality = AnswerQuality(**judge_result)

    # -----------------------------
    # 4) Auto-reject unsafe answers
    # -----------------------------
    if quality.hallucination_risk == "high":
        quality.rejected = True
        return SAFE_REFUSAL_MESSAGE, citations, quality

    return answer, citations, quality

