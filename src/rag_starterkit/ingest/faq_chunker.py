import re
import hashlib
from typing import List, Dict
import os

FAQ_PATTERN = re.compile(
    r"""
    (?P<prefix>^|\n)                           # start or newline
    \s*(?:Q\s*)?                               # optional "Q"
    (?P<num>\d{1,4})                           # question number
    \s*(?:[\.\)\-:])\s+                        # separator: . ) - :
    (?P<question>.{5,300}?)                    # question text (short-ish)
    \s*\n+                                     # line break(s)
    (?P<answer>.*?)                            # answer (anything)
    (?=\n\s*(?:Q\s*)?\d{1,4}\s*[\.\)\-:]\s+|\Z) # next question or end
    """,
    re.VERBOSE | re.DOTALL,
)

def _stable_chunk_id(doc_id: str, qnum: str, question: str) -> str:
    h = hashlib.sha1(f"{doc_id}|{qnum}|{question}".encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}::faq::{qnum}::{h}"


def _is_faq_doc(file_path: str, full_text: str) -> bool:
    base = os.path.basename(file_path).lower()
    if "faq" in base:
        return True
    head = (full_text or "").lower()[:3000]
    return ("frequently asked" in head) or ("\nfaq" in head) or (" faq " in head)

def chunk_faq_text(
    text: str,
    *,
    doc_id: str,
    source_path: str,
) -> List[Dict]:
    chunks: List[Dict] = []
    t = (text or "").strip()
    if not t:
        return chunks

    for m in FAQ_PATTERN.finditer(t):
        qnum = m.group("num").strip()
        question = re.sub(r"\s+", " ", m.group("question").strip())
        answer = m.group("answer").strip()

        # guard against junk answers
        if len(answer) < 20:
            continue

        chunk_text = f"Q{qnum}: {question}\nA: {answer}"

        chunks.append({
            "chunk_id": _stable_chunk_id(doc_id, qnum, question),
            "text": chunk_text,
            "metadata": {
                "doc_id": doc_id,
                "source_path": source_path,
                "section_type": "faq",
                "order_key": f"FAQ-{int(qnum):04d}",
                "policy_topic": f"{doc_id}_FAQ",
                "page_start": 1,
                "page_end": 1,
            },
        })

    return chunks
