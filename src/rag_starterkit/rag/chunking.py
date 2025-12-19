import re
from typing import List, Dict

FAQ_PATTERN = re.compile(
    r"\n?\s*(\d+)\.\s+(.*?)\n(.*?)(?=\n\s*\d+\.|\Z)",
    re.DOTALL
)

def chunk_faq_text(text: str) -> List[Dict]:
    """
    Splits FAQ-style documents into question-answer chunks.
    """
    chunks = []

    for match in FAQ_PATTERN.finditer(text):
        q_no = match.group(1).strip()
        question = match.group(2).strip()
        answer = match.group(3).strip()

        chunk_text = f"Q{q_no}: {question}\nA: {answer}"

        chunks.append({
            "id": f"faq_{q_no}",
            "text": chunk_text
        })

    return chunks
