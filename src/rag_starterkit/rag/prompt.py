def build_rag_prompt(query: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        [f"Source {i+1}:\n{c['text']}" for i, c in enumerate(contexts)]
    )

    prompt = f"""
You are a banking compliance assistant.

Answer the question strictly using the provided sources.
If the answer is not present in the sources, say:
"I do not have sufficient information from the provided documents."

Sources:
{context_block}

Question:
{query}

Answer (concise, factual):
"""
    return prompt.strip()
