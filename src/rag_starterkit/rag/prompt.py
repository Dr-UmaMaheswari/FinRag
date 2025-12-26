from collections import defaultdict
from typing import List, Dict


def _pretty_bank_name(raw: str) -> str:
    """
    Normalize internal bank codes (e.g. 'axis_bank', 'tmb', 'rbi') into
    human-readable names for use in prompts.
    """
    if not raw:
        return "Unknown Bank"

    b = raw.lower().strip()

    mapping = {
        # Regulator / infrastructure
        "rbi": "RBI (Reserve Bank of India)",
        "npci": "NPCI (Clearing Infrastructure)",
        "npci_bank": "NPCI (Clearing Infrastructure)",

        # Axis
        "axis": "Axis Bank",
        "axis_bank": "Axis Bank",

        # Bank of India
        "boi": "Bank of India",
        "boi_bank": "Bank of India",

        # Bank of Baroda (filename typo 'barado')
        "bank_of_barado_bank": "Bank of Baroda",

        # Canara
        "canara": "Canara Bank",
        "canara_bank": "Canara Bank",

        # CSB
        "csb": "CSB Bank",
        "csb_bank": "CSB Bank",

        # CUB
        "cub": "City Union Bank",
        "cub_bank": "City Union Bank",

        # Central Bank
        "central": "Central Bank of India",
        "central_bank": "Central Bank of India",

        # CSCM
        "cscm": "CSCM Bank",
        "cscm_bank": "CSCM Bank",

        # HDFC
        "hdfc": "HDFC Bank",
        "hdfc_bank": "HDFC Bank",

        # HPSCB
        "hpscb": "HP State Cooperative Bank",
        "hpscb_bank": "HP State Cooperative Bank",

        # ICICI
        "icici": "ICICI Bank",
        "icici_bank": "ICICI Bank",

        # Indian Bank
        "indian": "Indian Bank",
        "indian_bank": "Indian Bank",

        # IDBI
        "idbi": "IDBI Bank",

        # IDFC
        "idfc": "IDFC FIRST Bank",

        # IndusInd
        "indusind": "IndusInd Bank",
        "indusind_bank": "IndusInd Bank",

        # J&K Bank
        "jk": "Jammu & Kashmir Bank",
        "jk_bank": "Jammu & Kashmir Bank",

        # DCB
        "dcb": "DCB Bank",
        "dcb_bank": "DCB Bank",

        # Kotak (filename typo 'kodak')
        "kodak": "Kotak Mahindra Bank",
        "kodak_bank": "Kotak Mahindra Bank",

        # KVB
        "kvb": "Karur Vysya Bank",
        "kvb_bank": "Karur Vysya Bank",

        # Bank of Maharashtra (filename typo 'maharastra')
        "maharastra": "Bank of Maharashtra",
        "maharastra_bank": "Bank of Maharashtra",

        # MCB
        "mcb": "MCB Bank",
        "mcb_bank": "MCB Bank",

        # Punjab – you can refine if you know the exact bank
        "punjab": "Punjab National Bank",
        "punjab_bank": "Punjab National Bank",

        # TMB
        "tmb": "Tamilnad Mercantile Bank",

        # Union
        "union": "Union Bank of India",
        "union_bank": "Union Bank of India",

        # UCO
        "uco": "UCO Bank",
        "uco_bank": "UCO Bank",

        # YES
        "yes": "YES Bank",
        "yes_bank": "YES Bank",

        # Generic fallback
        "generic": "Generic Bank Policy",
        "generic_bank": "Generic Bank Policy",
    }

    if b in mapping:
        return mapping[b]

    # Fallback: strip suffixes, prettify
    pretty = raw.replace("_bank", "").replace("_", " ").title()
    return pretty or "Unknown Bank"


def build_rag_prompt(query: str, contexts: List[Dict]) -> str:
    """
    Build a bank-aware RAG prompt.

    - If only one non-RBI bank appears in contexts: answer for that bank only.
    - If multiple banks and/or RBI appear: produce a structured comparison:
      RBI → Bank A → Bank B → Similarities / Differences.
    """

    if not contexts:
        return (
            "You are a banking compliance assistant.\n\n"
            "No sources were retrieved. Answer:\n"
            "I do not have sufficient information from the provided documents."
        )

    # Group contexts by bank
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for c in contexts:
        bank = c.get("bank") or "unknown"
        grouped[bank].append(c)

    unique_banks = list(grouped.keys())
    has_rbi = any(b.lower() == "rbi" for b in unique_banks)

    # Build source blocks
    def build_bank_block(bank_key: str, items: List[Dict]) -> str:
        bank_name = _pretty_bank_name(bank_key)
        lines = [f"### Bank: {bank_name}"]
        for i, c in enumerate(items, start=1):
            page_info = ""
            if c.get("page_start") is not None:
                if c.get("page_end") and c["page_end"] != c["page_start"]:
                    page_info = f" (pages {c['page_start']}-{c['page_end']})"
                else:
                    page_info = f" (page {c['page_start']})"

            lines.append(
                f"Source {i}{page_info}:\n"
                f"{(c.get('text') or '').strip()}"
            )
        return "\n\n".join(lines)

    bank_blocks: List[str] = []
    # Keep RBI first if present, then other banks alphabetically
    ordered_banks = sorted(unique_banks, key=lambda b: (b.lower() != "rbi", b))
    for b in ordered_banks:
        bank_blocks.append(build_bank_block(b, grouped[b]))

    source_section = "\n\n".join(bank_blocks)

    # Decide mode: single-bank vs multi-bank comparison
    non_rbi_banks = [b for b in unique_banks if b.lower() != "rbi"]
    multi_bank_mode = len(non_rbi_banks) > 1 or (has_rbi and len(non_rbi_banks) >= 1)

    if not multi_bank_mode:
        # Single-bank answer
        bank_name = _pretty_bank_name(non_rbi_banks[0]) if non_rbi_banks else "the bank"
        prompt = f"""
You are a banking compliance assistant.

You must answer the question strictly using the provided sources for {bank_name}.
If the answer is not present in the sources, reply exactly:
"I do not have sufficient information from the provided documents."

Sources (grouped by bank):
{source_section}

User question:
{query}

Instructions for your answer:
- Focus on the policies and rules of {bank_name}.
- Be concise, factual, and policy-oriented.
- If the policy is silent on a point, say explicitly that it is not covered in the provided extracts.

Answer:
""".strip()
        return prompt

    # Multi-bank / comparison mode
    bank_names_readable = ", ".join(_pretty_bank_name(b) for b in non_rbi_banks)
    rbi_label = _pretty_bank_name("rbi") if has_rbi else "the regulator"

    prompt = f"""
You are a banking compliance and CTS policy comparison assistant.

You are given extracts from multiple banks' Cheque Collection / CTS policies,
and optionally from {rbi_label}. You must answer the question strictly using
the provided sources. If a detail is not present, do NOT guess.

Sources (grouped by bank):
{source_section}

User question:
{query}

Write a structured answer with the following sections:

1) High-level summary
   - Briefly summarise what the policies say about the question.

2) {rbi_label} view
   - Summarise what the RBI / CTS FAQ or regulatory source says (if present).
   - Clearly state if the regulator is silent on some aspect.

3) Bank-wise policies
   For each bank ({bank_names_readable}):
   - Describe what the bank's policy says that is relevant to the question.
   - Mention any explicit clauses, conditions, or exceptions if visible.
   - If the bank policy does not mention something the regulator mentions, say so.
   - If needed, quote short phrases (not more than a line) to anchor your statements.

4) Comparison and key differences
   - Compare each bank against the RBI guidance and against each other.
   - Highlight any stricter rules, additional safeguards, or gaps.
   - Phrase it in neutral, factual language (no opinions, only what you see).

5) Final compliance note
   - Briefly state how a bank compliance team might use these differences in practice,
     without speculating beyond the texts.

If the information is insufficient to answer, write:
"I do not have sufficient information from the provided documents."

Answer:
""".strip()

    return prompt
