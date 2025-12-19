from pathlib import Path
from rag_starterkit.data.pdf_loader import load_pdf_text
from rag_starterkit.rag.chunking import chunk_faq_text
from rag_starterkit.rag.vectorstore import add_documents

def ingest_path(path: str):
    p = Path(path)

    docs = []

    if not p.exists():
        raise ValueError(f"Path not found: {path}")

    # Case 1: Directory (mix of PDF + TXT)
    if p.is_dir():
        for fp in p.iterdir():

            # PDF → FAQ-aware chunking
            if fp.suffix.lower() == ".pdf":
                raw_text = load_pdf_text(str(fp))
                faq_chunks = chunk_faq_text(raw_text)
                docs.extend(faq_chunks)

            # TXT → simple text
            elif fp.suffix.lower() == ".txt":
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    docs.append({
                        "id": fp.name,
                        "text": text
                    })

    # Case 2: Single PDF
    elif p.is_file() and p.suffix.lower() == ".pdf":
        raw_text = load_pdf_text(str(p))
        docs = chunk_faq_text(raw_text)

    else:
        raise ValueError("Unsupported file type")

    if docs:
        stored = add_documents(docs)
    return {"chunks": len(docs), "stored": stored}
    
