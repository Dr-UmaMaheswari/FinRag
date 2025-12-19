import fitz  # PyMuPDF

def load_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text.strip())
    return "\n".join(pages)
