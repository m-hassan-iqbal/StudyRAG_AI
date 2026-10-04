"""
Chunking Module.
Splits extracted document sections into meaningful overlapping chunks while
retaining complete source metadata (subject_id, document_id, filename, page, chunk_id).
"""

from typing import List, Dict, Any


def split_text_with_overlap(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Splits text into chunks of approximately chunk_size characters with overlap.
    Attempts to break chunks on whitespace or sentence boundaries when feasible.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)
    step = max(1, chunk_size - overlap)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # If not at the end of the text, try to find a natural break near the end
        if end < text_len:
            best_break = -1
            for search_idx in range(end, max(start + step, end - 100), -1):
                if text[search_idx] in ("\n", ". ", "? ", "! "):
                    best_break = search_idx + 1
                    break
                elif text[search_idx] == " " and best_break == -1:
                    best_break = search_idx + 1

            if best_break != -1:
                end = best_break

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append(chunk_content)

        if end >= text_len:
            break

        start = start + step
        if start >= end:
            start = end

    return chunks


def create_chunks(
    extracted_documents: List[Dict[str, Any]],
    subject_id: str,
    document_id: str,
    chunk_size: int = 800,
    overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Creates structured chunks from extracted document items.
    
    Every resulting chunk retains:
    - subject_id: string
    - document_id: string
    - filename: string
    - page: int or None
    - chunk_id: unique identifier
    - text: chunk content
    """
    all_chunks = []
    global_seq = 0

    for doc_item in extracted_documents:
        filename = doc_item["filename"]
        page = doc_item.get("page")
        content = doc_item.get("text", "")

        text_pieces = split_text_with_overlap(content, chunk_size=chunk_size, overlap=overlap)
        
        for piece_idx, piece in enumerate(text_pieces):
            global_seq += 1
            page_str = f"p{page}" if page is not None else "pNA"
            chunk_id = f"{subject_id}_{document_id}_{page_str}_c{global_seq:04d}"

            all_chunks.append({
                "subject_id": subject_id,
                "document_id": document_id,
                "filename": filename,
                "page": page,
                "chunk_id": chunk_id,
                "text": piece
            })

    return all_chunks
