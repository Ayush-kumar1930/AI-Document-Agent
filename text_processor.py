from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """
    Reads PDF page-by-page and chunks each page while preserving its exact page number (for Breadcrumbs/citations).

    Returns:
    [
       {"text": "chunk text...", "page": 1},
       {"text": "chunk text...", "page": 2},
       ...
    ]
    """
    reader = PdfReader(pdf_path)
    all_chunks_with_metadata = []

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue 

        #  Split this page using our word-safe recursive chunker
        page_chunks = recursive_chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for c in page_chunks:
            all_chunks_with_metadata.append({
                "text": c,
                "page": page_idx  # 1-indexed page number !
            })
    return all_chunks_with_metadata


# Method 1: Fixed-Size Sliding Window Chunking
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """
    Splits text into chunks using a fixed sliding window approach with overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)

    if text_length <= chunk_size:
        return [text] if text else []

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += (chunk_size - chunk_overlap)

    return chunks


# Method 2: Recursive Character Chunking (The Industry Standard!)
def recursive_chunk_text(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 100,
    separators: list[str] = None
) -> list[str]:
    """
    Recursively splits text by natural separators and guarantees that words are never broken in half during overlap!
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Find highest priority separator present in the text
    chosen_separator = ""
    for sep in separators:
        if sep == "" or sep in text:
            chosen_separator = sep
            break

    # Split using chosen separator
    if chosen_separator:
        raw_splits = text.split(chosen_separator)
    else:
        raw_splits = list(text)

    # Remaining separators for sub-splits if a piece is still too big
    next_separators = separators[separators.index(chosen_separator) + 1:] if chosen_separator in separators else []

    chunks = []
    current_chunk = ""

    for split in raw_splits:
        split_piece = split if not chosen_separator else split + chosen_separator

        # If a single piece is STILL larger than chunk_size, split recursively!
        if len(split_piece) > chunk_size and next_separators:
            sub_chunks = recursive_chunk_text(split_piece, chunk_size, chunk_overlap, next_separators)
            for sc in sub_chunks:
                if len(current_chunk) + len(sc) <= chunk_size:
                    current_chunk += sc
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sc
            continue

        # Normal merging with overlap buffer
        if len(current_chunk) + len(split_piece) <= chunk_size:
            current_chunk += split_piece
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # --- WORD-SAFE OVERLAP (Never cuts words in half!) ---
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            next_space = current_chunk.find(" ", overlap_start)
            if next_space != -1 and next_space < len(current_chunk):
                overlap_start = next_space + 1
            current_chunk = current_chunk[overlap_start:] + split_piece

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


if __name__ == "__main__":
    sample_text = (
        "Artificial Intelligence is transforming modern industries rapidly.\n\n"
        "Retrieval-Augmented Generation (RAG) is an advanced architecture. "
        "It grounds large language models on external verified data, ensuring that answers "
        "remain factual, transparent, and completely free from hallucinations.\n\n"
        "Vector databases like ChromaDB store high-dimensional embeddings to enable "
        "sub-millisecond semantic similarity searches."
    )

    print("=== 1. FIXED-SIZE CHUNKING ===")
    fixed = chunk_text(sample_text, chunk_size=150, chunk_overlap=30)
    for i, c in enumerate(fixed):
        print(f"Chunk {i+1} ({len(c)} chars): {repr(c)}")

    print("\n=== 2. RECURSIVE CHARACTER CHUNKING ===")
    rec = recursive_chunk_text(sample_text, chunk_size=150, chunk_overlap=30)
    for i, c in enumerate(rec):
        print(f"Chunk {i+1} ({len(c)} chars): {repr(c)}")
