# ingestion/chunker.py
import json
import os

def chunk_section(label, text, max_chars=1500, overlap=200):
    """Split one section's text into overlapping chunks, breaking on paragraph
    boundaries (double newlines) where possible instead of mid-sentence."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            # start new chunk, carrying a small overlap from the end of the previous one
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + para + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks

def chunk_all_sections(sections_path="data/processed/NVDA_10K_sections.json",
                        max_chars=1500, overlap=200):
    with open(sections_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    all_chunks = []
    chunk_id = 0

    for label, text in sections.items():
        section_chunks = chunk_section(label, text, max_chars, overlap)
        for c in section_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "section": label,
                "text": c,
                "char_count": len(c)
            })
            chunk_id += 1

    return all_chunks

if __name__ == "__main__":
    chunks = chunk_all_sections()

    print(f"Total chunks created: {len(chunks)}\n")

    # show distribution per section
    from collections import Counter
    section_counts = Counter(c["section"] for c in chunks)
    for section, count in section_counts.items():
        print(f"{section:15s} -> {count:>3} chunks")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/NVDA_10K_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nSaved {len(chunks)} chunks to {out_path}")
    print(f"\nSample chunk:\n{json.dumps(chunks[10], indent=2)[:500]}")