# ingestion/section_splitter.py
import re
import json
import os

ITEM_PATTERN = re.compile(r"^\s*(Item\s+\d+[A-Z]?\.?)", re.IGNORECASE | re.MULTILINE)

def find_body_matches(text, toc_gap_threshold=500):
    matches = list(ITEM_PATTERN.finditer(text))

    toc_end_idx = 0
    for i in range(len(matches) - 1):
        gap = matches[i + 1].start() - matches[i].start()
        if gap >= toc_gap_threshold:
            toc_end_idx = i
            break

    body_matches = matches[toc_end_idx + 1:]
    return body_matches

def normalize_label(raw_label):
    # "Item 1A." -> "Item 1A" (strip trailing period, normalize spacing)
    return re.sub(r"\s+", " ", raw_label.strip()).rstrip(".")

def split_sections(text):
    body_matches = find_body_matches(text)
    sections = {}

    for i, m in enumerate(body_matches):
        label = normalize_label(m.group(1))
        start = m.start()
        end = body_matches[i + 1].start() if i + 1 < len(body_matches) else len(text)
        content = text[start:end].strip()

        # if this label already exists (duplicate heading), keep the LONGER version
        if label not in sections or len(content) > len(sections[label]):
            sections[label] = content

    return sections

if __name__ == "__main__":
    with open("data/raw/NVDA_10K.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sections = split_sections(text)

    print(f"Found {len(sections)} unique sections:\n")
    for label, content in sections.items():
        print(f"{label:15s} -> {len(content):>7} chars")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/NVDA_10K_sections.json", "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)

    print("\nSaved to data/processed/NVDA_10K_sections.json")