# ingestion/inspect_structure.py
import re

def inspect(path="data/raw/NVDA_10K.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Look for lines that look like "Item 1A." / "Item 7." etc.
    pattern = re.compile(r"^\s*(Item\s+\d+[A-Z]?\.?)", re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(text))

    print(f"Total characters: {len(text)}")
    print(f"Found {len(matches)} lines matching 'Item N' pattern\n")

    for m in matches[:40]:
        start = m.start()
        # print a bit of context after the match so we can see the title
        snippet = text[start:start+80].replace("\n", " ")
        print(f"[pos {start:>7}] {snippet}")

if __name__ == "__main__":
    inspect()