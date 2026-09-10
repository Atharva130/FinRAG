# ingestion/embed_and_store.py
import json
import chromadb
from sentence_transformers import SentenceTransformer

def load_chunks(path="data/processed/NVDA_10K_chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(chunks, collection_name="nvda_10k", persist_dir="data/chroma"):
    print("Loading embedding model (first run downloads it, ~80MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good enough for MVP

    print(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=persist_dir)

    # wipe any old collection with this name so re-runs don't duplicate data
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)

    ids = [str(c["chunk_id"]) for c in chunks]
    metadatas = [{"section": c["section"], "char_count": c["char_count"]} for c in chunks]

    print("Writing to ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas #type:ignore
    )

    print(f"Done. Collection '{collection_name}' now has {collection.count()} chunks.")
    return collection

if __name__ == "__main__":
    chunks = load_chunks()
    build_index(chunks)