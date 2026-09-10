# retrieval/retriever.py
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

STOPWORDS = {"what", "are", "is", "the", "a", "an", "to", "of", "in", "on",
             "for", "related", "and", "or", "main", "its", "their"}

def tokenize(t):
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOPWORDS]


class HybridRetriever:
    def __init__(self, chunks_path="data/processed/NVDA_10K_chunks.json",
                 collection_name="nvda_10k", persist_dir="data/chroma"):
        self.chunks = json.load(open(chunks_path, "r", encoding="utf-8"))
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}

        corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_collection(collection_name)

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, query, top_k=5, pool_size=20):
        # BM25 candidates
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool_size]
        bm25_ids = [self.chunks[i]["chunk_id"] for i in bm25_idx]

        # Vector candidates
        query_emb = self.embed_model.encode([query]).tolist()
        vec_results = self.collection.query(query_embeddings=query_emb, n_results=pool_size)
        vec_ids = [int(cid) for cid in vec_results["ids"][0]]

        # Merge (dedup, preserve order)
        merged_ids = list(dict.fromkeys(bm25_ids + vec_ids))
        candidates = [self.chunk_by_id[cid] for cid in merged_ids]

        # Rerank full pool
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in reranked[:top_k]:
            results.append({
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "text": chunk["text"],
                "rerank_score": float(score)
            })
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "What are NVIDIA's main risk factors related to competition?"
    results = retriever.retrieve(query)

    print(f"Query: {query}\n")
    for i, r in enumerate(results):
        print(f"--- Result {i+1} | chunk_id={r['chunk_id']} | section={r['section']} | score={r['rerank_score']:.3f} ---")
        print(r["text"][:250].replace("\n", " "))
        print()