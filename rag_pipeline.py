# rag_pipeline.py
from retrieval.retriever import HybridRetriever
from generation.generate_answer import generate_answer

class RAGPipeline:
    def __init__(self):
        print("Loading retriever (embedding model + reranker)...")
        self.retriever = HybridRetriever()

    def ask(self, question, top_k=5):
        chunks = self.retriever.retrieve(question, top_k=top_k)
        answer = generate_answer(question, chunks)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks
        }

if __name__ == "__main__":
    pipeline = RAGPipeline()

    questions = [
        "What are NVIDIA's main risk factors related to competition?",
        "What does NVIDIA say about export controls to China?",
        "Who is NVIDIA's auditor?",
    ]

    for q in questions:
        result = pipeline.ask(q)
        print("=" * 70)
        print(f"Q: {result['question']}")
        print("-" * 70)
        print(f"A: {result['answer']}")
        print()