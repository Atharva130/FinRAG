# generation/generate_answer.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a financial research assistant. You answer questions
using ONLY the provided excerpts from a company's SEC filing (10-K).

Rules:
- Use ONLY the information in the provided excerpts. Do not use outside knowledge.
- If the excerpts don't contain enough information to answer, say exactly:
  "This is not addressed in the retrieved sections of the filing."
- After your answer, cite which chunk_id(s) you used, like: [Sources: chunk_id 46, 121]
- Be concise and factual. Do not speculate.
"""

def generate_answer(question, retrieved_chunks, model="openai/gpt-oss-20b"):
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(
            f"[chunk_id={c['chunk_id']} | section={c['section']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"""Filing excerpts:

{context}

Question: {question}

Answer using only the excerpts above, and cite the chunk_id(s) you used."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,  # low temperature = stick to facts, don't get creative
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from retrieval.retriever import HybridRetriever

    retriever = HybridRetriever()
    question = "What are NVIDIA's main risk factors related to competition?"

    print("Retrieving chunks...")
    chunks = retriever.retrieve(question, top_k=5)

    print("Generating answer...\n")
    answer = generate_answer(question, chunks)

    print("=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    print(answer)