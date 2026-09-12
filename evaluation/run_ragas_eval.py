# evaluation/run_ragas_eval.py
import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from retrieval.retriever import HybridRetriever
from generation.generate_answer import generate_answer

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
judge_llm = llm_factory("openai/gpt-oss-120b", client=client)
faithfulness_scorer = Faithfulness(llm=judge_llm)


async def evaluate_faithfulness(question, retrieved_chunks, answer):
    contexts = [c["text"] for c in retrieved_chunks]
    result = await faithfulness_scorer.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    return result.value


async def main():
    with open("evaluation/eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    retriever = HybridRetriever()

    results = []
    faithfulness_scores = []

    for item in eval_set:
        question = item["question"]
        print(f"Evaluating: {question}")

        retrieved = retriever.retrieve(question, top_k=5)
        answer = generate_answer(question, retrieved)

        try:
            faithfulness = await evaluate_faithfulness(question, retrieved, answer)
            faithfulness_scores.append(faithfulness)
            print(f"  Faithfulness: {faithfulness:.2f}")
        except Exception as e:
            faithfulness = None
            print(f"  Faithfulness: FAILED ({type(e).__name__})")

        results.append({
            "question": question,
            "answer": answer,
            "faithfulness_score": faithfulness
        })

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    failed_count = len(eval_set) - len(faithfulness_scores)

    print(f"\n{'='*60}")
    print(f"Average Faithfulness across {len(faithfulness_scores)} successful questions: {avg_faithfulness:.2%}")
    if failed_count:
        print(f"({failed_count} question(s) failed judging and were excluded)")
    print("="*60)

    output = {
        "average_faithfulness": avg_faithfulness,
        "results": results
    }
    with open("evaluation/ragas_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Saved to evaluation/ragas_results.json")


if __name__ == "__main__":
    asyncio.run(main())