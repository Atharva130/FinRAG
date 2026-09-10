# evaluation/run_eval.py
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from retrieval.retriever import HybridRetriever
from generation.generate_answer import generate_answer


def load_eval_set(path="evaluation/eval_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_ids, expected_ids):
    """Fraction of expected chunk_ids that appear anywhere in the retrieved set.
    If expected_ids is empty, we skip this metric for that question (N/A)."""
    if not expected_ids:
        return None
    hits = sum(1 for eid in expected_ids if eid in retrieved_ids)
    return hits / len(expected_ids)


def answer_contains_check(answer, expected_keywords):
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in expected_keywords)


def run_eval(top_k=5):
    eval_set = load_eval_set()
    retriever = HybridRetriever()

    results = []
    recall_scores = []
    answer_correct_count = 0
    questions_with_expected_answer = 0

    for item in eval_set:
        question = item["question"]
        expected_chunk_ids = item["expected_chunk_ids"]
        expected_keywords = item.get("expected_answer_contains", [])

        print(f"Running: {question}")

        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]

        recall = recall_at_k(retrieved_ids, expected_chunk_ids)
        if recall is not None:
            recall_scores.append(recall)

        answer = generate_answer(question, retrieved)

        answer_correct = None
        if expected_keywords:
            questions_with_expected_answer += 1
            answer_correct = answer_contains_check(answer, expected_keywords)
            if answer_correct:
                answer_correct_count += 1

        results.append({
            "question": question,
            "expected_chunk_ids": expected_chunk_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "recall_at_k": recall,
            "generated_answer": answer,
            "expected_keywords": expected_keywords,
            "answer_correct": answer_correct
        })

    # --- Summary ---
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None
    answer_accuracy = (answer_correct_count / questions_with_expected_answer
                        if questions_with_expected_answer else None)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions: {len(eval_set)}")
    print(f"Questions with known source chunks: {len(recall_scores)}")
    if avg_recall is not None:
        print(f"Average Recall@{top_k}: {avg_recall:.2%}")
    print(f"Questions with expected keywords: {questions_with_expected_answer}")
    if answer_accuracy is not None:
        print(f"Answer keyword accuracy: {answer_accuracy:.2%}")

    print("\nPer-question breakdown:")
    for r in results:
        recall_str = f"{r['recall_at_k']:.0%}" if r['recall_at_k'] is not None else "N/A"
        correct_str = "✓" if r['answer_correct'] else ("✗" if r['answer_correct'] is False else "N/A")
        print(f"  Recall={recall_str:>5}  Correct={correct_str:>3}  | {r['question']}")

    output = {
        "top_k": top_k,
        "total_questions": len(eval_set),
        "average_recall_at_k": avg_recall,
        "answer_keyword_accuracy": answer_accuracy,
        "results": results
    }

    with open("evaluation/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to evaluation/eval_results.json")


if __name__ == "__main__":
    run_eval()