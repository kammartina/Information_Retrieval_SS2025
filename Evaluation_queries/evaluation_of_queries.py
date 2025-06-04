import json
import os
from dotenv import load_dotenv
from Retrieval_ranking_answer.hybrid_retr import HybridRetriever

# Load environment variables from .env file
load_dotenv()
EVAL_SET_PATH = os.getenv("EVAL_SET_PATH")

# Load the evaluation set from a .jsonl file
def load_eval_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Evaluate the retriever system
def evaluate_retriever(eval_data, retriever_fn, k=5):
    correct = 0
    total = len(eval_data)

    for entry in eval_data:
        query = entry["query"]
        gold_ids = set(entry["gold_utterance_ids"])
        retrieved_ids = set(retriever_fn(query, k=k))

        if gold_ids & retrieved_ids:
            correct += 1
            print(f"✅ {query}")
        else:
            print(f"❌ {query}")
            print(f"   Expected: {gold_ids}")
            print(f"   Got     : {retrieved_ids}")

    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")


retriever = HybridRetriever()
# Use the HybridRetriever for evaluation
def hybrid_retriever(query, k=5):
    # Returns top k retrieved utterance IDs
    results = retriever.search(query)
    return [uid for uid, _, _ in results[:k]]

# Load eval set from path in .env and run evaluation
eval_data = load_eval_set(EVAL_SET_PATH)
evaluate_retriever(eval_data, hybrid_retriever)