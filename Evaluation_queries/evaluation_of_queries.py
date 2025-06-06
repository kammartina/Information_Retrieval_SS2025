import json
import os
from dotenv import load_dotenv
from Retrieval_ranking_answer.hybrid_retr import HybridRetriever

# Load environment variables from .env file
load_dotenv()
EVAL_SET_PATH = os.getenv("EVAL_SET_PATH")
LOG_FILE = os.getenv("EVAL_LOG_PATH_RUN2")

# Load the evaluation set from a .jsonl file
def load_eval_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Evaluate the retriever system
def evaluate_retriever(eval_data, retriever_fn, k=5):
    correct = 0
    total = len(eval_data)

    with open(LOG_FILE, "w", encoding="utf-8") as log_file:
        for entry in eval_data:
            query = entry["query"]
            gold_ids = set(str(gid) for gid in entry["gold_utterance_ids"])
            retrieved_ids = set(str(uid) for uid in retriever_fn(query, k=k))

            matched_ids = gold_ids & retrieved_ids
            if matched_ids:
                correct += 1
                status = f"✅ {query}"
            else:
                status = f"❌ {query}"

            expected_line = f"   Expected: {gold_ids}"
            got_line = f"   Got     : {retrieved_ids}"

            # Print to terminal
            print(status)
            print(expected_line)
            print(got_line)

            # Write to log file
            log_file.write(status + "\n")
            log_file.write(expected_line + "\n")
            log_file.write(got_line + "\n\n")

        accuracy = correct / total
        summary = f"\nAccuracy: {accuracy:.2%} ({correct}/{total})"
        print(summary)
        log_file.write(summary + "\n")

retriever = HybridRetriever()
# Use the HybridRetriever for evaluation
def hybrid_retriever(query, k=5):
    # Returns top k retrieved utterance IDs
    results = retriever.search(query)
    return [uid for uid, _, _ in results[:k]]

# Load eval set from path in .env and run evaluation
eval_data = load_eval_set(EVAL_SET_PATH)
evaluate_retriever(eval_data, hybrid_retriever)