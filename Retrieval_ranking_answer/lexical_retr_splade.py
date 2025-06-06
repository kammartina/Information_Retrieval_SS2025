# lexical_retr_spladev3.py

import os
import torch
import apsw
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForMaskedLM


class SpladeRetriever:
    def __init__(self):
        self.sqlite_path = os.getenv("SQLITE_DB_PATH")
        self.model_name = os.getenv("SPLADE_MODEL")
        self.top_k = int(os.getenv("SPLADE_TOP_K"))

        print(f"Loading SPLADE model: {self.model_name}")

        self.conn = apsw.Connection(self.sqlite_path)

        ## Use token from huggingface-cli login automatically - I used, it did not work, I changed the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.documents = self.load_documents()

    # executes a SQL query to fetch all doc IDs and their content
    # returns a dictionary mapping doc IDs to their content
    def load_documents(self):
        cur = self.conn.cursor()
        cur.execute("SELECT utterance_id, utterance_content FROM utterances")
        return {str(row[0]): row[1] for row in cur}

    # encodes the input text into a sparse vector using the SPLADE model (truncating/padding to 256 tokens)
    # runs the text through the SPLADE model to get logits
    # max pooling over the sequence dimension to get a single sparse bag-of-words vector (vocab-size dimensionality)
    # applies ReLU to get non-negative scores 
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits.squeeze(0)
            scores, _ = logits.max(dim=0)
            sparse_vector = torch.nn.functional.relu(scores).cpu()
        return sparse_vector

    def search(self, query):
        print(f"Running SPLADE search for query: {query}")
        query_vector = self.encode(query)
        results = []
        
        # encodes the doc text into a sparse vector
        # computes the dot product between the query vector and each document vector
        # stores the doc ID, text, and score in results
        # returns the top-k results sorted by score
        for doc_id, text in tqdm(self.documents.items(), desc="Scoring documents"):
            doc_vector = self.encode(text)
            score = torch.dot(query_vector, doc_vector).item()
            results.append((doc_id, text, score))
        top_results = sorted(results, key=lambda x: x[2], reverse=True)[:self.top_k]
        return top_results


if __name__ == "__main__":
    load_dotenv()
    retriever = SpladeRetriever()
    query = "blunt force trauma"
    results = retriever.search(query)

    print("\nTop SPLADE results:")
    for i, (doc_id, text, score) in enumerate(results, 1):
        print(f"{i}. [ID: {doc_id}] Score: {score:.4f} → {text}")
