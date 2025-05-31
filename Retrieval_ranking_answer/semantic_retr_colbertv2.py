########
### this works for ChromaDB but NOT for SQLite retrieval
########

import os
import torch
import apsw
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class ColBERTReranker:
    def __init__(self):
        self.sqlite_path = os.getenv("SQLITE_DB_PATH")
        self.chroma_path = os.getenv("CHROMA_DB_PATH")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.model_name = os.getenv("COLBERT_MODEL")
        self.top_k = int(os.getenv("COLBERT_TOP_K"))
        self.rerank_k = int(os.getenv("COLBERT_RERANK_K"))

        self.conn = apsw.Connection(self.sqlite_path)
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=DefaultEmbeddingFunction()
        )

        print(f"Loading ColBERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    ## this function does not work
    def retrieve_candidates_sqlite(self, query: str):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT utterance_id, utterance_content 
            FROM utterances
            WHERE utterance_content LIKE ?
            LIMIT ?
        """, (f'%{query}%', self.top_k))
        return [(str(row[0]), row[1]) for row in cur]

    def retrieve_candidates_chroma(self, query: str):
        results = self.collection.query(query_texts=[query], n_results=self.top_k)
        return list(zip(results['ids'][0], results['documents'][0]))

    def encode(self, text: str):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            reps = outputs.last_hidden_state.squeeze(0)  # (seq_len, dim)

        return reps

    def late_interaction_score(self, query_vecs, doc_vecs):
        query_vecs = torch.nn.functional.normalize(query_vecs, p=2, dim=1)
        doc_vecs = torch.nn.functional.normalize(doc_vecs, p=2, dim=1)

        sim_matrix = torch.matmul(query_vecs, doc_vecs.T)
        max_sim = sim_matrix.max(dim=1).values
        return max_sim.sum().item()

    def rerank_with_colbert(self, query: str, candidates: list):
        query_vecs = self.encode(query)
        reranked = []

        for uid, text in tqdm(candidates, desc="Reranking"):
            doc_vecs = self.encode(text)
            score = self.late_interaction_score(query_vecs, doc_vecs)
            reranked.append((uid, text, score))

        return sorted(reranked, key=lambda x: x[2], reverse=True)[:self.rerank_k]

    def search(self, query: str):
        print(f"\nQuery: {query}")

        # Retrieve from both sources
        chroma_candidates = self.retrieve_candidates_chroma(query)
        sqlite_candidates = self.retrieve_candidates_sqlite(query)

        print(f"Chroma results: {len(chroma_candidates)} | SQLite results: {len(sqlite_candidates)}")

        ## THIS COMBINES THE RESULTS FROM BOTH SOURCES
        # Merge and deduplicate based on ID
        combined = {}
        for uid, text in chroma_candidates + sqlite_candidates:
            combined[uid] = text  # latest wins if duplicates

        print(f"Total unique candidates: {len(combined)}")
        return self.rerank_with_colbert(query, list(combined.items()))

if __name__ == "__main__":
    load_dotenv()
    reranker = ColBERTReranker()

    query = "blunt force trauma"
    results = reranker.search(query)

    print("\nTop reranked results:")
    for i, (uid, text, score) in enumerate(results, 1):
        print(f"{i}. [utterance_id: {uid}] Score: {score:.4f} → {text}")