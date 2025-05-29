import os
import torch
import apsw
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class ColBERTReranker:
    def __init__(self):
        load_dotenv()
        self.sqlite_path = os.getenv("SQLITE_DB_PATH")
        self.chroma_path = os.getenv("CHROMA_DB_PATH")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.model_name = os.getenv("COLBERT_MODEL")
        self.top_k = int(os.getenv("TOP_K"))
        self.rerank_k = int(os.getenv("RERANK_K"))

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

    def encode(self, text: str, is_query=False):
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

        return reps  # (seq_len, dim)

    def late_interaction_score(self, query_vecs, doc_vecs):
    # Apply L2 normalization (ColBERT uses dot-product on normalized vectors → cosine sim)
        query_vecs = torch.nn.functional.normalize(query_vecs, p=2, dim=1)
        doc_vecs = torch.nn.functional.normalize(doc_vecs, p=2, dim=1)

        # Dot product for all token pairs → (query_len, doc_len)
        sim_matrix = torch.matmul(query_vecs, doc_vecs.T)

        # MaxSim over doc tokens (ColBERT late interaction)
        max_sim = sim_matrix.max(dim=1).values  # (query_len,)
        return max_sim.sum().item()  # scalar

    def rerank_with_colbert(self, query: str, candidates: list):
        query_vecs = self.encode(query)
        reranked = []

        for uid, text in tqdm(candidates, desc="Reranking"):
            doc_vecs = self.encode(text)
            score = self.late_interaction_score(query_vecs, doc_vecs)
            reranked.append((uid, text, score))

        return sorted(reranked, key=lambda x: x[2], reverse=True)[:self.rerank_k]

    def search(self, query: str, use_chroma: bool = True):
        print(f"\nQuery: {query}")
        if use_chroma:
            candidates = self.retrieve_candidates_chroma(query)
        else:
            candidates = self.retrieve_candidates_sqlite(query)

        print(f"Retrieved {len(candidates)} candidates. Now reranking with ColBERT-style interaction…")
        return self.rerank_with_colbert(query, candidates)


if __name__ == "__main__":
    reranker = ColBERTReranker()
    query = "blunt force trauma"
    results = reranker.search(query, use_chroma=True)

    print("\nTop reranked results:")
    for i, (uid, text, score) in enumerate(results, 1):
        print(f"{i}. [ID: {uid}] Score: {score:.4f} → {text}")
