from colbert.modeling.colbert import ColBERT
from colbert.infra import ColBERTConfig
import torch
from typing import List, Tuple
from dotenv import load_dotenv
import os
import chromadb
import sqlite3


class ColBERTReranker:
    def __init__(self, colbert):
        self.colbert = os.getenv("COLBERT_MODEL")
        self.config = ColBERTConfig(colbert=colbert)

        self.model = ColBERT.from_pretrained(colbert, config=self.config)
        self.model.eval()

    def encode_query(self, query: str):
        with torch.no_grad():
            Q = self.model.queryFromText([query])
        return Q

    def encode_documents(self, documents: List[str]):
        with torch.no_grad():
            D = self.model.docFromText(documents)
        return D

    def maxsim_score(self, Q, D_batch):
        Q = Q.squeeze(0)  # shape: [q_tokens, dim]
        scores = []
        for D in D_batch:
            sim_matrix = torch.einsum('qd,nd->qn', Q, D)
            score = sim_matrix.max(dim=1).values.sum()
            scores.append(score.item())
        return scores

    def rerank(self, query: str, candidates: List[Tuple[str, str]], top_k: int = 5):
        Q = self.encode_query(query)
        doc_texts = [text for _, text in candidates]
        D = self.encode_documents(doc_texts)

        scores = self.maxsim_score(Q, D)

        reranked = sorted(
            zip([doc_id for doc_id, _ in candidates], scores, doc_texts),
            key=lambda x: -x[1]
        )

        return reranked[:top_k]
    
    def fetch_candidates_from_chroma(query: str, top_k: int = 10) -> List[Tuple[str, str]]:
        client = chromadb.Client()
        
        # Replace 'my_collection' with your real collection name
        collection = client.get_collection("chroma.sqlite3")
        
        # Retrieve top_k documents semantically
        results = collection.query(query_texts=[query], n_results=top_k)

        candidates = []
        for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
            candidates.append((doc_id, doc))
        return candidates
    
    def fetch_from_sqlite(query_vector: List[float], db_path: str = "vectors.db", top_k: int = 5) -> List[Tuple[str, str]]:
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        conn.execute("SELECT load_extension('vector0')")  # Load VectorLite

        cursor = conn.cursor()
        placeholders = ",".join("?" * len(query_vector))
        sql = f"""
            SELECT id, text
            FROM documents
            ORDER BY vss_search_l2(vec, vss_vector({placeholders}))
            LIMIT ?
        """
        cursor.execute(sql, (*query_vector, top_k))
        rows = cursor.fetchall()
        conn.close()
        return [(row[0], row[1]) for row in rows]


def main():
    load_dotenv()
    # Simulated results from your vector DB
    candidate_docs = [
        ("doc1", "Vector search enables semantic similarity using dense vectors."),
        ("doc2", "SQLite is a lightweight database used in mobile apps."),
        ("doc3", "ColBERT enables late interaction for better ranking."),
        ("doc4", "VectorLite is an extension to SQLite for fast vector search."),
    ]

    query = "How does semantic vector search work?"

    reranker = ColBERTReranker()
    top_docs = reranker.rerank(query, candidate_docs, top_k=3)

    print(f"\nTop results for query: \"{query}\"\n")
    for rank, (doc_id, score, text) in enumerate(top_docs, 1):
        print(f"{rank}. [{doc_id}] Score: {score:.4f}")
        print(f"   {text}\n")


if __name__ == "__main__":
    main()