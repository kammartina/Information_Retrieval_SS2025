import os
import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.utils.runs import Run
from colbert.utils.utils import print_message
from colbert.data import Queries
from chromadb import PersistentClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Settings
TOP_K = 10
CHROMA_PATH = os.getenv("CHROMA_DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "utterance_vectors_chroma")
COLBERT_MODEL = "colbert-ir/colbertv2.0"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # or any preferred LLM

# 1. Load LLM
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 2. Load ColBERT
print("Loading ColBERT...")
config = ColBERTConfig.load_from_hf(COLBERT_MODEL)
colbert = ColBERT.from_pretrained(COLBERT_MODEL, config=config).cuda()
colbert.eval()

# 3. Initialize Chroma
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

def rerank_with_colbert(query, candidates):
    with torch.no_grad():
        query_tok = colbert.query_tokenizer([query])
        query_vecs = colbert.encode_query(*query_tok)

        results = []
        for doc_id, doc_text in tqdm(candidates, desc="Reranking"):
            doc_tok = colbert.doc_tokenizer([doc_text])
            doc_vecs = colbert.encode_doc(*doc_tok)
            score = (query_vecs @ doc_vecs.transpose(0, 1)).max(dim=1).values.sum().item()
            results.append((doc_id, doc_text, score))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:TOP_K]

def retrieve_from_chroma(query):
    res = collection.query(query_texts=[query], n_results=50)
    return list(zip(res["ids"][0], res["documents"][0]))

def generate_answer(query, top_docs):
    context = "\n".join(f"- {doc}" for _, doc, _ in top_docs)
    prompt = f"""You are a helpful assistant. Based on the following context, answer the user's question.

Question: {query}

Context:
{context}

Answer:"""
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        print(f"\nRetrieving candidates for: {query}")
        candidates = retrieve_from_chroma(query)

        print("Reranking with ColBERT...")
        reranked = rerank_with_colbert(query, candidates)

        print("\nTop Reranked Results:")
        for i, (doc_id, doc, score) in enumerate(reranked, 1):
            print(f"{i}. [ID: {doc_id}] Score: {score:.2f} → {doc}")

        print("\nGenerating Answer from Reranked Results...")
        answer = generate_answer(query, reranked)
        print("\nAnswer:")
        print(answer)
        print("-" * 80)

if __name__ == "__main__":
    main()
