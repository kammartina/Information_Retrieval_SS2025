from Retrieval_ranking_answer.semantic_retr_colbert import ColBERTReranker
from Retrieval_ranking_answer.lexical_retr_splade import SpladeRetriever

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv


class HybridRetriever:
    def __init__(self):
        self.colbert = ColBERTReranker()
        self.splade = SpladeRetriever()
        #self.load_llm()  # DO NOT load the LLM here for GPU memory efficiency

    def search(self, query: str):
        print(f"\nQuery: {query}")

        splade_results = self.splade.search(query)
        colbert_candidates = self.colbert.retrieve_candidates_chroma(query)

        print(f"SPLADE results: {len(splade_results)} | Chroma results: {len(colbert_candidates)}")

        combined = {}
        for uid, text, _ in splade_results:
            if len(text.strip()) > 5:  # Filter super short lines
                combined[uid] = text
        for uid, text in colbert_candidates:
            combined[uid] = text

        print(f"Total unique candidates: {len(combined)}")

        reranked_results = self.colbert.rerank_with_colbert(query, list(combined.items()))
        return reranked_results

    def load_llm(self):
        print("Loading TinyLlama model...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.llm_model.eval()

    
    def generate_answer(self, query: str, retrieved_passages: list):
        context = "\n".join([f"{i+1}. {text}" for i, (_, text, _) in enumerate(retrieved_passages)])
        
        ## this is a manually crafted prompt, but many modern open-source chat models use a structured prompt format
        prompt = f"""
        <|system|>\n
        You are a knowledgeable assistant specialized in the TV series *Bones*. Your role is to answer questions strictly based on the provided passages. Do not use outside knowledge. If the passages do not fully answer the question, state that directly.\n

        <|user|>\n
        Please carefully review the following passages from the *Bones* TV script. Then synthesize a clear, concise, and informative answer to the question. Use direct quotes or paraphrasing from the passages to support your answer whenever possible. If the information is not fully available in the passages, acknowledge that.\n

        Passages:\n
        {context}

        \nQuestion: {query}\n
        <|assistant|>"""

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        with torch.no_grad():
            output = self.llm_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.2,
                top_k=20,
                top_p=0.8,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )
        response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip()


'''
if __name__ == "__main__":
    load_dotenv()
    retriever = HybridRetriever()
    retriever.load_llm() # load TinyLama here once

    # here is the query defined
    query = "Who is Booth and where does this person work?"
    results = retriever.search(query)

    print("\nTop reranked results:")
    for i, (uid, text, score) in enumerate(results, 1):
        print(f"{i}. [utterance_id: {uid}] Score: {score:.4f} → {text}")

    answer = retriever.generate_answer(query, results)
    print("\n🧠 Generated Answer:")
    print(answer)
'''