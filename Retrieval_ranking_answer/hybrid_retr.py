#from semantic_retr_colbertv2 import ColBERTReranker
from Retrieval_ranking_answer.semantic_retr_colbertv2 import ColBERTReranker
#from lexical_retr_spladev3 import SpladeRetriever
from Retrieval_ranking_answer.lexical_retr_spladev3 import SpladeRetriever
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
        self.llm_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.6")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.llm_model.eval()

    def generate_answer(self, query: str, retrieved_passages: list):
        # Combine top contexts into one prompt
        context = "\n".join([f"{i+1}. {text}" for i, (_, text, _) in enumerate(retrieved_passages)])
        
        ## VERY BASIC, ORIGINAL (FIRST TRY) PROMPT
        prompt = f"""<|system|>\nYou are an intelligent assistant helping with information retrieval.\n<|user|>\nPlease synthesize an answer from the following passages to answer the question:\n\n{context}\n\nQuestion: {query}\n<|assistant|>"""

        ## STILL WORKING ON THIS...

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        with torch.no_grad():
            output = self.llm_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )
        response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip()
    ##


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