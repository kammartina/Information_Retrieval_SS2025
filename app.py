import os
import sys
import streamlit as st

# Set page config at the very top
st.set_page_config(page_title="Hybrid RAG System", layout="wide")

# Add project root to sys.path for imports
sys.path.append(os.path.abspath("."))

# Try to import HybridRetriever with error handling
try:
    from Retrieval_ranking_answer.hybrid_retr import HybridRetriever
    st.success("HybridRetriever imported successfully!")
    hybrid_retriever_instance = HybridRetriever()
    hybrid_retriever_instance.load_llm()
except Exception as e:
    st.error(f"Error importing HybridRetriever: {e}")

# Streamlit app layout
st.title("🔍 Hybrid RAG System")

query = st.text_input("Enter your query:")

if query:
    st.write(f"You asked: {query}")
    
    with st.spinner("Retrieving and generating answer..."):
        try:
            results = hybrid_retriever_instance.search(query)
            if results:
                answer = hybrid_retriever_instance.generate_answer(query, results)
                st.subheader("🧠 Answer:")
                st.write(answer)
            else:
                st.warning("No results found. Please try another query.")
        except Exception as e:
            st.error(f"Error generating answer: {e}")
else:
    st.info("Please enter a query above to get started.")
