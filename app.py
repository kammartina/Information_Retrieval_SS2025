### Using Streamlit for basic user interface for a hybrid RAG system
# 
# This code sets up a Streamlit app for a our hybrid RAG system with very basic user interface

import os
import sys
import streamlit as st

# Set page config at the very top
st.set_page_config(page_title="Hybrid RAG System", layout="wide")

# Add project root to sys.path for imports
sys.path.append(os.path.abspath("."))

# Try to import HybridRetriever with error handling (from "hybrid_retr.py")
try:
    from Retrieval_ranking_answer.hybrid_retr import HybridRetriever
    st.success("HybridRetriever imported successfully!")
    hybrid_retriever_instance = HybridRetriever()
    hybrid_retriever_instance.load_llm()
except Exception as e:
    st.error(f"Error importing HybridRetriever: {e}")

# Display the image
st.image("bones.jpg", channels="RGB", width=450)

# Streamlit app layout
st.title("🔍 Hybrid Bones-Series RAG System")

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