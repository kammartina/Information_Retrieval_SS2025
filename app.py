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
    # Example usage to avoid "not accessed" warning
    hybrid_retriever_instance = HybridRetriever()
except Exception as e:
    st.error(f"Error importing HybridRetriever: {e}")

# Streamlit app layout
st.title("🔍 Hybrid RAG System")

query = st.text_input("Enter your query:")

if query:
    st.write(f"You asked: {query}")
else:
    st.info("Please enter a query above to get started.")
